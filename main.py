"""
Telegram-бот: щоденний аналіз робочого логу (chats/emails/calls) →
виявлення нових завдань і оновлень статусу існуючих →
послідовне підтвердження користувачем → запис у сховище.

Архітектура — див. документ "Архітектура_системи_завдань.md".
Webhook-інфраструктура (Flask + python-telegram-bot, окремий asyncio loop)
збережена з попередньої версії бота (репозиторій buyer-del/bot2) без змін,
оскільки вона вже надійно працює на Render.
"""

import os
import json
import logging
import asyncio
import threading

from flask import Flask, request
from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.error import BadRequest

from log_filter import build_selection
from task_analyzer import analyze_day, load_projects
from report import generate_report
import sheets_api as storage

# =========================
# ЛОГИ
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# ЗМІННІ СЕРЕДОВИЩА
# =========================
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # https://.../
PORT = int(os.getenv("PORT", 10000))
PROJECTS_FILE = os.getenv("PROJECTS_FILE", "projects.json")
CONTACTS_FILE = os.getenv("CONTACTS_FILE", "contacts.json")

if not TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN не задано")
if not WEBHOOK_URL or not WEBHOOK_URL.startswith("https://"):
    raise SystemExit("WEBHOOK_URL не задано або не HTTPS")

# =========================
# Flask
# =========================
flask_app = Flask(__name__)


@flask_app.route("/", methods=["GET", "HEAD"])
def root():
    return "ok", 200


# =========================
# Telegram Application
# =========================
bot_app = Application.builder().token(TOKEN).build()


# -------------------------
# ДОПОМІЖНЕ: керування чергою підтверджень
# -------------------------
def _queue(context: ContextTypes.DEFAULT_TYPE) -> list[dict]:
    """Черга елементів, що очікують підтвердження користувача (проєкти, потім завдання)."""
    return context.user_data.setdefault("confirm_queue", [])


def _pending(context: ContextTypes.DEFAULT_TYPE) -> dict | None:
    return context.user_data.get("pending_item")


async def _remove_old_keyboard(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.user_data.get("last_kb_chat_id")
    msg_id = context.user_data.get("last_kb_message_id")
    if not chat_id or not msg_id:
        return
    try:
        await context.bot.edit_message_reply_markup(chat_id=chat_id, message_id=msg_id, reply_markup=None)
    except BadRequest:
        pass
    except Exception as e:
        logger.exception("Не вдалося прибрати старі кнопки: %s", e)


async def _send_with_keyboard(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, keyboard: InlineKeyboardMarkup):
    await _remove_old_keyboard(context)
    sent = await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard)
    context.user_data["last_kb_chat_id"] = sent.chat_id
    context.user_data["last_kb_message_id"] = sent.message_id


# -------------------------
# ФОРМУВАННЯ ЧЕРГИ ПІДТВЕРДЖЕНЬ (Крок 4 архітектури)
# -------------------------
def _build_confirm_queue(analysis: dict) -> list[dict]:
    """
    Формує чергу підтверджень. Тільки два типи елементів:
    - new_task: нове завдання (project_id може бути конкретним або "other")
    - task_update: оновлення статусу існуючого завдання

    Жодних "нових проєктів" в автоматичному потоці — проєкти ведуться
    вручну через projects.json і команду синхронізації.
    Завдання без project_id автоматично отримують project_id="other".
    """
    queue = []

    new_tasks = analysis.get("new_tasks", [])
    task_updates = analysis.get("task_updates", [])

    # Відкидаємо оновлення де task_id не числовий — Gemini міг підставити назву замість id
    task_updates = [u for u in task_updates if str(u.get("task_id", "")).isdigit()]

    # Нормалізуємо project_id: null або рядок "null" → "other"
    for t in new_tasks:
        if not t.get("project_id") or t.get("project_id") == "null":
            t["project_id"] = "other"

    new_task_items = [{"type": "new_task", "payload": t} for t in new_tasks]
    update_items = [{"type": "task_update", "payload": u} for u in task_updates]

    # Чергуємо нові завдання і оновлення статусу
    max_len = max(len(new_task_items), len(update_items), 1)
    for i in range(max_len):
        if i < len(new_task_items):
            queue.append(new_task_items[i])
        if i < len(update_items):
            queue.append(update_items[i])

    return queue


def _kb_new_task(payload: dict, projects: list[dict]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("✅ Додати як відкрите", callback_data="nt_accept")],
        [InlineKeyboardButton("🔄 Додати з оновленим статусом", callback_data="nt_accept_with_status")],
        [InlineKeyboardButton("🔗 Змінити проєкт", callback_data="nt_change_project")],
        [InlineKeyboardButton("❌ Відхилити", callback_data="nt_reject")],
    ]
    return InlineKeyboardMarkup(rows)


def _kb_task_update() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Підтвердити оновлення", callback_data="tu_accept")],
        [InlineKeyboardButton("❌ Відхилити", callback_data="tu_reject")],
    ])


def _kb_project_list(projects: list[dict]) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(p["full_name"], callback_data=f"proj_pick_{p['id']}")]
        for p in projects
    ]
    rows.append([InlineKeyboardButton("« Назад", callback_data="proj_pick_cancel")])
    return InlineKeyboardMarkup(rows)


async def _present_next(update_or_chat_id, context: ContextTypes.DEFAULT_TYPE):
    """Бере наступний елемент черги і показує його користувачу на підтвердження."""
    queue = _queue(context)

    if isinstance(update_or_chat_id, int):
        chat_id = update_or_chat_id
    else:
        chat_id = update_or_chat_id.effective_chat.id

    if not queue:
        context.user_data["pending_item"] = None

        # Читаємо вибірку з сесії — не тримаємо в пам'яті процесу
        session = await asyncio.to_thread(storage.load_session)
        report_date = session["log_date"] if session else ""
        selection = session["selection"] if session else []

        if selection:
            await asyncio.to_thread(
                storage.save_session,
                "report_pending", selection, {}, [], None, report_date
            )
            await context.bot.send_message(chat_id=chat_id, text="✅ Усі пропозиції за сьогодні опрацьовано.\n📝 Складаю звіт за день…")
            try:
                report_tasks = await asyncio.to_thread(storage.get_open_tasks)
                report_text = await asyncio.to_thread(
                    generate_report,
                    selection,
                    report_date,
                    report_tasks,
                )
                # Крок 4: зберігаємо тільки текст звіту, вибірку не дублюємо
                await asyncio.to_thread(
                    storage.save_session,
                    "report_ready", [], {}, [], report_text, report_date
                )
                await context.bot.send_message(chat_id=chat_id, text=report_text)
                await asyncio.to_thread(storage.clear_session)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="Готово. Звіт сформовано.",
                    reply_markup=ReplyKeyboardMarkup(
                        [[KeyboardButton("/bag")]],
                        resize_keyboard=True,
                        one_time_keyboard=True
                    )
                )
            except Exception as e:
                logger.exception("Помилка генерації звіту: %s", e)
                await context.bot.send_message(chat_id=chat_id, text=f"❌ Не вдалося згенерувати звіт: {e}")
        else:
            await asyncio.to_thread(storage.clear_session)
            await context.bot.send_message(
                chat_id=chat_id,
                text="✅ Усі пропозиції за сьогодні опрацьовано.",
                reply_markup=ReplyKeyboardMarkup(
                    [[KeyboardButton("/bag")]],
                    resize_keyboard=True,
                    one_time_keyboard=True
                )
            )
        return

    # Зберігаємо поточний стан черги (Крок 3)
    # Вибірку не передаємо — вона вже збережена в сесії з Кроку 1-2
    session = await asyncio.to_thread(storage.load_session)
    if session:
        await asyncio.to_thread(
            storage.save_session,
            "queue",
            session.get("selection", []),
            {},
            queue,
            None,
            session.get("log_date", ""),
        )

    item = queue.pop(0)
    context.user_data["pending_item"] = item
    payload = item["payload"]

    if item["type"] == "new_task":
        proj = payload.get("project_id") or "other"
        projects = storage.get_projects()
        proj_name = next((p["full_name"] for p in projects if p["id"] == proj), proj)
        source_id = payload.get("source_id", "")
        text = (
            f"📌 Нове завдання:\n\n"
            f"{payload.get('title')}\n"
            f"Проєкт: {proj_name}\n"
            f"Від: {payload.get('source_sender', '')}\n"
            f"Джерело [{source_id}]: \"{payload.get('source_text', '')}\""
        )
        await _send_with_keyboard(context, chat_id, text, _kb_new_task(payload, projects))

    elif item["type"] == "task_update":
        task = storage.get_task_by_id(int(payload["task_id"])) if str(payload.get("task_id", "")).isdigit() else None
        task_title = task["title"] if task else payload.get("title") or f"id={payload.get('task_id')}"
        source_id = payload.get("source_id", "")
        text = (
            f"🔄 Оновлення статусу:\n\n"
            f"Завдання: {task_title}\n"
            f"Новий статус: {payload.get('new_status')}\n"
            f"Коментар: {payload.get('comment', '')}\n"
            f"Від: {payload.get('source_sender', '')}\n"
            f"Джерело [{source_id}]: \"{payload.get('source_text', '')}\""
        )
        await _send_with_keyboard(context, chat_id, text, _kb_task_update())


# -------------------------
# КОМАНДИ
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Бот готовий. Надішли JSON-файл логу (кнопка 'поділитися' з додатку логування), "
        "і я проаналізую день — знайду нові завдання та оновлення статусу."
    )


async def resume_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /bag — відновлення перерваної сесії."""
    chat_id = update.effective_chat.id
    session = await asyncio.to_thread(storage.load_session)

    if not session:
        await update.message.reply_text(
            "Незавершених сесій немає.",
            reply_markup=ReplyKeyboardMarkup([[KeyboardButton("/bag")]], resize_keyboard=True, one_time_keyboard=True)
        )
        return

    stage = session["stage"]

    if stage == "filtered":
        # Вибірка є але Gemini ще не викликали
        await update.message.reply_text("⏳ Відновлюю аналіз дня…")
        selection = session["selection"]
        log_date = session["log_date"]
        projects = load_projects(PROJECTS_FILE) if os.path.exists(PROJECTS_FILE) else storage.get_projects()
        open_tasks = storage.get_open_tasks()
        try:
            analysis = await asyncio.to_thread(analyze_day, selection, projects, open_tasks)
            queue = _build_confirm_queue(analysis)
            await asyncio.to_thread(storage.save_session, "queue", selection, analysis, queue, None, log_date)
            context.user_data["confirm_queue"] = queue
            # Вибірка збережена в сесії — не дублюємо в user_data
            n_new = len(analysis.get("new_tasks", []))
            n_upd = len(analysis.get("task_updates", []))
            await update.message.reply_text(f"Знайдено: {n_new} нових завдань, {n_upd} оновлень. Продовжую підтвердження…")
            await _present_next(update, context)
        except Exception as e:
            await update.message.reply_text(f"❌ Помилка аналізу: {e}")

    elif stage == "queue":
        # Черга підтверджень перервалась
        queue = session["queue"]
        if not queue:
            await update.message.reply_text("Черга порожня.")
            return
        context.user_data["confirm_queue"] = queue
        # Вибірка береться з сесії при генерації звіту — не в user_data
        await update.message.reply_text(f"⏳ Відновлюю підтвердження. Залишилось: {len(queue)} пунктів.")
        await _present_next(update, context)

    elif stage == "report_pending":
        # Черга завершена але звіт ще не сформовано
        await update.message.reply_text("⏳ Формую звіт…")
        selection = session["selection"]
        log_date = session["log_date"]
        try:
            report_tasks = storage.get_open_tasks()
            report_text = await asyncio.to_thread(generate_report, selection, log_date, report_tasks)
            await asyncio.to_thread(storage.save_session, "report_ready", selection, {}, [], report_text, log_date)
            await update.message.reply_text(report_text)
            await asyncio.to_thread(storage.clear_session)
            await context.bot.send_message(
                chat_id=chat_id,
                text="Готово.",
                reply_markup=ReplyKeyboardMarkup([[KeyboardButton("/bag")]], resize_keyboard=True, one_time_keyboard=True)
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Помилка формування звіту: {e}")

    elif stage == "report_ready":
        # Звіт є але не був надісланий
        await update.message.reply_text(session["report_text"])
        await asyncio.to_thread(storage.clear_session)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Готово.",
            reply_markup=ReplyKeyboardMarkup([[KeyboardButton("/bag")]], resize_keyboard=True, one_time_keyboard=True)
        )


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong ✅")


async def debug_log(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Надсилає файл gemini_debug.log прямо в чат для діагностики."""
    debug_path = os.environ.get("DEBUG_LOG", "gemini_debug.log")
    if not os.path.exists(debug_path):
        await update.message.reply_text("Файл debug-логу ще не створено — спочатку надішли лог для аналізу.")
        return
    try:
        with open(debug_path, "rb") as f:
            await update.message.reply_document(document=f, filename="gemini_debug.log")
    except Exception as e:
        await update.message.reply_text(f"Помилка при відправці логу: {e}")


async def clear_debug_log(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очищає debug-лог."""
    debug_path = os.environ.get("DEBUG_LOG", "gemini_debug.log")
    try:
        open(debug_path, "w").close()
        await update.message.reply_text("✅ Debug-лог очищено.")
    except Exception as e:
        await update.message.reply_text(f"Помилка: {e}")


# -------------------------
# ПРИЙОМ ЛОГУ (JSON-файл)
# -------------------------
async def log_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if not doc.file_name.endswith(".json"):
        return  # не наш файл — ігноруємо мовчки (можливо документ-аудіо обробляється іншим хендлером)

    await update.message.reply_text("📥 Лог отримано, аналізую день…")

    try:
        file = await doc.get_file()
        local_path = "incoming_log.json"
        await file.download_to_drive(local_path)

        with open(local_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        projects = load_projects(PROJECTS_FILE) if os.path.exists(PROJECTS_FILE) else storage.get_projects()
        open_tasks = storage.get_open_tasks()

        selection = await asyncio.to_thread(
            build_selection,
            log_data, CONTACTS_FILE, PROJECTS_FILE, open_tasks,
        )

        # Крок 1: зберігаємо відфільтровану вибірку
        await asyncio.to_thread(
            storage.save_session,
            "filtered", selection, None, None, None, log_data.get("export_date", "")
        )

        analysis = await asyncio.to_thread(
            analyze_day,
            selection, projects, open_tasks,
        )

        # Крок 2: зберігаємо результат аналізу Gemini
        queue = _build_confirm_queue(analysis)
        await asyncio.to_thread(
            storage.save_session,
            "queue", selection, analysis, queue, None, log_data.get("export_date", "")
        )

        context.user_data["confirm_queue"] = queue
        # Не зберігаємо вибірку в user_data — вона вже є в сесії (Google Таблиця)
        # Це знижує споживання пам'яті під час підтвердження

        n_new = len(analysis.get("new_tasks", []))
        n_upd = len(analysis.get("task_updates", []))
        await update.message.reply_text(
            f"Готово. Знайдено: {n_new} нових завдань, {n_upd} оновлень статусу.\nПочинаю підтвердження…"
        )

        await _present_next(update, context)

    except Exception as e:
        logger.exception("Помилка обробки логу: %s", e)
        await update.message.reply_text(f"❌ Помилка обробки логу: {e}")


# -------------------------
# КНОПКИ (Крок 4 — послідовне підтвердження)
# -------------------------
async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data
    await q.answer()

    # --- Вибір статусу — обробляємо до перевірки pending_item ---
    # бо може викликатись і з черги підтверджень, і з меню керування завданнями
    if data.startswith("set_status_"):
        rest = data[len("set_status_"):]
        task_id = rest.split("_")[-1]
        new_status = rest[:-(len(task_id)+1)]
        await asyncio.to_thread(storage.update_task_status, task_id, new_status)
        await _remove_old_keyboard(context)
        await q.message.reply_text(f"✅ Статус: {new_status}")
        # Якщо є активна черга підтверджень — продовжуємо її
        # Якщо ні — повертаємось до меню
        if _queue(context) or _pending(context):
            await _present_next(update, context)
        else:
            await _show_main_menu(update.effective_chat.id, context)
        return

    item = _pending(context)
    if not item:
        await q.message.reply_text("Немає активної пропозиції для підтвердження.")
        return

    payload = item["payload"]

    # --- Нове завдання: додати як відкрите ---
    if data == "nt_accept":
        storage.add_task(
            title=payload.get("title"),
            project_id=payload.get("project_id", "other"),
            source_text=payload.get("source_text", ""),
            source_sender=payload.get("source_sender", ""),
            status="відкрито",
        )
        await _remove_old_keyboard(context)
        await q.message.reply_text(f"✅ Завдання додано як відкрите: {payload.get('title')}")
        await _present_next(update, context)
        return

    # --- Нове завдання: додати з вибором статусу ---
    if data == "nt_accept_with_status":
        task = storage.add_task(
            title=payload.get("title"),
            project_id=payload.get("project_id", "other"),
            source_text=payload.get("source_text", ""),
            source_sender=payload.get("source_sender", ""),
            status="відкрито",
        )
        await _remove_old_keyboard(context)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Завдання додано. Обери статус для: {payload.get('title')}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🟡 В роботі", callback_data=f"set_status_в роботі_{task['id']}")],
                [InlineKeyboardButton("⏸️ Відкладено", callback_data=f"set_status_відкладено_{task['id']}")],
                [InlineKeyboardButton("🟢 Виконано", callback_data=f"set_status_виконано_{task['id']}")],
            ])
        )
        return

    if data == "nt_reject":
        await _remove_old_keyboard(context)
        await q.message.reply_text("❌ Відхилено.")
        await _present_next(update, context)
        return

    # --- Нове завдання: змінити проєкт ---
    if data == "nt_change_project":
        projects = storage.get_projects()
        if not projects:
            await q.message.reply_text("Список проєктів порожній.")
            return
        await _remove_old_keyboard(context)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Обери проєкт для цього завдання:",
            reply_markup=_kb_project_list(projects),
        )
        return

    if data.startswith("proj_pick_"):
        picked_id = data[len("proj_pick_"):]
        item = _pending(context)
        if item:
            item["payload"]["project_id"] = picked_id
        await _remove_old_keyboard(context)
        # Показуємо завдання знову з оновленим проєктом
        projects = storage.get_projects()
        payload = item["payload"]
        proj_name = next((p["full_name"] for p in projects if p["id"] == picked_id), picked_id)
        text = (
            f"📌 Нове завдання (проєкт оновлено):\n\n"
            f"{payload.get('title')}\n"
            f"Проєкт: {proj_name}\n"
            f"Джерело: \"{payload.get('source_text', '')}\"\n"
            f"Від: {payload.get('source_sender', '')}"
        )
        await _send_with_keyboard(context, update.effective_chat.id, text, _kb_new_task(payload, projects))
        return

    if data == "proj_pick_cancel":
        # Повертаємось до завдання без змін
        item = _pending(context)
        if item:
            projects = storage.get_projects()
            payload = item["payload"]
            proj = payload.get("project_id", "other")
            proj_name = next((p["full_name"] for p in projects if p["id"] == proj), proj)
            text = (
                f"📌 Нове завдання:\n\n"
                f"{payload.get('title')}\n"
                f"Проєкт: {proj_name}\n"
                f"Джерело: \"{payload.get('source_text', '')}\"\n"
                f"Від: {payload.get('source_sender', '')}"
            )
            await _send_with_keyboard(context, update.effective_chat.id, text, _kb_new_task(payload, projects))
        return

    # --- Оновлення статусу: підтвердити ---
    if data == "tu_accept":
        task_id = payload.get("task_id")
        # Захист: task_id має бути числом. Якщо Gemini повернув текст — ігноруємо оновлення
        if str(task_id).isdigit():
            storage.update_task_status(
                int(task_id),
                payload.get("new_status", "in_progress"),
                payload.get("comment", ""),
            )
            await _remove_old_keyboard(context)
            await q.message.reply_text("✅ Статус оновлено.")
        else:
            await _remove_old_keyboard(context)
            await q.message.reply_text(f"⚠️ Не вдалося оновити статус: завдання з id={task_id} не знайдено в таблиці.")
        await _present_next(update, context)
        return

    if data == "tu_reject":
        await _remove_old_keyboard(context)
        await q.message.reply_text("❌ Відхилено.")
        await _present_next(update, context)
        return


# -------------------------
# ТЕКСТ (використовується для введення нової назви проєкту)
# -------------------------
async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Обробка введення тексту для нового ручного завдання
    state = context.user_data.get("manual_task_state")

    if state == "awaiting_title":
        context.user_data["manual_task_title"] = update.message.text.strip()
        context.user_data["manual_task_state"] = "awaiting_project"
        projects = storage.get_projects()
        rows = [[InlineKeyboardButton(p["full_name"], callback_data=f"mt_proj_{p['id']}")]
                for p in projects]
        await update.message.reply_text(
            "Обери проєкт:",
            reply_markup=InlineKeyboardMarkup(rows)
        )
        return

    if state == "awaiting_comment":
        task_id = context.user_data.get("editing_task_id")
        comment = update.message.text.strip()
        if task_id:
            await asyncio.to_thread(storage.update_task_comment, task_id, comment)
            await update.message.reply_text("✅ Коментар оновлено.")
        context.user_data["manual_task_state"] = None
        context.user_data["editing_task_id"] = None
        await _show_main_menu(update.effective_chat.id, context)
        return

    await update.message.reply_text(
        "Надішли JSON-файл логу командою 'поділитися' з додатку логування."
    )


# ─── Головне меню після прокидання ──────────────────────────────────────────

async def _show_main_menu(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    await _remove_old_keyboard(context)
    await context.bot.send_message(
        chat_id=chat_id,
        text="Що робимо?",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📋 Актуальні завдання", callback_data="menu_open")],
            [InlineKeyboardButton("✅ Виконані завдання", callback_data="menu_done")],
            [InlineKeyboardButton("✏️ Нове завдання", callback_data="menu_new_task")],
        ])
    )


async def _show_task_card(chat_id: int, context: ContextTypes.DEFAULT_TYPE, task: dict, mode: str):
    """Показує картку завдання з кнопками. mode: 'open' або 'done'"""
    tasks = context.user_data.get("task_list", [])
    idx = context.user_data.get("task_idx", 0)
    total = len(tasks)

    projects = storage.get_projects()
    proj_name = next((p["full_name"] for p in projects if p["id"] == task.get("project_id")), task.get("project_id") or "—")

    text = (
        f"{'📋' if mode == 'open' else '✅'} [{idx+1}/{total}]\n\n"
        f"📌 {task['title']}\n"
        f"Проєкт: {proj_name}\n"
        f"Статус: {task['status']}\n"
        f"Дата: {task.get('created_date', '')}\n"
        f"Від: {task.get('source_sender', '')}\n"
        f"Коментар: {task.get('comment') or '—'}"
    )

    nav_row = []
    if idx > 0:
        nav_row.append(InlineKeyboardButton("◀️", callback_data=f"task_nav_prev_{mode}"))
    if idx < total - 1:
        nav_row.append(InlineKeyboardButton("▶️", callback_data=f"task_nav_next_{mode}"))

    if mode == "open":
        action_rows = [
            [InlineKeyboardButton("🔄 Змінити статус", callback_data=f"task_status_{task['id']}")],
            [InlineKeyboardButton("✏️ Редагувати коментар", callback_data=f"task_comment_{task['id']}")],
        ]
    else:
        action_rows = [
            [InlineKeyboardButton("🔄 Змінити статус", callback_data=f"task_status_{task['id']}")],
            [InlineKeyboardButton("🗑️ Видалити", callback_data=f"task_delete_{task['id']}")],
        ]

    keyboard = []
    if nav_row:
        keyboard.append(nav_row)
    keyboard.extend(action_rows)
    keyboard.append([InlineKeyboardButton("◀️ Назад", callback_data="menu_back")])

    await _send_with_keyboard(context, chat_id, text, InlineKeyboardMarkup(keyboard))


# ─── Обробники кнопок меню і завдань ────────────────────────────────────────

async def menu_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data
    await q.answer()
    chat_id = update.effective_chat.id

    # Прокидайся → головне меню
    if data == "wake_up":
        await _show_main_menu(chat_id, context)
        return

    # Головне меню


    if data == "menu_open":
        tasks = await asyncio.to_thread(storage.get_open_tasks)
        if not tasks:
            await q.message.reply_text("Актуальних завдань немає.")
            return
        context.user_data["task_list"] = tasks
        context.user_data["task_idx"] = 0
        context.user_data["task_mode"] = "open"
        await _show_task_card(chat_id, context, tasks[0], "open")
        return

    if data == "menu_done":
        tasks = await asyncio.to_thread(storage.get_done_tasks)
        if not tasks:
            await q.message.reply_text("Виконаних завдань немає.")
            return
        context.user_data["task_list"] = tasks
        context.user_data["task_idx"] = 0
        context.user_data["task_mode"] = "done"
        await _show_task_card(chat_id, context, tasks[0], "done")
        return

    if data == "menu_back":
        # Очищаємо task_list з пам'яті при виході з перегляду завдань
        context.user_data.pop("task_list", None)
        context.user_data.pop("task_idx", None)
        context.user_data.pop("task_mode", None)
        await _show_main_menu(chat_id, context)
        return

    if data == "menu_new_task":
        context.user_data["manual_task_state"] = "awaiting_title"
        await q.message.reply_text("Введи назву нового завдання:")
        return

    # Навігація між завданнями
    if data.startswith("task_nav_"):
        parts = data.split("_")
        direction = parts[2]  # prev або next
        mode = parts[3]
        tasks = context.user_data.get("task_list", [])
        idx = context.user_data.get("task_idx", 0)
        if direction == "prev" and idx > 0:
            idx -= 1
        elif direction == "next" and idx < len(tasks) - 1:
            idx += 1
        context.user_data["task_idx"] = idx
        await _show_task_card(chat_id, context, tasks[idx], mode)
        return

    # Зміна статусу
    if data.startswith("task_status_"):
        task_id = data[len("task_status_"):]
        context.user_data["editing_task_id"] = task_id
        await _remove_old_keyboard(context)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Обери новий статус:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔵 Відкрито", callback_data=f"set_status_відкрито_{task_id}")],
                [InlineKeyboardButton("🟡 В роботі", callback_data=f"set_status_в роботі_{task_id}")],
                [InlineKeyboardButton("⏸️ Відкладено", callback_data=f"set_status_відкладено_{task_id}")],
                [InlineKeyboardButton("🟢 Виконано", callback_data=f"set_status_виконано_{task_id}")],
            ])
        )
        return

    # Редагування коментаря
    if data.startswith("task_comment_"):
        task_id = data[len("task_comment_"):]
        context.user_data["editing_task_id"] = task_id
        context.user_data["manual_task_state"] = "awaiting_comment"
        await _remove_old_keyboard(context)
        await context.bot.send_message(chat_id=chat_id, text="Введи новий коментар:")
        return

    # Видалення завдання
    if data.startswith("task_delete_"):
        task_id = data[len("task_delete_"):]
        await _remove_old_keyboard(context)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Підтвердити видалення?",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Так, видалити", callback_data=f"confirm_delete_{task_id}")],
                [InlineKeyboardButton("❌ Скасувати", callback_data="menu_back")],
            ])
        )
        return

    if data.startswith("confirm_delete_"):
        task_id = data[len("confirm_delete_"):]
        await asyncio.to_thread(storage.delete_task, task_id)
        await _remove_old_keyboard(context)
        await context.bot.send_message(chat_id=chat_id, text="🗑️ Завдання видалено.")
        await _show_main_menu(chat_id, context)
        return

    # Вибір проєкту для нового ручного завдання
    if data.startswith("mt_proj_"):
        proj_id = data[len("mt_proj_"):]
        context.user_data["manual_task_project"] = proj_id
        context.user_data["manual_task_state"] = "awaiting_status"
        await _remove_old_keyboard(context)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Обери статус:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔵 Відкрито", callback_data="mt_status_відкрито")],
                [InlineKeyboardButton("🟡 В роботі", callback_data="mt_status_в роботі")],
                [InlineKeyboardButton("🟢 Виконано", callback_data="mt_status_виконано")],
            ])
        )
        return

    if data.startswith("mt_status_"):
        status = data[len("mt_status_"):]
        context.user_data["manual_task_status"] = status
        context.user_data["manual_task_state"] = "awaiting_comment_optional"
        await _remove_old_keyboard(context)
        await context.bot.send_message(
            chat_id=chat_id,
            text="Додай коментар (або пропусти):",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("Пропустити", callback_data="mt_skip_comment")
            ]])
        )
        return

    if data == "mt_skip_comment":
        await _save_manual_task(chat_id, context, comment="")
        return


async def _save_manual_task(chat_id: int, context: ContextTypes.DEFAULT_TYPE, comment: str):
    """Зберігає ручне завдання в таблицю."""
    title = context.user_data.get("manual_task_title", "")
    proj_id = context.user_data.get("manual_task_project", "other")
    status = context.user_data.get("manual_task_status", "відкрито")

    task = await asyncio.to_thread(
        storage.add_task,
        title, proj_id, "", "вручну", status
    )
    if comment:
        await asyncio.to_thread(storage.update_task_comment, task["id"], comment)

    # Очищаємо стан
    for key in ["manual_task_title", "manual_task_project", "manual_task_status", "manual_task_state"]:
        context.user_data.pop(key, None)

    await _remove_old_keyboard(context)
    await context.bot.send_message(chat_id=chat_id, text=f"✅ Завдання додано: {title}")
    await _show_main_menu(chat_id, context)


# =========================
# ASYNC LOOP
# =========================
ASYNC_LOOP = asyncio.new_event_loop()


def _run_loop_forever(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


# =========================
# WEBHOOK
# =========================
@flask_app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        update = Update.de_json(data, bot_app.bot)
        asyncio.run_coroutine_threadsafe(bot_app.process_update(update), ASYNC_LOOP)
    except Exception as e:
        logger.error("Webhook error", exc_info=e)
    return "ok"


# =========================
# ЗАПУСК
# =========================
def main():
    bot_app.add_handler(CommandHandler("start", start))
    bot_app.add_handler(CommandHandler("ping", ping))
    bot_app.add_handler(CommandHandler("bag", resume_session))
    bot_app.add_handler(CommandHandler("debug", debug_log))
    bot_app.add_handler(CommandHandler("cleardebug", clear_debug_log))
    bot_app.add_handler(MessageHandler(filters.Document.FileExtension("json"), log_document_message))
    bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))
    bot_app.add_handler(CallbackQueryHandler(menu_buttons, pattern="^(wake_up|menu_|task_nav_|task_status_|task_comment_|task_delete_|confirm_delete_|mt_proj_|mt_status_|mt_skip_comment)"))
    bot_app.add_handler(CallbackQueryHandler(buttons))

    threading.Thread(target=_run_loop_forever, args=(ASYNC_LOOP,), daemon=True).start()

    asyncio.run_coroutine_threadsafe(bot_app.initialize(), ASYNC_LOOP).result()
    asyncio.run_coroutine_threadsafe(bot_app.start(), ASYNC_LOOP).result()
    asyncio.run_coroutine_threadsafe(bot_app.bot.set_webhook(f"{WEBHOOK_URL}/webhook"), ASYNC_LOOP).result()

    logger.info("✅ PTB запущено; вебхук: %s/webhook", WEBHOOK_URL)
    flask_app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
