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

    # Нормалізуємо: null → "other"
    for t in new_tasks:
        if not t.get("project_id"):
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
    proj = payload.get("project_id", "other")
    rows = [
        [InlineKeyboardButton("✅ Додати завдання", callback_data="nt_accept")],
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

        report_messages = context.user_data.get("report_messages")
        report_date = context.user_data.get("report_date", "")

        if report_messages is not None:
            await context.bot.send_message(chat_id=chat_id, text="✅ Усі пропозиції за сьогодні опрацьовано.\n📝 Складаю звіт за день…")
            try:
                report_text = await asyncio.to_thread(generate_report, report_messages, report_date)
                await context.bot.send_message(chat_id=chat_id, text=report_text)
            except Exception as e:
                logger.exception("Помилка генерації звіту: %s", e)
                await context.bot.send_message(chat_id=chat_id, text=f"❌ Не вдалося згенерувати звіт: {e}")
            finally:
                context.user_data["report_messages"] = None
        else:
            await context.bot.send_message(chat_id=chat_id, text="✅ Усі пропозиції за сьогодні опрацьовано.")
        return

    item = queue.pop(0)
    context.user_data["pending_item"] = item
    payload = item["payload"]

    if item["type"] == "new_task":
        proj = payload.get("project_id") or "other"
        # Знаходимо повну назву проєкту для відображення
        projects = storage.get_projects()
        proj_name = next((p["full_name"] for p in projects if p["id"] == proj), proj)
        text = (
            f"📌 Нове завдання:\n\n"
            f"{payload.get('title')}\n"
            f"Проєкт: {proj_name}\n"
            f"Джерело: \"{payload.get('source_text', '')}\"\n"
            f"Від: {payload.get('source_sender', '')}"
        )
        await _send_with_keyboard(context, chat_id, text, _kb_new_task(payload, projects))

    elif item["type"] == "task_update":
        task = storage.get_task_by_id(int(payload["task_id"])) if str(payload.get("task_id", "")).isdigit() else None
        task_title = task["title"] if task else f"id={payload.get('task_id')}"
        text = (
            f"🔄 Оновлення статусу:\n\n"
            f"Завдання: {task_title}\n"
            f"Новий статус: {payload.get('new_status')}\n"
            f"Коментар: {payload.get('comment', '')}\n"
            f"Джерело: \"{payload.get('source_text', '')}\"\n"
            f"Від: {payload.get('source_sender', '')}"
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

        analysis = await asyncio.to_thread(
            analyze_day,
            selection, projects, open_tasks,
        )

        queue = _build_confirm_queue(analysis)
        context.user_data["confirm_queue"] = queue

        context.user_data["report_messages"] = selection
        context.user_data["report_date"] = log_data.get("export_date", "")

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

    item = _pending(context)
    if not item:
        await q.message.reply_text("Немає активної пропозиції для підтвердження.")
        return

    payload = item["payload"]

    # --- Нове завдання: прийняти ---
    if data == "nt_accept":
        storage.add_task(
            title=payload.get("title"),
            project_id=payload.get("project_id", "other"),
            source_text=payload.get("source_text", ""),
            source_sender=payload.get("source_sender", ""),
        )
        await _remove_old_keyboard(context)
        await q.message.reply_text(f"✅ Завдання додано: {payload.get('title')}")
        await _present_next(update, context)
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
        if str(task_id).isdigit():
            storage.update_task_status(int(task_id), payload.get("new_status", "in_progress"), payload.get("comment", ""))
        await _remove_old_keyboard(context)
        await q.message.reply_text("✅ Статус оновлено.")
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
    await update.message.reply_text(
        "Надішли JSON-файл логу командою 'поділитися' з додатку логування."
    )


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
    bot_app.add_handler(CommandHandler("debug", debug_log))
    bot_app.add_handler(CommandHandler("cleardebug", clear_debug_log))
    bot_app.add_handler(MessageHandler(filters.Document.FileExtension("json"), log_document_message))
    bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))
    bot_app.add_handler(CallbackQueryHandler(buttons))

    threading.Thread(target=_run_loop_forever, args=(ASYNC_LOOP,), daemon=True).start()

    asyncio.run_coroutine_threadsafe(bot_app.initialize(), ASYNC_LOOP).result()
    asyncio.run_coroutine_threadsafe(bot_app.start(), ASYNC_LOOP).result()
    asyncio.run_coroutine_threadsafe(bot_app.bot.set_webhook(f"{WEBHOOK_URL}/webhook"), ASYNC_LOOP).result()

    logger.info("✅ PTB запущено; вебхук: %s/webhook", WEBHOOK_URL)
    flask_app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
