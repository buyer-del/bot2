import os
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

from ai import transcribe_audio, extract_text_from_image, analyze_task_with_ai
from sheets_api import append_task, append_task_structured

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

if not TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN не задано")
if not WEBHOOK_URL or not WEBHOOK_URL.startswith("https://"):
    raise SystemExit("WEBHOOK_URL не задано або не HTTPS")

MAX_BUFFER_ITEMS = 3

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
# ДОПОМІЖНЕ
# -------------------------
def _buf(context: ContextTypes.DEFAULT_TYPE):
    return context.user_data.setdefault("buffer", [])

def _kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📌 Створити задачу", callback_data="new_task")],
        [InlineKeyboardButton("🧹 Очистити", callback_data="clear_buf")],
    ])

async def _remove_old_keyboard(context: ContextTypes.DEFAULT_TYPE):
    """Прибирає кнопки із попереднього бот-повідомлення."""
    chat_id = context.user_data.get("last_kb_chat_id")
    msg_id = context.user_data.get("last_kb_message_id")
    if not chat_id or not msg_id:
        return
    try:
        await context.bot.edit_message_reply_markup(
            chat_id=chat_id,
            message_id=msg_id,
            reply_markup=None
        )
    except BadRequest:
        pass
    except Exception as e:
        logger.exception("Не вдалося прибрати старі кнопки: %s", e)

def _buffer_has_space(context: ContextTypes.DEFAULT_TYPE):
    return len(_buf(context)) < MAX_BUFFER_ITEMS

async def _post_text_with_keyboard(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Надсилає повідомлення з текстом + кнопками, прибираючи попередні."""
    await _remove_old_keyboard(context)
    sent = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        reply_markup=_kb()
    )
    context.user_data["last_kb_chat_id"] = sent.chat_id
    context.user_data["last_kb_message_id"] = sent.message_id

# -------------------------
# ПАРСИНГ ВІДПОВІДІ AI (S2)
# -------------------------
def _parse_ai_structured_text(s: str):
    """
    Очікує формат:
      Назва: ...
      Тег: ...
      Дедлайн: ...
      Пріоритет: ...
      Опис: ...

    Повертає dict або None, якщо щось критично не заповнено.
    """
    if not s:
        return None

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    fields = {"name": "", "tag": "", "deadline": "", "priority": "", "description": ""}

    def take(prefix):
        for ln in lines:
            low = ln.lower()
            if low.startswith(prefix.lower()):
                return ln[len(prefix):].strip()
        return ""

    fields["name"] = take("Назва:")
    fields["tag"] = take("Тег:")
    fields["deadline"] = take("Дедлайн:")
    fields["priority"] = take("Пріоритет:")
    # опис може бути багаторядковим; якщо модель дала в один рядок — теж ок
    desc_start = None
    for idx, ln in enumerate(lines):
        if ln.lower().startswith("опис:"):
            desc_start = idx
            break
    if desc_start is not None:
        first = lines[desc_start][len("Опис:"):].strip()
        rest = lines[desc_start + 1 :]
        fields["description"] = ("\n".join([first] + rest)).strip()
    else:
        fields["description"] = take("Опис:")

    # Нормалізація
    if not fields["name"]:
        return None
    tag = fields["tag"] or "#інше"
    if tag and not tag.startswith("#"):
        tag = f"#{tag}"
    fields["tag"] = tag
    fields["deadline"] = fields["deadline"] or "не вказано"

    pr = (fields["priority"] or "").lower()
    if "висок" in pr:
        fields["priority"] = "високий"
    elif "сер" in pr:
        fields["priority"] = "середній"
    elif "звич" in pr or not pr:
        fields["priority"] = "звичайний"
    else:
        fields["priority"] = "звичайний"

    if not fields["description"]:
        fields["description"] = "(без опису)"

    return fields

# -------------------------
# КОМАНДИ
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Бот працює. Надішли текст, фото або голос — усе буде розпізнано.")
    await _post_text_with_keyboard(update, context, "Чорнетка порожня. Додавайте записи повідомленнями.")

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong ✅")

# -------------------------
# ТЕКСТ
# -------------------------
async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        await update.message.reply_text("❌ Порожній текст.")
        return
    if not _buffer_has_space(context):
        await update.message.reply_text("⚠️ Чернетка заповнена (3/3).")
        return
    _buf(context).append(text)
    await update.message.reply_text("✅ Додано в чернетку")
    await _post_text_with_keyboard(update, context, text)

# -------------------------
# ФОТО
# -------------------------
async def photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file = await update.message.photo[-1].get_file()
        local_path = "photo.jpg"
        await file.download_to_drive(local_path)
        recognized = (extract_text_from_image(local_path) or "").strip()
        if not recognized:
            await update.message.reply_text("❌ Нічого не розпізнано.")
            return
        if not _buffer_has_space(context):
            await update.message.reply_text("⚠️ Чернетка заповнена (3/3).")
            return
        _buf(context).append(recognized)
        await update.message.reply_text("🖼 Розпізнано текст")
        await _post_text_with_keyboard(update, context, recognized)
    except Exception as e:
        logger.exception("Помилка OCR: %s", e)
        await update.message.reply_text("❌ Помилка розпізнавання фото.")

# -------------------------
# ГОЛОС (voice)
# -------------------------
async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file = await update.message.voice.get_file()
        local_path = "voice.ogg"
        await file.download_to_drive(local_path)
        recognized = (transcribe_audio(local_path) or "").strip()
        if not recognized:
            await update.message.reply_text("❌ Голос не розпізнано.")
            return
        if not _buffer_has_space(context):
            await update.message.reply_text("⚠️ Чернетка заповнена (3/3).")
            return
        _buf(context).append(recognized)
        await update.message.reply_text("🎤 Розпізнано текст")
        await _post_text_with_keyboard(update, context, recognized)
    except Exception as e:
        logger.exception("Помилка голосу: %s", e)
        await update.message.reply_text("❌ Помилка розпізнавання голосу.")

# -------------------------
# АУДІО-ФАЙЛИ (m4a/mp3/wav як документ)
# -------------------------
async def audio_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file = await update.message.document.get_file()
        orig_name = update.message.document.file_name or "audio"
        local_path = f"input_{orig_name}"
        await file.download_to_drive(local_path)
        recognized = (transcribe_audio(local_path) or "").strip()
        if not recognized:
            await update.message.reply_text("❌ Не вдалося розпізнати аудіо-файл.")
            return
        if not _buffer_has_space(context):
            await update.message.reply_text("⚠️ Чернетка заповнена (3/3).")
            return
        _buf(context).append(recognized)
        await update.message.reply_text("🎧 Розпізнано текст з файлу")
        await _post_text_with_keyboard(update, context, recognized)
    except Exception as e:
        logger.exception("Помилка розпізнавання аудіо-файлу: %s", e)
        await update.message.reply_text("❌ Помилка розпізнавання аудіо-файлу.")

# -------------------------
# КНОПКИ
# -------------------------
AI_PROMPT = (
    "Ти — аналітик задач у виробничій команді.\n"
    "Отримуєш короткі або неформальні повідомлення українською.\n"
    "У текстах можуть бути зайві слова, жаргон, повтори чи імена людей — їх потрібно ігнорувати.\n"
    "Залишай лише суттєву, змістовну інформацію, зрозумілу людині.\n\n"
    "На основі повідомлення потрібно створити структурований опис із такими полями:\n\n"
    "1. Назва — коротко і змістовно описує суть дії (наприклад, \"Закупівля метизу\", \"Перевірка освітлення\", \"Ремонт дверей\").\n"
    "   У назві не використовуй номери об’єктів чи теги.\n"
    "2. Тег — якщо у тексті є номер ліфта або об’єкта (наприклад, 246), зроби його тегом у форматі #246.\n"
    "   Якщо номер відсутній, встанови тег #інше.\n"
    "3. Дедлайн — якщо дата або термін не згадані, пиши \"не вказано\".\n"
    "4. Пріоритет — оцінюй рівень терміновості за змістом повідомлення:\n"
    "   якщо згадано \"терміново\", \"негайно\", \"сьогодні\", \"зараз\" — вкажи \"високий\",\n"
    "   якщо \"цього тижня\", \"до кінця тижня\" — \"середній\",\n"
    "   інакше — \"звичайний\".\n"
    "5. Опис — сформулюй коротку інструкцію, яка пояснює, що потрібно зробити, без зайвих деталей і повторів.\n\n"
    "Формат відповіді строго такий:\n"
    "Назва: ...\n"
    "Тег: ...\n"
    "Дедлайн: ...\n"
    "Пріоритет: ...\n"
    "Опис: ..."
)

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data
    buf = _buf(context)

    if data == "clear_buf":
        buf.clear()
        await _remove_old_keyboard(context)
        await q.message.reply_text("🧹 Чернетку очищено.")
        return

    if data == "new_task":
        if not buf:
            await q.message.reply_text("⚠️ Чернетка порожня.")
            return

        raw_text = "\n".join(buf)

        # 1) Викликаємо AI (Vertex) у бекграунді, щоб не блокувати
        try:
            structured_text = await asyncio.to_thread(analyze_task_with_ai, AI_PROMPT, raw_text)
        except Exception as e:
            logger.exception("AI exception: %s", e)
            structured_text = None

        # 2) Якщо є структурований результат — парсимо і пишемо 5 колонок
        if structured_text:
            fields = _parse_ai_structured_text(structured_text)
            if fields:
                try:
                    append_task_structured(
                        fields["name"],
                        fields["tag"],
                        fields["deadline"],
                        fields["priority"],
                        fields["description"],
                    )
                    await _remove_old_keyboard(context)
                    await q.message.reply_text("✅ Задачу структуровано й додано в таблицю:\n\n" + structured_text)
                    buf.clear()
                    return
                except Exception as e:
                    logger.exception("Помилка запису структури у таблицю: %s", e)
                    # падати не будемо — перейдемо до фолбек-запису як є

        # 3) Фолбек: AI недоступний або парсинг не вдався — записуємо як є (в опис)
        try:
            append_task(raw_text)
            await _remove_old_keyboard(context)
            await q.message.reply_text("⚠️ AI недоступний. Задачу додано як є (в опис).")
            buf.clear()
        except Exception as e:
            logger.exception("Помилка фолбек-запису у таблицю: %s", e)
            await q.message.reply_text("❌ Помилка запису у таблицю.")
        return

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
        asyncio.run_coroutine_threadsafe(
            bot_app.process_update(update),
            ASYNC_LOOP
        )
    except Exception as e:
        logger.error("Webhook error", exc_info=e)

    return "ok"

# =========================
# ЗАПУСК
# =========================
def main():
    bot_app.add_handler(CommandHandler("start", start))
    bot_app.add_handler(CommandHandler("ping", ping))
    bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))
    bot_app.add_handler(MessageHandler(filters.PHOTO, photo_message))
    bot_app.add_handler(MessageHandler(filters.VOICE, voice_message))
    bot_app.add_handler(MessageHandler(filters.Document.AUDIO, audio_document_message))
    bot_app.add_handler(CallbackQueryHandler(buttons))

    threading.Thread(target=_run_loop_forever, args=(ASYNC_LOOP,), daemon=True).start()
    asyncio.run_coroutine_threadsafe(bot_app.initialize(), ASYNC_LOOP).result()
    asyncio.run_coroutine_threadsafe(bot_app.start(), ASYNC_LOOP).result()
    asyncio.run_coroutine_threadsafe(
        bot_app.bot.set_webhook(f"{WEBHOOK_URL}/webhook"),
        ASYNC_LOOP
    ).result()

    logger.info("✅ PTB запущено; вебхук: %s/webhook", WEBHOOK_URL)
    flask_app.run(host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()

