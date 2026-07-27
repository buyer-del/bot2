"""
Модуль формування вибірки з логу для передачі до Gemini.

Замість двох окремих гілок (А і Б) формує ОДНУ компактну вибірку у форматі JSON:
- Повідомлення від "особливих" контактів завжди включаються з міткою task_source=true
- Решта повідомлень проходить двоступеневу фільтрацію:
  1. Технічний фільтр — відсіює рекламний шум за ознаками відправника/теми
  2. Embedding-фільтр — відсіює семантично нерелевантні повідомлення
- Компактний формат: тільки потрібні поля (час, відправник, текст, чат, мітка)

Термінологія:
- лог — вхідний JSON-файл від Android-додатку
- вибірка — компактний JSON який передається до Gemini
"""

import json
import os
import re
import logging

logger = logging.getLogger(__name__)

# Порогове значення cosine similarity для embedding-фільтра
EMBEDDING_THRESHOLD = float(os.environ.get("EMBEDDING_THRESHOLD", "0.25"))


# ─── Завантаження довідників ─────────────────────────────────────────────────

def load_contacts(contacts_path: str) -> list[dict]:
    with open(contacts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["contacts"]


def load_projects(projects_path: str) -> list[dict]:
    if not os.path.exists(projects_path):
        return []
    with open(projects_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("projects", [])


# ─── Нормалізація ─────────────────────────────────────────────────────────────

def _normalize(value: str) -> str:
    return (value or "").strip().lower()


# ─── Технічний фільтр шуму ───────────────────────────────────────────────────

# Ознаки рекламних/автоматичних відправників
_SPAM_SENDER_PATTERNS = [
    r"noreply", r"no-reply", r"newsletter", r"notification",
    r"marketing", r"promo", r"support@", r"info@", r"hello@",
    r"sales@", r"deals@", r"updates@", r"mailer",
    r"rozetka", r"brain\.ua", r"prom\.ua", r"epicentrk",
    r"ukrposhta", r"novaposhta.*info", r"meest",
]

# Ознаки рекламних тем листів
_SPAM_SUBJECT_PATTERNS = [
    r"розпродаж", r"акція", r"знижк", r"промокод",
    r"unsubscribe", r"відпис", r"newsletter", r"рекламн",
    r"спеціальна пропозиція", r"тільки сьогодні",
    r"нові надходження", r"топ товар", r"купуй зараз",
]

# Ознаки рекламних повідомлень у Viber
_SPAM_VIBER_PATTERNS = [
    r"відпис", r"unsubscribe", r"натисніть тут", r"click here",
    r"переходьте за посиланням", r"замовляйте зараз",
]

# Маркери підпису — від цього місця відрізаємо все до кінця
_SIGNATURE_MARKERS = [
    "Kind Regards", "Best regards", "Best Regards",
    "Regards,", "BR\n", "BR,",
    "З повагою", "З повагою,",
    "Дякуємо", "Дякую,",
    "Thank you,", "Thanks,",
    "Sincerely,",
    "[image:", "[cid:",
    "Phone:", "Mobile:", "Tel.:",
    "NIP:", "KRS:", "REGON:",
    "Confidentiality Notice",
]

# Маркери початку цитати попереднього листа
_QUOTE_MARKERS = [
    "\nFrom:", "\nOd:", "\nVid:", "\nDe:",
    "\nSent:", "\nWysłane:", "\nНадіслано:",
    "\n-----", "\n_____",
    "\nOn ", "\n>",
    "\nчт,", "\nпн,", "\nвт,", "\nср,", "\nпт,", "\nсб,", "\nнд,",
    "\nPn,", "\nWt,", "\nŚr,", "\nCz,", "\nPt,",
]

# Технічні рядки які видаляємо навіть з середини тексту
_TECH_LINE_PATTERNS = [
    r"\[cid:[^\]]+\]",           # вбудовані зображення [cid:...]
    r"<https?://[^>]+>",         # посилання в кутових дужках <http://...>
    r"\[https?://[^\]]+\]",      # посилання в квадратних дужках [http://...]
    r"https?://\S+",             # голі посилання
    r"www\.\S+",                 # www посилання
    r"⚠.*?⚠",                   # попередження про зовнішній відправник
]


def _is_spam(message: dict) -> bool:
    """
    Технічний фільтр шуму — відсіює рекламні та автоматичні повідомлення.
    """
    sender_email = _normalize(message.get("sender_email", ""))
    subject = _normalize(message.get("chat_name", "") or "")
    text = message.get("text", "") or ""
    source = message.get("source", "")

    # Перевірка за email відправника
    for pattern in _SPAM_SENDER_PATTERNS:
        if re.search(pattern, sender_email):
            return True

    # Перевірка за темою листа (email)
    if source == "email":
        for pattern in _SPAM_SUBJECT_PATTERNS:
            if re.search(pattern, subject):
                return True
        # Якщо лист містить багато посилань — ознака реклами
        link_count = len(re.findall(r"https?://", text))
        if link_count > 5:
            return True

    # Перевірка за текстом (Viber-розсилки)
    if source == "chat":
        for pattern in _SPAM_VIBER_PATTERNS:
            if re.search(pattern, _normalize(text)):
                return True

    return False


# ─── Очищення тексту ─────────────────────────────────────────────────────────

def _clean_email_text(text: str) -> str:
    """
    Агресивне очищення email-тексту:
    1. Відрізаємо підпис і все що після нього (включно з підписом)
    2. Відрізаємо цитати попередніх листів
    3. Видаляємо технічні рядки (посилання, cid, попередження)
    """
    if not text:
        return ""

    # Крок 1: відрізаємо від першого маркера підпису або цитати
    cut_at = len(text)
    for marker in _SIGNATURE_MARKERS + _QUOTE_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)

    text = text[:cut_at].strip()

    # Крок 2: видаляємо технічні рядки через regex
    for pattern in _TECH_LINE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Крок 3: прибираємо порожні рядки що утворились після очищення
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = "\n".join(lines)

    return text.strip()


# ─── Розпізнавання групових повідомлень ──────────────────────────────────────

def _parse_group_message(chat_name: str, sender: str, text: str) -> tuple[str, str]:
    """
    У групових чатах Telegram sender = назва групи, а реальне ім'я
    відправника йде першим рядком у text.
    """
    if sender and chat_name and _normalize(sender) == _normalize(chat_name):
        lines = text.split("\n", 1)
        if len(lines) == 2:
            real_sender = lines[0].strip()
            clean_text = lines[1].strip()
            if real_sender:
                return real_sender, clean_text
    return sender, text


# ─── Перевірка "особливого" контакту ─────────────────────────────────────────

def is_allowed_sender(message: dict, contacts: list[dict]) -> bool:
    sender_name = _normalize(
        message.get("sender") or message.get("sender_name") or message.get("contact_name", "")
    )
    sender_email = _normalize(message.get("sender_email", ""))
    sender_phone = _normalize(message.get("phone_number", ""))

    for contact in contacts:
        names = [_normalize(n) for n in contact.get("names", [])]
        emails = [_normalize(e) for e in contact.get("emails", [])]
        phones = [_normalize(p) for p in contact.get("phones", [])]

        if sender_name and sender_name in names:
            return True
        if sender_email and sender_email in emails:
            return True
        if sender_phone and sender_phone in phones:
            return True

    return False


# ─── Embedding-фільтр ────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Кешований genai-клієнт — раніше _get_embeddings() створював новий
# genai.Client() на кожен виклик, а викликається вона окремо для КОЖНОГО
# повідомлення, що проходить embedding-фільтр (цикл у build_selection).
# Тобто на один лог могли будуватись десятки/сотні клієнтів поспіль —
# те саме джерело поступового росту пам'яті, що й у sheets_api.py.
_genai_client_cache = None


def _get_genai_client(api_key: str | None = None):
    global _genai_client_cache
    if _genai_client_cache is not None:
        return _genai_client_cache
    from google import genai
    _genai_client_cache = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    return _genai_client_cache


def _get_embeddings(texts: list[str], api_key: str | None = None) -> list[list[float]]:
    """Отримує embedding-вектори через Gemini text-embedding-004."""
    client = _get_genai_client(api_key)
    result = client.models.embed_content(
        model="models/text-embedding-004",
        contents=texts,
    )
    return [e.values for e in result.embeddings]


def _is_relevant_by_embedding(
    text: str,
    anchor_embeddings: list[list[float]],
    api_key: str | None = None,
) -> bool:
    """
    Перевіряє чи повідомлення семантично пов'язане з активними проєктами
    або відкритими завданнями через cosine similarity.
    """
    if not anchor_embeddings or not text.strip():
        return False
    try:
        msg_embeddings = _get_embeddings([text], api_key)
        msg_vec = msg_embeddings[0]
        max_sim = max(_cosine_similarity(msg_vec, anchor) for anchor in anchor_embeddings)
        return max_sim >= EMBEDDING_THRESHOLD
    except Exception as e:
        logger.warning("Embedding-фільтр: помилка при векторизації: %s", e)
        return True  # При помилці включаємо повідомлення (безпечніший варіант)


# ─── Збір і очищення повідомлень з логу ──────────────────────────────────────

def _collect_raw_messages(log_data: dict) -> list[dict]:
    """Збирає всі текстові повідомлення з логу в уніфікований формат."""
    unified = []

    for chat in log_data.get("chats", []):
        platform = chat.get("platform")
        chat_name = chat.get("chat_name")
        for m in chat.get("messages", []):
            if m.get("media_type") == "media" and not m.get("text"):
                continue
            if not m.get("text"):
                continue

            raw_sender = m.get("sender", "")
            raw_text = m.get("text", "")
            real_sender, clean_text = _parse_group_message(chat_name, raw_sender, raw_text)
            is_own = raw_sender == "Я (Ви)" or real_sender == "Я (Ви)"

            unified.append({
                "source": "chat",
                "platform": platform,
                "chat_name": chat_name,
                "sender": real_sender,
                "sender_email": "",
                "is_own": is_own,
                "time": m.get("time"),
                "text": clean_text,
                "raw_sender_for_check": real_sender,
                "raw_email_for_check": "",
            })

    for e in log_data.get("emails", []):
        if not e.get("text"):
            continue
        cleaned_text = _clean_email_text(e.get("text", ""))
        if not cleaned_text:
            continue
        is_own = e.get("direction") == "outgoing"
        unified.append({
            "source": "email",
            "platform": "Gmail",
            "chat_name": e.get("subject", ""),
            "sender": e.get("sender_name", ""),
            "sender_email": e.get("sender_email", ""),
            "is_own": is_own,
            "time": e.get("time"),
            "text": cleaned_text,
            "raw_sender_for_check": e.get("sender_name", ""),
            "raw_email_for_check": e.get("sender_email", ""),
        })

    return unified


# ─── Головна функція: формування вибірки ─────────────────────────────────────

def build_selection(
    log_data: dict,
    contacts_path: str,
    projects_path: str,
    open_tasks: list[dict] | None = None,
    api_key: str | None = None,
) -> list[dict]:
    """
    Формує компактну вибірку повідомлень для передачі до Gemini.

    Кожен елемент вибірки:
    {
        "t": "14:37",
        "from": "Ігор Чорний конструктор",
        "chat": "Viber / назва чату",
        "msg": "текст повідомлення",
        "task_source": true/false  # true = від особливого контакту, шукати нові завдання
    }
    """
    contacts = load_contacts(contacts_path)
    projects = load_projects(projects_path)
    all_messages = _collect_raw_messages(log_data)

    # Формуємо anchor-тексти для embedding-фільтра:
    # назви проєктів + тексти відкритих завдань
    anchor_texts = [p["full_name"] for p in projects if p.get("id") != "other"]
    if open_tasks:
        anchor_texts += [t.get("title", "") for t in open_tasks if t.get("title")]

    # Кешуємо anchor-embeddings (один виклик API для всіх anchor-текстів)
    anchor_embeddings = []
    if anchor_texts:
        try:
            anchor_embeddings = _get_embeddings(anchor_texts, api_key)
        except Exception as e:
            logger.warning("Не вдалось отримати anchor-embeddings: %s. Embedding-фільтр вимкнено.", e)

    selection = []
    msg_counter = {}  # для унікальності id якщо час однаковий

    for m in all_messages:
        text = m.get("text", "").strip()
        if not text:
            continue

        # Формуємо унікальний source_id: YYYYMMDD_HHMM_платформа_лічильник
        log_date = log_data.get("export_date", "")[:10].replace("-", "")
        time_str = (m.get("time", "") or "").replace(":", "")[:4]
        platform_str = (m.get("platform", "") or "").lower()[:5].replace(" ", "")
        base_id = f"{log_date}_{time_str}_{platform_str}"
        msg_counter[base_id] = msg_counter.get(base_id, 0) + 1
        source_id = f"{base_id}_{msg_counter[base_id]}"

        # Перевірка чи від особливого контакту
        check_msg = {
            "sender": m["raw_sender_for_check"],
            "sender_email": m["raw_email_for_check"],
        }
        is_special = is_allowed_sender(check_msg, contacts)

        if is_special:
            # Особливий контакт — завжди включаємо, ніяких фільтрів
            selection.append({
                "id": source_id,
                "t": m.get("time", ""),
                "from": m["sender"] or "Я",
                "chat": f"{m['platform']} / {m['chat_name']}",
                "msg": text,
                "task_source": not m["is_own"],
            })
        elif m["is_own"]:
            # Власні повідомлення — включаємо без embedding-фільтра але відсіюємо шум
            if not _is_spam(m):
                selection.append({
                    "id": source_id,
                    "t": m.get("time", ""),
                    "from": "Я",
                    "chat": f"{m['platform']} / {m['chat_name']}",
                    "msg": text,
                    "task_source": False,
                })
        else:
            # Інші відправники — технічний фільтр + embedding-фільтр
            if _is_spam(m):
                continue
            if anchor_embeddings and not _is_relevant_by_embedding(text, anchor_embeddings, api_key):
                continue
            selection.append({
                "id": source_id,
                "t": m.get("time", ""),
                "from": m["sender"] or "",
                "chat": f"{m['platform']} / {m['chat_name']}",
                "msg": text,
                "task_source": False,
            })

    return selection


# ─── Зворотна сумісність (для старого коду що міг викликати filter_log) ──────

def filter_log(log_data: dict, contacts_path: str) -> dict:
    """
    Залишено для сумісності. Новий код має використовувати build_selection().
    """
    contacts = load_contacts(contacts_path)
    all_messages = _collect_raw_messages(log_data)

    branch_a = [
        m for m in all_messages
        if not m["is_own"] and is_allowed_sender(
            {"sender": m["raw_sender_for_check"], "sender_email": m["raw_email_for_check"]},
            contacts,
        )
    ]

    return {
        "branch_a_new_tasks": branch_a,
        "branch_b_status_updates": all_messages,
    }


if __name__ == "__main__":
    import sys
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    contacts_path = sys.argv[2] if len(sys.argv) > 2 else None
    projects_path = sys.argv[3] if len(sys.argv) > 3 else "projects.json"

    if not log_path or not contacts_path:
        print("Використання: python log_filter.py <лог.json> <контакти.json> [проєкти.json]")
        sys.exit(1)

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    selection = build_selection(log_data, contacts_path, projects_path)
    special = [m for m in selection if m["task_source"]]
    print(f"Всього у вибірці: {len(selection)}")
    print(f"Від особливих контактів (task_source=true): {len(special)}")
    print()
    print("=== Перші 5 з task_source=true ===")
    for m in special[:5]:
        print(f"[{m['chat']}] {m['from']} ({m['t']}): {m['msg'][:80]}")
