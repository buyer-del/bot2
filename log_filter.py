"""
Модуль фільтрації логу за день.

Реалізує дворівневу фільтрацію (Розділ 5, Крок 1 архітектурного документа):
- Гілка А (для пошуку НОВИХ завдань): тільки повідомлення від "дозволених" контактів.
- Гілка Б (для оновлення СТАТУСУ існуючих завдань): усі повідомлення дня, без фільтрації.

Лог має три незалежні джерела з різною структурою полів: chats, emails, calls
(деталі структури — Розділ 3.1 архітектурного документа).
"""

import json


def load_contacts(contacts_path: str) -> list[dict]:
    """Завантажує довідник 'дозволених' контактів з JSON-файлу."""
    with open(contacts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["contacts"]


def _clean_email_signature(text: str) -> str:
    """
    Прибирає типові email-підписи (Kind Regards, телефони, банери),
    щоб не засмічувати контекст для LLM. Проста евристика: обрізає текст
    на першому входженні типових маркерів підпису.
    """
    markers = ["Kind Regards", "[image:", "З повагою", "Best regards"]
    cut_at = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)
    return text[:cut_at].strip()


def _parse_group_message(chat_name: str, sender: str, text: str) -> tuple[str, str]:
    """
    У групових чатах Telegram поле sender = назва групи, а реальне ім'я
    відправника йде першим рядком у text, після якого з нового рядка — сам текст.

    Наприклад:
      sender = "ЦЕХ"
      text   = "Шеф Андрій Петрович Скайліфт Бережний\nДуже схоже на то"

    Якщо sender збігається з chat_name — це ознака групового формату.
    Повертає (real_sender, clean_text).
    """
    if sender and chat_name and _normalize(sender) == _normalize(chat_name):
        lines = text.split("\n", 1)
        if len(lines) == 2:
            real_sender = lines[0].strip()
            clean_text = lines[1].strip()
            if real_sender:
                return real_sender, clean_text
    return sender, text


def _normalize(value: str) -> str:
    """Нормалізація рядка для порівняння (без зайвих пробілів, у нижньому регістрі)."""
    return (value or "").strip().lower()


def is_allowed_sender(message: dict, contacts: list[dict]) -> bool:
    """
    Перевіряє, чи повідомлення від когось зі списку 'дозволених' контактів.

    Збіг по будь-якому ОДНОМУ полю (ім'я / телефон / email) зараховує
    відправника як дозволеного — та сама людина в різних каналах
    ідентифікується по-різному (див. приклад 'Шеф' у test_contacts.json:
    у Viber/дзвінках — повне ім'я з телефонної книги, в email — латинкою).
    """
    sender_name = _normalize(message.get("sender") or message.get("sender_name") or message.get("contact_name", ""))
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


def _is_own_message(message: dict) -> bool:
    """Власні вихідні повідомлення позначені 'Я (Ви)' у chats, direction='outgoing' в emails."""
    sender = message.get("sender", "")
    return sender == "Я (Ви)" or message.get("direction") == "outgoing"


def collect_all_messages(log_data: dict) -> list[dict]:
    """
    Збирає УСІ повідомлення дня (chats + emails) в єдиний плаский список
    з уніфікованими полями для подальшої обробки.

    Дзвінки (calls) НЕ включаються сюди — вони без тексту (тільки recording_path),
    обробка дзвінків заморожена до етапу транскрипції (Варіант А, Розділ 3).
    Voice/відео в чатах (media_type == "media", без тексту) теж відсіюються тут.
    """
    unified = []

    for chat in log_data.get("chats", []):
        platform = chat.get("platform")
        chat_name = chat.get("chat_name")
        for m in chat.get("messages", []):
            # Пропускаємо голосові/відео без тексту — заморожено (Варіант А)
            if m.get("media_type") == "media" and not m.get("text"):
                continue
            # Пропускаємо вкладення без супровідного тексту (фото/документ без підпису)
            if not m.get("text"):
                continue

            raw_sender = m.get("sender", "")
            raw_text = m.get("text", "")

            # Для групових чатів: витягуємо реального відправника з першого рядка тексту
            real_sender, clean_text = _parse_group_message(chat_name, raw_sender, raw_text)

            # Власне повідомлення: або "Я (Ви)" в оригінальному sender, або після розпарсингу групи
            is_own = raw_sender == "Я (Ви)" or real_sender == "Я (Ви)" or m.get("direction") == "outgoing"

            # Зберігаємо реального відправника в raw для подальшої перевірки is_allowed_sender
            enriched_raw = {**m, "sender": real_sender}

            unified.append({
                "source": "chat",
                "platform": platform,
                "chat_name": chat_name,
                "sender": real_sender,
                "is_own": is_own,
                "time": m.get("time"),
                "text": clean_text,
                "raw": enriched_raw,
            })

    for e in log_data.get("emails", []):
        if not e.get("text"):
            continue
        cleaned_text = _clean_email_signature(e.get("text"))
        if not cleaned_text:
            continue
        unified.append({
            "source": "email",
            "platform": "Gmail",
            "chat_name": e.get("subject"),
            "sender": e.get("sender_name"),
            "sender_email": e.get("sender_email"),
            "is_own": _is_own_message(e),
            "time": e.get("time"),
            "text": cleaned_text,
            "raw": e,
        })

    return unified


def filter_log(log_data: dict, contacts_path: str) -> dict:
    """
    Головна функція дворівневої фільтрації.

    Повертає:
        {
            "branch_a_new_tasks": [...],   # тільки від дозволених контактів
            "branch_b_status_updates": [...],  # усі повідомлення дня
        }
    """
    contacts = load_contacts(contacts_path)
    all_messages = collect_all_messages(log_data)

    branch_a = [
        m for m in all_messages
        if not m["is_own"] and is_allowed_sender(m["raw"], contacts)
    ]

    # Гілка Б: усі повідомлення дня (включно з власними вихідними —
    # вони теж є сигналом про дію, наприклад "Ок" у відповідь на рахунок)
    branch_b = all_messages

    return {
        "branch_a_new_tasks": branch_a,
        "branch_b_status_updates": branch_b,
    }


if __name__ == "__main__":
    # Швидка перевірка на реальному лозі
    import sys

    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    contacts_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not log_path or not contacts_path:
        print("Використання: python log_filter.py <шлях_до_логу.json> <шлях_до_контактів.json>")
        sys.exit(1)

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    result = filter_log(log_data, contacts_path)

    print(f"Усього повідомлень дня (Гілка Б): {len(result['branch_b_status_updates'])}")
    print(f"Від дозволених контактів (Гілка А): {len(result['branch_a_new_tasks'])}")
    print()
    print("=== Приклади з Гілки А (нові завдання) ===")
    for m in result["branch_a_new_tasks"][:5]:
        print(f"[{m['platform']}] {m['sender']} ({m['time']}): {m['text'][:80]}")
