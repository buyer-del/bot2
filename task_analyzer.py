"""
Крок 2 архітектури: єдиний аналіз через Claude.

Один виклик LLM отримує:
1. Повідомлення Гілки А (від дозволених контактів) — для пошуку НОВИХ завдань.
2. Повідомлення Гілки Б (увесь масив дня) — для пошуку ОНОВЛЕНЬ статусу.
3. Список активних проєктів.
4. Поточну таблицю відкритих завдань.

І повертає два списки: new_tasks та task_updates.
"""

import json
import os
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"


def load_projects(projects_path: str) -> list[dict]:
    with open(projects_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["projects"]


def _format_messages_block(messages: list[dict], label: str) -> str:
    """Форматує список повідомлень у компактний текстовий блок для промта."""
    lines = [f"### {label} ({len(messages)} повідомлень)"]
    for m in messages:
        sender = "Я (сам користувач)" if m["is_own"] else m["sender"]
        lines.append(f"- [{m['platform']} | {m['chat_name']} | {m['time']}] {sender}: {m['text']}")
    return "\n".join(lines)


def _format_projects_block(projects: list[dict]) -> str:
    lines = ["### Активні проєкти"]
    for p in projects:
        lines.append(f"- id={p['id']} | коротка форма в переписці: \"{p['short_number']}\" | повна назва: {p['full_name']}")
    return "\n".join(lines)


def _format_open_tasks_block(open_tasks: list[dict]) -> str:
    if not open_tasks:
        return "### Відкриті завдання\n(наразі немає жодного відкритого завдання)"
    lines = ["### Відкриті завдання (вже існують, очікують виконання)"]
    for t in open_tasks:
        lines.append(f"- id={t['id']} | проєкт={t['project_id']} | {t['title']} | статус={t['status']}")
    return "\n".join(lines)


SYSTEM_PROMPT = """Ти аналізуєш робочу переписку закупівельника ліфтової компанії за один день.
Твоя задача — знайти НОВІ завдання та ОНОВЛЕННЯ статусу вже існуючих завдань.

КОНТЕКСТ:
- Користувач — закупівельник, відповідає за замовлення комплектуючих для ліфтів.
- Кожен ліфт = окремий проєкт. Один проєкт може мати багато дрібних завдань.
- В переписці люди часто згадують проєкт коротким номером ("259", "254", "об'єкт 259", "259-ий")
  замість повної назви — шукай ці згадки і зіставляй з наданим списком проєктів.
- Проєкт також можна впізнати по адресі, замовнику чи вазі ліфта зі списку проєктів.

ПРАВИЛА ДЛЯ НОВИХ ЗАВДАНЬ (аналізуй тільки блок "Повідомлення від дозволених контактів"):
- Завдання = конкретна дія, яку має виконати закупівельник (замовити, узгодити, перевірити, забрати тощо).
- Одне повідомлення може містити кілька різних завдань — знаходь усі.
- НЕ створюй завдання з того, що вже є у списку "Відкриті завдання" — це призведе до дублів.
- Відрізняй нову дію від простого обговорення чи питання без конкретної дії для закупівельника.
- Якщо в тексті є явна згадка проєкту (номер, адреса) — вкажи project_id з наданого списку.
  Якщо проєкт впізнати не вдалось — постав project_id: null.

ПРАВИЛА ДЛЯ ОНОВЛЕНЬ СТАТУСУ (аналізуй блок "Усі повідомлення дня"):
- Дивись, чи якесь повідомлення стосується ВЖЕ ІСНУЮЧОГО завдання зі списку "Відкриті завдання".
- Це може бути підтвердження виконання, рахунок від постачальника, інформація про термін поставки,
  власне повідомлення користувача про виконану дію тощо.
- Вказуй id існуючого завдання (з наданого списку) і коротко опиши, що саме змінилося.
- Не вигадуй оновлення, якщо зв'язок із конкретним завданням неочевидний.

ФОРМАТ ВІДПОВІДІ — лише валідний JSON, без жодного тексту до чи після:
{
  "new_tasks": [
    {
      "title": "коротка назва завдання",
      "project_id": "id проєкту зі списку або null",
      "source_text": "коротка цитата-джерело з повідомлення (до 100 символів)",
      "source_sender": "хто написав",
      "reasoning": "одне речення — чому це нове завдання і чому саме цей проєкт"
    }
  ],
  "task_updates": [
    {
      "task_id": "id існуючого завдання зі списку відкритих завдань",
      "new_status": "in_progress | done | other",
      "comment": "що саме сталося, коротко",
      "source_text": "коротка цитата-джерело (до 100 символів)",
      "source_sender": "хто написав"
    }
  ]
}

Якщо нових завдань чи оновлень немає — поверни порожні масиви. Без жодних пояснень поза JSON."""


def analyze_day(
    branch_a_messages: list[dict],
    branch_b_messages: list[dict],
    projects: list[dict],
    open_tasks: list[dict],
    api_key: str | None = None,
) -> dict:
    """
    Виконує Крок 2: один виклик Gemini, що повертає нові завдання й оновлення статусу.
    """
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    user_content = "\n\n".join([
        _format_messages_block(branch_a_messages, "Повідомлення від дозволених контактів (шукай НОВІ завдання)"),
        _format_messages_block(branch_b_messages, "Усі повідомлення дня (шукай ОНОВЛЕННЯ статусу)"),
        _format_projects_block(projects),
        _format_open_tasks_block(open_tasks),
    ])

    response = client.models.generate_content(
        model=MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=4000,
        ),
    )

    raw_text = response.text.strip()

    # На випадок, якщо модель все ж обгорне відповідь у markdown-код-блок
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini повернув не-JSON відповідь: {e}\nВідповідь: {raw_text[:500]}")


if __name__ == "__main__":
    import sys
    from log_filter import filter_log

    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    contacts_path = sys.argv[2] if len(sys.argv) > 2 else None
    projects_path = sys.argv[3] if len(sys.argv) > 3 else None

    if not all([log_path, contacts_path, projects_path]):
        print("Використання: python task_analyzer.py <лог.json> <контакти.json> <проєкти.json>")
        sys.exit(1)

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    filtered = filter_log(log_data, contacts_path)
    projects = load_projects(projects_path)
    open_tasks = []  # для першого тестового прогону таблиця завдань ще порожня

    result = analyze_day(
        branch_a_messages=filtered["branch_a_new_tasks"],
        branch_b_messages=filtered["branch_b_status_updates"],
        projects=projects,
        open_tasks=open_tasks,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
