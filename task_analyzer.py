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
import logging
from datetime import datetime
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"
logger = logging.getLogger(__name__)
DEBUG_LOG = os.environ.get("DEBUG_LOG", "gemini_debug.log")


def _write_debug(section: str, content: str):
    """Записує запит/відповідь Gemini у файл для діагностики."""
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {section}\n")
            f.write(f"{'='*60}\n")
            f.write(content)
            f.write("\n")
    except Exception as e:
        logger.warning("Не вдалось записати debug-лог: %s", e)


def load_projects(projects_path: str) -> list[dict]:
    with open(projects_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    projects = data["projects"]
    # Нормалізація: якщо є старе поле short_number — конвертуємо в short_numbers
    for p in projects:
        if "short_numbers" not in p and "short_number" in p:
            p["short_numbers"] = [p["short_number"]] if p["short_number"] else []
    return projects


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
        short_numbers = p.get("short_numbers") or p.get("short_number", "")
        if isinstance(short_numbers, list):
            short_str = ", ".join(short_numbers) if short_numbers else "—"
        else:
            short_str = short_numbers or "—"
        lines.append(f"- id={p['id']} | короткі номери в переписці: \"{short_str}\" | повна назва: {p['full_name']}")
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

ВХІДНІ ДАНІ — JSON-масив повідомлень. Кожне повідомлення має поля:
- "t" — час
- "from" — відправник ("Я" = сам користувач)
- "chat" — платформа і назва чату
- "msg" — текст повідомлення
- "task_source": true — це повідомлення від довіреного колеги/керівника, який може ставити НОВІ завдання

КОНТЕКСТ:
- Користувач — закупівельник, відповідає за замовлення комплектуючих для ліфтів.
- ПРОЄКТ = конкретний ліфт (фізичний об'єкт). Завжди має номер (259, 254, 243 тощо)
  і/або адресу чи назву замовника. Проєкт НЕ є дією — назва ніколи не містить дієслів.
- ЗАВДАННЯ = конкретна дія закупівельника (замовити, узгодити, перевірити тощо).
- Номер проєкту в переписці може бути у різних формах: "259", "259-ий", "об'єкт 259" тощо.

ПРАВИЛА ДЛЯ НОВИХ ЗАВДАНЬ — шукай ЛИШЕ в повідомленнях де "task_source": true:
- Завдання = конкретна дія, яку має виконати закупівельник.
- Одне повідомлення може містити кілька завдань — знаходь усі.
- НЕ дублюй те що вже є у списку відкритих завдань.
- Короткі відповіді ("ОК", "добре", менше 4 слів) без явної нової дії — НЕ є новими завданнями.
- Якщо є згадка проєкту — вкажи project_id зі списку. Якщо немає — project_id: null.
- Якщо номер ліфта є але його немає в списку проєктів — project_id: null, згадай номер у reasoning.

ПРАВИЛА ДЛЯ ОНОВЛЕНЬ СТАТУСУ — аналізуй УСІ повідомлення (з task_source і без):
- Чи стосується повідомлення вже існуючого завдання зі списку відкритих завдань?
- ВАЖЛИВО: task_updates стосується ВИКЛЮЧНО завдань які вже є у переданому списку
  відкритих завдань. Якщо список порожній — task_updates теж має бути порожнім масивом.
- НЕ створюй оновлення для завдань які ти щойно додав у new_tasks — вони ще не існують.
- task_id ЗАВЖДИ числовий (беремо поле "id" з переданого списку відкритих завдань).
  Якщо не знайшов відповідного id — не створюй оновлення взагалі.
- Підтвердження, рахунок, інформація про доставку, власна дія користувача — це оновлення.
- "ОК" від довіреного колеги у відповідь на запит — може бути підтвердженням дозволу на дію.
- Не вигадуй оновлення якщо зв'язок неочевидний.

ФОРМАТ ВІДПОВІДІ — лише валідний JSON, без жодного тексту до чи після:
{
  "new_tasks": [
    {
      "title": "коротка назва завдання (до 10 слів)",
      "project_id": "id проєкту зі списку або null",
      "source_text": "цитата до 60 символів",
      "source_sender": "хто написав",
      "reasoning": "до 10 слів чому це завдання і який проєкт"
    }
  ],
  "task_updates": [
    {
      "task_id": "числовий id завдання зі списку відкритих завдань",
      "title": "назва завдання зі списку відкритих завдань",
      "new_status": "in_progress | done | other",
      "comment": "до 10 слів що змінилось",
      "source_text": "цитата до 60 символів",
      "source_sender": "хто написав"
    }
  ]
}

Якщо нових завдань чи оновлень немає — поверни порожні масиви. Без жодних пояснень поза JSON."""


def _format_selection_block(selection: list[dict]) -> str:
    """Форматує вибірку у компактний JSON-рядок для промту."""
    compact = []
    for m in selection:
        item = {
            "t": m.get("t", ""),
            "from": m.get("from", ""),
            "chat": m.get("chat", ""),
            "msg": m.get("msg", ""),
        }
        if m.get("task_source"):
            item["task_source"] = True
        compact.append(item)
    return json.dumps(compact, ensure_ascii=False)


def analyze_day(
    selection: list[dict],
    projects: list[dict],
    open_tasks: list[dict],
    api_key: str | None = None,
) -> dict:
    """
    Виконує Крок 2: один виклик Gemini з компактною вибіркою.
    selection — результат build_selection() з log_filter.py
    """
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    user_content = "\n\n".join([
        "### Вибірка повідомлень (повідомлення з task_source=true — від довірених осіб, можуть ставити нові завдання)",
        _format_selection_block(selection),
        _format_projects_block(projects),
        _format_open_tasks_block(open_tasks),
    ])

    _write_debug("SYSTEM PROMPT", SYSTEM_PROMPT)
    _write_debug("ЗАПИТ ДО GEMINI (user content)", user_content)

    # Для зручності читання — окремо логуємо вибірку у форматованому вигляді
    try:
        selection_start = user_content.find("[{")
        if selection_start != -1:
            selection_end = user_content.find("]", selection_start) + 1
            raw_selection = user_content[selection_start:selection_end]
            parsed = json.loads(raw_selection)
            _write_debug("ВИБІРКА (форматована)", json.dumps(parsed, ensure_ascii=False, indent=2))
    except Exception:
        pass

    response = client.models.generate_content(
        model=MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=16000,
        ),
    )

    raw_text = response.text.strip()
    _write_debug("СИРА ВІДПОВІДЬ GEMINI", raw_text)

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        result = json.loads(raw_text)
        _write_debug("РОЗПАРСЕНИЙ РЕЗУЛЬТАТ", json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except json.JSONDecodeError as e:
        _write_debug("ПОМИЛКА ПАРСИНГУ JSON", f"Помилка: {e}\nТекст: {raw_text}")
        raise ValueError(f"Gemini повернув не-JSON відповідь: {e}\nВідповідь: {raw_text[:500]}")


if __name__ == "__main__":
    import sys
    from log_filter import build_selection

    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    contacts_path = sys.argv[2] if len(sys.argv) > 2 else None
    projects_path = sys.argv[3] if len(sys.argv) > 3 else None

    if not all([log_path, contacts_path, projects_path]):
        print("Використання: python task_analyzer.py <лог.json> <контакти.json> <проєкти.json>")
        sys.exit(1)

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    projects = load_projects(projects_path)
    selection = build_selection(log_data, contacts_path, projects_path)
    open_tasks = []

    result = analyze_day(selection=selection, projects=projects, open_tasks=open_tasks)
    print(json.dumps(result, ensure_ascii=False, indent=2))
