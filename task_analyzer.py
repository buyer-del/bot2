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

КОНТЕКСТ:
- Користувач — закупівельник, відповідає за замовлення комплектуючих для ліфтів.
- ПРОЄКТ = конкретний ліфт, що будується або монтується. Проєкт завжди має номер
  (наприклад 259, 254, 243) і/або адресу чи назву замовника. Проєкт НЕ є дією,
  НЕ є завданням, НЕ є питанням чи обговоренням — це фізичний об'єкт (ліфт).
  Назва проєкту НІКОЛИ не містить дієслів (замовити, оплатити, перевірити тощо).
- ЗАВДАННЯ = конкретна дія закупівельника (замовити, узгодити, перевірити тощо),
  яка може відноситись до одного з проєктів або бути без прив'язки до конкретного ліфта.
- В переписці проєкт згадують коротким номером ("259", "об'єкт 259", "259-ий", "243")
  або по адресі/замовнику зі списку проєктів.

ПРАВИЛА ДЛЯ НОВИХ ЗАВДАНЬ (аналізуй тільки блок "Повідомлення від дозволених контактів"):
- Завдання = конкретна дія, яку має виконати закупівельник (замовити, узгодити, перевірити, забрати тощо).
- Одне повідомлення може містити кілька різних завдань — знаходь усі.
- НЕ створюй завдання з того, що вже є у списку "Відкриті завдання" — це призведе до дублів.
- Відрізняй нову дію від простого обговорення чи питання без конкретної дії для закупівельника.
- Якщо в тексті є явна згадка проєкту (номер ліфта, адреса) — вкажи project_id з наданого
  списку проєктів. Якщо проєкт впізнати не вдалось або згадки немає — постав project_id: null.
- ВАЖЛИВО: якщо в переписці згадується номер ліфта якого немає в списку проєктів (наприклад
  "243") — це означає, що проєкт існує але ще не доданий до списку. Постав project_id: null,
  але згадай номер у полі reasoning. НЕ вигадуй назву проєкту — це не твоя задача.
- КОРОТКІ ПОВІДОМЛЕННЯ ("ОК", "добре", "зрозумів", "так", "ні" — менше 4 слів): якщо
  немає чіткого контексту попередніх повідомлень у тому самому чаті/листі — НЕ створюй
  з них нових завдань. Такі відповіді частіше є підтвердженням вже існуючих дій, а не
  постановкою нових завдань.

ПРАВИЛА ДЛЯ ОНОВЛЕНЬ СТАТУСУ (аналізуй блок "Усі повідомлення дня"):
- Дивись, чи якесь повідомлення стосується ВЖЕ ІСНУЮЧОГО завдання зі списку "Відкриті завдання".
- Це може бути підтвердження виконання, рахунок від постачальника, інформація про термін поставки,
  власне повідомлення користувача про виконану дію тощо.
- Вказуй id існуючого завдання (з наданого списку) і коротко опиши, що саме змінилося.
- Не вигадуй оновлення, якщо зв'язок із конкретним завданням неочевидний.
- "ОК", "добре" від дозволеного контакту у відповідь на запит — це може бути підтвердження
  виконання або дозвіл на дію, шукай відповідне відкрите завдання зі списку.

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
      "task_id": "id існуючого завдання",
      "new_status": "in_progress | done | other",
      "comment": "до 10 слів що змінилось",
      "source_text": "цитата до 60 символів",
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

    # Логуємо запит
    _write_debug("ЗАПИТ ДО GEMINI", user_content)

    response = client.models.generate_content(
        model=MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=16000,
        ),
    )

    raw_text = response.text.strip()

    # Логуємо сиру відповідь
    _write_debug("СИРА ВІДПОВІДЬ GEMINI", raw_text)

    # На випадок, якщо модель все ж обгорне відповідь у markdown-код-блок
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        result = json.loads(raw_text)
        # Логуємо розпарсений результат
        _write_debug("РОЗПАРСЕНИЙ РЕЗУЛЬТАТ", json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except json.JSONDecodeError as e:
        _write_debug("ПОМИЛКА ПАРСИНГУ JSON", f"Помилка: {e}\nТекст: {raw_text}")
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
