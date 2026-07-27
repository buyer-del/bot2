"""
Крок 6 архітектури: складання щоденного звіту.

Винесено в окремий модуль свідомо — промт потребуватиме подальшого
доопрацювання незалежно від решти логіки (фільтрація, аналіз завдань).
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

# Кешований genai-клієнт (той самий підхід, що й у log_filter.py /
# task_analyzer.py) — один клієнт на весь модуль.
_genai_client_cache = None


def _get_genai_client(api_key: str | None = None):
    global _genai_client_cache
    if _genai_client_cache is not None:
        return _genai_client_cache
    _genai_client_cache = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
    return _genai_client_cache

REPORT_SYSTEM_PROMPT = """Твоя задача: На основі наданої переписки та списку завдань скласти короткий щоденний звіт про роботу закупівельника.

Вхідні дані:
- JSON-масив повідомлень з полями: "t" (час), "from" (відправник), "chat" (платформа і чат), "msg" (текст)
- Список завдань які були відкриті або змінились за день

Контекст: Головна особа — автор повідомлень підписаний як "Я" або "Максим". Звіт складається виключно про його дії.

Перед тим як писати звіт — прочитай всю переписку і зіставте її зі списком завдань. Зрозумій що відбувалось за день: які завдання просувались, що виникло нового, де були труднощі. Ця внутрішня картина — основа для звіту.

Правила написання: Писати минулим часом. Переважно недоконаний вид — опрацьовував, узгоджував, уточнював. Але ~30% дієслів — доконаний вид там де дія явно завершена — надіслав, отримав, погодив, організував. Короткими природними реченнями, діловим стилем. КОЖНЕ РЕЧЕННЯ З НОВОГО РЯДКА — жодних речень підряд в одному рядку. Жодних тире, списків, маркерів, крапок з комою між реченнями. 6-8 речень загалом. Кожне речення — 5-10 слів, максимум 12. Не вказувати з ким саме узгоджувалось — тільки суть дії. Одна тема в кількох чатах — одне речення у звіті.

Ігнорувати: привітання, прощання, "ок", "зрозумів", порожні відповіді, безрезультатні запити.

Не згадувати: імена людей, номери телефонів, назви чатів, месенджери, посади людей крім конструктора та постачальника.

Заборонені слова: обговорив, переговорив, поговорив, поспілкувався.

Додатково: Якщо є номери проектів або ліфтів (#225, #253 тощо) — згадуй їх у реченні. Якщо є згадки про оплату — тільки так: отримав рахунок / рахунок на оплаті / підтверджував оплату.

Дату бери з переданих даних.

Формат виходу: ЗВІТ ЗА {дата}: [речення] [речення] ..."""


def _write_debug(section: str, content: str):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {section}\n")
            f.write(f"{'='*60}\n")
            f.write(content)
            f.write("\n")
    except Exception as e:
        logger.warning("Не вдалось записати debug-лог звіту: %s", e)


def _format_selection_for_report(selection: list[dict]) -> str:
    """
    Форматує вибірку у JSON для промту звіту — той самий формат що й для
    аналізу завдань, щоб Gemini бачив повний контекст: хто, коли, з якого
    каналу. Поле task_source не передаємо — для звіту воно не потрібне.
    """
    compact = []
    for m in selection:
        compact.append({
            "id": m.get("id", ""),
            "t": m.get("t", ""),
            "from": m.get("from", ""),
            "chat": m.get("chat", ""),
            "msg": m.get("msg", ""),
        })
    return json.dumps(compact, ensure_ascii=False)


def _format_tasks_for_report(tasks: list[dict]) -> str:
    """Форматує список завдань для промту звіту."""
    if not tasks:
        return "[]"
    compact = []
    for t in tasks:
        compact.append({
            "id": t.get("id", ""),
            "title": t.get("title", ""),
            "project_id": t.get("project_id", ""),
            "status": t.get("status", ""),
        })
    return json.dumps(compact, ensure_ascii=False)


def generate_report(
    selection: list[dict],
    log_date: str,
    tasks: list[dict] | None = None,
    api_key: str | None = None,
) -> str:
    """
    Генерує текст щоденного звіту.
    selection — результат build_selection() з log_filter.py (вже очищений від шуму).
    log_date — дата з файлу логу (наприклад log_data["export_date"]).
    tasks — список завдань які були відкриті або змінились за день.
    """
    client = _get_genai_client(api_key)

    formatted_messages = _format_selection_for_report(selection)
    formatted_tasks = _format_tasks_for_report(tasks or [])

    user_content = (
        f"Дата: {log_date}\n\n"
        f"Переписка за день:\n{formatted_messages}\n\n"
        f"Завдання (відкриті або змінені за день):\n{formatted_tasks}"
    )

    _write_debug("SYSTEM PROMPT (ЗВІТ)", REPORT_SYSTEM_PROMPT)
    _write_debug("ЗАПИТ ДО GEMINI (ЗВІТ)", user_content)

    response = client.models.generate_content(
        model=MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=REPORT_SYSTEM_PROMPT,
            max_output_tokens=8000,
        ),
    )

    result = response.text.strip()
    _write_debug("ВІДПОВІДЬ GEMINI (ЗВІТ)", result)
    return result
