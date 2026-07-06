"""
Крок 6 архітектури: складання щоденного звіту.

Винесено в окремий модуль свідомо — цей промт уже робочий, але ще
потребуватиме доопрацювання незалежно від решти логіки (фільтрація,
аналіз завдань). Тримати його окремо дозволяє редагувати і тестувати
звіт, не торкаючись Кроків 1-5.
"""

import os
import logging
from datetime import datetime
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash"
logger = logging.getLogger(__name__)
DEBUG_LOG = os.environ.get("DEBUG_LOG", "gemini_debug.log")

REPORT_SYSTEM_PROMPT = """Твоя задача: На основі наданих переписок написати короткий щоденний звіт.

Контекст: Головна особа — автор повідомлень підписаний як "Я" або "Максим". Звіт складається виключно про його дії.

Правила: Писати минулим часом. Переважно недоконаний вид — опрацьовував, узгоджував, уточнював. Але ~30% дієслів — доконаний вид там де дія явно завершена — надіслав, отримав, погодив, організував. Короткими природними реченнями, діловим стилем. Жодних тире, списків, маркерів — лише прості речення, кожне з нового рядка. 6-8 речень загалом. Кожне речення — 5-10 слів, максимум 12. Не вказувати з ким саме узгоджувалось — тільки суть дії. Одна тема в кількох чатах — одне речення у звіті.

Ігнорувати: привітання, прощання, "ок", "зрозумів", дрібні робочі рахунки, порожні відповіді, безрезультатні запити, дії спрямовані до керівництва.

Не згадувати: імена людей, номери телефонів, назви чатів, месенджери, посади людей крім конструктора та постачальника.

Заборонені слова: обговорив, переговорив, поговорив, поспілкувався.

Додатково: Якщо є номери проектів або ліфтів (#225, #253 тощо) — згадуй їх у реченні. Якщо є згадки про оплату — тільки так: отримав рахунок / рахунок на оплаті / підтверджував оплату.

Дату бери з файлу логу.

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
    Форматує вибірку для промту звіту.
    Вибірка вже очищена від шуму в log_filter.py.
    """
    lines = []
    for m in selection:
        sender = m.get("from", "")
        lines.append(f"[{m.get('t', '')}] {sender}: {m.get('msg', '')}")
    return "\n".join(lines)


def generate_report(selection: list[dict], log_date: str, api_key: str | None = None) -> str:
    """
    Генерує текст щоденного звіту на основі вибірки повідомлень.
    selection — результат build_selection() з log_filter.py (вже очищений від шуму).
    log_date — дата з файлу логу (наприклад log_data["export_date"]).
    """
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    formatted = _format_selection_for_report(selection)
    user_content = f"Дата логу: {log_date}\n\nПереписка за день:\n{formatted}"

    _write_debug("ЗАПИТ ДО GEMINI (ЗВІТ)", user_content)

    response = client.models.generate_content(
        model=MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=REPORT_SYSTEM_PROMPT,
            max_output_tokens=4000,
        ),
    )

    result = response.text.strip()
    _write_debug("ВІДПОВІДЬ GEMINI (ЗВІТ)", result)
    return result
