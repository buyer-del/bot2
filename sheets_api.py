"""
Модуль роботи з Google Таблицею для зберігання завдань.

Структура колонок таблиці:
A: ID | B: Дата створення | C: Проєкт | D: Назва завдання |
E: Статус | F: Від кого | G: Джерело (цитата) | H: Коментар

Статуси (українською):
- відкрито  — нове завдання, ще нічого не робилось
- в роботі  — розпочато: надіслав запит, чекаю відповіді, веду переговори
- виконано  — завдання закрите, дія завершена
- відкладено — поки не актуально, але не закрите
"""

import os
import json
import logging
from datetime import date
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

logger = logging.getLogger(__name__)

SPREADSHEET_ID = os.environ.get("SHEETS_SPREADSHEET_ID")
SHEET_NAME = "Завдання"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Статуси українською
STATUS_OPEN = "відкрито"
STATUS_IN_PROGRESS = "в роботі"
STATUS_DONE = "виконано"
STATUS_POSTPONED = "відкладено"

# Відображення англійських статусів (від Gemini) на українські
STATUS_MAP = {
    "open": STATUS_OPEN,
    "in_progress": STATUS_IN_PROGRESS,
    "done": STATUS_DONE,
    "other": STATUS_POSTPONED,
    # На випадок якщо Gemini поверне українські одразу
    "відкрито": STATUS_OPEN,
    "в роботі": STATUS_IN_PROGRESS,
    "виконано": STATUS_DONE,
    "відкладено": STATUS_POSTPONED,
}

# Індекси колонок
COL_ID = 0        # A
COL_DATE = 1      # B
COL_PROJECT = 2   # C
COL_TITLE = 3     # D
COL_STATUS = 4    # E
COL_SENDER = 5    # F
COL_SOURCE = 6    # G
COL_COMMENT = 7   # H


def normalize_status(status: str) -> str:
    """Перетворює будь-який формат статусу на українську назву."""
    return STATUS_MAP.get(status.strip().lower(), STATUS_OPEN)


# Кешований service-об'єкт (побудований один раз, перевикористовується
# всіма функціями модуля). Раніше _get_service() будував новий service
# при КОЖНОМУ виклику (кожне read/write у таблицю) — це створювало
# постійний потік важких тимчасових об'єктів (парсинг discovery-документа
# Sheets API) і призводило до поступового росту споживання пам'яті
# процесу, особливо помітного під час довгого діалогу підтвердження
# (де на кожен тап кнопки викликалось по кілька storage-функцій).
#
# Токен доступу Google оновлюється бібліотекою google-auth автоматично
# "під капотом" при кожному запиті, незалежно від того, новий це service
# чи перевикористаний старий — тож кешування тут ніяк не впливає на
# коректність оновлення доступу.
_service_cache = None


def _get_service():
    global _service_cache
    if _service_cache is not None:
        return _service_cache

    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not creds_json:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON не задано")
    creds_data = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_data, scopes=SCOPES)
    _service_cache = build("sheets", "v4", credentials=creds)
    return _service_cache


def _get_all_rows() -> list[list]:
    service = _get_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A2:H",
    ).execute()
    return result.get("values", [])


def _ensure_header():
    service = _get_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A1:H1",
    ).execute()
    existing = result.get("values", [])
    if not existing or not existing[0]:
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{SHEET_NAME}!A1:H1",
            valueInputOption="RAW",
            body={"values": [["ID", "Дата", "Проєкт", "Назва завдання", "Статус", "Від кого", "Джерело", "Коментар"]]},
        ).execute()


def _next_id(rows: list[list]) -> int:
    if not rows:
        return 1
    ids = []
    for row in rows:
        try:
            ids.append(int(row[COL_ID]))
        except (IndexError, ValueError):
            pass
    return max(ids) + 1 if ids else 1


def _row_to_dict(row: list) -> dict:
    while len(row) < 8:
        row.append("")
    return {
        "id": row[COL_ID],
        "created_date": row[COL_DATE],
        "project_id": row[COL_PROJECT],
        "title": row[COL_TITLE],
        "status": row[COL_STATUS],
        "source_sender": row[COL_SENDER],
        "source_text": row[COL_SOURCE],
        "comment": row[COL_COMMENT],
    }


# ─── Проєкти ────────────────────────────────────────────────────────────────

def get_projects() -> list[dict]:
    projects_path = os.environ.get("PROJECTS_FILE", "projects.json")
    if not os.path.exists(projects_path):
        return []
    with open(projects_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("projects", [])


def set_projects(projects: list[dict]):
    projects_path = os.environ.get("PROJECTS_FILE", "projects.json")
    try:
        with open(projects_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"projects": []}
    data["projects"] = projects
    with open(projects_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── Завдання ────────────────────────────────────────────────────────────────

def get_open_tasks() -> list[dict]:
    """Повертає всі завдання крім виконаних."""
    rows = _get_all_rows()
    return [
        _row_to_dict(list(row))
        for row in rows
        if row and row[COL_STATUS] != STATUS_DONE
    ]


def get_done_tasks() -> list[dict]:
    """Повертає виконані завдання."""
    rows = _get_all_rows()
    return [
        _row_to_dict(list(row))
        for row in rows
        if row and row[COL_STATUS] == STATUS_DONE
    ]


def get_all_tasks() -> list[dict]:
    """Повертає всі завдання."""
    rows = _get_all_rows()
    return [_row_to_dict(list(row)) for row in rows if row]


def add_task(
    title: str,
    project_id: str | None,
    source_text: str,
    source_sender: str,
    status: str = STATUS_OPEN,
) -> dict:
    """Додає нове завдання в таблицю."""
    _ensure_header()
    rows = _get_all_rows()
    task_id = _next_id(rows)
    today = str(date.today())
    project = project_id or ""
    normalized_status = normalize_status(status)

    service = _get_service()
    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A:H",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [[
            task_id, today, project, title,
            normalized_status, source_sender, source_text, ""
        ]]},
    ).execute()

    logger.info("Додано завдання id=%s: %s [%s]", task_id, title, normalized_status)
    return {
        "id": task_id,
        "created_date": today,
        "project_id": project,
        "title": title,
        "status": normalized_status,
        "source_sender": source_sender,
        "source_text": source_text,
        "comment": "",
    }


def update_task_status(task_id: int | str, new_status: str, comment: str = ""):
    """Оновлює статус і коментар існуючого завдання."""
    service = _get_service()
    rows = _get_all_rows()
    normalized = normalize_status(new_status)

    for i, row in enumerate(rows):
        if not row:
            continue
        try:
            if str(row[COL_ID]) == str(task_id):
                row_num = i + 2
                while len(row) < 8:
                    row.append("")
                service.spreadsheets().values().update(
                    spreadsheetId=SPREADSHEET_ID,
                    range=f"{SHEET_NAME}!E{row_num}:H{row_num}",
                    valueInputOption="RAW",
                    body={"values": [[
                        normalized,
                        row[COL_SENDER],
                        row[COL_SOURCE],
                        comment if comment else row[COL_COMMENT],
                    ]]},
                ).execute()
                logger.info("Оновлено id=%s: статус=%s", task_id, normalized)
                return
        except (IndexError, ValueError):
            continue

    logger.warning("Завдання id=%s не знайдено", task_id)


def update_task_comment(task_id: int | str, comment: str):
    """Оновлює тільки коментар завдання."""
    service = _get_service()
    rows = _get_all_rows()

    for i, row in enumerate(rows):
        if not row:
            continue
        try:
            if str(row[COL_ID]) == str(task_id):
                row_num = i + 2
                service.spreadsheets().values().update(
                    spreadsheetId=SPREADSHEET_ID,
                    range=f"{SHEET_NAME}!H{row_num}",
                    valueInputOption="RAW",
                    body={"values": [[comment]]},
                ).execute()
                logger.info("Оновлено коментар id=%s", task_id)
                return
        except (IndexError, ValueError):
            continue

    logger.warning("Завдання id=%s не знайдено для оновлення коментаря", task_id)


def delete_task(task_id: int | str):
    """Видаляє рядок завдання з таблиці."""
    service = _get_service()
    rows = _get_all_rows()

    for i, row in enumerate(rows):
        if not row:
            continue
        try:
            if str(row[COL_ID]) == str(task_id):
                row_num = i + 2  # +1 заголовок, +1 бо індекс з 0
                # Отримуємо sheetId для batchUpdate
                spreadsheet = service.spreadsheets().get(
                    spreadsheetId=SPREADSHEET_ID
                ).execute()
                sheet_id = next(
                    s["properties"]["sheetId"]
                    for s in spreadsheet["sheets"]
                    if s["properties"]["title"] == SHEET_NAME
                )
                service.spreadsheets().batchUpdate(
                    spreadsheetId=SPREADSHEET_ID,
                    body={"requests": [{
                        "deleteDimension": {
                            "range": {
                                "sheetId": sheet_id,
                                "dimension": "ROWS",
                                "startIndex": row_num - 1,
                                "endIndex": row_num,
                            }
                        }
                    }]},
                ).execute()
                logger.info("Видалено завдання id=%s", task_id)
                return
        except (IndexError, ValueError):
            continue

    logger.warning("Завдання id=%s не знайдено для видалення", task_id)


def get_task_by_id(task_id: int | str) -> dict | None:
    rows = _get_all_rows()
    for row in rows:
        if not row:
            continue
        try:
            if str(row[COL_ID]) == str(task_id):
                return _row_to_dict(list(row))
        except (IndexError, ValueError):
            continue
    return None


# ─── Сесія (резервне збереження стану) ──────────────────────────────────────

SESSION_SHEET = "Сесія"

# Колонки листа Сесія
# A: Етап | B: Вибірка | C: Результат Gemini | D: Черга | E: Звіт | F: Дата логу


def _ensure_session_sheet():
    """Створює лист Сесія якщо його немає."""
    service = _get_service()
    spreadsheet = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
    sheets = [s["properties"]["title"] for s in spreadsheet["sheets"]]
    if SESSION_SHEET not in sheets:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={"requests": [{"addSheet": {"properties": {"title": SESSION_SHEET}}}]},
        ).execute()
        # Заголовок
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{SESSION_SHEET}!A1:F1",
            valueInputOption="RAW",
            body={"values": [["Етап", "Вибірка", "Результат Gemini", "Черга", "Звіт", "Дата логу"]]},
        ).execute()


def save_session(
    stage: str,
    selection: list | None = None,
    gemini_result: dict | None = None,
    queue: list | None = None,
    report_text: str | None = None,
    log_date: str = "",
):
    """
    Зберігає поточний стан сесії в Google Таблицю (один рядок, перезаписується).
    stage: 'filtered' | 'analysed' | 'queue' | 'report_pending' | 'report_ready' | 'done'
    """
    _ensure_session_sheet()
    service = _get_service()
    service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SESSION_SHEET}!A2:F2",
        valueInputOption="RAW",
        body={"values": [[
            stage,
            json.dumps(selection or [], ensure_ascii=False),
            json.dumps(gemini_result or {}, ensure_ascii=False),
            json.dumps(queue or [], ensure_ascii=False),
            report_text or "",
            log_date,
        ]]},
    ).execute()
    logger.info("Сесія збережена: етап=%s", stage)


def load_session() -> dict | None:
    """
    Завантажує збережений стан сесії.
    Повертає None якщо сесії немає або вона завершена.
    """
    try:
        _ensure_session_sheet()
        service = _get_service()
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{SESSION_SHEET}!A2:F2",
        ).execute()
        rows = result.get("values", [])
        if not rows or not rows[0]:
            return None
        row = rows[0]
        while len(row) < 6:
            row.append("")
        stage = row[0]
        if stage == "done" or not stage:
            return None
        return {
            "stage": stage,
            "selection": json.loads(row[1]) if row[1] else [],
            "gemini_result": json.loads(row[2]) if row[2] else {},
            "queue": json.loads(row[3]) if row[3] else [],
            "report_text": row[4],
            "log_date": row[5],
        }
    except Exception as e:
        logger.warning("Не вдалось завантажити сесію: %s", e)
        return None


def clear_session():
    """Очищає стан сесії після успішного завершення."""
    try:
        _ensure_session_sheet()
        service = _get_service()
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{SESSION_SHEET}!A2:F2",
            valueInputOption="RAW",
            body={"values": [["done", "", "", "", "", ""]]},
        ).execute()
        logger.info("Сесія очищена")
    except Exception as e:
        logger.warning("Не вдалось очистити сесію: %s", e)
