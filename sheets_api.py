"""
Модуль роботи з Google Таблицею для зберігання завдань.

Замінює тимчасовий storage.py (локальний JSON-файл).
Інтерфейс функцій лишається тим самим, що і в storage.py —
решта коду бота не змінюється.

Структура колонок таблиці:
A: ID | B: Дата створення | C: Проєкт | D: Назва завдання |
E: Статус | F: Від кого | G: Джерело (цитата) | H: Коментар

Авторизація: сервісний ключ з GOOGLE_CREDENTIALS_JSON (Render Environment).
ID таблиці: з SHEETS_SPREADSHEET_ID (Render Environment).
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

# Індекси колонок (0-based для читання, 1-based для запису через A1 нотацію)
COL_ID = 0        # A
COL_DATE = 1      # B
COL_PROJECT = 2   # C
COL_TITLE = 3     # D
COL_STATUS = 4    # E
COL_SENDER = 5    # F
COL_SOURCE = 6    # G
COL_COMMENT = 7   # H


def _get_service():
    """Авторизація через сервісний акаунт з GOOGLE_CREDENTIALS_JSON."""
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not creds_json:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON не задано в змінних середовища")
    creds_data = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_data, scopes=SCOPES)
    return build("sheets", "v4", credentials=creds)


def _get_all_rows() -> list[list]:
    """Повертає всі рядки таблиці (без заголовка)."""
    service = _get_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A2:H",
    ).execute()
    return result.get("values", [])


def _ensure_header():
    """Перевіряє і створює заголовок таблиці, якщо він відсутній."""
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
    """Визначає наступний вільний ID на основі наявних рядків."""
    if not rows:
        return 1
    ids = []
    for row in rows:
        try:
            ids.append(int(row[COL_ID]))
        except (IndexError, ValueError):
            pass
    return max(ids) + 1 if ids else 1


# ─── Публічний інтерфейс (аналогічний storage.py) ───────────────────────────

def get_projects() -> list[dict]:
    """
    Повертає список проєктів із файлу projects.json.
    Проєкти зберігаються у файлі, а не в таблиці —
    таблиця лише для завдань.
    """
    projects_path = os.environ.get("PROJECTS_FILE", "projects.json")
    if not os.path.exists(projects_path):
        return []
    with open(projects_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("projects", [])


def set_projects(projects: list[dict]):
    """Оновлює список проєктів у файлі projects.json."""
    projects_path = os.environ.get("PROJECTS_FILE", "projects.json")
    try:
        with open(projects_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"projects": []}
    data["projects"] = projects
    with open(projects_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_open_tasks() -> list[dict]:
    """Повертає всі завдання зі статусом, що не є 'done'."""
    rows = _get_all_rows()
    tasks = []
    for row in rows:
        # Доповнюємо рядок до потрібної довжини якщо є порожні колонки
        while len(row) < 8:
            row.append("")
        status = row[COL_STATUS]
        if status != "done":
            tasks.append({
                "id": row[COL_ID],
                "created_date": row[COL_DATE],
                "project_id": row[COL_PROJECT],
                "title": row[COL_TITLE],
                "status": status,
                "source_sender": row[COL_SENDER],
                "source_text": row[COL_SOURCE],
                "comment": row[COL_COMMENT],
            })
    return tasks


def add_task(title: str, project_id: str | None, source_text: str, source_sender: str) -> dict:
    """Додає нове завдання в таблицю і повертає його як dict."""
    _ensure_header()
    rows = _get_all_rows()
    task_id = _next_id(rows)
    today = str(date.today())
    project = project_id or ""
    status = "open"

    service = _get_service()
    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A:H",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [[
            task_id, today, project, title, status, source_sender, source_text, ""
        ]]},
    ).execute()

    logger.info("Додано завдання id=%s: %s", task_id, title)
    return {
        "id": task_id,
        "created_date": today,
        "project_id": project,
        "title": title,
        "status": status,
        "source_sender": source_sender,
        "source_text": source_text,
        "comment": "",
    }


def update_task_status(task_id: int | str, new_status: str, comment: str):
    """Оновлює статус і коментар існуючого завдання за його ID."""
    service = _get_service()
    rows = _get_all_rows()

    for i, row in enumerate(rows):
        if not row:
            continue
        try:
            if str(row[COL_ID]) == str(task_id):
                # Рядок у таблиці = i+2 (рядок 1 — заголовок, рядки нумеруються з 1)
                row_num = i + 2
                service.spreadsheets().values().update(
                    spreadsheetId=SPREADSHEET_ID,
                    range=f"{SHEET_NAME}!E{row_num}:H{row_num}",
                    valueInputOption="RAW",
                    body={"values": [[new_status, row[COL_SENDER] if len(row) > COL_SENDER else "", row[COL_SOURCE] if len(row) > COL_SOURCE else "", comment]]},
                ).execute()
                logger.info("Оновлено завдання id=%s: статус=%s", task_id, new_status)
                return
        except (IndexError, ValueError):
            continue

    logger.warning("Завдання id=%s не знайдено для оновлення", task_id)


def get_task_by_id(task_id: int | str) -> dict | None:
    """Повертає завдання за ID або None якщо не знайдено."""
    rows = _get_all_rows()
    for row in rows:
        if not row:
            continue
        try:
            if str(row[COL_ID]) == str(task_id):
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
        except (IndexError, ValueError):
            continue
    return None
