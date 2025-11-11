import os
import json
from typing import List
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# =========================
# Налаштування Google Sheets
# =========================

def _get_credentials():
    """
    Отримує облікові дані з GOOGLE_CREDENTIALS_JSON (як у інших модулях).
    """
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not creds_json:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON не задано")
    info = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    return Credentials.from_service_account_info(info, scopes=scopes)

_SPREADSHEET_ID = os.environ.get("SHEETS_SPREADSHEET_ID")
if not _SPREADSHEET_ID:
    raise RuntimeError("SHEETS_SPREADSHEET_ID не задано")

# Діапазони за замовчуванням
_RANGE_STRUCTURED = os.environ.get("SHEETS_RANGE_STRUCTURED", "Tasks!A:E")  # 5 колонок
_RANGE_LEGACY = os.environ.get("SHEETS_RANGE_LEGACY", "Tasks!A:C")         # 3 колонки (сумісність)

def _append_values(range_a1: str, values: List[List[str]]):
    creds = _get_credentials()
    service = build("sheets", "v4", credentials=creds)
    body = {"values": values}
    service.spreadsheets().values().append(
        spreadsheetId=_SPREADSHEET_ID,
        range=range_a1,
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()

# =========================
# Старий інтерфейс (сумісність)
# =========================
def append_task(title: str, description: str, tag: str):
    """
    Старий варіант: 3 колонки.
    Не чіпаємо для сумісності з існуючими місцями виклику.
    """
    _append_values(_RANGE_LEGACY, [[title, description, tag]])

# =========================
# Новий інтерфейс (5 колонок, S2)
# =========================
def append_task_structured(name: str, tag: str, deadline: str, priority: str, description: str):
    """
    Новий варіант: запис у 5 колонок:
      A: Назва
      B: Тег
      C: Дедлайн
      D: Пріоритет
      E: Опис
    """
    _append_values(_RANGE_STRUCTURED, [[name, tag, deadline, priority, description]])
