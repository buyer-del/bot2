import os
import json
from datetime import datetime
from typing import List
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# =========================
# Налаштування Google Sheets
# =========================

def _get_credentials():
    """
    Отримує облікові дані з GOOGLE_CREDENTIALS_JSON.
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

# Діапазони
_RANGE_STRUCTURED = os.environ.get("SHEETS_RANGE_STRUCTURED", "Tasks!A:F")  # 6 колонок (час + 5 полів)
_RANGE_FALLBACK = os.environ.get("SHEETS_RANGE_FALLBACK", "Tasks!A:F")

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
# Запис fallback (AI недоступний)
# =========================
def append_task(raw_description: str):
    """
    Коли AI недоступний — записує лише час і опис.
    Інші стовпці залишаються порожні.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [ts, "", "", "", "", raw_description]
    _append_values(_RANGE_FALLBACK, [row])

# =========================
# Запис структурованих задач (AI)
# =========================
def append_task_structured(name: str, tag: str, deadline: str, priority: str, description: str):
    """
    Новий варіант: запис у 6 колонок:
      A: Час
      B: Назва
      C: Тег
      D: Дедлайн
      E: Пріоритет
      F: Опис
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [ts, name, tag, deadline, priority, description]
    _append_values(_RANGE_STRUCTURED, [row])
