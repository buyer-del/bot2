"""
Сховище завдань і проєктів.

ТИМЧАСОВЕ РІШЕННЯ для першого збирання бота: локальний JSON-файл замість
Google Таблиці. Це дозволяє перевірити всю логіку (Кроки 1-4 архітектури)
без додаткової складності авторизації Google Sheets API на цьому етапі.

Коли основна логіка підтвердиться робочою — обмін цього модуля на
sheets_api.py (запис у Google Таблицю) робиться без зміни решти бота,
бо інтерфейс (get_open_tasks/add_task/update_task/...) лишається той самий.
"""

import json
import os
from datetime import date

STORAGE_PATH = os.environ.get("STORAGE_PATH", "storage.json")


def _load() -> dict:
    if not os.path.exists(STORAGE_PATH):
        return {"tasks": [], "projects": [], "next_task_id": 1}
    with open(STORAGE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict):
    with open(STORAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_projects() -> list[dict]:
    return _load()["projects"]


def set_projects(projects: list[dict]):
    """Перезаписує список проєктів (використовується командою синхронізації, Розділ 4)."""
    data = _load()
    data["projects"] = projects
    _save(data)


def get_open_tasks() -> list[dict]:
    data = _load()
    return [t for t in data["tasks"] if t["status"] != "done"]


def add_task(title: str, project_id: str | None, source_text: str, source_sender: str) -> dict:
    data = _load()
    task_id = data["next_task_id"]
    task = {
        "id": task_id,
        "title": title,
        "project_id": project_id,
        "status": "open",
        "source_text": source_text,
        "source_sender": source_sender,
        "created_date": str(date.today()),
        "history": [],
    }
    data["tasks"].append(task)
    data["next_task_id"] = task_id + 1
    _save(data)
    return task


def update_task_status(task_id: int, new_status: str, comment: str):
    data = _load()
    for t in data["tasks"]:
        if t["id"] == task_id:
            t["status"] = new_status
            t["history"].append({"date": str(date.today()), "comment": comment, "new_status": new_status})
            break
    _save(data)


def get_task_by_id(task_id: int) -> dict | None:
    data = _load()
    for t in data["tasks"]:
        if t["id"] == task_id:
            return t
    return None
