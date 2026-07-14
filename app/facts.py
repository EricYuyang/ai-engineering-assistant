"""
Long-term fact storage: persists facts extracted from conversations so
they survive across sessions.  Separate from the LangGraph checkpointer
(short-term, per-thread history) by design — different lifecycles,
different access patterns.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from app.config import settings


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.facts_db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  project_name TEXT NOT NULL,"
        "  fact TEXT NOT NULL,"
        "  created_at TEXT NOT NULL"
        ")"
    )
    return conn


def save_facts(project_name: str, facts: list[str]) -> None:
    if not facts:
        return
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT INTO facts (project_name, fact, created_at) VALUES (?, ?, ?)",
        [(project_name, fact, now) for fact in facts],
    )
    conn.commit()
    conn.close()


def load_facts(project_name: str) -> list[str]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT fact FROM facts WHERE project_name = ? ORDER BY created_at",
        (project_name,),
    ).fetchall()
    conn.close()
    return [row[0] for row in rows]
