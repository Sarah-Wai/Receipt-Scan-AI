# backend/db/sqlite_reader.py

from __future__ import annotations

import sqlite3
from pathlib import Path

from config import DB_PATH


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """
    Initialise the database schema.
    Single source of truth lives in sqlite_writer.init_sqlite_schema().
    This function is the one called at app startup.
    """
    from db.sqlite_writer import init_sqlite_schema  # avoid circular import at module level
    init_sqlite_schema(Path(DB_PATH))