"""SQLite catalog for DAS phase picks.

A single flat ``picks`` table stores every pick produced by the pickers in
:mod:`dasieve.picker`, together with the source ``file_name`` and the
``author`` (the picking method, e.g. "trigger", "ar", "phasenet"). Re-running
the same file with the same method *replaces* the previous picks for that
(file_name, author) pair, so the catalog always holds the latest result per
method per file while keeping results from other files/methods intact.

Typical use::

    from dasieve.catalog import save_picks
    df = trigger_picker(patch, ...)
    save_picks(df, file_name="…/event.h5", author="trigger")

The database is a plain SQLite file (default ``~/DASieve/picks.sqlite``); query
it with any SQLite tool or with :func:`load_picks`.
"""

import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .picker import PICK_COLUMNS

DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~/DASieve"), "picks.sqlite")

# Order of columns persisted to the picks table. The first three are catalog
# metadata; the rest are the shared picker schema (PICK_COLUMNS).
_META_COLUMNS = ["file_name", "author", "created_at"]
_TABLE_COLUMNS = _META_COLUMNS + list(PICK_COLUMNS)

# Column -> SQLite type. onset_time / off_time are stored as ISO-8601 text
# (sortable); sample indices as INTEGER; everything else REAL/TEXT.
_COLUMN_TYPES = {
    "file_name": "TEXT NOT NULL",
    "author": "TEXT NOT NULL",
    "created_at": "TEXT NOT NULL",
    "distance": "REAL",
    "phase": "TEXT",
    "onset_sample": "INTEGER",
    "onset_time": "TEXT",
    "score": "REAL",
    "cft_at_onset": "REAL",
    "off_sample": "INTEGER",
    "off_time": "TEXT",
    "cft_at_off": "REAL",
}


def _connect(db_path):
    """Open a connection with sane defaults, creating parent dirs as needed."""
    db_path = os.path.abspath(os.path.expanduser(db_path))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")   # safer concurrent reads
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path=DEFAULT_DB_PATH):
    """Create the ``picks`` table and indexes if they don't already exist.

    Safe to call repeatedly. Returns the absolute database path.
    """
    cols_sql = ",\n    ".join(f'"{c}" {_COLUMN_TYPES[c]}' for c in _TABLE_COLUMNS)
    with _connect(db_path) as conn:
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS picks (\n'
            f'    id INTEGER PRIMARY KEY AUTOINCREMENT,\n'
            f'    {cols_sql}\n'
            f');'
        )
        # speeds up the replace-on-rerun delete and per-file/method queries
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_picks_file_author '
            'ON picks (file_name, author);'
        )
        conn.commit()
    return os.path.abspath(os.path.expanduser(db_path))


def _iso_or_none(value):
    """Coerce a datetime-like value to an ISO-8601 string, or None for NaT."""
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else ts.isoformat()


def _int_or_none(value):
    return None if pd.isna(value) else int(value)


def _float_or_none(value):
    return None if pd.isna(value) else float(value)


def _prepare_rows(df, file_name, author, created_at):
    """Turn a picks DataFrame into a list of value-tuples matching
    _TABLE_COLUMNS, with proper type coercion and NaN/NaT -> NULL."""
    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                file_name,
                author,
                created_at,
                _float_or_none(r.get("distance")),
                None if pd.isna(r.get("phase")) else str(r.get("phase")),
                _int_or_none(r.get("onset_sample")),
                _iso_or_none(r.get("onset_time")),
                _float_or_none(r.get("score")),
                _float_or_none(r.get("cft_at_onset")),
                _int_or_none(r.get("off_sample")),
                _iso_or_none(r.get("off_time")),
                _float_or_none(r.get("cft_at_off")),
            )
        )
    return rows


def save_picks(df, file_name, author, db_path=DEFAULT_DB_PATH, replace=True):
    """Persist a picks DataFrame to the catalog.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of a picker (columns == dasieve.picker.PICK_COLUMNS). May be
        empty (the previous picks for this file+author are still cleared when
        ``replace`` is True, so the catalog reflects the latest empty result).
    file_name : str
        Source data file the picks came from (full path or basename).
    author : str
        Picking method, e.g. "trigger", "ar", or "phasenet".
    db_path : str
        SQLite file path (created if missing).
    replace : bool
        If True (default), delete existing picks for this (file_name, author)
        before inserting, so only the latest run is kept. If False, append.

    Returns
    -------
    int : number of pick rows inserted.
    """
    init_db(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    rows = _prepare_rows(df, file_name, author, created_at)

    placeholders = ", ".join(["?"] * len(_TABLE_COLUMNS))
    col_list = ", ".join(f'"{c}"' for c in _TABLE_COLUMNS)
    insert_sql = f"INSERT INTO picks ({col_list}) VALUES ({placeholders});"

    with _connect(db_path) as conn:
        if replace:
            conn.execute(
                "DELETE FROM picks WHERE file_name = ? AND author = ?;",
                (file_name, author),
            )
        if rows:
            conn.executemany(insert_sql, rows)
        conn.commit()
    return len(rows)


def load_picks(db_path=DEFAULT_DB_PATH, file_name=None, author=None):
    """Read picks back into a DataFrame, optionally filtered by file/method."""
    if not os.path.exists(os.path.expanduser(db_path)):
        return pd.DataFrame(columns=["id"] + _TABLE_COLUMNS)

    clauses, params = [], []
    if file_name is not None:
        clauses.append("file_name = ?")
        params.append(file_name)
    if author is not None:
        clauses.append("author = ?")
        params.append(author)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"SELECT * FROM picks{where} ORDER BY file_name, author, distance, onset_sample;"

    with _connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    for col in ("onset_time", "off_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
