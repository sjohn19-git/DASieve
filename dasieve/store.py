"""SQLite store for DAS picks and association results.

A single flat ``picks`` table stores every pick produced by the pickers in
:mod:`dasieve.picker`, labelled with the source ``file_name`` and the
``method`` (the picking method, e.g. "sta_lta", "phasenetdas"). Re-running
with the same ``(file_name, method)`` pair *replaces* the previous rows for
that pair -- no duplicates accumulate. The ``events`` / ``assignments``
tables written by :func:`save_associations` follow the same scheme, keyed on
``(file_name, method)`` where ``method`` is the associator (e.g. "gamma").

    from dasieve.store import save_picks
    df = trigger_picker(patch, ..., db_save=False)
    save_picks(df, file_name=".../event.h5", method="sta_lta")

The database is a plain SQLite file (default ``~/DASieve/dasieve.sqlite``);
query it with any SQLite tool or with :func:`load_picks_by_ids`.
"""

import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .picker import PICK_COLUMNS

DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~/DASieve"), "dasieve.sqlite")

# Order of columns persisted to the picks table. The first three are run
# metadata; the rest are the shared picker schema (PICK_COLUMNS).
_META_COLUMNS = ["file_name", "method", "created_at"]
_TABLE_COLUMNS = _META_COLUMNS + list(PICK_COLUMNS)

# Column -> SQLite type. onset_time / off_time are stored as ISO-8601 text
# (sortable); sample indices as INTEGER; everything else REAL/TEXT.
_COLUMN_TYPES = {
    "file_name": "TEXT NOT NULL",
    "method": "TEXT NOT NULL",
    "created_at": "TEXT NOT NULL",
    "distance": "REAL",
    "x": "REAL",
    "y": "REAL",
    "z": "REAL",
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
            'CREATE INDEX IF NOT EXISTS idx_picks_file_method '
            'ON picks (file_name, method);'
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


def _prepare_rows(df, file_name, method, created_at):
    """Turn a picks DataFrame into a list of value-tuples matching
    _TABLE_COLUMNS, with proper type coercion and NaN/NaT -> NULL."""
    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                file_name,
                method,
                created_at,
                _float_or_none(r.get("distance")),
                _float_or_none(r.get("x")),
                _float_or_none(r.get("y")),
                _float_or_none(r.get("z")),
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


def save_picks(df, file_name, method, db_path=DEFAULT_DB_PATH, replace=True):
    """Persist a picks DataFrame to the store.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of a picker (columns == dasieve.picker.PICK_COLUMNS). May be
        empty (the previous picks for this file+method are still cleared when
        ``replace`` is True, so the store reflects the latest empty result).
    file_name : str
        Source data file the picks came from (full path or basename).
    method : str
        Picking method, e.g. "sta_lta", "ar", or "phasenetdas".
    db_path : str
        SQLite file path (created if missing).
    replace : bool
        If True (default), delete existing picks for this (file_name, method)
        before inserting, so only the latest run is kept and no duplicates
        accumulate. If False, append.

    Returns
    -------
    int : number of pick rows inserted.
    """
    init_db(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    rows = _prepare_rows(df, file_name, method, created_at)

    placeholders = ", ".join(["?"] * len(_TABLE_COLUMNS))
    col_list = ", ".join(f'"{c}"' for c in _TABLE_COLUMNS)
    insert_sql = f"INSERT INTO picks ({col_list}) VALUES ({placeholders});"

    with _connect(db_path) as conn:
        if replace:
            _drop_associations_referencing(conn, file_name, method)
            conn.execute(
                "DELETE FROM picks WHERE file_name = ? AND method = ?;",
                (file_name, method),
            )
        if rows:
            conn.executemany(insert_sql, rows)
        conn.commit()
    return len(rows)


def _drop_associations_referencing(conn, file_name, method):
    """Delete association runs whose assignments reference picks about to be
    replaced.

    ``assignments.pick_id`` has a FOREIGN KEY to ``picks.id``, so the
    replace-on-rerun delete in :func:`save_picks` would otherwise fail with an
    IntegrityError -- and even without the constraint, association results
    built on picks that no longer exist are stale. Each affected run is
    removed whole (its ``events`` and ``assignments`` rows, keyed on the
    run's own (file_name, method)); re-run the associator to regenerate it.
    """
    has_assignments = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='assignments';"
    ).fetchone()
    if not has_assignments:
        return

    runs = conn.execute(
        "SELECT DISTINCT a.file_name, a.method FROM assignments a "
        "JOIN picks p ON a.pick_id = p.id "
        "WHERE p.file_name = ? AND p.method = ?;",
        (file_name, method),
    ).fetchall()
    for run_file, run_method in runs:
        conn.execute(
            "DELETE FROM assignments WHERE file_name = ? AND method = ?;",
            (run_file, run_method),
        )
        conn.execute(
            "DELETE FROM events WHERE file_name = ? AND method = ?;",
            (run_file, run_method),
        )
        print(
            f"save_picks: dropped stale association run "
            f"(file_name={run_file!r}, method={run_method!r}) that referenced "
            f"the replaced picks -- re-run the associator to regenerate it"
        )


def select_pick_ids(
    db_path=DEFAULT_DB_PATH,
    *,
    method=None,
    time_start=None,
    time_end=None,
    phase=None,
    file_name=None,
    min_score=None,
):
    """Return the ``picks.id`` values matching the filters.

    This is the single place where "which picks?" is decided; consumers
    (e.g. :mod:`dasieve.associator`, future locators) take the returned ids
    and load the rows they need via :func:`load_picks_by_ids`.

    Parameters
    ----------
    db_path : str
        Catalog database path.
    method : str or list of str, optional
        Picker method(s) to include (the ``picks.method`` column), e.g.
        "phasenet" or ["phasenet", "eqtransformer_sb"].
    time_start, time_end : str or datetime, optional
        Inclusive ``onset_time`` window.
    phase : str, optional
        Restrict to one phase label (e.g. "P").
    file_name : str, optional
        Restrict to picks from one source data file.
    min_score : float, optional
        Keep only picks with ``score >= min_score``.

    Returns
    -------
    list of int : matching pick ids, ordered by onset_time.
    """
    clauses, params = [], []
    if method is not None:
        if isinstance(method, (list, tuple, set)):
            methods = list(method)
            placeholders = ", ".join(["?"] * len(methods))
            clauses.append(f"method IN ({placeholders})")
            params.extend(methods)
        else:
            clauses.append("method = ?")
            params.append(method)
    if time_start is not None:
        clauses.append("onset_time >= ?")
        params.append(pd.Timestamp(time_start).isoformat())
    if time_end is not None:
        clauses.append("onset_time <= ?")
        params.append(pd.Timestamp(time_end).isoformat())
    if phase is not None:
        clauses.append("phase = ?")
        params.append(phase)
    if file_name is not None:
        clauses.append("file_name = ?")
        params.append(file_name)
    if min_score is not None:
        clauses.append("score >= ?")
        params.append(float(min_score))

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"SELECT id FROM picks{where} ORDER BY onset_time;"

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [r[0] for r in rows]


# SQLite's default variable limit is 999; chunk IN (...) queries below it.
_SQL_CHUNK = 900


def load_picks_by_ids(pick_ids, db_path=DEFAULT_DB_PATH):
    """Load pick rows for the given ``picks.id`` values into a DataFrame.

    ``onset_time`` / ``off_time`` are parsed to datetime; rows come back
    ordered by onset_time. Unknown ids are silently absent from the result.
    """
    pick_ids = [int(i) for i in pick_ids]
    if not pick_ids:
        return pd.DataFrame(columns=["id"] + _TABLE_COLUMNS)

    frames = []
    with _connect(db_path) as conn:
        for i in range(0, len(pick_ids), _SQL_CHUNK):
            chunk = pick_ids[i : i + _SQL_CHUNK]
            placeholders = ", ".join(["?"] * len(chunk))
            frames.append(
                pd.read_sql_query(
                    f"SELECT * FROM picks WHERE id IN ({placeholders});",
                    conn,
                    params=chunk,
                )
            )
    df = pd.concat(frames, ignore_index=True)
    for col in ("onset_time", "off_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.sort_values("onset_time", ignore_index=True)


# ---------------------------------------------------------------------------
# Association results (events / assignments tables)
# ---------------------------------------------------------------------------
def _init_association_tables(db_path):
    """Create the ``events`` / ``assignments`` tables if missing.

    ``method`` records which associator produced the rows (e.g. "gamma"),
    mirroring the ``method`` column of the picks table, so results from
    different associators can coexist and be filtered."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                method TEXT NOT NULL DEFAULT 'gamma',
                created_at TEXT NOT NULL,
                event_index INTEGER,
                time TEXT,
                x_km REAL,
                y_km REAL,
                z_km REAL,
                gamma_score REAL,
                sigma_time REAL,
                magnitude REAL,
                number_picks INTEGER,
                number_p_picks INTEGER,
                number_s_picks INTEGER
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                method TEXT NOT NULL DEFAULT 'gamma',
                created_at TEXT NOT NULL,
                event_index INTEGER,
                pick_id INTEGER REFERENCES picks(id),
                gamma_score REAL
            );
            """
        )
        for table in ("events", "assignments"):
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_file_method "
                f"ON {table} (file_name, method);"
            )
        conn.commit()


def save_associations(
    catalog_df, assignments_df, db_path=DEFAULT_DB_PATH, *, file_name, method="gamma"
):
    """Persist an association run to the ``events`` / ``assignments`` tables.

    Normally called automatically by :meth:`dasieve.associator.BaseAssociator.run`
    (``db_save=True``), which fills ``file_name`` and ``method`` from what it
    was called with. Existing rows with the same (file_name, method) are
    deleted first (replace-on-rerun, like :func:`save_picks`), so re-running
    association on the same data overwrites instead of duplicating.

    Returns
    -------
    (n_events, n_assignments) : rows inserted into each table.
    """
    _init_association_tables(db_path)
    created_at = datetime.now(timezone.utc).isoformat()

    event_rows = []
    for _, r in catalog_df.iterrows():
        event_rows.append(
            (
                file_name,
                method,
                created_at,
                _int_or_none(r.get("event_index")),
                None if pd.isna(r.get("time")) else str(r.get("time")),
                _float_or_none(r.get("x(km)")),
                _float_or_none(r.get("y(km)")),
                _float_or_none(r.get("z(km)")),
                _float_or_none(r.get("gamma_score")),
                _float_or_none(r.get("sigma_time")),
                _float_or_none(r.get("magnitude")),
                _int_or_none(r.get("number_picks")),
                _int_or_none(r.get("number_p_picks")),
                _int_or_none(r.get("number_s_picks")),
            )
        )

    assignment_rows = []
    for _, r in assignments_df.iterrows():
        assignment_rows.append(
            (
                file_name,
                method,
                created_at,
                _int_or_none(r.get("event_index")),
                _int_or_none(r.get("pick_id")),
                _float_or_none(r.get("gamma_score")),
            )
        )

    with _connect(db_path) as conn:
        conn.execute(
            "DELETE FROM events WHERE file_name = ? AND method = ?;",
            (file_name, method),
        )
        conn.execute(
            "DELETE FROM assignments WHERE file_name = ? AND method = ?;",
            (file_name, method),
        )
        if event_rows:
            conn.executemany(
                "INSERT INTO events (file_name, method, created_at, event_index, "
                "time, x_km, y_km, z_km, gamma_score, sigma_time, magnitude, "
                "number_picks, number_p_picks, number_s_picks) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                event_rows,
            )
        if assignment_rows:
            conn.executemany(
                "INSERT INTO assignments (file_name, method, created_at, "
                "event_index, pick_id, gamma_score) VALUES (?, ?, ?, ?, ?, ?);",
                assignment_rows,
            )
        conn.commit()
    return len(event_rows), len(assignment_rows)
