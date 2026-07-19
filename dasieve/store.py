"""SQLite store for DAS picks and association results.

Data is identified by *which fiber* and *which span of time*, not by file
path: ``cable_id`` (e.g. "16BConst") plus the ``(time_start, time_end)``
window the patch covers. A file is just one way to slice that -- two files
from the same cable are two windows of the same ``cable_id``.

A single flat ``picks`` table stores every pick produced by the pickers in
:mod:`dasieve.picking`, labelled with ``cable_id``, the time window, and the
``method`` (the picking method, e.g. "sta_lta", "phasenetdas"). Re-running
with the same ``(cable_id, time_start, time_end, method)`` *replaces* the
previous rows for that key -- no duplicates accumulate. The ``events`` /
``assignments`` tables written by :func:`save_associations` extend the same
scheme with ``pick_method``: a run is keyed on
``(cable_id, time_start, time_end, method, pick_method)`` where ``method`` is
the associator (e.g. "gamma") and ``pick_method`` the picker whose picks were
associated (e.g. "phasenetdas"), so one associator's runs on different pick
sets, cables, or time windows all coexist.

    from dasieve.store import save_picks
    df = trigger_picker(patch, ..., db_save=False)
    save_picks(df, cable_id="16BConst", method="sta_lta",
               time_start="2024-04-07T07:20:54", time_end="2024-04-07T07:21:54")

Each ``assignments`` row also carries ``event_id``, a foreign key to
``events.id``, so a pick's event can be joined directly rather than via
(run key, event_index).

The database is a plain SQLite file (default ``~/DASieve/dasieve.sqlite``);
query it with any SQLite tool or with :func:`load_picks_by_ids`. The schema
changed when ``file_name`` was replaced by ``cable_id`` + time window, and
there is no migration: delete an older database file (or use a new
``db_path``) and re-run the pickers.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .picking import PICK_COLUMNS, time_window_from_patch

DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~/DASieve"), "dasieve.sqlite")

# The run key shared by every table: which fiber, which span of time, which
# method. time_start/time_end are ISO-8601 text, so they sort and range-query
# lexicographically.
_KEY_COLUMNS = ["cable_id", "time_start", "time_end"]

# Order of columns persisted to the picks table. The first five are run
# metadata; the rest are the shared picker schema (PICK_COLUMNS).
_META_COLUMNS = _KEY_COLUMNS + ["method", "created_at"]
_TABLE_COLUMNS = _META_COLUMNS + list(PICK_COLUMNS)

# Column -> SQLite type. onset_time is stored as ISO-8601 text (sortable);
# the sample index as INTEGER; everything else REAL/TEXT.
_COLUMN_TYPES = {
    "cable_id": "TEXT NOT NULL",
    "time_start": "TEXT NOT NULL",
    "time_end": "TEXT NOT NULL",
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
}


@contextmanager
def _connect(db_path):
    """Open a connection with sane defaults, creating parent dirs as needed.

    Always used as a context manager. On exit the transaction is committed
    (rolled back if the body raised) *and* the connection is closed --
    sqlite3's own ``with conn`` only ends the transaction and never closes,
    so a long-running pipeline calling into the store once per file would
    otherwise leak a handle (and a WAL descriptor) per call until it hit the
    process fd limit.
    """
    db_path = os.path.abspath(os.path.expanduser(db_path))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        # PRAGMAs must run outside a transaction, hence before `with conn`
        conn.execute("PRAGMA journal_mode=WAL;")   # safer concurrent reads
        conn.execute("PRAGMA foreign_keys=ON;")
        with conn:      # commit on clean exit, rollback on exception
            yield conn
    finally:
        conn.close()


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
        # speeds up the replace-on-rerun delete and per-cable/window queries
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_picks_run '
            'ON picks (cable_id, time_start, time_end, method);'
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


def _prepare_rows(df, cable_id, time_start, time_end, method, created_at):
    """Turn a picks DataFrame into a list of value-tuples matching
    _TABLE_COLUMNS, with proper type coercion and NaN/NaT -> NULL."""
    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                cable_id,
                time_start,
                time_end,
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
            )
        )
    return rows


def save_picks(
    df, cable_id, method, db_path=DEFAULT_DB_PATH, *,
    time_start=None, time_end=None, patch=None, replace=True,
):
    """Persist a picks DataFrame to the store.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of a picker (columns == dasieve.picking.PICK_COLUMNS). May be
        empty (the previous picks for this run key are still cleared when
        ``replace`` is True, so the store reflects the latest empty result).
    cable_id : str
        Which fiber the data came from, e.g. "16BConst". Deliberately not a
        file path: several files from one cable share a cable_id and differ
        only by their time window.
    method : str
        Picking method, e.g. "sta_lta", "ar", or "phasenetdas".
    db_path : str
        SQLite file path (created if missing).
    time_start, time_end : str or datetime
        The span of data the picks were produced from, stored as ISO-8601.
        Required unless ``patch`` is given.
    patch : dascore.Patch, optional
        Derive the time window from this patch's time coordinate instead of
        passing it explicitly (see
        :func:`dasieve.picking.time_window_from_patch`).
    replace : bool
        If True (default), delete existing picks for this
        (cable_id, time_start, time_end, method) before inserting, so only the
        latest run is kept and no duplicates accumulate. If False, append.

    Returns
    -------
    int : number of pick rows inserted.
    """
    if cable_id is None:
        raise ValueError("save_picks requires cable_id")
    if patch is not None and (time_start is None or time_end is None):
        time_start, time_end = time_window_from_patch(patch)
    if time_start is None or time_end is None:
        raise ValueError(
            "save_picks requires a time window: pass time_start/time_end, "
            "or patch=... to derive them from the data"
        )
    time_start, time_end = _iso_or_none(time_start), _iso_or_none(time_end)
    if time_start is None or time_end is None:
        raise ValueError("time_start/time_end must be parseable timestamps")

    init_db(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    rows = _prepare_rows(df, cable_id, time_start, time_end, method, created_at)

    placeholders = ", ".join(["?"] * len(_TABLE_COLUMNS))
    col_list = ", ".join(f'"{c}"' for c in _TABLE_COLUMNS)
    insert_sql = f"INSERT INTO picks ({col_list}) VALUES ({placeholders});"

    key = (cable_id, time_start, time_end, method)
    with _connect(db_path) as conn:
        if replace:
            _drop_associations_referencing(conn, *key)
            conn.execute(
                "DELETE FROM picks WHERE cable_id = ? AND time_start = ? "
                "AND time_end = ? AND method = ?;",
                key,
            )
        if rows:
            conn.executemany(insert_sql, rows)
        conn.commit()
    return len(rows)


def _delete_association_run(conn, cable_id, time_start, time_end, method,
                            pick_method):
    """Delete one association run whole -- its ``assignments`` rows first
    (they reference ``events.id``), then its ``events`` rows."""
    key = (cable_id, time_start, time_end, method, pick_method)
    for table in ("assignments", "events"):
        conn.execute(
            f"DELETE FROM {table} WHERE cable_id = ? AND time_start = ? "
            "AND time_end = ? AND method = ? AND pick_method IS ?;",
            key,
        )


def _drop_associations_referencing(conn, cable_id, time_start, time_end, method):
    """Delete association runs whose assignments reference picks about to be
    replaced.
    """
    has_assignments = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='assignments';"
    ).fetchone()
    if not has_assignments:
        return

    runs = conn.execute(
        "SELECT DISTINCT a.cable_id, a.time_start, a.time_end, a.method, "
        "a.pick_method FROM assignments a JOIN picks p ON a.pick_id = p.id "
        "WHERE p.cable_id = ? AND p.time_start = ? AND p.time_end = ? "
        "AND p.method = ?;",
        (cable_id, time_start, time_end, method),
    ).fetchall()
    for run in runs:
        _delete_association_run(conn, *run)
        print(
            f"save_picks: dropped stale association run (cable_id={run[0]!r}, "
            f"time_window={run[1]}..{run[2]}, method={run[3]!r}, "
            f"pick_method={run[4]!r}) that referenced the replaced picks -- "
            f"re-run the associator to regenerate it"
        )


def _in_clause(column, value, clauses, params):
    """Append ``column = ?`` or ``column IN (?, ...)`` for a scalar or an
    iterable of values."""
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        placeholders = ", ".join(["?"] * len(values))
        clauses.append(f"{column} IN ({placeholders})")
        params.extend(values)
    else:
        clauses.append(f"{column} = ?")
        params.append(value)


def normalize_time_windows(time_window):
    """Coerce a time-window filter into a list of ``(start_iso, end_iso)``.

    Accepts a single ``(start, end)`` pair or an iterable of them; each bound
    is parsed with pandas and re-emitted as ISO-8601 so it matches the stored
    text exactly.
    """
    if time_window is None:
        return None
    pairs = time_window
    if (len(time_window) == 2
            and not isinstance(time_window[0], (list, tuple))):
        pairs = [time_window]        # a bare (start, end) pair
    out = []
    for start, end in pairs:
        s, e = _iso_or_none(start), _iso_or_none(end)
        if s is None or e is None:
            raise ValueError(f"unparseable time window: {(start, end)!r}")
        out.append((s, e))
    return out


def select_pick_ids(
    db_path=DEFAULT_DB_PATH,
    *,
    method=None,
    onset_start=None,
    onset_end=None,
    phase=None,
    cable_id=None,
    time_window=None,
    min_score=None,
):
    """Return the ``picks.id`` values matching the filters.

    This is the single place where "which picks?" is decided; consumers
    (e.g. :mod:`dasieve.association`, future locators) take the returned ids
    and load the rows they need via :func:`load_picks_by_ids`.

    Parameters
    ----------
    db_path : str
        Catalog database path.
    method : str or list of str, optional
        Picker method(s) to include (the ``picks.method`` column), e.g.
        "phasenet" or ["phasenet", "eqtransformer_sb"].
    onset_start, onset_end : str or datetime, optional
        Inclusive filter on each pick's own ``onset_time``. Unrelated to
        ``time_window``, which matches the run the pick belongs to.
    phase : str, optional
        Restrict to one phase label (e.g. "P").
    cable_id : str or list of str, optional
        Restrict to picks from one fiber, or from any of several.
    time_window : (start, end) or list of (start, end), optional
        Restrict to picks stored under exactly these run windows -- an
        identity match on the (time_start, time_end) key, not a range query.
        Combine with ``cable_id`` to name specific files' worth of picks.
    min_score : float, optional
        Keep only picks with ``score >= min_score``.

    Returns
    -------
    list of int : matching pick ids, ordered by onset_time.
    """
    clauses, params = [], []
    if method is not None:
        _in_clause("method", method, clauses, params)
    if onset_start is not None:
        clauses.append("onset_time >= ?")
        params.append(pd.Timestamp(onset_start).isoformat())
    if onset_end is not None:
        clauses.append("onset_time <= ?")
        params.append(pd.Timestamp(onset_end).isoformat())
    if phase is not None:
        clauses.append("phase = ?")
        params.append(phase)
    if cable_id is not None:
        _in_clause("cable_id", cable_id, clauses, params)
    windows = normalize_time_windows(time_window)
    if windows is not None:
        ors = " OR ".join(["(time_start = ? AND time_end = ?)"] * len(windows))
        clauses.append(f"({ors})" if windows else "0")
        for s, e in windows:
            params.extend((s, e))
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

    ``onset_time`` is parsed to datetime; rows come back ordered by
    onset_time. Unknown ids are silently absent from the result.
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
    if "onset_time" in df.columns:
        df["onset_time"] = pd.to_datetime(df["onset_time"], errors="coerce")
    return df.sort_values("onset_time", ignore_index=True)


def load_picks(
    db_path=DEFAULT_DB_PATH,
    *,
    method=None,
    onset_start=None,
    onset_end=None,
    phase=None,
    cable_id=None,
    time_window=None,
    min_score=None,
):
    """Select and load picks in one call.

    Convenience wrapper: :func:`select_pick_ids` with the same filters,
    followed by :func:`load_picks_by_ids`. Returns the picks DataFrame
    (including the ``id`` column), ordered by onset_time.
    """
    pick_ids = select_pick_ids(
        db_path,
        method=method,
        onset_start=onset_start,
        onset_end=onset_end,
        phase=phase,
        cable_id=cable_id,
        time_window=time_window,
        min_score=min_score,
    )
    return load_picks_by_ids(pick_ids, db_path)


# ---------------------------------------------------------------------------
# Association results (events / assignments tables)
# ---------------------------------------------------------------------------
def _init_association_tables(db_path):
    """Create the ``events`` / ``assignments`` tables if missing.

    ``method`` records which associator produced the rows (e.g. "gamma") and
    ``pick_method`` which picker's picks it was run on (e.g. "phasenetdas"),
    mirroring the ``method`` column of the picks table. A run is identified by
    (``cable_id``, ``time_start``, ``time_end``, ``method``, ``pick_method``),
    so the same associator run on different pick sets, cables or time windows
    -- or different associators -- coexist, and only a rerun of the same key
    replaces previous rows.

    ``assignments.event_id`` references ``events.id``, so a pick joins to its
    event directly; ``event_index`` is kept as the associator's own numbering
    within the run."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cable_id TEXT NOT NULL,
                time_start TEXT NOT NULL,
                time_end TEXT NOT NULL,
                method TEXT NOT NULL DEFAULT 'gamma',
                pick_method TEXT,
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
                cable_id TEXT NOT NULL,
                time_start TEXT NOT NULL,
                time_end TEXT NOT NULL,
                method TEXT NOT NULL DEFAULT 'gamma',
                pick_method TEXT,
                created_at TEXT NOT NULL,
                event_index INTEGER,
                event_id INTEGER REFERENCES events(id),
                pick_id INTEGER REFERENCES picks(id),
                gamma_score REAL
            );
            """
        )
        for table in ("events", "assignments"):
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_run "
                f"ON {table} (cable_id, time_start, time_end, method, pick_method);"
            )
        conn.commit()


def save_associations(
    catalog_df, assignments_df, db_path=DEFAULT_DB_PATH, *, cable_id,
    time_start, time_end, method="gamma", pick_method=None
):
    """Persist an association run to the ``events`` / ``assignments`` tables.

    Normally called automatically by :meth:`dasieve.association.BaseAssociator.run`
    (``db_save=True``), which fills ``cable_id``, the time window, ``method``
    (the associator) and ``pick_method`` (the picker whose picks were
    associated) from what it was called with. Existing rows with the same
    (cable_id, time_start, time_end, method, pick_method) are deleted first
    (replace-on-rerun, like :func:`save_picks`), so re-running the same
    associator over the same data overwrites instead of duplicating -- while a
    different pick set, cable, or window is kept as a separate run.

    Events are inserted first so each assignment can carry ``event_id``, a
    foreign key to the ``events`` row its ``event_index`` refers to.

    Returns
    -------
    (n_events, n_assignments) : rows inserted into each table.
    """
    _init_association_tables(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    time_start, time_end = _iso_or_none(time_start), _iso_or_none(time_end)
    if cable_id is None or time_start is None or time_end is None:
        raise ValueError(
            "save_associations requires cable_id and a parseable time window"
        )
    key = (cable_id, time_start, time_end, method, pick_method)

    event_rows = []
    for _, r in catalog_df.iterrows():
        event_rows.append(
            (
                cable_id,
                time_start,
                time_end,
                method,
                pick_method,
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

    with _connect(db_path) as conn:
        _delete_association_run(conn, *key)
        if event_rows:
            conn.executemany(
                "INSERT INTO events (cable_id, time_start, time_end, method, "
                "pick_method, created_at, event_index, time, x_km, y_km, z_km, "
                "gamma_score, sigma_time, magnitude, number_picks, "
                "number_p_picks, number_s_picks) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                event_rows,
            )

        # event_index (the associator's own numbering) -> events.id, so each
        # assignment can point at its event row directly
        event_ids = dict(
            conn.execute(
                "SELECT event_index, id FROM events WHERE cable_id = ? AND "
                "time_start = ? AND time_end = ? AND method = ? "
                "AND pick_method IS ?;",
                key,
            ).fetchall()
        )

        assignment_rows = []
        for _, r in assignments_df.iterrows():
            event_index = _int_or_none(r.get("event_index"))
            assignment_rows.append(
                (
                    cable_id,
                    time_start,
                    time_end,
                    method,
                    pick_method,
                    created_at,
                    event_index,
                    event_ids.get(event_index),
                    _int_or_none(r.get("pick_id")),
                    _float_or_none(r.get("gamma_score")),
                )
            )

        if assignment_rows:
            conn.executemany(
                "INSERT INTO assignments (cable_id, time_start, time_end, "
                "method, pick_method, created_at, event_index, event_id, "
                "pick_id, gamma_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                assignment_rows,
            )
        conn.commit()
    return len(event_rows), len(assignment_rows)
