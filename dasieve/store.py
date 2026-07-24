"""SQLite store for DAS picks, association results, and PSD QC products.

The catalog follows the QuakeML chain -- a pick is associated to an origin,
and an origin belongs to an event::

    picks --< associations >-- origins --< events
                                       (preferred_origin_id)

* ``picks``        -- one row per phase arrival on one channel
* ``associations`` -- one row per (pick, origin) link
* ``origins``      -- one located hypocentre, built from its associated picks
* ``events``       -- one earthquake, pointing at its preferred origin

Data is identified by *which fiber* and *which span of time*, not by file
path: ``cable_id`` (e.g. "16BConst") plus the
``(file_starttime, file_endtime)`` window the patch covers. A file is just
one way to slice that -- two files from the same cable are two windows of
the same ``cable_id``.

Re-picking replaces
-------------------
Saving picks for a ``(cable_id, file_starttime, file_endtime, pick_method)``
key that already exists *replaces* the previous rows, and the replacement
cascades down the chain (:func:`save_picks`)::

    replaced picks
      -> their associations                              deleted
      -> every origin owning any of those associations   deleted whole
      -> those origins' events                           deleted

Re-running an associator over picks that have not changed cascades the same
way: existing associations with the same ``association_method`` whose
``pick_id`` is in the new run's pick set are superseded, and the same rule
applies (:func:`save_associations`).

The rule is *any overlap deletes the whole origin*, so an origin is always
the product of a single run -- never half-old and half-new. The cost is that
an origin built from picks in two files, only one of which is re-processed,
is deleted without being rebuilt (the other file's picks were not in the
run). That is reported loudly when it happens.

Every picker in :mod:`dasieve.picking` returns exactly the ``picks`` column
schema (:data:`dasieve.picking.PICK_COLUMNS`), so a DataFrame straight from
a picker and a row read back out of the database look the same::

    from dasieve.store import save_picks
    df = trigger_picker(patch, cable_id="16BConst", db_save=False)
    save_picks(df)          # run key is read from the DataFrame itself

The database is a plain SQLite file (default ``~/DASieve/dasieve.sqlite``);
query it with any SQLite tool or with :func:`load_picks`. There is no
migration from older schema versions: delete the database file (or use a new
``db_path``) and re-run the pickers.
"""

import os
import sqlite3
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .picking import PICK_COLUMNS

DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~/DASieve"), "dasieve.sqlite")

# Which fiber, which span of data, which picker: the key a pick run is stored
# under and replaced on. file_starttime/file_endtime are ISO-8601 text, so
# they sort and range-query lexicographically.
PICK_RUN_KEY = ["cable_id", "file_starttime", "file_endtime", "pick_method"]

# Column -> SQLite type for the picks table. Columns and their order are
# PICK_COLUMNS, the same schema the pickers emit.
_PICK_COLUMN_TYPES = {
    "cable_id": "TEXT NOT NULL",
    "distance": "REAL",
    "pick_method": "TEXT NOT NULL",
    "phase": "TEXT",
    "onset_time": "TEXT",
    "x": "REAL",
    "y": "REAL",
    "z": "REAL",
    "probability": "REAL",
    "file_starttime": "TEXT NOT NULL",
    "file_endtime": "TEXT NOT NULL",
}

# SQLite's default variable limit is 999; chunk IN (...) queries below it.
_SQL_CHUNK = 900


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
    """Create the catalog tables and indexes if they don't already exist.

    Safe to call repeatedly. ``events`` and ``origins`` reference each other
    (an origin's event, an event's preferred origin), so both are created
    before anything is inserted into either. Returns the absolute database
    path.
    """
    cols_sql = ",\n    ".join(
        f'"{c}" {_PICK_COLUMN_TYPES[c]}' for c in PICK_COLUMNS
    )
    with _connect(db_path) as conn:
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS picks (\n'
            f'    id INTEGER PRIMARY KEY AUTOINCREMENT,\n'
            f'    {cols_sql}\n'
            f');'
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preferred_origin_id INTEGER REFERENCES origins(id),
                magnitude REAL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS origins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER REFERENCES events(id),
                origin_method TEXT NOT NULL,
                origin_time TEXT,
                x REAL,
                y REAL,
                z REAL,
                number_picks INTEGER
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pick_id INTEGER NOT NULL REFERENCES picks(id),
                origin_id INTEGER NOT NULL REFERENCES origins(id),
                association_method TEXT NOT NULL,
                probability REAL
            );
            """
        )
        # the replace-on-rerun delete and per-cable/window queries
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_picks_run '
            'ON picks (cable_id, file_starttime, file_endtime, pick_method);'
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_picks_onset ON picks (onset_time);'
        )
        # both directions of the cascade: pick -> associations -> origin
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_assoc_pick '
            'ON associations (pick_id, association_method);'
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_assoc_origin '
            'ON associations (origin_id);'
        )
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_origins_event ON origins (event_id);'
        )
        conn.commit()
    return os.path.abspath(os.path.expanduser(db_path))


def _iso_or_none(value):
    """Coerce a datetime-like value to an ISO-8601 string, or None for NaT."""
    ts = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(ts) else ts.isoformat()


def _parse_stored_time(values):
    """Parse a column of stored ISO-8601 timestamps to datetime.

    ``format="ISO8601"`` is required, not cosmetic: :func:`_iso_or_none` writes
    ``Timestamp.isoformat()``, which omits the fractional part on a whole
    second. A column mixing "...:03" with "...:02.500000" makes pandas infer
    the format from the first row and silently coerce every row that does not
    match to NaT -- so any pick landing exactly on a second would vanish.
    """
    return pd.to_datetime(values, format="ISO8601", errors="coerce")


def _int_or_none(value):
    return None if pd.isna(value) else int(value)


def _float_or_none(value):
    return None if pd.isna(value) else float(value)


def _text_or_none(value):
    return None if pd.isna(value) else str(value)


def _chunks(values):
    """Split a sequence into IN (...)-sized chunks."""
    values = list(values)
    for i in range(0, len(values), _SQL_CHUNK):
        yield values[i : i + _SQL_CHUNK]


def _select_ids(conn, sql_template, ids, params=()):
    """Run an ``id IN (...)`` SELECT over chunks, returning the flat first
    column. ``sql_template`` contains a single ``{placeholders}`` slot; any
    ``params`` are bound *before* the chunk."""
    out = []
    for chunk in _chunks(ids):
        placeholders = ", ".join(["?"] * len(chunk))
        sql = sql_template.format(placeholders=placeholders)
        out.extend(
            r[0] for r in conn.execute(sql, tuple(params) + tuple(chunk)).fetchall()
        )
    return out


def _exec_ids(conn, sql_template, ids):
    """Run an ``id IN (...)`` DELETE/UPDATE over chunks."""
    for chunk in _chunks(ids):
        placeholders = ", ".join(["?"] * len(chunk))
        conn.execute(sql_template.format(placeholders=placeholders), chunk)


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------
def _cascade_delete(conn, association_ids, *, trigger=""):
    """Delete superseded associations and everything downstream of them.

    ``association_ids`` are the associations made stale by whatever the caller
    is about to replace -- picks being re-picked, or an associator being
    re-run. Every origin owning *any* of them is deleted **whole**, including
    its associations to picks that were not superseded, and an event is
    deleted once every one of its origins is gone.

    Deleting a whole origin on partial overlap is deliberate: an origin is a
    joint fit over its picks, so one stale pick invalidates it, and rebuilding
    it from the surviving subset would silently produce a different (worse)
    hypocentre than either run. The straddling case -- an origin losing picks
    that the current run will not regenerate -- is printed, since nothing
    rebuilds that origin.

    ``trigger`` labels the cause in those messages. Returns
    ``(n_associations, n_origins, n_events)`` deleted.
    """
    association_ids = [int(i) for i in association_ids]
    if not association_ids:
        return 0, 0, 0

    origin_ids = sorted(set(_select_ids(
        conn,
        "SELECT DISTINCT origin_id FROM associations WHERE id IN ({placeholders});",
        association_ids,
    )))
    if not origin_ids:
        return 0, 0, 0

    # An origin whose association count exceeds the number superseded here
    # keeps picks the current run will not touch: deleting it leaves a hole.
    superseded_per_origin = {}
    for chunk in _chunks(association_ids):
        placeholders = ", ".join(["?"] * len(chunk))
        for origin_id, n in conn.execute(
            f"SELECT origin_id, COUNT(*) FROM associations "
            f"WHERE id IN ({placeholders}) GROUP BY origin_id;",
            chunk,
        ).fetchall():
            superseded_per_origin[origin_id] = (
                superseded_per_origin.get(origin_id, 0) + n
            )
    straddlers = []
    for chunk in _chunks(origin_ids):
        placeholders = ", ".join(["?"] * len(chunk))
        for origin_id, total in conn.execute(
            f"SELECT origin_id, COUNT(*) FROM associations "
            f"WHERE origin_id IN ({placeholders}) GROUP BY origin_id;",
            chunk,
        ).fetchall():
            kept = total - superseded_per_origin.get(origin_id, 0)
            if kept > 0:
                straddlers.append((origin_id, kept, total))
    if straddlers:
        detail = ", ".join(
            f"origin {oid} ({kept}/{total} picks unaffected)"
            for oid, kept, total in straddlers[:5]
        )
        more = "" if len(straddlers) <= 5 else f", +{len(straddlers) - 5} more"
        print(
            f"store: {len(straddlers)} origin(s) deleted{trigger} extended "
            f"beyond the affected picks and will NOT be rebuilt by this run "
            f"-- {detail}{more}. Re-run over the full pick set to regenerate "
            f"them."
        )

    # events whose origins are all being deleted die with them; an event
    # keeping at least one origin survives, repointed if it preferred a
    # deleted one
    doomed = set(origin_ids)
    event_ids = [
        e for e in set(_select_ids(
            conn,
            "SELECT DISTINCT event_id FROM origins WHERE id IN ({placeholders});",
            origin_ids,
        )) if e is not None
    ]
    doomed_events, survivors = [], {}
    for event_id in event_ids:
        remaining = [
            o for (o,) in conn.execute(
                "SELECT id FROM origins WHERE event_id = ?;", (event_id,)
            ).fetchall()
            if o not in doomed
        ]
        if remaining:
            survivors[event_id] = remaining[0]
        else:
            doomed_events.append(event_id)

    # Order matters: nothing may reference a row when it is deleted. Clear the
    # events -> origins pointers first, then delete origins (which frees the
    # origins -> events pointers), then the events themselves.
    _exec_ids(
        conn, "DELETE FROM associations WHERE origin_id IN ({placeholders});",
        origin_ids,
    )
    # a surviving event that preferred a deleted origin adopts one it keeps
    for event_id, origin_id in survivors.items():
        preferred = conn.execute(
            "SELECT preferred_origin_id FROM events WHERE id = ?;", (event_id,)
        ).fetchone()
        if preferred and preferred[0] in doomed:
            conn.execute(
                "UPDATE events SET preferred_origin_id = ? WHERE id = ?;",
                (origin_id, event_id),
            )
    # any remaining pointer into the doomed set belongs to a doomed event
    _exec_ids(
        conn,
        "UPDATE events SET preferred_origin_id = NULL "
        "WHERE preferred_origin_id IN ({placeholders});",
        origin_ids,
    )
    _exec_ids(conn, "DELETE FROM origins WHERE id IN ({placeholders});",
              origin_ids)
    _exec_ids(conn, "DELETE FROM events WHERE id IN ({placeholders});",
              doomed_events)

    n_assoc = len(association_ids)
    print(
        f"store: cascade{trigger} deleted {len(origin_ids)} origin(s) and "
        f"{len(doomed_events)} event(s) from {n_assoc} superseded association(s)"
    )
    return n_assoc, len(origin_ids), len(doomed_events)


def _cascade_delete_for_picks(conn, pick_ids, *, trigger=""):
    """Cascade from picks: delete every association on these picks, and
    everything downstream. Used when picks are replaced."""
    pick_ids = [int(i) for i in pick_ids]
    if not pick_ids:
        return 0, 0, 0
    association_ids = _select_ids(
        conn,
        "SELECT id FROM associations WHERE pick_id IN ({placeholders});",
        pick_ids,
    )
    return _cascade_delete(conn, association_ids, trigger=trigger)


# ---------------------------------------------------------------------------
# Picks
# ---------------------------------------------------------------------------
def _resolve_run_key(df, cable_id, pick_method, file_starttime, file_endtime,
                     patch):
    """Work out the (cable_id, file_starttime, file_endtime, pick_method) a
    save is keyed on.

    Explicit arguments win; otherwise the value is read off the DataFrame,
    which carries these columns as part of the shared pick schema. The window
    can also be derived from ``patch``. A DataFrame mixing several runs is
    rejected -- the key decides what gets replaced, so it must be unambiguous.
    """
    from .picking import time_window_from_patch

    resolved = {
        "cable_id": cable_id,
        "file_starttime": file_starttime,
        "file_endtime": file_endtime,
        "pick_method": pick_method,
    }
    for column, value in resolved.items():
        if value is not None or column not in df.columns or not len(df):
            continue
        distinct = pd.unique(df[column].dropna())
        if len(distinct) > 1:
            raise ValueError(
                f"picks DataFrame mixes {len(distinct)} values of {column!r} "
                f"({sorted(map(str, distinct))[:4]}...); save one run at a "
                f"time, or pass {column}=... explicitly"
            )
        if len(distinct):
            resolved[column] = distinct[0]

    if patch is not None and (resolved["file_starttime"] is None
                              or resolved["file_endtime"] is None):
        resolved["file_starttime"], resolved["file_endtime"] = (
            time_window_from_patch(patch)
        )

    for column in ("file_starttime", "file_endtime"):
        resolved[column] = _iso_or_none(resolved[column])

    missing = [c for c, v in resolved.items() if v is None]
    if missing:
        raise ValueError(
            f"save_picks cannot determine {', '.join(missing)}: pass them "
            f"explicitly (or patch=... for the time window), or supply a "
            f"DataFrame carrying those columns"
        )
    return (str(resolved["cable_id"]), resolved["file_starttime"],
            resolved["file_endtime"], str(resolved["pick_method"]))


def save_picks(
    df, cable_id=None, pick_method=None, db_path=DEFAULT_DB_PATH, *,
    file_starttime=None, file_endtime=None, patch=None, replace=True,
):
    """Persist a picks DataFrame to the store.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of a picker (columns == :data:`dasieve.picking.PICK_COLUMNS`).
        May be empty (the previous picks for this run key are still cleared
        when ``replace`` is True, so the store reflects the latest empty
        result).
    cable_id, pick_method, file_starttime, file_endtime : optional
        The run key. Each defaults to the value carried in the corresponding
        DataFrame column, so a picker's output normally saves with no extra
        arguments. Pass them explicitly to override, or when ``df`` is empty
        and so carries no values.
    patch : dascore.Patch, optional
        Derive the time window from this patch's time coordinate (see
        :func:`dasieve.picking.time_window_from_patch`).
    replace : bool
        If True (default), delete existing picks for this run key before
        inserting, so only the latest run is kept and no duplicates
        accumulate. The delete **cascades**: associations on those picks, the
        origins owning them, and those origins' events all go too (see
        :func:`_cascade_delete`). If False, append.

    Returns
    -------
    int : number of pick rows inserted.
    """
    key = _resolve_run_key(df, cable_id, pick_method, file_starttime,
                           file_endtime, patch)
    cable_id, file_starttime, file_endtime, pick_method = key

    rows = []
    for _, r in df.iterrows():
        rows.append(
            (
                cable_id,
                _float_or_none(r.get("distance")),
                pick_method,
                _text_or_none(r.get("phase")),
                _iso_or_none(r.get("onset_time")),
                _float_or_none(r.get("x")),
                _float_or_none(r.get("y")),
                _float_or_none(r.get("z")),
                _float_or_none(r.get("probability")),
                file_starttime,
                file_endtime,
            )
        )

    init_db(db_path)
    placeholders = ", ".join(["?"] * len(PICK_COLUMNS))
    col_list = ", ".join(f'"{c}"' for c in PICK_COLUMNS)
    insert_sql = f"INSERT INTO picks ({col_list}) VALUES ({placeholders});"
    where = " AND ".join(f"{c} = ?" for c in PICK_RUN_KEY)

    with _connect(db_path) as conn:
        if replace:
            old = [
                r[0] for r in conn.execute(
                    f"SELECT id FROM picks WHERE {where};", key
                ).fetchall()
            ]
            _cascade_delete_for_picks(
                conn, old,
                trigger=f" from re-picking {cable_id} "
                        f"{file_starttime}..{file_endtime} ({pick_method})",
            )
            conn.execute(f"DELETE FROM picks WHERE {where};", key)
        if rows:
            conn.executemany(insert_sql, rows)
        conn.commit()
    return len(rows)


def _in_clause(column, value, clauses, params):
    """Append ``column = ?`` or ``column IN (?, ...)`` for a scalar or an
    iterable of values. An empty iterable matches nothing (``IN ()`` would be
    a SQLite syntax error)."""
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if not values:
            clauses.append("0")     # always false: no value can match
            return
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
    pick_method=None,
    onset_start=None,
    onset_end=None,
    phase=None,
    cable_id=None,
    time_window=None,
    min_probability=None,
):
    """Return the ``picks.id`` values matching the filters.

    This is the single place where "which picks?" is decided; consumers
    (e.g. :mod:`dasieve.association`, future locators) take the returned ids
    and load the rows they need via :func:`load_picks_by_ids`.

    Parameters
    ----------
    db_path : str
        Catalog database path.
    pick_method : str or list of str, optional
        Picker method(s) to include (the ``picks.pick_method`` column), e.g.
        "phasenetdas" or ["phasenetdas", "eqtransformer_sb"].
    onset_start, onset_end : str or datetime, optional
        Inclusive filter on each pick's own ``onset_time``. Unrelated to
        ``time_window``, which matches the file the pick came from.
    phase : str, optional
        Restrict to one phase label (e.g. "P").
    cable_id : str or list of str, optional
        Restrict to picks from one fiber, or from any of several.
    time_window : (start, end) or list of (start, end), optional
        Restrict to picks stored under exactly these file windows -- an
        identity match on the (file_starttime, file_endtime) key, not a range
        query. Combine with ``cable_id`` to name specific files' worth of
        picks.
    min_probability : float, optional
        Keep only picks with ``probability >= min_probability``.

    Returns
    -------
    list of int : matching pick ids, ordered by onset_time.
    """
    clauses, params = [], []
    if pick_method is not None:
        _in_clause("pick_method", pick_method, clauses, params)
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
        ors = " OR ".join(
            ["(file_starttime = ? AND file_endtime = ?)"] * len(windows)
        )
        clauses.append(f"({ors})" if windows else "0")
        for s, e in windows:
            params.extend((s, e))
    if min_probability is not None:
        clauses.append("probability >= ?")
        params.append(float(min_probability))

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    query = f"SELECT id FROM picks{where} ORDER BY onset_time;"

    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [r[0] for r in rows]


def load_picks_by_ids(pick_ids, db_path=DEFAULT_DB_PATH):
    """Load pick rows for the given ``picks.id`` values into a DataFrame.

    ``onset_time`` is parsed to datetime; rows come back ordered by
    onset_time. Unknown ids are silently absent from the result.
    """
    pick_ids = [int(i) for i in pick_ids]
    if not pick_ids:
        return pd.DataFrame(columns=["id"] + PICK_COLUMNS)

    frames = []
    with _connect(db_path) as conn:
        for chunk in _chunks(pick_ids):
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
        df["onset_time"] = _parse_stored_time(df["onset_time"])
    return df.sort_values("onset_time", ignore_index=True)


def load_picks(
    db_path=DEFAULT_DB_PATH,
    *,
    pick_method=None,
    onset_start=None,
    onset_end=None,
    phase=None,
    cable_id=None,
    time_window=None,
    min_probability=None,
):
    """Select and load picks in one call.

    Convenience wrapper: :func:`select_pick_ids` with the same filters,
    followed by :func:`load_picks_by_ids`. Returns the picks DataFrame
    (including the ``id`` column), ordered by onset_time.
    """
    pick_ids = select_pick_ids(
        db_path,
        pick_method=pick_method,
        onset_start=onset_start,
        onset_end=onset_end,
        phase=phase,
        cable_id=cable_id,
        time_window=time_window,
        min_probability=min_probability,
    )
    return load_picks_by_ids(pick_ids, db_path)


# ---------------------------------------------------------------------------
# Association results (associations / origins / events tables)
# ---------------------------------------------------------------------------
#: Columns of the origins DataFrame the associators emit. ``origin_index`` is
#: the in-memory link to the associations DataFrame -- it is *not* stored;
#: :func:`save_associations` resolves it to the inserted ``origins.id``. An
#: optional extra ``magnitude`` column, if present, lands on the created
#: event rather than the origin.
ORIGIN_COLUMNS = [
    "origin_index", "origin_method", "origin_time", "x", "y", "z",
    "number_picks",
]

#: Columns of the associations DataFrame the associators emit. ``pick_id``
#: maps back to ``picks.id`` (or to a row position when the picks came from a
#: DataFrame rather than the store).
ASSOCIATION_COLUMNS = [
    "pick_id", "origin_index", "association_method", "probability",
]


def save_associations(
    origins_df, associations_df, db_path=DEFAULT_DB_PATH, *,
    association_method, origin_method=None, pick_ids=None,
    create_events=True, replace=True,
):
    """Persist an association run to the ``associations`` / ``origins`` /
    ``events`` tables.

    Normally called by :meth:`dasieve.association.BaseAssociator.run`
    (``db_save=True``). Each origin is inserted first so its associations can
    carry ``origin_id``; with ``create_events`` an ``events`` row is then
    created per origin, pointing at it as ``preferred_origin_id`` with a NULL
    magnitude, and the origin's ``event_id`` is set back to it.

    Replace-on-rerun (``replace=True``) supersedes existing associations with
    the same ``association_method`` whose ``pick_id`` is in this run's pick
    set, then cascades: the origins owning them are deleted whole, along with
    their events (see :func:`_cascade_delete`). Re-running the same associator
    over the same picks therefore leaves one set of origins, while a different
    associator's results on those picks are left alone.

    Parameters
    ----------
    origins_df : pandas.DataFrame
        One row per located origin, with :data:`ORIGIN_COLUMNS`.
    associations_df : pandas.DataFrame
        One row per (pick, origin) link, with :data:`ASSOCIATION_COLUMNS`.
    association_method : str
        Which associator produced these links, e.g. "gamma". Also the
        replace key.
    origin_method : str, optional
        Which method located the origins; defaults to ``association_method``
        (the associator located them itself). A later relocation writes its
        own origins with a different ``origin_method``.
    pick_ids : iterable of int, optional
        The run's full pick set, used as the replace scope. Defaults to the
        pick ids appearing in ``associations_df`` -- pass the whole selection
        when a rerun may associate *fewer* picks than before, so the previously
        associated ones are still superseded.
    create_events : bool
        If True (default), create one event per origin.
    replace : bool
        If False, append without superseding anything.

    Returns
    -------
    (n_origins, n_associations, n_events) : rows inserted into each table.
    """
    if origin_method is None:
        origin_method = association_method

    init_db(db_path)
    if pick_ids is None:
        pick_ids = (
            associations_df["pick_id"].dropna().astype(int).tolist()
            if "pick_id" in associations_df.columns else []
        )
    pick_ids = [int(i) for i in pick_ids]

    n_origins = n_assoc = n_events = 0
    with _connect(db_path) as conn:
        if replace and pick_ids:
            superseded = _select_ids(
                conn,
                "SELECT id FROM associations WHERE association_method = ? "
                "AND pick_id IN ({placeholders});",
                pick_ids,
                params=(association_method,),
            )
            _cascade_delete(
                conn, superseded,
                trigger=f" from re-running {association_method!r}",
            )

        # origin_index (the associator's own numbering) -> origins.id, so each
        # association can point at its origin row directly
        origin_ids = {}
        for _, r in origins_df.iterrows():
            cur = conn.execute(
                "INSERT INTO origins (event_id, origin_method, origin_time, "
                "x, y, z, number_picks) VALUES (NULL, ?, ?, ?, ?, ?, ?);",
                (
                    origin_method,
                    _iso_or_none(r.get("origin_time")),
                    _float_or_none(r.get("x")),
                    _float_or_none(r.get("y")),
                    _float_or_none(r.get("z")),
                    _int_or_none(r.get("number_picks")),
                ),
            )
            origin_ids[_int_or_none(r.get("origin_index"))] = cur.lastrowid
            n_origins += 1

            if create_events:
                # events and origins reference each other, so the event is
                # inserted pointing at the origin and the origin updated after.
                # A "magnitude" column on origins_df is optional -- associators
                # that estimate one (GaMMA with use_amplitude) pass it through,
                # everything else leaves the event's magnitude NULL.
                ev = conn.execute(
                    "INSERT INTO events (preferred_origin_id, magnitude) "
                    "VALUES (?, ?);",
                    (cur.lastrowid, _float_or_none(r.get("magnitude"))),
                )
                conn.execute(
                    "UPDATE origins SET event_id = ? WHERE id = ?;",
                    (ev.lastrowid, cur.lastrowid),
                )
                n_events += 1

        association_rows = []
        for _, r in associations_df.iterrows():
            origin_id = origin_ids.get(_int_or_none(r.get("origin_index")))
            pick_id = _int_or_none(r.get("pick_id"))
            if origin_id is None or pick_id is None:
                continue    # an assignment to an origin that was not emitted
            association_rows.append(
                (
                    pick_id,
                    origin_id,
                    association_method,
                    _float_or_none(r.get("probability")),
                )
            )
        if association_rows:
            conn.executemany(
                "INSERT INTO associations (pick_id, origin_id, "
                "association_method, probability) VALUES (?, ?, ?, ?);",
                association_rows,
            )
        n_assoc = len(association_rows)
        conn.commit()
    return n_origins, n_assoc, n_events


def load_origins(db_path=DEFAULT_DB_PATH, *, origin_method=None,
                 t_start=None, t_end=None):
    """Load the ``origins`` table into a DataFrame.

    Returns the table's own columns and nothing else -- ``id``, ``event_id``,
    ``origin_method``, ``origin_time``, ``x``, ``y``, ``z``,
    ``number_picks``. Join to the other tables yourself on ``event_id`` /
    ``origins.id`` when you want a combined view.

    ``t_start`` / ``t_end`` range-filter on ``origin_time``, which is parsed
    to datetime on the way out (the stored ISO text, same values).
    """
    init_db(db_path)
    clauses, params = [], []
    if origin_method is not None:
        _in_clause("origin_method", origin_method, clauses, params)
    if t_start is not None:
        clauses.append("origin_time >= ?")
        params.append(pd.Timestamp(t_start).isoformat())
    if t_end is not None:
        clauses.append("origin_time <= ?")
        params.append(pd.Timestamp(t_end).isoformat())
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM origins{where} ORDER BY origin_time;",
            conn,
            params=params,
        )
    if len(df):
        df["origin_time"] = _parse_stored_time(df["origin_time"])
    return df


def load_associations(db_path=DEFAULT_DB_PATH, *, origin_id=None,
                      pick_id=None, association_method=None):
    """Load the ``associations`` table into a DataFrame.

    Returns the table's own columns and nothing else -- ``id``, ``pick_id``,
    ``origin_id``, ``association_method``, ``probability``. To see the picks
    behind an origin, merge with :func:`load_picks` on
    ``pick_id`` -> ``picks.id``.
    """
    init_db(db_path)
    clauses, params = [], []
    if origin_id is not None:
        _in_clause("origin_id", origin_id, clauses, params)
    if pick_id is not None:
        _in_clause("pick_id", pick_id, clauses, params)
    if association_method is not None:
        _in_clause("association_method", association_method, clauses, params)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with _connect(db_path) as conn:
        return pd.read_sql_query(
            f"SELECT * FROM associations{where} ORDER BY origin_id, id;",
            conn,
            params=params,
        )


def load_events(db_path=DEFAULT_DB_PATH):
    """Load the ``events`` table into a DataFrame.

    Returns the table's own columns and nothing else -- ``id``,
    ``preferred_origin_id``, ``magnitude``. For the catalog view (each event
    with its preferred origin's time and location), merge with
    :func:`load_origins` on ``preferred_origin_id`` -> ``origins.id``.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM events ORDER BY id;", conn)


# ---------------------------------------------------------------------------
# PSD QC store (psd_runs / psd tables)
# ---------------------------------------------------------------------------
# Blob dtypes are a fixed convention of the schema: decoding reads them back
# with np.frombuffer, so they must never change without a schema migration.
_PSD_FREQ_DTYPE = np.float64
_PSD_DTYPE = np.float32


def _init_psd_tables(db_path):
    """Create the ``psd_runs`` / ``psd`` tables if missing.

    One ``psd_runs`` row per processed patch, keyed on
    (``cable_id``, ``file_starttime``, ``file_endtime``) -- the span of data
    the PSD was computed over, named exactly as in the picks table. The
    frequency vector is stored once there as a float64 blob, not per channel.
    One ``psd`` row per channel, holding that channel's PSD curve (dB) as a
    float32 blob of length ``n_freq``, keyed back to its run by ``run_id``.
    """
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS psd_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cable_id TEXT NOT NULL,
                file_starttime TEXT NOT NULL,
                file_endtime TEXT NOT NULL,
                created_at TEXT NOT NULL,
                n_freq INTEGER NOT NULL,
                n_ch INTEGER NOT NULL,
                freqs BLOB NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS psd (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES psd_runs(id),
                ch INTEGER NOT NULL,
                psd BLOB NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_psd_runs_run "
            "ON psd_runs (cable_id, file_starttime, file_endtime);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_psd_run ON psd (run_id, ch);"
        )
        conn.commit()


def save_psd(
    freqs, psd_db, db_path=DEFAULT_DB_PATH, *,
    cable_id, file_starttime, file_endtime, replace=True,
):
    """Persist one patch's per-channel PSDs (e.g. from
    :func:`dasieve.qc.compute_psd`).

    Parameters
    ----------
    freqs : np.ndarray, shape (n_freq,)
        Frequency vector in Hz (shared by every channel).
    psd_db : np.ndarray, shape (n_freq, n_channel)
        PSD in dB per channel; column index is the channel index.
    cable_id : str
        Which fiber the data came from (same meaning as in the picks table);
        required, like everywhere else in the store.
    file_starttime, file_endtime : datetime-like
        The span of data the PSD was computed over, stored as ISO-8601.
        Together with ``cable_id`` this is the run key, mirroring the picks
        table -- so a PSD's averaging extent is always known, and a trimmed
        patch is a separate run rather than silently replacing the full one.
    replace : bool
        If True (default), delete any previously stored run for this
        (cable_id, file_starttime, file_endtime) first, so re-processing the
        same span overwrites instead of duplicating.

    Returns
    -------
    int : number of channel rows inserted.
    """
    if cable_id is None:
        raise ValueError("save_psd requires cable_id")
    file_starttime = _iso_or_none(file_starttime)
    file_endtime = _iso_or_none(file_endtime)
    if file_starttime is None or file_endtime is None:
        raise ValueError(
            "save_psd requires a parseable time window "
            "(file_starttime/file_endtime)"
        )
    freqs = np.ascontiguousarray(freqs, dtype=_PSD_FREQ_DTYPE)
    psd_db = np.ascontiguousarray(psd_db, dtype=_PSD_DTYPE)
    if psd_db.ndim != 2 or psd_db.shape[0] != freqs.size:
        raise ValueError(
            f"psd_db must be (n_freq, n_channel) with n_freq == len(freqs); "
            f"got psd_db {psd_db.shape} vs {freqs.size} frequencies"
        )
    n_freq, n_ch = psd_db.shape

    _init_psd_tables(db_path)
    from datetime import datetime, timezone
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        if replace:
            old = [
                r[0] for r in conn.execute(
                    "SELECT id FROM psd_runs WHERE cable_id = ? AND "
                    "file_starttime = ? AND file_endtime = ?;",
                    (cable_id, file_starttime, file_endtime),
                ).fetchall()
            ]
            if old:
                placeholders = ", ".join(["?"] * len(old))
                conn.execute(
                    f"DELETE FROM psd WHERE run_id IN ({placeholders});", old
                )
                conn.execute(
                    f"DELETE FROM psd_runs WHERE id IN ({placeholders});", old
                )
        cur = conn.execute(
            "INSERT INTO psd_runs (cable_id, file_starttime, file_endtime, "
            "created_at, n_freq, n_ch, freqs) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (cable_id, file_starttime, file_endtime, created_at, n_freq, n_ch,
             freqs.tobytes()),
        )
        run_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO psd (run_id, ch, psd) VALUES (?, ?, ?);",
            [
                (run_id, ch, np.ascontiguousarray(psd_db[:, ch]).tobytes())
                for ch in range(n_ch)
            ],
        )
        conn.commit()
    return n_ch


def load_psd_runs(db_path=DEFAULT_DB_PATH, *, cable_id=None, t_start=None,
                  t_end=None):
    """Inventory of stored PSD runs -- one row per
    (cable_id, file_starttime, file_endtime).

    Cheap: reads only run metadata (``n_freq``, ``n_ch``, ``created_at``),
    no PSD blobs. ``file_starttime`` / ``file_endtime`` are parsed to
    datetime. ``t_start`` / ``t_end`` range-filter on each run's start time.
    """
    _init_psd_tables(db_path)
    clauses, params = [], []
    if cable_id is not None:
        _in_clause("cable_id", cable_id, clauses, params)
    if t_start is not None:
        clauses.append("file_starttime >= ?")
        params.append(pd.Timestamp(t_start).isoformat())
    if t_end is not None:
        clauses.append("file_starttime <= ?")
        params.append(pd.Timestamp(t_end).isoformat())
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with _connect(db_path) as conn:
        df = pd.read_sql_query(
            f"SELECT cable_id, file_starttime, file_endtime, created_at, "
            f"n_freq, n_ch FROM psd_runs{where} ORDER BY file_starttime;",
            conn,
            params=params,
        )
    for col in ("file_starttime", "file_endtime"):
        # format="ISO8601": stored strings vary in fractional-second
        # precision (isoformat drops ".000"), which a single inferred
        # format would reject
        df[col] = pd.to_datetime(df[col], format="ISO8601", errors="coerce")
    return df


def load_psd(db_path=DEFAULT_DB_PATH, *, cable_id=None, channel=None,
             t_start=None, t_end=None):
    """Load stored PSDs into a DataFrame: one row per (run, channel).

    * ``file_starttime`` / ``file_endtime`` -- pd.Timestamp, the span of data
      the run's PSD was computed over
    * ``ch``    -- int channel index
    * ``freqs`` -- np.ndarray (n_freq,), shared per run (decoded once)
    * ``psds``  -- np.ndarray (n_freq,), PSD in dB

    ``channel`` restricts the read to one channel (or a list of them) --
    only those rows' blobs are fetched, so pulling one channel's history
    stays cheap however many channels are stored. ``t_start`` / ``t_end``
    range-filter on each run's start time.
    """
    _init_psd_tables(db_path)
    clauses, params = [], []
    if cable_id is not None:
        _in_clause("r.cable_id", cable_id, clauses, params)
    if channel is not None:
        _in_clause("p.ch", channel, clauses, params)
    if t_start is not None:
        clauses.append("r.file_starttime >= ?")
        params.append(pd.Timestamp(t_start).isoformat())
    if t_end is not None:
        clauses.append("r.file_starttime <= ?")
        params.append(pd.Timestamp(t_end).isoformat())
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT r.file_starttime, r.file_endtime, p.ch, r.freqs, p.psd "
            f"FROM psd p JOIN psd_runs r ON p.run_id = r.id{where} "
            f"ORDER BY r.file_starttime, p.ch;",
            params,
        ).fetchall()

    if not rows:
        return pd.DataFrame(
            columns=["file_starttime", "file_endtime", "ch", "freqs", "psds"]
        )

    # decode each run's frequency blob once; every row of the run shares it
    freq_memo = {}
    starts, ends, chs, freqs_col, psds_col = [], [], [], [], []
    for start_txt, end_txt, ch, freq_blob, psd_blob in rows:
        f = freq_memo.get(freq_blob)
        if f is None:
            f = np.frombuffer(freq_blob, dtype=_PSD_FREQ_DTYPE)
            freq_memo[freq_blob] = f
        starts.append(start_txt)
        ends.append(end_txt)
        chs.append(ch)
        freqs_col.append(f)
        psds_col.append(np.frombuffer(psd_blob, dtype=_PSD_DTYPE))
    return pd.DataFrame(
        {
            "file_starttime": pd.to_datetime(starts, format="ISO8601"),
            "file_endtime": pd.to_datetime(ends, format="ISO8601"),
            "ch": chs,
            "freqs": freqs_col,
            "psds": psds_col,
        }
    )
