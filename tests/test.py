"""
Minimal DASieve walkthrough on a single file (cell-style).

    load h5 -> preprocess -> PSD QC -> picks (STA/LTA + PhaseNet-DAS)
    -> GaMMA association, both ways:
         (a) db_save   -- picks read from the store, origins/associations/
                          events written back
         (b) DataFrame -- picks handed over directly, database never touched

Everything lands in one SQLite file (``~/DASieve/dasieve.sqlite``) whose
catalog follows the QuakeML chain::

    picks --< associations >-- origins --< events

Run cell-by-cell in VS Code / Jupyter, or top-to-bottom:
    python tests/test.py
"""
# %%
%load_ext autoreload
%autoreload 2

import logging

import numpy as np
import dascore as dc

import dasieve as sieve
from dasieve import association, store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

source_file = "/Users/sj201/Downloads/16BConst_Stimulation_UTC_20240407_072054.163.h5"
survey_path = "/Users/sj201/Downloads/survey.csv"

# Which fiber the data came from. Picks are keyed on
# (cable_id, file_starttime, file_endtime, pick_method), so every file from
# this cable shares the cable_id and is told apart by its own file window.
cable_id = "16B"

print("store:", store.DEFAULT_DB_PATH)

# %% ------------------------------------------------------------------------
# 1. Load + preprocess
# ----------------------------------------------------------------------------
patch = dc.spool(source_file)[0]
survey = sieve.processing.load_survey(survey_path)
patch = sieve.processing.attach_geometry(patch, survey)

patch = patch.select(distance=(1000, 3000))
patch = sieve.processing.to_strain_rate(patch)
patch = sieve.processing.remove_cmod(
    patch, dim="distance", window=5000, method="median", plot=False)
patch = sieve.processing.decimate(
    patch, target_fs=500, target_dx=2, plot=False, lateral_stacking=True,
    pws_power=0)

# association needs x/y/z per channel; channels off the survey stay NaN and
# are dropped by the associator
_x = np.asarray(patch.coords.get_array("x"), dtype=float)
logging.info("geometry: %d/%d channels have x/y/z",
             int(np.sum(~np.isnan(_x))), len(_x))

# %% ------------------------------------------------------------------------
# 2. QC: per-channel PSD -> psd_runs / psd tables
#    Keyed on (cable_id, file_starttime, file_endtime), like the picks.
# ----------------------------------------------------------------------------
freqs, psd_db = sieve.qc.compute_psd(
    patch, cable_id=cable_id, plot=True, vmax=0.8, ylim=(-160, -132.5))
logging.info("PSD: %d frequencies x %d channels", *psd_db.shape)

print(store.load_psd_runs(cable_id=cable_id))

# %% ------------------------------------------------------------------------
# 3. Picks -> picks table. Each picker returns the table's own schema, so the
#    DataFrame and a row read back out of the database look identical.
#    Re-running a picker replaces that file's picks for that pick_method, and
#    cascades: associations on the old picks, their origins, and those
#    origins' events are deleted too.
# ----------------------------------------------------------------------------
df_trig = sieve.picking.trigger_picker(
    patch,
    method="sta_lta",
    sta=0.3,
    lta=2.0,
    thr_on=4.0,
    thr_off=1.0,
    plot=True,
    cable_id=cable_id,
    db_save=True
)
logging.info("STA/LTA: %d picks (all labelled P)", len(df_trig))

df_pn = sieve.picking.phasenet_das_picker(
    patch, min_prob=0.3, plot=True, plot_channel=280, cable_id=cable_id)

logging.info("PhaseNet-DAS: %d picks (P=%d, S=%d)", len(df_pn),
             (df_pn["phase"] == "P").sum(), (df_pn["phase"] == "S").sum())

print(df_pn.head())
print("\nstored picks by method:")
print(store.load_picks(cable_id=cable_id).groupby("pick_method").size())

# %% ------------------------------------------------------------------------
# 4. Detection: scan the picks and emit association windows.
#    Runs before the associator -- association is then done once per window,
#    on the picks inside it, and picks outside every window are never
#    associated. Windows come from the store here, but det.detect(picks=df_pn)
#    takes a DataFrame just like the associator does.
# ----------------------------------------------------------------------------
det = sieve.detection.EventDetector(
    method="count",
    window=1.0,            # detection window (s)
    stride=0.25,           # overlapping stride (s)
    look_ahead=2.0,        # length of each emitted association window (s)
    min_channels=100,       # distinct channels picking to trigger
)
windows = det.detect(pick_method="phasenetdas", cable_id=cable_id,
                     plot=True, patch=patch)
logging.info("detector: %d association window(s)", len(windows))
print(windows)

# %% ------------------------------------------------------------------------
# 5a. GaMMA with db_save: picks are SELECTED FROM THE STORE and the results
#     are written back to origins / associations / events.
#
#     Re-running this cell supersedes gamma's existing associations on these
#     picks and cascades to their origins/events, so results replace rather
#     than duplicate. A different associator on the same picks would coexist.
#
#     Drop `windows=` to associate the whole file in one go instead.
# ----------------------------------------------------------------------------
assoc = association.GammaAssociator.from_preset(
    "default",
    dbscan_eps=3.0,
    dbscan_min_samples=5,
    min_picks_per_eq=100,
    vel={"p": 6.0, "s": 6.0 / 1.75},
)

origins_df, associations_df = assoc.run(
    pick_method="phasenetdas",       # which picker's picks to associate
    cable_id=cable_id,
    min_probability=0.3,
    windows=windows,                 # from the detector above
    db_save=True,
    plot=True,
    patch=patch,
)
logging.info("GaMMA (db_save): %d origins, %d associations",
             len(origins_df), len(associations_df))
print(origins_df)

# %% ------------------------------------------------------------------------
# 5b. GaMMA with a DataFrame: picks handed over directly.
#     The database is never touched -- no selection, no save (db_save is
#     ignored on this path). Use it to try settings without writing anything.
# ----------------------------------------------------------------------------
n_events_before = len(store.load_events())

origins_mem, associations_mem = assoc.run(
    picks=df_pn,                     # straight from the picker, no store read
    windows=windows,                 # same detector windows as above
    plot=True,
    patch=patch,
)
logging.info("GaMMA (DataFrame): %d origins, %d associations",
             len(origins_mem), len(associations_mem))
print(origins_mem)

# pick_id here indexes back into df_pn's rows, not picks.id
assert len(store.load_events()) == n_events_before, "DataFrame path wrote to the DB!"
print(f"\nevents in store unchanged: {n_events_before} -- nothing was written")

# %% ------------------------------------------------------------------------
# 6. Read the catalog back out of the store.
#    Each loader returns exactly its own table -- same columns, same order as
#    in SQLite. Combined views are plain pandas merges along the chain:
#        picks --< associations >-- origins --< events
# ----------------------------------------------------------------------------
picks_tbl = store.load_picks(cable_id=cable_id)
origins_tbl = store.load_origins(origin_method="gamma")
assoc_tbl = store.load_associations(association_method="gamma")
events_tbl = store.load_events()

print("picks:      ", list(picks_tbl.columns))
print("associations:", list(assoc_tbl.columns))
print("origins:    ", list(origins_tbl.columns))
print("events:     ", list(events_tbl.columns))

print(f"\nevents ({len(events_tbl)}):")
print(events_tbl)
print(f"\norigins ({len(origins_tbl)}):")
print(origins_tbl)

# the catalog view: each event with its preferred origin's time and location
catalog = events_tbl.merge(
    origins_tbl, left_on="preferred_origin_id", right_on="id",
    suffixes=("_event", "_origin"))
print(f"\ncatalog (events joined to their preferred origin):")
print(catalog[["id_event", "magnitude", "origin_time", "x", "y", "z",
               "number_picks"]])

# the picks behind each origin: associations joined back to picks
assoc_picks = assoc_tbl.merge(
    picks_tbl, left_on="pick_id", right_on="id", suffixes=("_assoc", "_pick"))
print(f"\nassociations: {len(assoc_tbl)} rows")
if len(assoc_picks):
    print(assoc_picks[["origin_id", "pick_id", "phase", "onset_time",
                       "distance", "probability_assoc",
                       "probability_pick"]].head())
    print("\npicks per origin:")
    print(assoc_picks.groupby("origin_id").size())
# %%
