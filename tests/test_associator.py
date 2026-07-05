# %%
"""
Real-data GaMMA association workflow (cell-style, like test.py).

Pipeline:  load h5 -> preprocess -> PhaseNet-DAS picks (plotted, saved to
dasieve.sqlite) -> select picks from the store -> GaMMA association ->
event-colored waterfall + event location map.

Run cell-by-cell in VS Code / Jupyter, or top-to-bottom:
    python tests/test_associator.py
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dascore as dc

import dasieve as sieve
from dasieve import associator, store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

source_file = "/Users/sj201/Downloads/16BConst_Stimulation_UTC_20240407_072054.163.h5"
survey_path = "/Users/sj201/Downloads/survey.csv"

# %% ------------------------------------------------------------------------
# 1. Load + preprocess (same chain as test.py)
# ----------------------------------------------------------------------------
patch = dc.spool(source_file)[0]
survey = sieve.processing.load_survey(survey_path)
patch = sieve.processing.attach_geometry(patch, survey)

patch = patch.select(distance=(1000, 3000))
patch = sieve.processing.to_strain_rate(patch)
patch = sieve.processing.cmd_remove(
    patch, dim="distance", window=5000, method="median", plot=False
)
patch = sieve.processing.decimate(
    patch, target_fs=500, target_dx=5, plot=False, lateral_stacking=True, pws_power=0
)

# sanity check: geometry must survive the processing chain, otherwise picks
# are stored with NaN x/y/z and association has no stations to work with
_x = np.asarray(patch.coords.get_array("x"), dtype=float)
logging.info("geometry after preprocessing: %d/%d channels have x/y/z",
             int(np.sum(~np.isnan(_x))), len(_x))

# %% ------------------------------------------------------------------------
# 2. PhaseNet-DAS picks -> plotted + saved to ~/DASieve/dasieve.sqlite
#    (picks now carry x/y/z from the attached geometry)
# ----------------------------------------------------------------------------
# saves automatically to the store (method="phasenetdas"); re-running the
# same file replaces its previous phasenetdas picks
df_pn = sieve.picker.phasenet_das_picker(
    patch, min_prob=0.3, plot=True, plot_channel=None, file_name=source_file
)
logging.info("PhaseNet-DAS: %d picks (P=%d, S=%d)", len(df_pn),
             (df_pn["phase"] == "P").sum(), (df_pn["phase"] == "S").sum())

# %% ------------------------------------------------------------------------
# 3. Select picks from the store and run GaMMA
# ----------------------------------------------------------------------------
# pick rows are only needed for the plots below; associate() does its own
# selection from the same filters
pick_ids = store.select_pick_ids(
    method="phasenetdas",
    file_name=source_file,
    min_score=0.3,          # raise to feed GaMMA only high-confidence picks
)
picks = store.load_picks_by_ids(pick_ids)

# DAS/microseismic tuning on top of the "default" preset. All knobs are
# editable here (or later via assoc.update_config(...)).
#   dbscan_eps        : max time gap (s) linking picks into one cluster;
#                       25 s (regional default) is far too wide for
#                       stimulation microseismicity -> a few seconds.
#   min_picks_per_eq  : DAS has hundreds of channels, so demand more picks.
#   vel               : FORGE granite ~6.0 km/s P, Vp/Vs ~1.75.
# NOTE z convention: bounds are auto-derived from the survey z. If your
# survey z is elevation (negative down-hole), pass explicit
# **{"z(km)": (zmin, zmax)} / bfgs_bounds overrides here.
assoc = associator.GammaAssociator.from_preset(
    "default",
    dbscan_eps=3.0,
    dbscan_min_samples=5,
    min_picks_per_eq=10,
    vel={"p": 6.0, "s": 6.0 / 1.75},
)
# e.g. to constrain the x search bounds (km) explicitly:
#   assoc.update_config(**{"x(km)": (4258.0, 4263.0)})

# associates AND saves to events/assignments (db_save=True default);
# results are keyed on (file_name, method="gamma") -> rerun replaces
catalog_df, assignments_df = assoc.run(
    pick_method="phasenetdas",   # whose picks to associate
    file_name=source_file,
    min_score=0.3,
)
logging.info("GaMMA: %d events, %d associated picks",
             len(catalog_df), len(assignments_df))
if len(catalog_df):
    _show = [c for c in ("event_index", "time", "x(km)", "y(km)", "z(km)",
                         "gamma_score", "number_picks", "number_p_picks",
                         "number_s_picks") if c in catalog_df.columns]
    print(catalog_df[_show])

# (already persisted by associate above -- pass db_save=False to skip saving)

# %% ------------------------------------------------------------------------
# 4. Plot: waterfall with picks colored by associated event
# ----------------------------------------------------------------------------
time_vals = patch.coords.get_array("time")
dist_vals = patch.coords.get_array("distance")
t_sec = (time_vals - time_vals[0]) / np.timedelta64(1, "s")
t0 = pd.Timestamp(time_vals[0])

dist_axis = patch.dims.index("distance")
time_axis = patch.dims.index("time")
data2d = np.moveaxis(patch.data, [dist_axis, time_axis], [0, 1])  # (nx, nt)
vmax = np.percentile(np.abs(data2d), 99)
extent = [t_sec[0], t_sec[-1], dist_vals[-1], dist_vals[0]]

# join picks <- assignments to get each pick's event_index (NaN = unassociated)
picks_ev = picks.merge(
    assignments_df[["pick_id", "event_index"]],
    left_on="id", right_on="pick_id", how="left",
)
picks_ev["t_s"] = (picks_ev["onset_time"] - t0).dt.total_seconds()

fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(data2d, aspect="auto", extent=extent, cmap="RdBu",
          vmin=-vmax, vmax=vmax, interpolation="nearest")

unassoc = picks_ev[picks_ev["event_index"].isna()]
ax.scatter(unassoc["t_s"], unassoc["distance"], marker="|", s=30,
           color="0.5", linewidths=0.8, label="unassociated", zorder=3)

event_ids = sorted(picks_ev["event_index"].dropna().unique())
cmap = plt.get_cmap("tab20")
for k, ev in enumerate(event_ids):
    sub = picks_ev[picks_ev["event_index"] == ev]
    ax.scatter(sub["t_s"], sub["distance"], marker="|", s=45,
               color=cmap(k % 20), linewidths=1.4,
               label=f"event {int(ev)} ({len(sub)})", zorder=4)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Distance (m)")
ax.set_title(f"GaMMA association — {len(event_ids)} events, "
             f"{len(picks_ev) - len(unassoc)}/{len(picks_ev)} picks associated")
ax.legend(loc="upper right", fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# %% ------------------------------------------------------------------------
# 5. Plot: event locations vs the fiber (plan view + depth section)
# ----------------------------------------------------------------------------
if len(catalog_df):
    st_x = picks["x"].to_numpy() / 1000.0   # fiber channels, km
    st_y = picks["y"].to_numpy() / 1000.0
    st_z = picks["z"].to_numpy() / 1000.0

    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(13, 6))

    ax_xy.plot(st_x, st_y, ".", color="0.6", markersize=2, label="fiber")
    ax_xz.plot(st_x, st_z, ".", color="0.6", markersize=2, label="fiber")
    for k, (_, ev) in enumerate(catalog_df.iterrows()):
        c = cmap(k % 20)
        ax_xy.scatter(ev["x(km)"], ev["y(km)"], s=90, marker="*", color=c,
                      edgecolor="k", linewidth=0.5, zorder=4,
                      label=f"event {int(ev['event_index'])}")
        ax_xz.scatter(ev["x(km)"], ev["z(km)"], s=90, marker="*", color=c,
                      edgecolor="k", linewidth=0.5, zorder=4)

    ax_xy.set_xlabel("x (km)"); ax_xy.set_ylabel("y (km)")
    ax_xy.set_title("Plan view"); ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.legend(fontsize=8)
    ax_xz.set_xlabel("x (km)"); ax_xz.set_ylabel("z (km)")
    ax_xz.set_title("Depth section")
    plt.tight_layout()
    plt.show()
else:
    logging.info("no events to plot — loosen dbscan_eps / min_picks_per_eq "
                 "or lower min_score and re-run the association cell")

# %% ------------------------------------------------------------------------
# 6. Read results back from the DB (sanity check)
# ----------------------------------------------------------------------------
import sqlite3

with sqlite3.connect(store.DEFAULT_DB_PATH) as conn:
    ev_db = pd.read_sql_query(
        "SELECT file_name, method, event_index, time, x_km, y_km, z_km, number_picks "
        "FROM events ORDER BY id DESC LIMIT 10;", conn)
    n_asg = pd.read_sql_query("SELECT COUNT(*) AS n FROM assignments;", conn)
print(ev_db)
print(f"assignments rows in DB: {n_asg['n'].iloc[0]}")
