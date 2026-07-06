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
    patch, min_prob=0.3, plot=False, file_name=source_file,plot_channel=10
)
logging.info("PhaseNet-DAS: %d picks (P=%d, S=%d)", len(df_pn),
             (df_pn["phase"] == "P").sum(), (df_pn["phase"] == "S").sum())

# %% ------------------------------------------------------------------------
# 3. Select picks from the store and run GaMMA
# ----------------------------------------------------------------------------

assoc = associator.GammaAssociator.from_preset(
    "default",
    dbscan_eps=3.0,
    dbscan_min_samples=5,
    min_picks_per_eq=10,
    vel={"p": 6.0, "s": 6.0 / 1.75},
)

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

# %% ------------------------------------------------------------------------
# 4. Combined view: well geometry + events in 3D | waterfall with P/S picks
#    (color = phase: blue P / red S, dark = associated, light = unassociated;
#     marker = event, same symbols in both panels)
# ----------------------------------------------------------------------------
# picks from the store (same filters as the association run) joined to the
# assignments -> each pick's event_index (NaN = unassociated)
pick_ids = store.select_pick_ids(
    method="phasenetdas", file_name=source_file, min_score=0.3,
)
picks = store.load_picks_by_ids(pick_ids)
picks_ev = picks.merge(
    assignments_df[["pick_id", "event_index"]],
    left_on="id", right_on="pick_id", how="left",
)

# waterfall arrays
time_vals = patch.coords.get_array("time")
dist_vals = patch.coords.get_array("distance")
t_sec = (time_vals - time_vals[0]) / np.timedelta64(1, "s")
t0 = pd.Timestamp(time_vals[0])
picks_ev["t_s"] = (picks_ev["onset_time"] - t0).dt.total_seconds()

dist_axis = patch.dims.index("distance")
time_axis = patch.dims.index("time")
data2d = np.moveaxis(patch.data, [dist_axis, time_axis], [0, 1])  # (nx, nt)
vmax = np.percentile(np.abs(data2d), 99)
extent = (t_sec[0], t_sec[-1], dist_vals[-1], dist_vals[0])

event_ids = sorted(picks_ev["event_index"].dropna().unique())

# encoding: color = phase (dark = associated, light = unassociated),
#           marker = event
P_DARK, P_LIGHT = "darkblue", "lightskyblue"
S_DARK, S_LIGHT = "darkred", "lightcoral"
EVENT_MARKERS = ["x", "_", "+", "v", "^", "s", "D", "o", "*", "d"]

fig = plt.figure(figsize=(16, 7))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax_w = fig.add_subplot(1, 2, 2)

# --- left: fiber geometry (patch coords, m -> km) + event locations
fib_x = np.asarray(patch.coords.get_array("x"), dtype=float) / 1000.0
fib_y = np.asarray(patch.coords.get_array("y"), dtype=float) / 1000.0
fib_z = np.asarray(patch.coords.get_array("z"), dtype=float) / 1000.0
ax3d.plot(fib_x, fib_y, fib_z, color="0.4", lw=1.5, label="fiber")

for k, ev in enumerate(event_ids):
    row = catalog_df[catalog_df["event_index"] == ev].iloc[0]
    ax3d.scatter(row["x(km)"], row["y(km)"], row["z(km)"], s=120,
                 marker=EVENT_MARKERS[k % len(EVENT_MARKERS)], color="k",
                 linewidths=1.5, label=f"event {int(ev)}")

# box aspect ~ true spatial proportions (fiber + events together)
_ev = catalog_df if len(catalog_df) else None
_spans = []
for arr, col in ((fib_x, "x(km)"), (fib_y, "y(km)"), (fib_z, "z(km)")):
    vals = np.concatenate([arr, _ev[col].to_numpy()]) if _ev is not None else arr
    _spans.append(max(np.nanmax(vals) - np.nanmin(vals), 1e-3))
ax3d.set_box_aspect(_spans)

ax3d.set_xlabel("x (km)")
ax3d.set_ylabel("y (km)")
ax3d.set_zlabel("z (km)")
ax3d.set_title("Well geometry + event locations")
ax3d.legend(fontsize=8)

# --- right: waterfall with P/S picks; associated picks colored by event
ax_w.imshow(data2d, aspect="auto", extent=extent, cmap="gray",
            vmin=-vmax, vmax=vmax, interpolation="nearest")

for ph, color in (("P", P_LIGHT), ("S", S_LIGHT)):
    sub = picks_ev[(picks_ev["phase"] == ph) & picks_ev["event_index"].isna()]
    ax_w.scatter(sub["t_s"], sub["distance"], marker="|", s=30, color=color,
                 linewidths=0.8, label=f"{ph} unassociated", zorder=3)
for k, ev in enumerate(event_ids):
    mk = EVENT_MARKERS[k % len(EVENT_MARKERS)]
    for ph, color in (("P", P_DARK), ("S", S_DARK)):
        sub = picks_ev[(picks_ev["phase"] == ph)
                       & (picks_ev["event_index"] == ev)]
        if len(sub):
            ax_w.scatter(sub["t_s"], sub["distance"], marker=mk, s=45,
                         color=color, linewidths=1.4, zorder=4,
                         label=f"event {int(ev)} {ph} ({len(sub)})")
ax_w.set_xlabel("Time (s)")
ax_w.set_ylabel("Distance (m)")
ax_w.set_title("P (blue) / S (red) picks — dark = associated, marker = event")
ax_w.legend(loc="upper right", fontsize=7, ncol=2)

plt.tight_layout()
plt.show()


# %%
