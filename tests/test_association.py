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
from dasieve import associator, detection, store

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
    patch, target_fs=500, target_dx=2, plot=False, lateral_stacking=True, pws_power=0
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

df_trig = sieve.picker.trigger_picker(
    patch,
    sta=0.3,
    lta=2.0,
    thr_on=4.0,
    thr_off=1,
    plot=True,
    plot_channel=None,
    file_name=source_file,
)

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
    min_picks_per_eq=100,
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
# 3b. STA/LTA trigger picks -> GaMMA, straight from the picker DataFrame
#     (picks=... bypasses the store: nothing is read from or written to the
#      database, the event/assignment tables are only returned;
#      assignments' pick_id indexes back into df_trig rows)
# ----------------------------------------------------------------------------
assoc_trig = associator.GammaAssociator.from_preset(
    "default",
    dbscan_eps=3.0,
    dbscan_min_samples=5,
    min_picks_per_eq=100,
    vel={"p": 6.0, "s": 6.0 / 1.75},
)
catalog_trig, assignments_trig = assoc_trig.run(picks=df_trig)
logging.info("GaMMA on STA/LTA triggers: %d events, %d associated picks",
             len(catalog_trig), len(assignments_trig))
if len(catalog_trig):
    _show = [c for c in ("event_index", "time", "x(km)", "y(km)", "z(km)",
                         "gamma_score", "number_picks", "number_p_picks",
                         "number_s_picks") if c in catalog_trig.columns]
    print(catalog_trig[_show])

# %% ------------------------------------------------------------------------
# 3c. Detection-gated association: only associate where enough of the fiber
#     picked inside a sliding window (see dasieve/detection.py)
# ----------------------------------------------------------------------------
det = detection.EventDetector(
    method="vote",            # or "count" with min_channels=...
    window=0.5,               # detection window (s)
    stride=0.25,              # detection stride (s), overlapping
    look_ahead=2.0,           # emitted association window (s)
    n_segments=8,
    seg_min_channels=3,       # distinct channels for a segment to vote True
    min_votes=5,
    channels=patch.coords.get_array("distance"),
)

# inspect the windows first (plot marks them on the waterfall; vote method
# draws green/red brackets per segment), then associate only inside them
windows = det.detect(picks=df_trig, plot=True, patch=patch)
print(windows)

catalog_det, assignments_det = assoc_trig.run(picks=df_trig, windows=windows)
logging.info("gated GaMMA on STA/LTA: %d events, %d associated picks",
             len(catalog_det), len(assignments_det))
# equivalent one-step form: assoc_trig.run(picks=df_trig, detector=det)

# %% ------------------------------------------------------------------------
# 4. Compare events + associations between the two pick types
#    (PhaseNet-DAS store-backed run vs STA/LTA DataFrame run)
# ----------------------------------------------------------------------------

def merge_assignments(picks_df, assignments_df):
    """Attach each pick's ``event_index`` (NaN = unassociated).

    Works for both workflows: store picks join on their ``id`` column; picker
    DataFrames without ``id`` join on row position, matching how the
    associator numbers ``pick_id`` for DataFrame input."""
    picks_df = picks_df.reset_index(drop=True)
    if "id" not in picks_df.columns:
        picks_df = picks_df.assign(id=np.arange(len(picks_df)))
    return picks_df.merge(
        assignments_df[["pick_id", "event_index"]],
        left_on="id", right_on="pick_id", how="left",
    )


def association_summary(label, picks_ev, catalog_df):
    """One summary row per run: pick counts, association rate, event count."""
    n = len(picks_ev)
    n_assoc = int(picks_ev["event_index"].notna().sum())
    n_ev = len(catalog_df)
    return {
        "run": label,
        "picks": n,
        "associated": n_assoc,
        "unassociated": n - n_assoc,
        "assoc_rate_%": round(100 * n_assoc / n, 1) if n else np.nan,
        "events": n_ev,
        "picks/event": round(n_assoc / n_ev, 1) if n_ev else np.nan,
    }


def match_events(cat_a, cat_b, max_dt=5.0):
    """Pair events across two catalogs by nearest origin time.

    Greedy: each event in ``cat_a`` takes the closest-in-time unused event in
    ``cat_b`` within ``max_dt`` seconds. Returns
    ``(matched_df, unmatched_a, unmatched_b)`` where matched_df has one row
    per pair with time and location offsets (b minus a) and the unmatched
    lists hold the leftover event indices from either catalog.
    """
    a = cat_a.assign(_t=pd.to_datetime(cat_a["time"]))
    b = cat_b.assign(_t=pd.to_datetime(cat_b["time"]))
    used_b, rows = set(), []
    for _, ea in a.sort_values("_t").iterrows():
        cand = b[~b.index.isin(used_b)]
        if not len(cand):
            break
        dt = (cand["_t"] - ea["_t"]).dt.total_seconds()
        j = dt.abs().idxmin()
        if abs(dt.loc[j]) > max_dt:
            continue
        used_b.add(j)
        eb = b.loc[j]
        dx = eb["x(km)"] - ea["x(km)"]
        dy = eb["y(km)"] - ea["y(km)"]
        dz = eb["z(km)"] - ea["z(km)"]
        rows.append({
            "a_event": int(ea["event_index"]),
            "b_event": int(eb["event_index"]),
            "dt_s": round(dt.loc[j], 3),
            "dx_km": round(dx, 3),
            "dy_km": round(dy, 3),
            "dz_km": round(dz, 3),
            "d3d_km": round(float(np.sqrt(dx**2 + dy**2 + dz**2)), 3),
        })
    matched = pd.DataFrame(rows)
    matched_a = set(matched["a_event"]) if len(matched) else set()
    matched_b = set(matched["b_event"]) if len(matched) else set()
    unmatched_a = sorted(set(cat_a["event_index"].astype(int)) - matched_a)
    unmatched_b = sorted(set(cat_b["event_index"].astype(int)) - matched_b)
    return matched, unmatched_a, unmatched_b


def matched_pick_overlap(matched, picks_ev_a, picks_ev_b):
    """Per matched event pair: picks each run associated and how many
    channels (distances) the two associations share."""
    rows = []
    for _, m in matched.iterrows():
        in_a = picks_ev_a["event_index"] == m["a_event"]
        in_b = picks_ev_b["event_index"] == m["b_event"]
        ch_a = set(picks_ev_a.loc[in_a, "distance"])
        ch_b = set(picks_ev_b.loc[in_b, "distance"])
        union = len(ch_a | ch_b)
        rows.append({
            "a_event": int(m["a_event"]),
            "b_event": int(m["b_event"]),
            "picks_a": int(in_a.sum()),
            "picks_b": int(in_b.sum()),
            "shared_channels": len(ch_a & ch_b),
            "channel_jaccard": round(len(ch_a & ch_b) / union, 2)
                               if union else np.nan,
        })
    return pd.DataFrame(rows)


# per-pick event_index for both runs: PhaseNet picks reloaded from the store
# (same filters as the association run), STA/LTA picks straight from df_trig
picks_pn = store.load_picks_by_ids(
    store.select_pick_ids(method="phasenetdas", file_name=source_file,
                          min_score=0.3)
)
picks_ev_pn = merge_assignments(picks_pn, assignments_df)
picks_ev_trig = merge_assignments(df_trig, assignments_trig)

# --- pick/association summary
summary = pd.DataFrame([
    association_summary("PhaseNet-DAS", picks_ev_pn, catalog_df),
    association_summary("STA/LTA", picks_ev_trig, catalog_trig),
])
print("\n=== association summary ===")
print(summary.to_string(index=False))

# --- event-level comparison: a = PhaseNet-DAS, b = STA/LTA
matched, only_pn, only_trig = match_events(catalog_df, catalog_trig, max_dt=5.0)
print("\n=== matched events (a=PhaseNet-DAS, b=STA/LTA; offsets = b - a) ===")
if len(matched):
    print(matched.to_string(index=False))
else:
    print("no events matched within 5.0 s")
print(f"events only in PhaseNet-DAS catalog: {only_pn}")
print(f"events only in STA/LTA catalog:      {only_trig}")

# --- association overlap for matched events
if len(matched):
    overlap = matched_pick_overlap(matched, picks_ev_pn, picks_ev_trig)
    print("\n=== per-event association overlap ===")
    print(overlap.to_string(index=False))

# %% ------------------------------------------------------------------------
# 5. Events on top of the data
#    left:  well geometry + event locations (one color per event, depth down)
#    right: waterfall imshow + picks -- gray = unassociated, colored = its
#           event (colors match the left panel)
# ----------------------------------------------------------------------------

def geom_limits(patch, catalogs, pad_km=0.1):
    """Shared 3D extent -- (easting, northing, depth) (lo, hi) pairs in km --
    covering the fiber and every catalog's events, so multiple figures use
    identical axis limits and stay visually comparable."""
    east = [np.asarray(patch.coords.get_array("y"), float) / 1000.0]
    north = [np.asarray(patch.coords.get_array("x"), float) / 1000.0]
    dep = [np.asarray(patch.coords.get_array("z"), float) / 1000.0]
    for cat in catalogs:
        if len(cat):
            east.append(cat["y(km)"].to_numpy())
            north.append(cat["x(km)"].to_numpy())
            dep.append(cat["z(km)"].to_numpy())
    lims = []
    for vals in (np.concatenate(east), np.concatenate(north),
                 np.concatenate(dep)):
        lims.append((np.nanmin(vals) - pad_km, np.nanmax(vals) + pad_km))
    return lims


def plot_events_on_data(patch, picks_ev, catalog_df, title="", lims=None):
    """Waterfall + picks colored by event, next to the event locations.

    picks_ev comes from :func:`merge_assignments` (event_index NaN =
    unassociated). Picks are drawn with two scatter calls (one gray for
    unassociated, one vectorized-color for associated), so it stays fast for
    thousands of picks. z is depth (positive down) -> 3D z-axis inverted.

    lims : optional (easting, northing, depth) (lo, hi) pairs from
        :func:`geom_limits`; pass the same value to several calls to keep
        the 3D axes identical across figures. Default: this run's extent.
    """
    time_vals = patch.coords.get_array("time")
    dist_vals = patch.coords.get_array("distance")
    t0 = pd.Timestamp(time_vals[0])
    t_sec = (time_vals - time_vals[0]) / np.timedelta64(1, "s")
    data2d = np.moveaxis(
        patch.data,
        [patch.dims.index("distance"), patch.dims.index("time")], [0, 1],
    )
    vmax = np.percentile(np.abs(data2d), 99)
    extent = (t_sec[0], t_sec[-1], dist_vals[-1], dist_vals[0])

    picks_ev = picks_ev.copy()
    picks_ev["t_s"] = (
        pd.to_datetime(picks_ev["onset_time"]) - t0
    ).dt.total_seconds()
    event_ids = sorted(picks_ev["event_index"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    ev_color = {ev: cmap(k % 10) for k, ev in enumerate(event_ids)}

    fig = plt.figure(figsize=(15, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_w = fig.add_subplot(1, 2, 2)

    # --- left: well geometry + event locations (m -> km)
    # survey convention (see well_plot.py): the survey's x_m column is
    # NORTHING (xN ~ 4.26e6 m) and y_m is EASTING (yE ~ 3.3e5 m), so plot
    # easting on the horizontal axis and northing on the vertical to get a
    # real map view -- same as well_plot.py's scatter(yE, xN, -tvd)
    f_north = np.asarray(patch.coords.get_array("x"), float) / 1000.0
    f_east = np.asarray(patch.coords.get_array("y"), float) / 1000.0
    f_dep = np.asarray(patch.coords.get_array("z"), float) / 1000.0
    ax3d.plot(f_east, f_north, f_dep, color="0.4", lw=1.5, label="fiber")
    for ev in event_ids:
        row = catalog_df[catalog_df["event_index"] == ev].iloc[0]
        ax3d.scatter(row["y(km)"], row["x(km)"], row["z(km)"], s=120,
                     marker="*", color=ev_color[ev], edgecolor="k",
                     linewidths=0.5, label=f"event {int(ev)}")
    # fixed limits + true spatial proportions: without the aspect the
    # ~0.3 km lateral deviation is stretched to match the 2.4 km depth and
    # the well looks bent
    if lims is None:
        lims = geom_limits(patch, [catalog_df])
    (e_lim, n_lim, d_lim) = lims
    ax3d.set_xlim(e_lim)
    ax3d.set_ylim(n_lim)
    ax3d.set_zlim(d_lim)
    ax3d.set_box_aspect([max(hi - lo, 1e-3) for lo, hi in lims])
    ax3d.invert_zaxis()  # z is depth: wellhead (z=0) on top
    # small tick/label text, full coordinate values (no "+4.263e3" offset);
    # tick count follows each axis's drawn length (true-proportion aspect
    # makes the northing axis short -- 5 ticks there collide)
    spans = [max(hi - lo, 1e-3) for lo, hi in lims]
    for axis, span in zip((ax3d.xaxis, ax3d.yaxis, ax3d.zaxis), spans):
        nbins = int(np.clip(round(6 * np.sqrt(span / max(spans))), 3, 5))
        axis.set_major_locator(plt.MaxNLocator(nbins))
        axis.get_major_formatter().set_useOffset(False)
    ax3d.tick_params(labelsize=7, pad=0)
    ax3d.set_xlabel("Easting (km)", fontsize=8, labelpad=4)
    ax3d.set_ylabel("Northing (km)", fontsize=8, labelpad=4)
    ax3d.set_zlabel("depth (km)", fontsize=8, labelpad=2)
    ax3d.set_title(f"{title}: event locations", fontsize=10)
    ax3d.legend(fontsize=7, loc="upper left")

    # --- right: waterfall + picks
    ax_w.imshow(data2d, aspect="auto", extent=extent, cmap="gray",
                vmin=-vmax, vmax=vmax, interpolation="nearest")
    un = picks_ev[picks_ev["event_index"].isna()]
    if len(un):
        ax_w.scatter(un["t_s"], un["distance"], marker="|", s=25,
                     linewidths=0.8, color="0.75",
                     label=f"unassociated ({len(un)})", zorder=3)
    asc = picks_ev.dropna(subset=["event_index"])
    if len(asc):
        ax_w.scatter(asc["t_s"], asc["distance"], marker="|", s=40,
                     linewidths=1.2, zorder=4,
                     c=[ev_color[e] for e in asc["event_index"]])
    # legend swatches per event, colors matching the 3D panel
    for ev in event_ids:
        n = int((asc["event_index"] == ev).sum())
        ax_w.scatter([], [], marker="|", s=40, linewidths=1.2,
                     color=ev_color[ev], label=f"event {int(ev)} ({n})")
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("Distance (m)")
    ax_w.set_title(f"{title}: picks on data (gray = unassociated)")
    ax_w.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()
    return fig


# one shared 3D extent (fiber + both catalogs) so the two figures are
# directly comparable; assigning the returned figures also stops Jupyter
# from re-displaying the last one as the cell output
lims = geom_limits(patch, [catalog_df, catalog_trig])
fig_pn = plot_events_on_data(patch, picks_ev_pn, catalog_df,
                             title="PhaseNet-DAS + GaMMA", lims=lims)
fig_trig = plot_events_on_data(patch, picks_ev_trig, catalog_trig,
                               title="STA/LTA + GaMMA", lims=lims)


# %%
