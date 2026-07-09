# %%
"""
Real-data GaMMA association workflow (cell-style, like test.py).

Pipeline:  load h5 -> preprocess -> PhaseNet-DAS picks (plotted, saved to
dasieve.sqlite) -> select picks from the store -> GaMMA association ->
event-colored waterfall + event location map.

Run cell-by-cell in VS Code / Jupyter, or top-to-bottom:
    python tests/test_association.py
"""

import logging

import numpy as np
import pandas as pd
import dascore as dc

import dasieve as sieve
from dasieve import association, detection, store

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

df_trig = sieve.picking.trigger_picker(
    patch,
    sta=0.3,
    lta=2.0,
    thr_on=4.0,
    thr_off=1,
    plot=False,
    plot_channel=None,
    file_name=source_file,
)

df_pn = sieve.picking.phasenet_das_picker(
    patch, min_prob=0.3, plot=False, file_name=source_file,plot_channel=10
)
logging.info("PhaseNet-DAS: %d picks (P=%d, S=%d)", len(df_pn),
             (df_pn["phase"] == "P").sum(), (df_pn["phase"] == "S").sum())

# %% ------------------------------------------------------------------------
# 3. Select picks from the store and run GaMMA
# ----------------------------------------------------------------------------

assoc = association.GammaAssociator.from_preset(
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
    plot=True,
    patch=patch,
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
assoc_trig = association.GammaAssociator.from_preset(
    "default",
    dbscan_eps=3.0,
    dbscan_min_samples=5,
    min_picks_per_eq=100,
    vel={"p": 6.0, "s": 6.0 / 1.75},
)
# plot=True draws the built-in association figure (event locations on the
# cable geometry | picks-on-data colored by event); patch = waterfall bg
catalog_trig, assignments_trig = assoc_trig.run(picks=df_trig, plot=True,
                                                patch=patch)
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
    method="count",           # or "vote" (uses the segment params below)
    window=0.5,               # detection window (s)
    stride=0.25,              # detection stride (s), overlapping
    look_ahead=2.0,           # emitted association window (s)
    min_channels=20,          # count: distinct channels to trigger
    n_segments=8,             # vote: segment layout
    seg_min_channels=3,       # vote: distinct channels for a True vote
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
picks_pn = store.load_picks(method="phasenetdas", file_name=source_file,
                            min_score=0.3)
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

# per-run figures now come from the associator itself:
# run(..., plot=True, patch=patch) -> association.plot_association

# %%
