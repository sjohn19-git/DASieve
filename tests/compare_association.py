# %%
"""
Compare GaMMA association of two pick sets (PhaseNet-DAS vs STA/LTA), gated by
the same vote-method detection windows.

Pipeline:
    load h5 -> preprocess
    -> PhaseNet-DAS picks        (saved to the store, method="phasenetdas")
    -> STA/LTA trigger picks     (saved to the store, method="sta_lta")
    -> vote-method detection     (emits association windows)
    -> GaMMA association of BOTH pick sets inside those windows
       (read from the store, saved back under distinct associator names)
    -> compare the two runs      (summary / matched events / pick overlap)

Run cell-by-cell in VS Code / Jupyter, or top-to-bottom:
    python tests/compare_association.py
"""

# Re-import dasieve on every cell run, so edits to the package take effect
# without restarting the kernel (no-op outside IPython/Jupyter).
try:
    _ip = get_ipython()  # noqa: F821
    _ip.run_line_magic("load_ext", "autoreload")
    _ip.run_line_magic("autoreload", "2")
except NameError:
    pass

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

# Which fiber the data came from. The store keys picks on
# (cable_id, the patch's time window, method), so several files from this
# cable coexist as separate time windows under one cable_id.
cable_id = "16B"

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
    patch, target_fs=500, target_dx=2, plot=False, lateral_stacking=True, pws_power=0)

_x = np.asarray(patch.coords.get_array("x"), dtype=float)
logging.info("geometry after preprocessing: %d/%d channels have x/y/z",
             int(np.sum(~np.isnan(_x))), len(_x))

# %% ------------------------------------------------------------------------
# 2. PhaseNet-DAS picks -> saved to the store (method="phasenetdas")
# ----------------------------------------------------------------------------
df_pn = sieve.picking.phasenet_das_picker(
    patch, min_prob=0.3, plot=False, cable_id=cable_id)
logging.info("PhaseNet-DAS: %d picks (P=%d, S=%d)", len(df_pn),
             (df_pn["phase"] == "P").sum(), (df_pn["phase"] == "S").sum())

# %% ------------------------------------------------------------------------
# 3. STA/LTA trigger picks -> saved to the store (method="sta_lta")
# ----------------------------------------------------------------------------
df_trig = sieve.picking.trigger_picker(
    patch, sta=0.3, lta=2.0, thr_on=4.0, thr_off=1.0,
    plot=True, cable_id=cable_id)
logging.info("STA/LTA: %d trigger picks", len(df_trig))

# %% ------------------------------------------------------------------------
# 3b. Channel mask -- TEST ONLY: flag the last 30 channels as bad
# ----------------------------------------------------------------------------
dist = patch.coords.get_array("distance")
channel_mask = np.ones(dist.size, dtype=bool)
channel_mask[-100:] = False
logging.info("channel_mask keeps %d / %d channels (last 30 excluded)",
             int(channel_mask.sum()), channel_mask.size)

# %% ------------------------------------------------------------------------
# 4. Vote-method detection -> association windows
#    (run on the dense PhaseNet-DAS picks so every fiber segment can vote)
# ----------------------------------------------------------------------------
det = detection.EventDetector(
    method="vote",
    window=0.5,                # detection window (s)
    stride=0.25,               # detection stride (s), overlapping
    look_ahead=2.0,            # emitted association window (s)
    n_segments=8,              # fiber split into 8 equal channel-count segments
    seg_min_channels=3,        # distinct channels for a segment to vote True
    min_votes=5,               # segments that must vote to trigger
    channels=dist,
    channel_mask=channel_mask,
    
)
windows = det.detect(picks=df_pn, plot=True, patch=patch)
print(windows)

# %% ------------------------------------------------------------------------
# 5. GaMMA association inside the detection windows, for BOTH pick sets.
#    Picks are read from the store and origins/associations/events saved back.
#    Both runs use the same associator ("gamma"), but they operate on disjoint
#    pick sets (the "phasenetdas" picks vs the "sta_lta" ones), and the replace
#    only supersedes associations on the picks of the run at hand -- so the two
#    results coexist.
# ----------------------------------------------------------------------------
assoc = association.GammaAssociator.from_preset(
    "default",
    dbscan_eps=3.0,
    dbscan_min_samples=5,
    min_picks_per_eq=100,
    vel={"p": 6.0, "s": 6.0 / 1.75},
)

# --- PhaseNet-DAS picks, gated by the detection windows
org_pn, asc_pn = assoc.run(
    pick_method="phasenetdas", cable_id=cable_id, min_probability=0.3,
    windows=windows, plot=True, patch=patch)

logging.info("gated GaMMA on PhaseNet-DAS: %d origins, %d associated picks",
             len(org_pn), len(asc_pn))

# --- STA/LTA picks, same windows
org_tr, asc_tr = assoc.run(
    pick_method="sta_lta", cable_id=cable_id,
    windows=windows, plot=True, patch=patch)
logging.info("gated GaMMA on STA/LTA: %d origins, %d associated picks",
             len(org_tr), len(asc_tr))

