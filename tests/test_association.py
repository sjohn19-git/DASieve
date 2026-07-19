# %%
"""
Minimal GaMMA association workflow (cell-style, like test.py).

Pipeline:  load h5 -> preprocess -> PhaseNet-DAS picks (saved to the store)
-> GaMMA association (read from the store, saved back) -> event-colored
waterfall + event location map.

For the PhaseNet-DAS vs STA/LTA comparison with detection-gated windows, see
compare_association.py.

Run cell-by-cell in VS Code / Jupyter, or top-to-bottom:
    python tests/test_association.py
"""

import logging

import numpy as np
import dascore as dc

import dasieve as sieve
from dasieve import association

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

source_file = "/Users/sj201/Downloads/16BConst_Stimulation_UTC_20240407_072054.163.h5"
survey_path = "/Users/sj201/Downloads/survey.csv"

# Which fiber the data came from. The store keys picks on
# (cable_id, the patch's time window, method), so several files from this
# cable coexist as separate time windows under one cable_id.
cable_id = "16BConst"

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
# 3. GaMMA association: read the picks from the store, save events back.
#    Results are keyed on (cable_id, time window, method="gamma",
#    pick_method="phasenetdas")
#    -> rerunning on the same picks replaces them instead of duplicating.
# ----------------------------------------------------------------------------
assoc = association.GammaAssociator.from_preset(
    "default",
    dbscan_eps=3.0,
    dbscan_min_samples=5,
    min_picks_per_eq=100,
    vel={"p": 6.0, "s": 6.0 / 1.75},
)
catalog_df, assignments_df = assoc.run(
    pick_method="phasenetdas", cable_id=cable_id, picks=df_pn, min_score=0.3,
    plot=True, patch=patch)



logging.info("GaMMA: %d events, %d associated picks",
             len(catalog_df), len(assignments_df))

if len(catalog_df):
    _show = [c for c in ("event_index", "time", "x(km)", "y(km)", "z(km)",
                         "gamma_score", "number_picks", "number_p_picks",
                         "number_s_picks") if c in catalog_df.columns]
    print(catalog_df[_show])

# %%