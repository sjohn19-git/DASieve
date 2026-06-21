"""
Batch PSD QC for all .h5 files in an input directory.

Run:
    python test.py
"""

import sys
import os
import glob
import logging
import dascore as dc
import importlib
import dasieve.qc
import dasieve.processing
import dasieve.picker

importlib.reload(dasieve.qc)
importlib.reload(dasieve.processing)
importlib.reload(dasieve.picker)

from dasieve.qc import compute_psd, append_to_store, plot_pdf, plot_patch
from dasieve.processing import normalize_patch, decimate, cmd_remove
from dasieve.picker import trigger_picker, pick_phasenet
from dasieve.catalog import save_picks

sys.path.insert(0, os.path.expanduser("~/DASieve"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


h5_files = [
    "/home/sjohn/data/16BConst_Stimulation_UTC_20240407_072054.163.h5"
]
patch = dc.spool(h5_files[0])[0]
patch = patch.select(distance=(0, 3000))
patch = normalize_patch(patch)
#


# plot_path = os.path.join(save_dir, "psd_pdf.png")
# plot_pdf(store_path=store_path, out_path=plot_path, vmax=0.8, ylim=(-160, -132.5))


patch = decimate(
    patch, target_fs=500, target_dx=5, plot=False, lateral_stacking=True, pws_power=0
)
# freqs, psd_db = compute_psd(patch)
# save_dir = os.path.expanduser("~/Downloads")


# store_path = os.path.join(save_dir, "psd_qc.pkl")
# timestamp = patch.coords.min("time")
# append_to_store(store_path, timestamp, freqs, psd_db)


# # inject 3e-6 spike at mid time sample across all channels for testing
# _data = patch.data.copy()
# _time_axis = patch.dims.index("time")
# _mid = _data.shape[_time_axis] // 2
# _data[_mid : _mid + 10, :] += 3e-6
# patch = patch.new(data=_data)
# plot_patch(patch)

patch = cmd_remove(patch, dim="distance", window=5000, method="median")
# plot_patch(patch)


# source_file = h5_files[0]

# df_trig = trigger_picker(
#     patch, sta=0.3, lta=2.0, thr_on=4.0, thr_off=1, plot=False, plot_channel=None
# )
# save_picks(df_trig, file_name=source_file, author="trigger")

# df_ar = trigger_picker(
#     patch,
#     method="ar",
#     f1=100.0,
#     f2=200.0,
#     lta_p=3,
#     sta_p=0.03,

#     lta_s=3,
#     sta_s=0.03,
#     m_p=2,
#     m_s=2,
#     l_p=0.03,
#     l_s=0.03,
#     plot=True,
#     plot_channel=200,
#     s_pick=False
# )


# save_picks(df_ar, file_name=source_file, author="ar")

df_pn = pick_phasenet(patch, min_prob=0.3, plot=True, plot_channel=280)

save_picks(df_pn, file_name=source_file, author="phasenet")


########

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h5_loc = "/mnt/nas3/sjohn/data/utah_forge/raw_data/das/16b/2024_04_stimulation/07"
results_dir = "/mnt/nas3/sjohn/results"
os.makedirs(results_dir, exist_ok=True)

batch_files = sorted(glob.glob(os.path.join(h5_loc, "**", "*.h5"), recursive=True))[:100]
if not batch_files:
    logging.warning("No .h5 files found in %s", h5_loc)
else:
    logging.info("Found %d .h5 files to process", len(batch_files))
    for source_file in batch_files:
        stem = os.path.splitext(os.path.basename(source_file))[0]
        logging.info("Processing %s", stem)
        try:
            patch = dc.spool(source_file)[0]
            patch = patch.select(distance=(0, 3000))
            patch = normalize_patch(patch)
            patch = decimate(
                patch, target_fs=500, target_dx=5, plot=False,
                lateral_stacking=False, pws_power=2,
            )
            patch = cmd_remove(patch, dim="distance", window=5000, method="median")

            df_pn = pick_phasenet(patch, min_prob=0.3, plot=True, plot_channel=None)

            save_path = os.path.join(results_dir, f"{stem}_phasenet.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            logging.info("  plot saved: %s", save_path)

        except Exception:
            logging.exception("Failed on %s — skipping", stem)
            continue

    logging.info("Batch done.")
