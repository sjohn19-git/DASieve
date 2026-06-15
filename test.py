"""
Batch PSD QC for all .h5 files in an input directory.

Run:
    python test.py
"""

import sys
import os
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
    "/Users/sebinjohn/Downloads/16BConst_Stimulation_UTC_20240407_072054.163.h5"
]
patch = dc.spool(h5_files[0])[0]
patch = patch.select(distance=(0, 3000))
patch = normalize_patch(patch)
patch = decimate(
    patch, target_fs=500, target_dx=5, plot=True, lateral_stacking=False, pws_power=2
)

# inject 3e-6 spike at mid time sample across all channels for testing
_data = patch.data.copy()
_time_axis = patch.dims.index("time")
_mid = _data.shape[_time_axis] // 2
_data[_mid : _mid + 10, :] += 3e-6
patch = patch.new(data=_data)
plot_patch(patch)

patch = cmd_remove(patch, dim="distance", window=5000, method="median")
plot_patch(patch)


source_file = h5_files[0]

df_trig = trigger_picker(
    patch, sta=0.3, lta=2.0, thr_on=4.0, thr_off=1, plot=True, plot_channel=None
)
save_picks(df_trig, file_name=source_file, author="trigger")

df_ar = trigger_picker(
    patch,
    method="ar",
    f1=1.0,
    f2=30.0,
    lta_p=2.0,
    sta_p=0.3,
    lta_s=2,
    sta_s=0.3,
    m_p=2,
    m_s=8,
    l_p=0.1,
    l_s=0.2,
    plot=True,
    plot_channel=None,
)
save_picks(df_ar, file_name=source_file, author="ar")

df_pn = pick_phasenet(patch, min_prob=0.3, plot=True, plot_channel=None)
save_picks(df_pn, file_name=source_file, author="phasenet")

# if __name__ == "__main__":
#     input_dir = "/Volumes/Elements"
#     save_dir = os.path.expanduser("~/Downloads")

#     store_path = os.path.join(save_dir, "psd_qc.h5")
#     plot_path = os.path.join(save_dir, "psd_pdf.png")

#     max_files = 30  # set to None to process all files

#     h5_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.h5"), recursive=True))
#     if not h5_files:
#         logging.warning("No .h5 files found in %s", input_dir)
#         sys.exit(1)

#     logging.info("Found %d .h5 files total", len(h5_files))
#     if max_files is not None:
#         h5_files = h5_files[:max_files]
#         logging.info("Processing first %d files", len(h5_files))

#     for h5_path in tqdm(h5_files, desc="PSD batch", unit="file"):
#         logging.info("Processing %s", os.path.basename(h5_path))
#         try:
#             patch = dc.spool(h5_path)[0]
#             patch = patch.select(distance=(0, 3000))
#             patch_dec = normalize_patch(patch)

#             freqs, psd_db = compute_psd(patch_dec)
#             timestamp = patch.coords.min("time")
#             append_to_store(store_path, timestamp, freqs, psd_db)

#         except Exception:
#             logging.exception("Failed on %s — skipping", os.path.basename(h5_path))
#             continue

#     # all channels
#     plot_pdf(store_path=store_path, out_path=plot_path, vmax=0.6)

#     print(f"Store : {store_path}")
#     print(f"Plot  : {plot_path}")
