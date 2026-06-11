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

importlib.reload(dasieve.qc)
importlib.reload(dasieve.processing)

from dasieve.qc import compute_psd, append_to_store, plot_pdf, plot_patch
from dasieve.processing import normalize_patch, decimate, cmd_remove
from dasieve.picker import trigger_picker

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
patch_dec = decimate(
    patch, target_fs=500, target_dx=5, plot=True, lateral_stacking=False, pws_power=2
)

# inject 3e-6 spike at mid time sample across all channels for testing
_data = patch_dec.data.copy()
_time_axis = patch_dec.dims.index("time")
_mid = _data.shape[_time_axis] // 2
_data[_mid : _mid + 10, :] += 3e-6
patch_dec = patch_dec.new(data=_data)
plot_patch(patch_dec)

patch = cmd_remove(patch_dec, dim="distance", window=5000, method="median")
plot_patch(patch)
trigger_picker(
    patch, sta=0.05, lta=1.0, thr_on=3.0, thr_off=0.3, plot=True, plot_channel=None
)


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
