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
from tqdm import tqdm
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


source_file = h5_files[0]

df_trig = trigger_picker(
    patch, sta=0.3, lta=2.0, thr_on=4.0, thr_off=1, plot=False, plot_channel=None
)
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

df_pn = pick_phasenet(patch, min_prob=0.3, plot=True, plot_channel=None)
save_picks(df_pn, file_name=source_file, author="phasenet")


########

input_dir = "/Volumes/Elements"
batch_files = sorted(glob.glob(os.path.join(input_dir, "07", "*.h5"), recursive=True))[
    :20
]
save_dir = os.path.expanduser("~/Downloads")
store_path = os.path.join(save_dir, "psd_qc.pkl")

if not batch_files:
    logging.warning("No .h5 files found in %s", input_dir)
else:
    logging.info("Found %d .h5 files — starting batch", len(batch_files))
    for h5_path in tqdm(batch_files, desc="PSD batch", unit="file"):
        logging.info("Processing %s", os.path.basename(h5_path))
        try:
            patch = dc.spool(h5_path)[0]
            patch = patch.select(distance=(0, 3000))
            patch = normalize_patch(patch)
            freqs, psd_db = compute_psd(patch)
            timestamp = patch.coords.min("time")
            append_to_store(store_path, timestamp, freqs, psd_db)
        except Exception:
            logging.exception("Failed on %s — skipping", os.path.basename(h5_path))

    # Plot 1: PDF across all channels (distance dimension)
    plot_pdf(
        store_path=store_path,
        out_path=os.path.join(save_dir, "psd_pdf_distance.png"),
        vmax=0.8,
        dimension="distance",
    )

    # Plot 2: PDF for mid channel across time
    plot_pdf(
        store_path=store_path,
        out_path=os.path.join(save_dir, "psd_pdf_time.png"),
        vmax=0.8,
        dimension="time",
    )

    print(f"Store  : {store_path}")
    print(f"Plots  : {save_dir}/psd_pdf_distance.png, psd_pdf_time.png")


import pandas as pd

df = pd.read_pickle(store_path)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

mid_ch = df["ch"].max() // 2
ch_df = df[df["ch"] == mid_ch].sort_values("time").reset_index(drop=True)

times = ch_df["time"].values
n = len(times)
colors = cm.turbo(np.linspace(0, 1, n))

freqs = ch_df["freqs"].iloc[0]
fmin, fmax = 1.0, 250.0
freq_mask = (freqs >= fmin) & (freqs <= fmax)
freqs_masked = freqs[freq_mask]

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

for i, (_, row) in enumerate(ch_df.iterrows()):
    ax.plot(
        freqs_masked, row["psds"][freq_mask], color=colors[i], linewidth=0.6, alpha=0.7
    )

sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(vmin=0, vmax=n - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
cbar.set_label("Time index")
cbar.set_ticks([0, n - 1])
cbar.set_ticklabels([str(times[0])[:19], str(times[-1])[:19]])

ax.set_xscale("log")
ax.set_xlim(fmin, fmax)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB)")
ax.set_title(f"PSD — channel {mid_ch} — all {n} time steps")
ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
plt.tight_layout()
plt.show()
