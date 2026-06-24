"""
Batch PSD QC for all .h5 files in an input directory.

Run:
    python test.py
"""
#%%
%load_ext autoreload
%autoreload 2

import os
import glob
import logging

import dascore as dc
import dasieve as sieve



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
h5_files = [
    "/Users/sj201/Downloads/16BConst_Stimulation_UTC_20240407_072054.163.h5"
]
source_file = h5_files[0]

patch = dc.spool(source_file)[0]
patch = patch.select(distance=(0, 3000))
patch = sieve.processing.to_strain_rate(patch)
patch = sieve.processing.cmd_remove(
    patch, dim="distance", window=5000, method="median", plot=True
)
patch = sieve.processing.decimate(
    patch, target_fs=500, target_dx=5, plot=True, lateral_stacking=True, pws_power=0
)
freqs, psd_db = sieve.qc.compute_psd(patch, plot=True, vmax=0.8, ylim=(-160, -132.5))




df_trig = sieve.picker.trigger_picker(
    patch,
    sta=0.3,
    lta=2.0,
    thr_on=4.0,
    thr_off=1,
    plot=False,
    plot_channel=None,
    file_name=source_file,
)

# df_ar = sieve.picker.trigger_picker(
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
#     s_pick=False,
#     file_name=source_file,
# )

df_pn = sieve.picker.phasenet_picker(
    patch, min_prob=0.3, plot=True, plot_channel=280, file_name=source_file
)

# EQTransformer (SeisBench) on the same patch, visualized with the shared
# picker plotter. device auto-detects MPS on Apple Silicon.
df_eqt = sieve.picker.seisbench_picker(
    patch,
    model="eqtransformer",
    pretrained="original",
    min_prob=0.3,
    plot=True,
    plot_channel=280,
    file_name=source_file,
)


########





import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

h5_loc = "/mnt/nas3/sjohn/data/utah_forge/raw_data/das/16b/2024_04_stimulation/07"
results_dir = "/mnt/nas3/sjohn/results"
os.makedirs(results_dir, exist_ok=True)

batch_files = sorted(glob.glob(os.path.join(h5_loc, "**", "*.h5"), recursive=True))[
    :100
]
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
            patch = sieve.processing.to_strain_rate(patch)
            patch = sieve.processing.decimate(
                patch,
                target_fs=500,
                target_dx=5,
                plot=False,
                lateral_stacking=False,
                pws_power=2,
            )
            patch = sieve.processing.cmd_remove(
                patch, dim="distance", window=5000, method="median"
            )

            df_pn = sieve.picker.phasenet_picker(
                patch, min_prob=0.3, plot=True, plot_channel=None, file_name=source_file
            )

            save_path = os.path.join(results_dir, f"{stem}_phasenet.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            logging.info("  plot saved: %s", save_path)

        except Exception:
            logging.exception("Failed on %s — skipping", stem)
            continue

    logging.info("Batch done.")
