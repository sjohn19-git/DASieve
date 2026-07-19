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
survey_path = "/Users/sj201/Downloads/survey.csv"

# Which fiber the data came from. Picks/events are keyed on (cable_id, the
# patch's time window, method) -- so every file from this cable shares the
# cable_id and is distinguished by its own time window.
cable_id = "16BConst"

patch = dc.spool(source_file)[0]
survey = sieve.processing.load_survey(survey_path)
patch = sieve.processing.attach_geometry(patch, survey)



patch = patch.select(distance=(1000, 3000))
patch = sieve.processing.to_strain_rate(patch)
patch = sieve.processing.remove_cmod(
    patch, dim="distance", window=5000, method="median", plot=True
)

patch = sieve.processing.decimate(
    patch, target_fs=500, target_dx=5, plot=True, lateral_stacking=True, pws_power=0
)


#%%

freqs, psd_db = sieve.qc.compute_psd(patch, plot=True, vmax=0.8, ylim=(-160, -132.5))


df_trig = sieve.picking.trigger_picker(
    patch,
    sta=0.3,
    lta=2.0,
    thr_on=4.0,
    thr_off=1,
    plot=False,
    plot_channel=None,
    cable_id=cable_id,
)

# df_ar = sieve.picking.trigger_picker(
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
#     cable_id=cable_id,
# )

df_pn = sieve.picking.phasenet_das_picker(
    patch, min_prob=0.3, plot=True, plot_channel=280, cable_id=cable_id
)

# EQTransformer (SeisBench) on the same patch, visualized with the shared
# picker plotter. device auto-detects MPS on Apple Silicon.
df_eqt = sieve.picking.seisbench_picker(
    patch,
    model="eqtransformer",
    pretrained="original",
    min_prob=0.3,
    plot=True,
    cable_id=cable_id,
)


# string key — loads pretrained weights automatically
df = sieve.picking.seisbench_picker(
    patch,
    model="phasenet",
    pretrained="original",
    plot=True,
    cable_id=cable_id)


#%%

def test_phasenet(patch, source_file=None, min_prob=0.3, max_match_s=1.0):
    """Compare in-memory phasenet_das_picker vs disk-based phasenet_das_picker_disk.

    Plots two waterfall panels (one per method with picks overlaid) and a
    histogram of matched-pick time differences (in-memory minus disk-based).
    """
    import time
    import numpy as np
    import matplotlib.pyplot as plt

    # ── run both pickers ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    df_mem = sieve.picking.phasenet_das_picker(
        patch, min_prob=min_prob, plot=False, db_save=False
    )
    t_mem = time.perf_counter() - t0

    t0 = time.perf_counter()
    df_disk = sieve.picking.phasenet_das_picker_disk(
        patch, min_prob=min_prob, plot=False, db_save=False
    )
    t_disk = time.perf_counter() - t0

    print(f"in-memory  : {len(df_mem):5d} picks  {t_mem:.2f}s")
    print(f"disk-based : {len(df_disk):5d} picks  {t_disk:.2f}s")

    # ── match picks by (distance, phase) nearest-neighbour in time ────────────
    time_vals = patch.coords.get_array("time")
    dt_s = float(np.median(np.diff(time_vals)) / np.timedelta64(1, "s"))
    max_match_samp = int(max_match_s / dt_s)

    diffs_ms = []
    for _, row in df_disk.iterrows():
        sub = df_mem[
            (df_mem["distance"] == row["distance"]) & (df_mem["phase"] == row["phase"])
        ]
        if sub.empty:
            continue
        delta = (sub["onset_sample"] - int(row["onset_sample"])).abs()
        best = delta.idxmin()
        if delta[best] <= max_match_samp:
            diffs_ms.append(
                (sub.loc[best, "onset_sample"] - row["onset_sample"]) * dt_s * 1e3
            )
    diffs_ms = np.array(diffs_ms)

    if len(diffs_ms):
        print(
            f"matched    : {len(diffs_ms):5d} pairs  "
            f"mean={np.mean(np.abs(diffs_ms)):.2f} ms  "
            f"max={np.max(np.abs(diffs_ms)):.2f} ms"
        )
    else:
        print("matched    :     0 pairs")

    # ── build common plot arrays ───────────────────────────────────────────────
    dist_vals = patch.coords.get_array("distance")
    t_sec = (time_vals - time_vals[0]) / np.timedelta64(1, "s")

    dist_axis = patch.dims.index("distance")
    time_axis = patch.dims.index("time")
    data2d = np.moveaxis(patch.data, [dist_axis, time_axis], [0, 1])  # (nx, nt)
    vmax = np.percentile(np.abs(data2d), 99)
    extent = [t_sec[0], t_sec[-1], dist_vals[-1], dist_vals[0]]

    PHASE_COLORS = {"P": "cyan", "S": "lime"}

    def _waterfall(ax, df, title):
        ax.imshow(
            data2d,
            aspect="auto",
            extent=extent,
            cmap="gray",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        for phase, col in PHASE_COLORS.items():
            sub = df[df["phase"] == phase]
            if not sub.empty:
                ax.scatter(
                    t_sec[sub["onset_sample"].astype(int)],
                    sub["distance"],
                    marker="|",
                    s=40,
                    color=col,
                    linewidths=1.0,
                    label=phase,
                    zorder=3,
                )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    _, axes = plt.subplots(
        1, 3, figsize=(20, 7),
        gridspec_kw={"width_ratios": [5, 5, 2]},
    )

    _waterfall(
        axes[0], df_mem,
        f"phasenet_das_picker (in-memory)\n{len(df_mem)} picks  {t_mem:.1f}s",
    )
    _waterfall(
        axes[1], df_disk,
        f"phasenet_das_picker_disk\n{len(df_disk)} picks  {t_disk:.1f}s",
    )

    ax_h = axes[2]
    if len(diffs_ms):
        ax_h.hist(
            diffs_ms, bins=40, color="steelblue", edgecolor="none",
            orientation="horizontal",
        )
        ax_h.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax_h.set_xlabel("Count")
        ax_h.set_ylabel("Δt  (ms)  [in-memory − disk]")
        ax_h.set_title(
            f"{len(diffs_ms)} matched\n"
            f"μ = {np.mean(diffs_ms):.2f} ms\n"
            f"σ = {np.std(diffs_ms):.2f} ms"
        )
    else:
        ax_h.text(0.5, 0.5, "no matched picks", ha="center", va="center",
                  transform=ax_h.transAxes)

    plt.tight_layout()
    plt.show()
    return df_mem, df_disk, diffs_ms


df_mem, df_disk, diffs_ms = test_phasenet(patch, source_file=source_file)

########


#%%


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

h5_loc = "/mnt/nas3/sjohn/data/utah_forge/raw_data/das/16b/2024_04_stimulation/07"
results_dir = "/mnt/nas3/sjohn/results"
os.makedirs(results_dir, exist_ok=True)

batch_files = sorted(glob.glob(os.path.join(h5_loc, "**", "*.h5"), recursive=True))[
    :100
]
survey = sieve.processing.load_survey(survey_path)
if not batch_files:
    logging.warning("No .h5 files found in %s", h5_loc)
else:
    logging.info("Found %d .h5 files to process", len(batch_files))
    for source_file in batch_files:
        stem = os.path.splitext(os.path.basename(source_file))[0]
        logging.info("Processing %s", stem)
        try:
            patch = dc.spool(source_file)[0]
            patch = sieve.processing.attach_geometry(patch, survey)
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
            patch = sieve.processing.remove_cmod(
                patch, dim="distance", window=5000, method="median"
            )

            df_pn = sieve.picking.phasenet_das_picker(
                patch, min_prob=0.3, plot=True, plot_channel=None,
                cable_id=cable_id,
            )

            save_path = os.path.join(results_dir, f"{stem}_phasenet.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            logging.info("  plot saved: %s", save_path)

        except Exception:
            logging.exception("Failed on %s — skipping", stem)
            continue

    logging.info("Batch done.")
