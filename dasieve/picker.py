import os
import sys
import glob as _glob
import tempfile
import importlib.util

import numpy as np
import pandas as pd
from obspy.signal.trigger import classic_sta_lta, trigger_onset, ar_pick
from dascore.utils.io import patch_to_obspy

# Shared output schema for every picking method so that the returned
# DataFrame is consistent across "sta_lta", "ar", and "phasenet". Methods
# that don't produce a given field leave it as NaN.
PICK_COLUMNS = [
    "distance",
    "phase",
    "onset_sample",
    "onset_time",
    "score",
    "cft_at_onset",
    "off_sample",
    "off_time",
    "cft_at_off",
]


def _pick_sta_lta(trace, sta, lta, thr_on, thr_off):
    """STA/LTA trigger picks for one trace. Returns (list_of_pick_dicts, cft)."""
    fs = trace.stats.sampling_rate
    nsta = int(sta * fs)
    nlta = int(lta * fs)
    cft = classic_sta_lta(trace.data, nsta, nlta)

    picks = []
    for on, off in trigger_onset(cft, thr_on, thr_off):
        picks.append(
            {
                "phase": "trigger",
                "onset_sample": int(on),
                "off_sample": int(off),
                "cft_at_onset": cft[on],
                "cft_at_off": cft[off],
            }
        )
    return picks, cft


def _pick_ar(
    trace, f1, f2, lta_p, sta_p, lta_s, sta_s, m_p, m_s, l_p, l_s, s_pick=True
):
    """AR-AIC P/S picks for one trace. Single-component DAS: the same channel
    data is passed for all three ar_pick components so that S picking stays
    enabled. Returns (list_of_pick_dicts, None)."""
    fs = trace.stats.sampling_rate
    data = np.ascontiguousarray(trace.data, dtype=np.float64)

    p_time, s_time = ar_pick(
        data,
        data,
        data,
        fs,
        f1,
        f2,
        lta_p,
        sta_p,
        lta_s,
        sta_s,
        m_p,
        m_s,
        l_p,
        l_s,
        s_pick=s_pick,
    )

    picks = []
    if p_time and p_time > 0:
        picks.append({"phase": "P", "onset_sample": int(p_time * fs)})
    if s_time and s_time > 0:
        picks.append({"phase": "S", "onset_sample": int(s_time * fs)})
    return picks, None


def _plot_picks(patch, df, ch_idx, cft_plot, thr_on, thr_off):
    """Single plotting routine used for every method. Plots P/S picks when
    present, otherwise trigger onsets."""
    import matplotlib.pyplot as plt

    dist_vals = patch.coords.get_array("distance")
    time_vals = patch.coords.get_array("time")
    t_sec = (time_vals - time_vals[0]) / np.timedelta64(1, "s")

    phase_color = {"P": "blue", "S": "red", "trigger": "blue"}

    # patch data oriented as (distance, time) for imshow
    time_axis = patch.dims.index("time")
    dist_axis = patch.dims.index("distance")
    data_dt = np.moveaxis(patch.data, [dist_axis, time_axis], [0, 1])

    n_panels = 3 if cft_plot is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 10), sharex=True)
    ax_img, ax_trace = axes[0], axes[1]
    ax_cft = axes[2] if cft_plot is not None else None

    # --- top: imshow of full patch ---
    extent = [t_sec[0], t_sec[-1], dist_vals[-1], dist_vals[0]]
    vmax = np.percentile(np.abs(data_dt), 99)
    ax_img.imshow(
        data_dt,
        aspect="auto",
        extent=extent,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    # tick markers for every pick, colored by phase
    for _, row in df.iterrows():
        ax_img.plot(
            t_sec[int(row["onset_sample"])],
            row["distance"],
            "|",
            color=phase_color.get(row["phase"], "blue"),
            markersize=10,
            markeredgewidth=1.5,
        )

    ax_img.axhline(
        dist_vals[ch_idx],
        color="green",
        linestyle="--",
        linewidth=1.2,
        label=f"ch {ch_idx}",
    )
    ax_img.set_ylabel("Distance")
    ax_img.legend(loc="upper right", fontsize=8)

    # --- middle: selected channel waveform ---
    ax_trace.plot(t_sec, data_dt[ch_idx], color="black", linewidth=0.6)
    ch_picks = df[df["distance"] == dist_vals[ch_idx]]
    for _, row in ch_picks.iterrows():
        phase = row["phase"]
        ax_trace.axvline(
            t_sec[int(row["onset_sample"])],
            color=phase_color.get(phase, "blue"),
            linewidth=1.2,
            linestyle="--",
            label=phase,
        )
        # draw trigger-off marker only for STA/LTA triggers
        if phase == "trigger" and not pd.isna(row.get("off_sample")):
            ax_trace.axvline(
                t_sec[int(row["off_sample"])],
                color="red",
                linewidth=1,
                linestyle="--",
            )
    ax_trace.set_ylabel("Amplitude")
    # de-duplicate legend labels
    handles, labels = ax_trace.get_legend_handles_labels()
    if labels:
        uniq = dict(zip(labels, handles))
        ax_trace.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

    # --- bottom: CFT with thresholds (STA/LTA only) ---
    if ax_cft is not None:
        ax_cft.plot(t_sec, cft_plot, color="steelblue", linewidth=0.7)
        ax_cft.axhline(
            thr_on,
            color="blue",
            linestyle="--",
            linewidth=1,
            label=f"thr_on={thr_on}",
        )
        ax_cft.axhline(
            thr_off,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"thr_off={thr_off}",
        )
        ax_cft.set_ylabel("CFT")
        ax_cft.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def trigger_picker(
    patch,
    method="sta_lta",
    sta=0.05,
    lta=1.0,
    thr_on=3.0,
    thr_off=0.3,
    f1=1.0,
    f2=20.0,
    lta_p=1.0,
    sta_p=0.1,
    lta_s=1.0,
    sta_s=0.5,
    m_p=2,
    m_s=8,
    l_p=0.1,
    l_s=0.2,
    plot=False,
    plot_channel=None,
    s_pick=True,
):
    """Pick arrivals on a DAS patch.

    method : "sta_lta" (classic STA/LTA + trigger_onset) or
             "ar" (AR-AIC P/S picker; single-component data is passed for all
             three ar_pick components).
    """
    stream = patch_to_obspy(patch)
    dist_vals = patch.coords.get_array("distance")
    time_vals = patch.coords.get_array("time")

    n_traces = len(stream)
    ch_idx = plot_channel if plot_channel is not None else n_traces // 2

    rows = []
    cft_plot = None
    for i, trace in enumerate(stream):
        if method == "sta_lta":
            picks, cft = _pick_sta_lta(trace, sta, lta, thr_on, thr_off)
            if i == ch_idx:
                cft_plot = cft
        elif method == "ar":
            picks, _ = _pick_ar(
                trace,
                f1,
                f2,
                lta_p,
                sta_p,
                lta_s,
                sta_s,
                m_p,
                m_s,
                l_p,
                l_s,
                s_pick=s_pick,
            )
        else:
            raise ValueError(f"unknown method: {method!r}")

        for p in picks:
            on = p["onset_sample"]
            off = p.get("off_sample")
            rows.append(
                {
                    "distance": dist_vals[i],
                    "phase": p["phase"],
                    "onset_sample": on,
                    "onset_time": time_vals[on],
                    "score": p.get("score", np.nan),
                    "cft_at_onset": p.get("cft_at_onset", np.nan),
                    "off_sample": off,
                    "off_time": time_vals[off] if off is not None else np.nan,
                    "cft_at_off": p.get("cft_at_off", np.nan),
                }
            )

    df = pd.DataFrame(
        rows,
        columns=PICK_COLUMNS,
    )

    if plot:
        _plot_picks(patch, df, ch_idx, cft_plot, thr_on, thr_off)

    return df


# ---------------------------------------------------------------------------
# PhaseNet-DAS
# ---------------------------------------------------------------------------
DEFAULT_EQNET_DIR = os.path.join(os.path.dirname(__file__), "EQNet")


def _import_eqnet_predict(eqnet_dir):
    """Import EQNet's predict.py as a module *in-process* (no os.system).

    EQNet's predict.py does bare ``import eqnet`` / ``import utils``, so the
    EQNet directory must be on sys.path. We load predict.py by file path under
    a private module name to avoid clashing with any other ``predict`` module.
    """
    eqnet_dir = os.path.abspath(eqnet_dir)
    predict_path = os.path.join(eqnet_dir, "predict.py")
    if not os.path.isfile(predict_path):
        raise FileNotFoundError(f"EQNet predict.py not found at {predict_path}")
    if eqnet_dir not in sys.path:
        sys.path.insert(0, eqnet_dir)
    spec = importlib.util.spec_from_file_location("eqnet_predict", predict_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_to_eqnet_h5(patch, h5_path):
    """Write a dascore patch to the h5 layout EQNet's DASIterableDataset reads:
    a 2-D ``data`` dataset shaped (n_distance, n_time) with dt_s / dx_m /
    begin_time attributes. Returns (dist_vals, time_vals)."""
    import h5py

    dist_vals = patch.coords.get_array("distance")
    time_vals = patch.coords.get_array("time")

    dist_axis = patch.dims.index("distance")
    time_axis = patch.dims.index("time")
    data = np.moveaxis(patch.data, [dist_axis, time_axis], [0, 1])  # (nx, nt)
    data = np.ascontiguousarray(data, dtype=np.float32)

    dt_s = float(np.median(np.diff(time_vals)) / np.timedelta64(1, "s"))
    dx_m = float(np.median(np.diff(dist_vals)))
    begin_time = pd.Timestamp(time_vals[0]).to_pydatetime().isoformat()

    with h5py.File(h5_path, "w") as fp:
        ds = fp.create_dataset("data", data=data)  # (nx, nt)
        ds.attrs["dt_s"] = dt_s
        ds.attrs["dx_m"] = dx_m
        ds.attrs["begin_time"] = begin_time

    return dist_vals, time_vals


def pick_phasenet(
    patch,
    eqnet_dir=DEFAULT_EQNET_DIR,
    min_prob=0.3,
    device=None,
    phases=("P", "S"),
    highpass_filter=0.0,
    cut_patch=False,
    batch_size=1,
    work_dir=None,
    keep_files=False,
    plot=False,
    plot_channel=None,
):
    """PhaseNet-DAS picks on a DAS patch, returned in the shared PICK_COLUMNS
    schema (same as ``trigger_picker``).

    Unlike the per-trace ``sta_lta`` / ``ar`` methods, PhaseNet-DAS runs once
    on the whole (channel x time) patch. EQNet's prediction pipeline is invoked
    by **importing** predict.py and calling ``predict.main`` in-process (no
    ``os.system``); the patch is staged as a temporary h5 file in the format
    EQNet's ``DASIterableDataset`` expects, and the resulting picks CSV is read
    back and mapped into the shared schema.

    Parameters
    ----------
    patch : dascore Patch with "distance" and "time" coords.
    eqnet_dir : path to the EQNet repo (the directory containing predict.py).
    min_prob : minimum phase probability for a pick (EQNet --min_prob).
    device : "cuda" / "cpu" / "mps"; auto-detected (cuda if available else cpu).
    phases : phase labels the model outputs, default ("P", "S").
    highpass_filter : EQNet highpass corner in Hz (0.0 = no filter).
    cut_patch : tile very large patches into EQNet patches before predicting.
    batch_size : number of tiles processed per GPU step. Only has effect when
        cut_patch=True (a single un-tiled patch is always batch 1). Increase
        on GPUs with more VRAM until you hit OOM; has no effect on CPU speed.
    work_dir : staging directory; a temp dir is used (and cleaned) if None.
    keep_files : if True, keep the staged h5 / results instead of deleting them.

    Returns
    -------
    pandas.DataFrame with columns == PICK_COLUMNS. PhaseNet picks populate
    ``score`` (phase probability); cft_* and off_* columns are NaN.
    """
    import torch
    import matplotlib

    _mpl_backend = matplotlib.get_backend()
    predict = _import_eqnet_predict(eqnet_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    owns_dir = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="phasenet_das_")
    os.makedirs(work_dir, exist_ok=True)
    data_dir = os.path.join(work_dir, "data")
    result_dir = os.path.join(work_dir, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    h5_path = os.path.join(data_dir, "patch.h5")
    dist_vals, time_vals = _patch_to_eqnet_h5(patch, h5_path)
    n_time = len(time_vals)

    files_list = os.path.join(data_dir, "files.txt")
    with open(files_list, "w") as f:
        f.write(h5_path)

    # Build a full args namespace from EQNet's own parser (so every attribute
    # main() expects is present), then override what we need.
    args = predict.get_args_parser().parse_args([])
    args.model = "phasenet_das"
    args.data_path = data_dir
    args.data_list = files_list
    args.result_path = result_dir
    args.format = "h5"
    args.batch_size = batch_size if cut_patch else 1
    args.workers = 0
    args.device = device
    args.min_prob = min_prob
    args.phases = list(phases)
    args.highpass_filter = highpass_filter
    args.cut_patch = cut_patch
    args.plot_figure = False
    # No "rank" attr + no torchrun env -> utils.init_distributed_mode() sets
    # args.distributed = False, so this stays single-process.

    cwd = os.getcwd()
    try:
        os.chdir(eqnet_dir)  # model weights cache to ./model_phasenet_das
        predict.main(args)
    finally:
        os.chdir(cwd)
        matplotlib.use(_mpl_backend)

    # EQNet writes picks under result_dir/picks_phasenet_das[/_patch]/...csv.
    csv_files = _glob.glob(os.path.join(result_dir, "**", "*.csv"), recursive=True)
    frames = []
    for csv in csv_files:
        if os.path.getsize(csv) == 0:
            continue
        try:
            frames.append(pd.read_csv(csv))
        except pd.errors.EmptyDataError:
            continue

    df = _phasenet_picks_to_df(frames, dist_vals, time_vals, n_time)

    if not keep_files and owns_dir:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)

    if plot:
        n_ch = len(patch.coords.get_array("distance"))
        ch_idx = plot_channel if plot_channel is not None else n_ch // 2
        _plot_picks(patch, df, ch_idx, None, None, None)

    return df


def _phasenet_picks_to_df(frames, dist_vals, time_vals, n_time):
    """Map EQNet PhaseNet-DAS pick rows (channel_index, phase_index,
    phase_score, phase_type) into the shared PICK_COLUMNS schema."""
    if not frames:
        return pd.DataFrame(columns=PICK_COLUMNS)

    picks = pd.concat(frames, ignore_index=True)
    n_dist = len(dist_vals)

    rows = []
    for _, r in picks.iterrows():
        ch = int(r["channel_index"])
        smp = int(r["phase_index"])
        if ch < 0 or ch >= n_dist or smp < 0 or smp >= n_time:
            # guard against padding-region picks
            continue
        rows.append(
            {
                "distance": dist_vals[ch],
                "phase": r["phase_type"],
                "onset_sample": smp,
                "onset_time": time_vals[smp],
                "score": float(r["phase_score"]),
                "cft_at_onset": np.nan,
                "off_sample": np.nan,
                "off_time": np.nan,
                "cft_at_off": np.nan,
            }
        )

    df = pd.DataFrame(rows, columns=PICK_COLUMNS)
    if not df.empty:
        df.sort_values(["distance", "onset_sample"], inplace=True, ignore_index=True)
    return df
