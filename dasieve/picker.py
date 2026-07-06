import os
import sys
import glob as _glob
import tempfile
import importlib.util
import warnings
from contextlib import nullcontext

import numpy as np
import pandas as pd
from obspy.signal.trigger import classic_sta_lta, trigger_onset, ar_pick
from dascore.utils.io import patch_to_obspy

# Shared output schema for every picking method so that the returned
# DataFrame is consistent across "sta_lta", "ar", and "phasenet". Methods
# that don't produce a given field leave it as NaN.
PICK_COLUMNS = [
    "distance",
    "x",
    "y",
    "z",
    "phase",
    "onset_sample",
    "onset_time",
    "score",
    "cft_at_onset",
    "off_sample",
    "off_time",
    "cft_at_off",
]


def _geom_arrays(patch):
    """Return (x, y, z) 1-D arrays aligned to the patch's ``distance`` dim.

    Geometry coords are attached by :func:`dasieve.processing.attach_geometry`;
    a coord that is missing (or the wrong length) yields a NaN-filled array so
    pickers work unchanged on patches without geometry."""
    n = len(patch.coords.get_array("distance"))
    out = {}
    for name in ("x", "y", "z"):
        try:
            arr = np.asarray(patch.coords.get_array(name), dtype=float)
            if arr.shape[0] != n:
                arr = np.full(n, np.nan)
        except Exception:
            arr = np.full(n, np.nan)
        out[name] = arr
    return out["x"], out["y"], out["z"]


def _auto_device():
    """Pick the best available torch device: CUDA (NVIDIA) > MPS (Apple
    Silicon) > CPU. Imported lazily so torch is only required when a
    deep-learning picker actually runs."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _save_to_store(df, file_name, method, db_path=None):
    """Persist picks via :func:`dasieve.store.save_picks`. Imported lazily to
    avoid a circular import (dasieve.store imports PICK_COLUMNS from this
    module). Replaces existing rows for (file_name, method)."""
    if file_name is None:
        raise ValueError(
            "db_save=True requires file_name (the source data file). "
            "Pass file_name=... or db_save=False."
        )
    from .store import save_picks

    kwargs = {} if db_path is None else {"db_path": db_path}
    return save_picks(df, file_name=file_name, method=method, **kwargs)


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
    file_name=None,
    db_save=True,
    db_path=None,
):
    """Pick arrivals on a DAS patch.

    method : "sta_lta" (classic STA/LTA + trigger_onset) or
             "ar" (AR-AIC P/S picker; single-component data is passed for all
             three ar_pick components).

    db_save : if True (default), the picks are saved to the store via
        ``store.save_picks`` with this ``method`` as the store method,
        replacing previous rows for (file_name, method). ``db_path`` overrides
        the default store location. Pass db_save=False to only return the
        DataFrame.
    """
    stream = patch_to_obspy(patch)
    dist_vals = patch.coords.get_array("distance")
    time_vals = patch.coords.get_array("time")
    x_arr, y_arr, z_arr = _geom_arrays(patch)

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
                    "x": x_arr[i],
                    "y": y_arr[i],
                    "z": z_arr[i],
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

    if db_save:
        _save_to_store(df, file_name, method, db_path=db_path)

    return df


# ---------------------------------------------------------------------------
# PhaseNet-DAS
# ---------------------------------------------------------------------------
DEFAULT_EQNET_DIR = os.path.join(os.path.dirname(__file__), "EQNet")

_MODEL_CACHE: dict = {}


def _ensure_eqnet_on_path(eqnet_dir):
    eqnet_dir = os.path.abspath(eqnet_dir)
    if eqnet_dir not in sys.path:
        sys.path.insert(0, eqnet_dir)
    return eqnet_dir


def _load_model(eqnet_dir, device, location, phases):
    import torch
    import eqnet

    cache_key = (eqnet_dir, device, location, tuple(phases))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = eqnet.models.__dict__["phasenet_das"].build_model(
        backbone="unet",
        in_channels=1,
        out_channels=len(phases) + 1,
    )

    if location is None:
        model_url = (
            "https://github.com/AI4EPS/models/releases/download/"
            "PhaseNet-DAS-v1/PhaseNet-DAS-v1.pth"
        )
    elif location == "forge":
        model_url = (
            "https://github.com/AI4EPS/models/releases/download/"
            "PhaseNet-DAS-ConvertedPhase/model_99.pth"
        )
    else:
        raise ValueError(f"no pretrained model for location={location!r}")

    cwd = os.getcwd()
    try:
        os.chdir(eqnet_dir)
        checkpoint = torch.hub.load_state_dict_from_url(
            model_url,
            model_dir="./model_phasenet_das",
            progress=True,
            check_hash=True,
            map_location="cpu",
        )
    finally:
        os.chdir(cwd)

    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    return model


def _preprocess(patch, highpass_filter=0.0):
    import torch
    import scipy.signal
    from eqnet.data.das import padding

    dist_vals = patch.coords.get_array("distance")
    time_vals = patch.coords.get_array("time")

    dist_axis = patch.dims.index("distance")
    time_axis = patch.dims.index("time")
    data = np.moveaxis(patch.data, [dist_axis, time_axis], [0, 1])  # (nx, nt)
    data = np.ascontiguousarray(data, dtype=np.float32)

    dt_s = float(np.median(np.diff(time_vals)) / np.timedelta64(1, "s"))
    begin_time = pd.Timestamp(time_vals[0]).to_pydatetime().isoformat(timespec="milliseconds")

    data = data - np.mean(data, axis=-1, keepdims=True)
    data = data - np.median(data, axis=-2, keepdims=True)
    if highpass_filter > 0.0:
        b, a = scipy.signal.butter(2, highpass_filter, "hp", fs=1.0 / dt_s)
        data = scipy.signal.filtfilt(b, a, data, axis=-1)

    data = data.T[np.newaxis, :, :]  # (1, nt, nx)
    tensor = torch.from_numpy(data)

    nt, nx = tensor.shape[1], tensor.shape[2]
    tensor = padding(tensor, min_nt=1024, min_nx=1024)

    return tensor, nt, nx, dt_s, begin_time, dist_vals, time_vals


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


def phasenet_das_picker(
    patch,
    eqnet_dir=DEFAULT_EQNET_DIR,
    min_prob=0.3,
    device=None,
    highpass_filter=0.0,
    location=None,
    plot=False,
    plot_channel=None,
    file_name=None,
    db_save=True,
    db_path=None,
):
    """PhaseNet-DAS picks on a DAS patch, entirely in memory (no temp files).

    The model is loaded once per process and cached; subsequent calls reuse
    the cached weights.

    Parameters
    ----------
    patch : dascore Patch with "distance" and "time" coords.
    eqnet_dir : path to the EQNet repo directory.
    min_prob : minimum phase probability threshold.
    device : "cuda" / "mps" / "cpu"; auto-detected if None.
    highpass_filter : highpass corner in Hz; 0.0 = no filter.
    location : pretrained model variant; None for default, "forge" for FORGE.

    Returns
    -------
    pandas.DataFrame with columns == PICK_COLUMNS. When ``db_save`` is True
    (default) the picks are also saved to the store with method
    ``"phasenetdas"``, replacing previous rows for (file_name, method).
    """
    import torch
    phases=("P", "S")
    phases = list(phases)
    eqnet_dir = _ensure_eqnet_on_path(eqnet_dir)

    if device is None:
        device = _auto_device()

    model = _load_model(eqnet_dir, device, location, phases)

    data, nt, nx, dt_s, begin_time, dist_vals, time_vals = _preprocess(
        patch, highpass_filter
    )

    data_batched = data.unsqueeze(0).to(device)

    dtype_str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=ptdtype)

    with torch.inference_mode(), ctx:
        output = model({"data": data_batched})

    phase_out = output["phase"][:, :, :nt, :nx].cpu()
    scores = torch.softmax(phase_out, dim=1)

    from eqnet.utils import detect_peaks, extract_picks

    topk_scores, topk_inds = detect_peaks(scores, vmin=min_prob, kernel=21)

    picks_ = extract_picks(
        topk_inds,
        topk_scores,
        file_name=["in_memory"],
        begin_time=[begin_time],
        begin_time_index=torch.tensor([0]),
        begin_channel_index=torch.tensor([0]),
        dt=dt_s,
        vmin=min_prob,
        phases=phases,
    )

    frames = []
    if picks_[0]:
        df_raw = pd.DataFrame(picks_[0])
        df_raw["channel_index"] = df_raw["station_id"].apply(lambda x: int(x))
        frames.append(df_raw)

    x_arr, y_arr, z_arr = _geom_arrays(patch)
    df = _phasenet_picks_to_df(
        frames, dist_vals, time_vals, len(time_vals), x_arr, y_arr, z_arr
    )

    if plot:
        n_ch = len(dist_vals)
        ch_idx = plot_channel if plot_channel is not None else n_ch // 2
        _plot_picks(patch, df, ch_idx, None, None, None)

    if db_save:
        _save_to_store(df, file_name, "phasenetdas", db_path=db_path)

    return df


def phasenet_das_picker_disk(
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
    file_name=None,
    db_save=True,
    db_path=None,
):
    """PhaseNet-DAS picks via EQNet's disk-based pipeline (temp h5 + CSV).

    Stages the patch as a temporary h5 file, runs EQNet's predict.main
    in-process, reads back the CSV picks. Prefer ``phasenet_das_picker``
    (in-memory) for normal use; this variant is kept for comparison and
    cut_patch tiling support.

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
    ``score`` (phase probability); cft_* and off_* columns are NaN. When
    ``db_save`` is True (default) the picks are also saved to the store with
    method ``"phasenetdas"``, replacing previous rows for
    (file_name, method).
    """
    import torch
    import matplotlib

    _mpl_backend = matplotlib.get_backend()
    predict = _import_eqnet_predict(eqnet_dir)

    if device is None:
        device = _auto_device()

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

    x_arr, y_arr, z_arr = _geom_arrays(patch)
    df = _phasenet_picks_to_df(
        frames, dist_vals, time_vals, n_time, x_arr, y_arr, z_arr
    )

    if not keep_files and owns_dir:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)

    if plot:
        n_ch = len(patch.coords.get_array("distance"))
        ch_idx = plot_channel if plot_channel is not None else n_ch // 2
        _plot_picks(patch, df, ch_idx, None, None, None)

    if db_save:
        _save_to_store(df, file_name, "phasenetdas", db_path=db_path)

    return df


def _phasenet_picks_to_df(frames, dist_vals, time_vals, n_time, x_arr, y_arr, z_arr):
    """Map EQNet PhaseNet-DAS pick rows (channel_index, phase_index,
    phase_score, phase_type) into the shared PICK_COLUMNS schema.
    ``x_arr/y_arr/z_arr`` are the per-channel geometry arrays from
    :func:`_geom_arrays`, indexed by channel."""
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
                "x": x_arr[ch],
                "y": y_arr[ch],
                "z": z_arr[ch],
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


# ---------------------------------------------------------------------------
# SeisBench models on DAS (generic wrapper)
# ---------------------------------------------------------------------------
# SeisBench ships single-station phase pickers (PhaseNet, EQTransformer, ...)
# plus an *experimental* DAS API, DASWaveformModelWrapper, that applies any
# of them to DAS data channel-by-channel. ``seisbench_picker`` wraps that:
# choose any SeisBench model, run it on a dascore patch, get picks back in the
# shared PICK_COLUMNS schema.
#
# IMPORTANT scope notes:
#   * The wrapper only accepts models whose ``output_type == "array"`` (it
#     reads the dense probability curves). That is PhaseNet / EQTransformer and
#     their relatives; point-output models like GPD are NOT supported.
#   * Each channel is treated independently (no spatial coherence). This is a
#     *different, generally weaker* approach than the dedicated 2-D PhaseNet-DAS
#     model in EQNet (see ``phasenet_das_picker``); it does not replace it.
#
# Requires the optional dependencies ``seisbench`` and ``xdas`` (imported
# lazily, like ``torch`` in ``phasenet_das_picker``).

# Friendly name -> SeisBench class name. ``model=`` may also be a pre-built
# SeisBench model instance, bypassing this table.
_SEISBENCH_MODELS = {
    "eqtransformer": "EQTransformer",
    "phasenet": "PhaseNet",
    "phasenetlight": "PhaseNetLight",
    "variablelengthphasenet": "VariableLengthPhaseNet",
    "obstransformer": "OBSTransformer",
    "skynet": "Skynet",
}


def _patch_to_xdas(patch):
    """Convert a dascore patch into the ``xdas.DataArray`` the SeisBench DAS
    API expects, keeping the patch's native dim order (the wrapper accepts both
    ("time", "distance") and ("distance", "time")). The time axis gets an
    interpolated coord; distance a dense one.

    Returns (data_array, dist_vals, time_vals, fs, t0_ns)."""
    import xdas

    dist_vals = patch.coords.get_array("distance")
    time_vals = patch.coords.get_array("time")

    nt = len(time_vals)
    dt_s = float(np.median(np.diff(time_vals)) / np.timedelta64(1, "s"))
    fs = 1.0 / dt_s
    t0 = np.datetime64(pd.Timestamp(time_vals[0]).to_pydatetime())
    t1 = np.datetime64(pd.Timestamp(time_vals[-1]).to_pydatetime())
    t0_ns = time_vals[0].astype("datetime64[ns]").astype("int64")

    time_coord = {"tie_indices": [0, nt - 1], "tie_values": [t0, t1]}
    dist_coord = np.asarray(dist_vals, dtype=float)

    # Build coords in the patch's own dim order -> no transpose, no assumption
    # about whether the patch is (time, distance) or (distance, time).
    coords = {
        d: (time_coord if d == "time" else dist_coord) for d in patch.dims
    }
    data = np.ascontiguousarray(patch.data, dtype=np.float32)

    da = xdas.DataArray(data=data, coords=coords)
    return da, dist_vals, time_vals, fs, t0_ns


def _model_window_seconds(base):
    """Required input-window length (seconds) of a SeisBench model, or None if
    the model does not expose a fixed window (e.g. variable-length nets)."""
    in_samples = getattr(base, "in_samples", None)
    model_fs = getattr(base, "sampling_rate", None)
    if not in_samples or not model_fs:
        return None
    return in_samples / model_fs


def _pad_patch_time(patch, target_samples):
    """Zero-pad a dascore patch along ``time`` up to ``target_samples`` samples,
    extending the time coordinate at the native sample spacing. The original
    data sits at the start; appended samples are zeros. Returns a new patch."""
    time_vals = patch.coords.get_array("time")
    nt = len(time_vals)
    n_pad = int(target_samples) - nt
    if n_pad <= 0:
        return patch

    dt = np.median(np.diff(time_vals))
    extra_time = time_vals[-1] + dt * np.arange(1, n_pad + 1)
    new_time = np.concatenate([time_vals, extra_time])

    time_axis = patch.dims.index("time")
    pad_width = [(0, 0)] * patch.data.ndim
    pad_width[time_axis] = (0, n_pad)
    new_data = np.pad(patch.data, pad_width, mode="constant", constant_values=0)

    new_coords = {d: patch.coords.get_array(d) for d in patch.dims}
    new_coords["time"] = new_time

    # propagate non-dim coords (x/y/z geometry, any file metadata) so padding
    # drops nothing: distance-attached pass through, time-attached numeric
    # coords are NaN-padded to the new length
    dim_map = getattr(patch.coords, "dim_map", {})
    for name, dims in dim_map.items():
        if name in patch.dims:
            continue
        arr = np.asarray(patch.coords.get_array(name))
        dims = tuple(dims)
        if dims == ("distance",):
            new_coords[name] = ("distance", arr)
        elif dims == ("time",) and np.issubdtype(arr.dtype, np.number):
            new_coords[name] = (
                "time",
                np.concatenate([arr.astype(float), np.full(n_pad, np.nan)]),
            )
        else:
            warnings.warn(
                f"coord {name!r} (dims={dims}) cannot be carried through "
                f"time padding and was dropped.",
                stacklevel=2,
            )
    return patch.new(data=new_data, coords=new_coords, dims=patch.dims)


def _das_picks_to_df(picks, dist_vals, time_vals, fs, t0_ns, n_time, x_arr, y_arr, z_arr):
    """Map SeisBench ``DASPick`` objects (time=datetime64, channel=<distance
    coord value>, confidence, phase) into the shared PICK_COLUMNS schema.
    ``x_arr/y_arr/z_arr`` are the per-channel geometry arrays from
    :func:`_geom_arrays`, indexed by channel.

    ``channel`` is snapped to the nearest distance value and the pick time to
    the nearest sample index so the result drops into PICK_COLUMNS and the
    shared ``_plot_picks`` channel-equality logic keeps working."""
    n_dist = len(dist_vals)
    dist_arr = np.asarray(dist_vals, dtype=float)

    rows = []
    for p in picks:
        # snap channel coordinate -> nearest channel index
        ch = int(np.argmin(np.abs(dist_arr - float(p.channel))))
        if ch < 0 or ch >= n_dist:
            continue

        peak_ns = pd.Timestamp(np.datetime64(p.time)).value
        smp = int(round((peak_ns - t0_ns) * 1e-9 * fs))
        if smp < 0 or smp >= n_time:
            continue

        rows.append(
            {
                "distance": dist_vals[ch],
                "x": x_arr[ch],
                "y": y_arr[ch],
                "z": z_arr[ch],
                "phase": p.phase,
                "onset_sample": smp,
                "onset_time": time_vals[smp],
                "score": float(p.confidence),
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


def _resolve_seisbench_model(model, pretrained):
    """Return (base_model, model_key).

    ``model`` may be a registry key string or a pre-built SeisBench instance.
    """
    import seisbench.models as sbm

    if isinstance(model, str):
        key = model.lower()
        cls_name = _SEISBENCH_MODELS.get(key)
        if cls_name is None:
            raise ValueError(
                f"unknown SeisBench model {model!r}. Known keys: "
                f"{sorted(_SEISBENCH_MODELS)}. You may also pass a pre-built "
                f"SeisBench model instance."
            )
        base = getattr(sbm, cls_name).from_pretrained(pretrained)
    else:
        base = model
        key = getattr(base, "name", base.__class__.__name__).lower()

    output_type = getattr(base, "output_type", None)
    if output_type != "array":
        raise ValueError(
            f"model {key!r} has output_type={output_type!r}; the SeisBench DAS "
            f"wrapper only supports output_type=='array' models (e.g. "
            f"eqtransformer, phasenet). Point-output models like GPD are not "
            f"supported."
        )
    return base, key


def seisbench_picker(
    patch,
    model="eqtransformer",
    pretrained="original",
    component_strategy="clone",
    min_prob=0.3,
    min_time_separation=1.0,
    pad_short=True,
    device=None,
    plot=False,
    plot_channel=None,
    file_name=None,
    db_save=True,
    db_path=None,
    method=None,
    **classify_kwargs,
):
    """Run any SeisBench phase picker on a DAS patch via the experimental
    ``DASWaveformModelWrapper``, returning picks in the shared PICK_COLUMNS
    schema (same as ``trigger_picker`` / ``phasenet_das_picker``).

    The chosen single-station model is applied **channel by channel**: each DAS
    channel's single component is expanded to the model's component count
    (``component_strategy``), the model's per-channel probability curves are
    peak-picked by SeisBench, and the picks are mapped back to (distance, time).
    Each channel is treated independently -- this is distinct from, and
    generally weaker than, the dedicated 2-D PhaseNet-DAS model in
    ``phasenet_das_picker``.

    Parameters
    ----------
    patch : dascore Patch with "distance" and "time" coords.
    model : SeisBench model. Either a registry key (one of
        ``eqtransformer``, ``phasenet``, ``phasenetlight``,
        ``variablelengthphasenet``, ``obstransformer``, ``skynet``) or a
        pre-built SeisBench model instance. Must have ``output_type == "array"``.
    pretrained : pretrained weight set passed to ``<Model>.from_pretrained``
        when ``model`` is a key, e.g. "original", "stead", "instance". Ignored
        when ``model`` is already an instance.
    component_strategy : how to turn the single DAS component into the model's
        multi-component input. "clone" (replicate to every component) or "pad"
        (first component is the data, the rest zeros).
    min_prob : confidence threshold for picking (SeisBench ``thresholds``).
    min_time_separation : minimum spacing (s) between two same-phase picks on a
        channel (SeisBench ``min_time_separation``).
    pad_short : if the patch is shorter than the model's fixed input window
        (e.g. EQTransformer needs 60 s = 6000 samples @ 100 Hz), zero-pad it in
        time up to one full window so the model can run. If ``False``, the patch
        is left as-is and a warning is emitted (SeisBench will return no picks
        when there is not a single complete window). Picks landing in the padded
        region are unreliable; prefer feeding a patch that is already long
        enough.
    device : "cuda" / "cpu" / "mps"; auto-detected (cuda if available else cpu).
    plot, plot_channel : diagnostic plot via the shared ``_plot_picks``.
    file_name, db_save, db_path : store persistence. When ``db_save`` is
        True (default) the picks are saved, replacing previous rows for
        (file_name, method).
    method : store method label. Defaults to ``"<model_key>_sb"`` (e.g.
        "eqtransformer_sb", "phasenet_sb"); the ``_sb`` suffix keeps SeisBench
        results separate from the 2-D PhaseNet-DAS picks ("phasenetdas").
    **classify_kwargs : forwarded to the wrapper's ``classify`` (e.g.
        ``overlap_samples``, ``blinding``).

    Returns
    -------
    pandas.DataFrame with columns == PICK_COLUMNS. Picks populate ``score``
    (confidence); cft_* and off_* columns are NaN.
    """
    import asyncio
    from seisbench.models import DASWaveformModelWrapper, DASPickingCallback

    if device is None:
        device = _auto_device()

    base, model_key = _resolve_seisbench_model(model, pretrained)
    if method is None:
        method = f"{model_key}_sb"

    runner = DASWaveformModelWrapper(base, component_strategy=component_strategy)
    runner.to(device)
    runner.eval()

    # capture geometry up front (padding now preserves coords, but this keeps
    # the pick rows independent of any later patch rebuilding)
    x_arr, y_arr, z_arr = _geom_arrays(patch)

    window_s = _model_window_seconds(base)
    if window_s is not None:
        t_vals = patch.coords.get_array("time")
        nt = len(t_vals)
        patch_fs = 1.0 / (np.median(np.diff(t_vals)) / np.timedelta64(1, "s"))
        patch_s = nt / patch_fs
        if patch_s < window_s:
            need_samples = int(np.ceil(window_s * patch_fs))
            if pad_short:
                warnings.warn(
                    f"{model_key}: patch is {patch_s:.2f}s but the model needs a "
                    f"{window_s:.2f}s window; zero-padding to {window_s:.2f}s. "
                    f"Picks in the padded region are unreliable.",
                    stacklevel=2,
                )
                patch = _pad_patch_time(patch, need_samples)
            else:
                warnings.warn(
                    f"{model_key}: patch is {patch_s:.2f}s, shorter than the "
                    f"model's {window_s:.2f}s window; no complete window exists "
                    f"so no picks will be returned. Pass a longer patch or "
                    f"pad_short=True.",
                    stacklevel=2,
                    )

    da, dist_vals, time_vals, fs, t0_ns = _patch_to_xdas(patch)
    n_time = len(time_vals)

    callback = DASPickingCallback(
        thresholds=min_prob,
        min_time_separation=min_time_separation,
    )
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, runner.annotate_async(da, callback, **classify_kwargs)).result()
    except RuntimeError:
        asyncio.run(runner.annotate_async(da, callback, **classify_kwargs))
    results_dict = callback.get_results_dict()

    all_keys = getattr(runner, "annotate_keys", None) or ["P", "S"]
    # "Detection" is EQTransformer's event-presence head, not a phase arrival;
    # exclude it so only phase picks reach the shared schema.
    phase_keys = [k for k in all_keys if k.lower() != "detection"]
    picks = []
    for key in phase_keys:
        plist = results_dict.get(key)
        if plist:
            picks.extend(list(plist))

    df = _das_picks_to_df(
        picks, dist_vals, time_vals, fs, t0_ns, n_time, x_arr, y_arr, z_arr
    )

    if plot:
        n_ch = len(dist_vals)
        ch_idx = plot_channel if plot_channel is not None else n_ch // 2
        _plot_picks(patch, df, ch_idx, None, None, None)

    if db_save:
        _save_to_store(df, file_name, method, db_path=db_path)

    return df


def eqtransformer_picker(patch, pretrained="original", **kwargs):
    """EQTransformer P/S picks on a DAS patch. Thin wrapper over
    :func:`seisbench_picker` with ``model="eqtransformer"`` (store method
    "eqtransformer_sb"). See ``seisbench_picker`` for all keyword
    arguments."""
    return seisbench_picker(
        patch, model="eqtransformer", pretrained=pretrained, **kwargs
    )
