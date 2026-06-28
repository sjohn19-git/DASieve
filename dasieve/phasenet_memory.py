"""
In-memory PhaseNet-DAS inference on a dascore patch.

The patch data is never written to disk; picks are returned directly as a
DataFrame.  The model is loaded once per (eqnet_dir, device, location, phases)
and kept alive for the lifetime of the process.
"""

import os
import sys
import numpy as np
import pandas as pd

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
    """Convert a dascore patch to a (1, nt, nx) tensor using the same
    preprocessing as DASIterableDataset.sample() for h5 files."""
    import torch
    import torch.nn.functional as F
    import scipy.signal

    dist_vals = patch.coords.get_array("distance")
    time_vals = patch.coords.get_array("time")

    dist_axis = patch.dims.index("distance")
    time_axis = patch.dims.index("time")
    data = np.moveaxis(patch.data, [dist_axis, time_axis], [0, 1])  # (nx, nt)
    data = np.ascontiguousarray(data, dtype=np.float32)

    dt_s = float(np.median(np.diff(time_vals)) / np.timedelta64(1, "s"))
    begin_time = pd.Timestamp(time_vals[0]).to_pydatetime().isoformat(timespec="milliseconds")

    data -= np.mean(data, axis=-1, keepdims=True)
    data -= np.median(data, axis=-2, keepdims=True)
    if highpass_filter > 0.0:
        b, a = scipy.signal.butter(2, highpass_filter, "hp", fs=1.0 / dt_s)
        data = scipy.signal.filtfilt(b, a, data, axis=-1)

    data = data.T[np.newaxis, :, :]  # (1, nt, nx)
    tensor = torch.from_numpy(data)

    nt, nx = tensor.shape[1], tensor.shape[2]
    pad_nt = (1024 - nt % 1024) % 1024
    pad_nx = (1024 - nx % 1024) % 1024
    with torch.no_grad():
        tensor = F.pad(tensor, (0, pad_nx, 0, pad_nt), mode="constant")

    return tensor, nt, nx, dt_s, begin_time, dist_vals, time_vals


def predict(
    patch,
    eqnet_dir=DEFAULT_EQNET_DIR,
    min_prob=0.3,
    device=None,
    phases=("P", "S"),
    highpass_filter=0.0,
    location=None,
):
    """PhaseNet-DAS picks on a dascore patch, entirely in memory.

    The model is loaded once and cached; subsequent calls on the same process
    reuse the cached weights.

    Parameters
    ----------
    patch           : dascore Patch with "distance" and "time" coords.
    eqnet_dir       : path to the EQNet repo directory.
    min_prob        : minimum phase probability threshold.
    device          : "cuda" / "mps" / "cpu"; auto-detected if None.
    phases          : phase labels the model outputs, default ("P", "S").
    highpass_filter : highpass corner in Hz; 0.0 = no filter.
    location        : pretrained model variant; None for default, "forge" for FORGE.

    Returns
    -------
    pandas.DataFrame with columns matching dasieve.picker.PICK_COLUMNS.
    """
    import torch
    from contextlib import nullcontext

    phases = list(phases)
    eqnet_dir = _ensure_eqnet_on_path(eqnet_dir)

    if device is None:
        from dasieve.picker import _auto_device
        device = _auto_device()

    model = _load_model(eqnet_dir, device, location, phases)

    data, nt, nx, dt_s, begin_time, dist_vals, time_vals = _preprocess(
        patch, highpass_filter
    )

    # (1, 1, nt_pad, nx_pad) — batch dim required by the model
    data_batched = data.unsqueeze(0).to(device)

    # Only CUDA supports autocast; treat MPS like CPU
    if device in ("cpu", "mps"):
        ctx = nullcontext()
    else:
        dtype_str = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
        ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)

    with torch.inference_mode(), ctx:
        output = model({"data": data_batched})

    # Trim padding from the phase scores: (1, nphases+1, nt, nx)
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

    # picks_[0]: list of dicts with station_id, phase_index, phase_score, phase_type
    from dasieve.picker import _phasenet_picks_to_df

    frames = []
    if picks_[0]:
        df_raw = pd.DataFrame(picks_[0])
        df_raw["channel_index"] = df_raw["station_id"].apply(lambda x: int(x))
        frames.append(df_raw)

    return _phasenet_picks_to_df(frames, dist_vals, time_vals, len(time_vals))
