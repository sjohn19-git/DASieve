"""
qc.py

PSD QC product for DASieve.

Computes per-channel Welch PSD from a DASCore patch and appends it to a
persistent HDF5 store. One row per 30s file, shape (time, channel, freq).

Pipeline position: runs on the raw patch (before resample/decimate),
corresponding to the "PSD QC Product — always runs" branch in the architecture.

Store layout (psd_qc.h5):
    /psd          (n_time, n_channel, n_freq)  float32   gzip compressed
    /timestamps   (n_time,)                    int64     nanoseconds since epoch
    /frequencies  (n_freq,)                    float64   Hz

Authors: Sebin John, 2025
"""

import logging
from pathlib import Path
from dascore.utils.patch import get_dim_sampling_rate
import numpy as np
import h5py
import scipy.signal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dascore as dc


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PSD computation
# ---------------------------------------------------------------------------


def compute_psd(patch: dc.Patch) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel Welch PSD from a DASCore patch.

    Processing chain per channel:
        1. Linear detrend
        2. 5% Tukey taper
        3. Welch (Hann window, 50% overlap, nperseg ~ fs/0.5 rounded to power of 2)
        4. Convert to dB: 10 * log10(Pxx)

    Parameters
    ----------
    patch : dc.Patch
        Raw DASCore patch (distance × time layout).

    Returns
    -------
    frequencies : np.ndarray, shape (n_freq,)
        Frequency vector in Hz.
    psd_db : np.ndarray, shape (n_channel, n_freq), float32
        PSD in dB for each channel.
    """
    fs = get_dim_sampling_rate(patch, "time")
    data = patch.data  # (n_time, n_channel)
    n_t, n_ch = data.shape

    # nperseg: nearest power of 2 to fs/0.5
    nperseg = int(2 ** np.round(np.log2(fs * 2)))
    noverlap = nperseg // 2

    log.info(
        f"Welch: fs={fs:.1f} Hz, nperseg={nperseg}, Δf={fs / nperseg:.3f} Hz, "
        f"n_channels={n_ch}, n_samples={n_t}"
    )

    data = scipy.signal.detrend(np.array(data, dtype=np.float32), type="linear", axis=0)

    freqs, Pxx = scipy.signal.welch(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend=False,
        axis=0,
    )
    psd_db = (10.0 * np.log10(np.maximum(Pxx, 1e-30))).astype(np.float32)
    return freqs, psd_db


# ---------------------------------------------------------------------------
# HDF5 store
# ---------------------------------------------------------------------------


def append_to_store(
    store_path: str,
    timestamp: np.datetime64,
    frequencies: np.ndarray,
    psd_db: np.ndarray,
) -> None:
    """
    Append one PSD time step to the HDF5 store.

    Creates the store and datasets on first call. Subsequent calls extend
    the time axis. Thread-unsafe — designed for single-writer pipeline use.

    Parameters
    ----------
    store_path : str
        Path to the HDF5 file (created if it does not exist).
    timestamp : np.datetime64
        UTC start time of the processed file.
    frequencies : np.ndarray, shape (n_freq,)
        Frequency vector in Hz (written once, verified on subsequent calls).
    psd_db : np.ndarray, shape (n_channel, n_freq), float32
        PSD in dB to append.
    """
    store_path = str(store_path)
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)

    n_freq, n_ch = psd_db.shape
    ts_ns = np.datetime64(timestamp, "ns").view(np.int64)

    with h5py.File(store_path, "a") as f:
        # --- First write: create datasets ---
        if "psd" not in f:
            f.create_dataset(
                "psd",
                data=psd_db[np.newaxis, :, :],  # (1, n_freq, n_ch)
                maxshape=(None, n_freq, n_ch),
                chunks=(120, n_freq, n_ch),
                dtype=np.float32,
            )
            f.create_dataset(
                "timestamps",
                data=np.array([ts_ns], dtype=np.int64),
                maxshape=(None,),
                chunks=(120,),
            )
            f.create_dataset(
                "frequencies",
                data=frequencies.astype(np.float64),
            )
            log.info(
                f"Created PSD store: {store_path}  (n_channels={n_ch}, n_freq={n_freq})"
            )

        # --- Subsequent writes: extend along time axis ---
        else:
            # Sanity check frequencies match
            stored_freq = f["frequencies"][:]
            if not np.allclose(stored_freq, frequencies, rtol=1e-4):
                raise ValueError(
                    "Frequency vector mismatch — store was created with different fs or nperseg"
                )

            existing = np.where(f["timestamps"][:] == ts_ns)[0]
            if existing.size:
                idx = existing[0]
                log.warning(
                    f"Timestamp {timestamp} already in store — overwriting row {idx}"
                )
                f["psd"][idx, :, :] = psd_db
                return

            t = f["psd"].shape[0]
            f["psd"].resize(t + 1, axis=0)
            f["psd"][t, :, :] = psd_db

            f["timestamps"].resize(t + 1, axis=0)
            f["timestamps"][t] = ts_ns

    log.debug(f"Appended t={timestamp} to {store_path}")
    _trim_store(store_path, max_bytes=85 * 1024**3)


# ---------------------------------------------------------------------------
# Store trimming
# ---------------------------------------------------------------------------


def _trim_store(store_path: str, max_bytes: int = 85 * 1024**3) -> None:
    """
    If the HDF5 store exceeds max_bytes, drop the oldest rows until it fits.

    HDF5 cannot reclaim space in-place, so this rewrites the file to a
    temporary path then replaces the original. Drops rows in chunks of 120
    (1 hour) to avoid trimming one row at a time.

    Parameters
    ----------
    store_path : str
        Path to the HDF5 store.
    max_bytes : int
        Maximum allowed file size in bytes. Default 250 GB.
    """
    import os
    import tempfile

    size = os.path.getsize(store_path)
    if size <= max_bytes:
        return

    log.warning(
        f"Store size {size / 1024**3:.1f} GB exceeds limit "
        f"{max_bytes / 1024**3:.0f} GB — trimming oldest rows"
    )

    with h5py.File(store_path, "r") as f:
        n_time = f["psd"].shape[0]
        n_ch = f["psd"].shape[1]
        n_freq = f["psd"].shape[2]
        freqs = f["frequencies"][:]

        # Estimate bytes per row and how many rows to drop to reach 80% of limit
        bytes_per_row = (n_ch * n_freq * 4) + 8  # float32 psd + int64 timestamp
        rows_to_drop = int(np.ceil((size - max_bytes * 0.8) / bytes_per_row))
        # Drop in multiples of 120 (chunk size) to keep chunk alignment
        rows_to_drop = int(np.ceil(rows_to_drop / 120) * 120)
        rows_to_drop = min(rows_to_drop, n_time - 1)  # always keep at least 1 row

        keep_from = rows_to_drop
        psd_keep = f["psd"][keep_from:, :, :]
        ts_keep = f["timestamps"][keep_from:]

    log.info(f"Dropping {rows_to_drop} oldest rows, keeping {n_time - rows_to_drop}")

    # Write surviving data to a temp file then atomically replace
    tmp_path = store_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset(
            "psd",
            data=psd_keep,
            maxshape=(None, n_ch, n_freq),
            chunks=(120, n_ch, n_freq),
            dtype=np.float32,
        )
        f.create_dataset(
            "timestamps",
            data=ts_keep,
            maxshape=(None,),
            chunks=(120,),
        )
        f.create_dataset("frequencies", data=freqs)

    import os

    os.replace(tmp_path, store_path)
    new_size = os.path.getsize(store_path)
    log.info(f"Store trimmed: {new_size / 1024**3:.1f} GB")


# ---------------------------------------------------------------------------
# Three-panel diagnostic plot
# ---------------------------------------------------------------------------


def plot_patch(
    patch: dc.Patch,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    channel_idx: int | None = None,
    space_dim: str = "distance",
    show: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Three-panel plot: waterfall | single-channel waveform | spectrogram.

    Assumes patch.data shape is (n_time, n_ch) — dim 0 is time.
    Units label is fixed to strain rate (m/m/s).
    """
    data = np.asarray(patch.data, dtype=np.float32)  # (n_time, n_ch)
    time_arr = patch.coords.get_array("time")
    dist_arr = patch.coords.get_array(space_dim)
    t0 = time_arr[0]
    time_s = (time_arr - t0) / np.timedelta64(1, "s")

    if vmin is None and vmax is None:
        clim = float(np.nanpercentile(np.abs(data), 98))
        vmin, vmax = -clim, clim

    if channel_idx is None:
        channel_idx = data.shape[1] // 2
    mid_dist = float(dist_arr[channel_idx])
    waveform = data[:, channel_idx]

    fs = float(get_dim_sampling_rate(patch, "time"))
    nperseg = int(2 ** np.round(np.log2(fs)))
    freqs_spec, t_spec, Sxx = scipy.signal.spectrogram(
        waveform,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 4,
        detrend="linear",
        scaling="density",
    )
    t_spec = t_spec + float(time_s[0])
    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-30))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )

    fig, (ax_im, ax_wf, ax_psd) = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.15},
    )

    # panel 1: waterfall — transpose to (n_ch, n_time) for imshow rows=space
    im = ax_im.imshow(
        data.T,
        aspect="auto",
        origin="lower",
        extent=[
            float(time_s[0]),
            float(time_s[-1]),
            float(dist_arr[0]),
            float(dist_arr[-1]),
        ],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="none",
    )
    ax_im.axhline(
        mid_dist, color="red", linewidth=1.0, linestyle="--", label=f"{mid_dist:.0f} m"
    )
    ax_im.legend(loc="upper right", fontsize=9, framealpha=0.7)
    div_im = make_axes_locatable(ax_im)
    cax_im = div_im.append_axes("right", size="2%", pad=0.05)
    cb = fig.colorbar(im, cax=cax_im)
    cb.set_label("Strain rate (m/m/s)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    ax_im.set_ylabel("Distance (m)", fontsize=10)
    ax_im.set_title(
        "DAS waterfall — strain rate", fontsize=11, fontweight="bold", loc="left"
    )
    plt.setp(ax_im.get_xticklabels(), visible=False)

    # panel 2: single-channel waveform
    ax_wf.plot(time_s, waveform, color="#222222", linewidth=0.7)
    ax_wf.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_wf.set_ylabel("Strain rate (m/m/s)", fontsize=10)
    ax_wf.set_title(f"Channel at {mid_dist:.0f} m", fontsize=10, loc="left")
    div_wf = make_axes_locatable(ax_wf)
    cax_wf = div_wf.append_axes("right", size="2%", pad=0.05)
    cax_wf.set_visible(False)
    plt.setp(ax_wf.get_xticklabels(), visible=False)

    # panel 3: spectrogram
    sg = ax_psd.pcolormesh(
        t_spec, freqs_spec, Sxx_db, shading="gouraud", cmap="nipy_spectral"
    )
    div_psd = make_axes_locatable(ax_psd)
    cax_psd = div_psd.append_axes("right", size="2%", pad=0.05)
    cb_sg = fig.colorbar(sg, cax=cax_psd)
    cb_sg.set_label("dB re 1 (m/m/s)²/Hz", fontsize=9)
    cb_sg.ax.tick_params(labelsize=8)
    ax_psd.set_xlabel(f"Time relative to {str(t0)[:19]} UTC (s)", fontsize=10)
    ax_psd.set_ylabel("Frequency (Hz)", fontsize=10)
    ax_psd.set_title(
        f"Spectrogram — channel at {mid_dist:.0f} m", fontsize=10, loc="left"
    )
    ax_psd.set_ylim(0, fs / 2)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax_im, ax_wf, ax_psd)


# ---------------------------------------------------------------------------
# PDF plot
# ---------------------------------------------------------------------------


def plot_pdf(
    store_path: str,
    out_path: str,
    channel: int | None = None,
    t_start=None,
    t_end=None,
    p_min: float = -20,
    p_max: float = 40,
    fmin: float = 1.0,
    fmax: float = 250.0,
    bins: tuple = (300, 200),
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot a 2-D PSD probability density function from the QC store."""
    with h5py.File(store_path, "r") as f:
        freqs = f["frequencies"][:]
        ts_ns = f["timestamps"][:]
        n_channels = f["psd"].shape[2]

        ch = n_channels // 2 if channel is None else channel

        mask = np.ones(len(ts_ns), dtype=bool)
        if t_start is not None:
            mask &= ts_ns >= np.datetime64(t_start, "ns").view(np.int64)
        if t_end is not None:
            mask &= ts_ns <= np.datetime64(t_end, "ns").view(np.int64)

        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[freq_mask]
        psd_ch = f["psd"][:, :, ch][mask][:, freq_mask]

    all_f = np.tile(freqs, psd_ch.shape[0])
    all_p = psd_ch.ravel()
    H, f_edges, p_edges = np.histogram2d(
        all_f, all_p, bins=bins, range=[[fmin, fmax], [p_min, p_max]]
    )
    H /= H.max()

    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=300)
    pcm = ax.pcolormesh(
        f_edges,
        p_edges,
        H.T,
        cmap="jet",
        shading="auto",
        vmin=0 if vmin is None else vmin,
        vmax=H.max() if vmax is None else vmax,
        rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(f"PSD–PDF  |  channel {ch}")
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
    ax.tick_params(axis="x", which="minor", length=3)
    ax.tick_params(axis="x", which="major", length=6)
    fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.035).set_label("Probability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()
    log.info("Saved: %s", out_path)
