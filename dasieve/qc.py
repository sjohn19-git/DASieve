"""
qc.py

PSD QC product for DASieve.

Computes per-channel Welch PSD from a DASCore patch and appends it to a
persistent pandas pickle store. One row per (time, channel) pair.

Pipeline position: runs on the raw patch (before resample/decimate),
corresponding to the "PSD QC Product — always runs" branch in the architecture.

Store layout (psd_qc.pkl) — pandas DataFrame:
    time   : pd.Timestamp   UTC start time of the processed file
    ch     : int            channel index
    freqs  : np.ndarray     frequency vector in Hz, shape (n_freq,)
    psds   : np.ndarray     PSD in dB, shape (n_freq,)

Authors: Sebin John, 2025
"""

import logging
from pathlib import Path
from dascore.utils.patch import get_dim_sampling_rate
import numpy as np
import pandas as pd
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
        2. Welch (Hann window, 50% overlap, nperseg ~ fs/0.5 rounded to power of 2)
        3. Convert to dB: 10 * log10(Pxx)

    Parameters
    ----------
    patch : dc.Patch
        Raw DASCore patch (time × channel layout).

    Returns
    -------
    frequencies : np.ndarray, shape (n_freq,)
        Frequency vector in Hz.
    psd_db : np.ndarray, shape (n_freq, n_channel), float32
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
    Append one PSD time step to the pandas pickle store.

    Creates the store on first call. Subsequent calls extend the DataFrame.
    Thread-unsafe — designed for single-writer pipeline use.

    Parameters
    ----------
    store_path : str
        Path to the pickle file (created if it does not exist).
    timestamp : np.datetime64
        UTC start time of the processed file.
    frequencies : np.ndarray, shape (n_freq,)
        Frequency vector in Hz.
    psd_db : np.ndarray, shape (n_freq, n_channel), float32
        PSD in dB to append.
    """
    store_path = str(store_path)
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)

    n_freq, n_ch = psd_db.shape
    t = pd.Timestamp(timestamp)

    rows = pd.DataFrame(
        {
            "time": t,
            "ch": np.arange(n_ch),
            "freqs": [frequencies] * n_ch,
            "psds": [psd_db[:, i] for i in range(n_ch)],
        }
    )

    if Path(store_path).exists():
        existing = pd.read_pickle(store_path)
        dup = existing["time"] == t
        if dup.any():
            log.warning(f"Timestamp {timestamp} already in store — overwriting")
            existing = existing[~dup]
        df = pd.concat([existing, rows], ignore_index=True)
    else:
        df = rows
        log.info(
            f"Created PSD store: {store_path}  (n_channels={n_ch}, n_freq={n_freq})"
        )

    df.to_pickle(store_path)
    log.debug(f"Appended t={timestamp} to {store_path}")
    _trim_store(store_path, max_bytes=85 * 1024**3)


# ---------------------------------------------------------------------------
# Store trimming
# ---------------------------------------------------------------------------


def _trim_store(store_path: str, max_bytes: int = 85 * 1024**3) -> None:
    """
    If the pickle store exceeds max_bytes, drop the oldest timestamps until it fits.

    Parameters
    ----------
    store_path : str
        Path to the pickle store.
    max_bytes : int
        Maximum allowed file size in bytes. Default 85 GB.
    """
    import os

    size = os.path.getsize(store_path)
    if size <= max_bytes:
        return

    log.warning(
        f"Store size {size / 1024**3:.1f} GB exceeds limit "
        f"{max_bytes / 1024**3:.0f} GB — trimming oldest timestamps"
    )

    df = pd.read_pickle(store_path)
    unique_times = sorted(df["time"].unique())
    n_times = len(unique_times)

    fraction_to_keep = (max_bytes * 0.8) / size
    keep_n = max(1, int(n_times * fraction_to_keep))
    keep_times = set(unique_times[-keep_n:])

    dropped = n_times - keep_n
    df = df[df["time"].isin(keep_times)].reset_index(drop=True)
    df.to_pickle(store_path)

    new_size = os.path.getsize(store_path)
    log.info(
        f"Dropped {dropped} timestamps — store trimmed to {new_size / 1024**3:.1f} GB"
    )


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
    t_start=None,
    t_end=None,
    p_min: float | None = None,
    p_max: float | None = None,
    fmin: float = 1.0,
    fmax: float = 250.0,
    bins: tuple = (300, 200),
    vmin: float | None = None,
    vmax: float | None = None,
    ylim: tuple | None = None,
    channel: int | None = None,
) -> None:
    """
    Two-row plot: 2-D PSD-PDF across all channels (top) and median PSD line
    for one channel (bottom).

    Parameters
    ----------
    ylim : tuple (ymin, ymax), optional
        Explicit y-axis limits in dB applied to both panels.
    channel : int, optional
        Channel index for the line plot. Defaults to the mid channel.
    """
    df = pd.read_pickle(store_path)

    if t_start is not None:
        df = df[df["time"] >= pd.Timestamp(t_start)]
    if t_end is not None:
        df = df[df["time"] <= pd.Timestamp(t_end)]

    freqs = df["freqs"].iloc[0]
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_masked = freqs[freq_mask]

    psd_matrix = np.vstack(df["psds"].to_numpy())  # (n_rows, n_freq)
    psd_masked = psd_matrix[:, freq_mask]  # (n_rows, n_masked_freq)

    all_f = np.tile(freqs_masked, psd_masked.shape[0])
    all_p = psd_masked.ravel()

    if p_min is None:
        p_min = float(np.percentile(all_p, 1))
    if p_max is None:
        p_max = float(np.percentile(all_p, 99))
    log.info("PSD range: p_min=%.1f dB, p_max=%.1f dB", p_min, p_max)

    H, f_edges, p_edges = np.histogram2d(
        all_f, all_p, bins=bins, range=[[fmin, fmax], [p_min, p_max]]
    )
    H /= H.max()

    n_channels = df["ch"].nunique()
    ch = n_channels // 2 if channel is None else channel

    ch_rows = df[df["ch"] == ch]
    psd_ch = np.vstack(ch_rows["psds"].to_numpy())[:, freq_mask]
    median_psd = np.median(psd_ch, axis=0)

    fig, (ax_pdf, ax_line) = plt.subplots(
        2, 1, figsize=(7.5, 7), dpi=300, gridspec_kw={"hspace": 0.4}
    )

    # --- top: 2-D PDF ---
    pcm = ax_pdf.pcolormesh(
        f_edges,
        p_edges,
        H.T,
        cmap="jet",
        shading="auto",
        vmin=0 if vmin is None else vmin,
        vmax=1.0 if vmax is None else vmax,
        rasterized=True,
    )
    ax_pdf.set_xscale("log")
    ax_pdf.set_xlim(fmin, fmax)
    if ylim is not None:
        ax_pdf.set_ylim(ylim)
    ax_pdf.set_xlabel("Frequency (Hz)")
    ax_pdf.set_ylabel("Power (dB)")
    ax_pdf.set_title(f"PSD–PDF  |  all {n_channels} channels")
    ax_pdf.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
    ax_pdf.tick_params(axis="x", which="minor", length=3)
    ax_pdf.tick_params(axis="x", which="major", length=6)
    fig.colorbar(pcm, ax=ax_pdf, pad=0.02, fraction=0.035).set_label("Probability")

    # --- bottom: single-channel median PSD ---
    ax_line.plot(freqs_masked, median_psd, color="#1f77b4", linewidth=1.0)
    ax_line.set_xscale("log")
    ax_line.set_xlim(fmin, fmax)
    if ylim is not None:
        ax_line.set_ylim(ylim)
    ax_line.set_xlabel("Frequency (Hz)")
    ax_line.set_ylabel("Power (dB)")
    ax_line.set_title(f"Median PSD  |  channel {ch}")
    ax_line.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
    ax_line.tick_params(axis="x", which="minor", length=3)
    ax_line.tick_params(axis="x", which="major", length=6)
    ax_line.grid(True, which="both", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()
    log.info("Saved: %s", out_path)
