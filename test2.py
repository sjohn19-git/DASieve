import os
import numpy as np
import dascore as dc
import matplotlib.pyplot as plt
import scipy.signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dascore.units import Hz, m, s
from dascore.utils.patch import get_dim_sampling_rate
from obspy import read
import matplotlib.pyplot as plt

# ── select quantity ──────────────────────────────────────────────────────────
# Options: "strain_rate" | "velocity" | "acceleration"
MODE = "velocity"
# ────────────────────────────────────────────────────────────────────────────

_LABELS = {
    "strain_rate": (
        "Strain rate (m/m/s)",
        "dB re 1 (m/m/s)²/Hz",
        "DAS waterfall — strain rate",
    ),
    "velocity": ("Velocity (m/s)", "dB re 1 (m/s)²/Hz", "DAS waterfall — velocity"),
    "acceleration": (
        "Acceleration (m/s²)",
        "dB re 1 (m/s²)²/Hz",
        "DAS waterfall — acceleration",
    ),
}


def plot_patch(
    patch: dc.Patch,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdBu_r",
    ax: plt.Axes | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    data2d = patch.data
    if patch.dims[0] != "channel":
        data2d = data2d.T

    time_arr = patch.coords.get_array("time")
    dist_arr = patch.coords.get_array("distance")
    t0 = time_arr[0]
    time_s = (time_arr - t0) / np.timedelta64(1, "s")

    if vmin is None and vmax is None:
        clim = float(np.nanpercentile(np.abs(data2d), 98))
        vmin, vmax = -clim, clim

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    else:
        fig = ax.get_figure()

    im = ax.imshow(
        data2d,
        aspect="auto",
        origin="lower",
        extent=[time_s[0], time_s[-1], float(dist_arr[0]), float(dist_arr[-1])],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="none",
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.ax.tick_params(labelsize=9)
    ax.set_xlabel(f"Time relative to {str(t0)[:19]} UTC (s)", fontsize=10)
    ax.set_ylabel("Distance (m)", fontsize=10)

    if show:
        plt.show()
    return fig, ax


def _normalize_patch(patch: dc.Patch) -> dc.Patch:
    fs_hz = get_dim_sampling_rate(patch, "time") * Hz
    gl = patch.attrs.gauge_length * m
    scale_factor = 1 / 8192
    norm = scale_factor * (116 * fs_hz) / gl
    return (patch * norm * 1e-9).set_units("m/m/s")


file_path = os.path.expanduser(
    "~/Downloads/16BConst_Stimulation_UTC_20240407_072054.163.h5"
)

patch_sr = dc.spool(file_path)[0]


patch_sr = patch_sr.select(distance=(1500, 2000))
patch_sr = _normalize_patch(patch_sr)

fig, ax = plot_patch(patch_sr)

if MODE == "strain_rate":
    p = patch_sr
elif MODE == "velocity":
    p = patch_sr.integrate(dim="distance").set_units(m / s)
elif MODE == "acceleration":
    patch_vel = patch_sr.integrate(dim="distance").set_units(m / s)
    p = patch_vel.differentiate(dim="time")
else:
    raise ValueError(
        f"Unknown MODE {MODE!r}. Choose 'strain_rate', 'velocity', or 'acceleration'."
    )

ylabel, psd_ylabel, title = _LABELS[MODE]

# --- data extraction ---
time_arr = p.coords.get_array("time")
dist_arr = p.coords.get_array("distance")
t0 = time_arr[0]
time_s = (time_arr - t0) / np.timedelta64(1, "s")

data2d = p.data
if p.dims[0] != "channel":
    data2d = data2d.T

clim = float(np.nanpercentile(np.abs(data2d), 98))
VMIN, VMAX = -clim, clim

# random channel
rng = np.random.default_rng()
mid_idx = int(rng.integers(0, len(dist_arr)))
mid_dist = float(dist_arr[mid_idx])
waveform = data2d[mid_idx, :]

dt_s = float(p.coords.step("time") / np.timedelta64(1, "s"))
fs = 1.0 / dt_s
nperseg = int(2 ** np.round(np.log2(fs)))
freqs_spec, t_spec, Sxx = scipy.signal.spectrogram(
    waveform,
    fs=fs,
    window="hann",
    nperseg=512,
    noverlap=nperseg // 4,
    detrend="linear",
    scaling="density",
)
t_spec = t_spec + time_s[0]

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

# --- top: waterfall ---
im = ax_im.imshow(
    data2d,
    aspect="auto",
    origin="lower",
    extent=[time_s[0], time_s[-1], dist_arr[0], dist_arr[-1]],
    vmin=VMIN,
    vmax=VMAX,
    cmap="RdBu_r",
    interpolation="none",
)
ax_im.axhline(
    mid_dist, color="red", linewidth=1.0, linestyle="--", label=f"{mid_dist:.0f} m"
)
ax_im.legend(loc="upper right", fontsize=9, framealpha=0.7)
div_im = make_axes_locatable(ax_im)
cax_im = div_im.append_axes("right", size="2%", pad=0.05)
cb = fig.colorbar(im, cax=cax_im)
cb.set_label(ylabel, fontsize=10)
cb.ax.tick_params(labelsize=9)
ax_im.set_ylabel("Distance (m)", fontsize=10)
ax_im.set_title(title, fontsize=11, fontweight="bold", loc="left")
plt.setp(ax_im.get_xticklabels(), visible=False)

# --- middle: waveform ---
ax_wf.plot(time_s, waveform, color="#222222", linewidth=0.7)
ax_wf.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax_wf.set_ylabel(ylabel, fontsize=10)
ax_wf.set_title(f"Channel at {mid_dist:.0f} m", fontsize=10, loc="left")
div_wf = make_axes_locatable(ax_wf)
cax_wf = div_wf.append_axes("right", size="2%", pad=0.05)
cax_wf.set_visible(False)
plt.setp(ax_wf.get_xticklabels(), visible=False)

# --- bottom: spectrogram ---
Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-30))
sg = ax_psd.pcolormesh(
    t_spec, freqs_spec, Sxx_db, shading="gouraud", cmap="nipy_spectral"
)
div_psd = make_axes_locatable(ax_psd)
cax_psd = div_psd.append_axes("right", size="2%", pad=0.05)
cb_sg = fig.colorbar(sg, cax=cax_psd)
cb_sg.set_label(psd_ylabel, fontsize=9)
cb_sg.ax.tick_params(labelsize=8)
ax_psd.set_xlabel(f"Time relative to {str(t0)[:19]} UTC (s)", fontsize=10)
ax_psd.set_ylabel("Frequency (Hz)", fontsize=10)
ax_psd.set_title(f"Spectrogram — channel at {mid_dist:.0f} m", fontsize=10, loc="left")
ax_psd.set_ylim(0, fs / 2)

fig.tight_layout()
plt.show()
