import logging

import numpy as np
import matplotlib.pyplot as plt
import dascore as dc

log = logging.getLogger(__name__)


def wavelet_denoise(
    patch: dc.Patch,
    method: str = "BayesShrink",
    wavelet: str = "db1",
    mode: str = "soft",
    sigma: float | None = None,
    sigma_scale: float = 1.0,
    wavelet_levels: int | None = None,
    plot: bool = False,
    plot_channel: int | None = None,
    cmap: str = "RdBu_r",
) -> dc.Patch:
    """Denoise a DAS patch with 2D wavelet thresholding on the waterfall.

    The (distance × time) array is treated as a single 2D image and denoised
    with either of two threshold-selection rules:

    - "BayesShrink": adaptive, one threshold per wavelet subband. Gentler,
      preserves more signal; ``sigma`` is estimated internally per subband.
    - "VisuShrink": a single universal threshold sigma*sqrt(2*ln(n)). Removes
      noise aggressively and can over-smooth; scale it down with
      ``sigma_scale`` (e.g. 0.5 or 0.25) to retain more signal.

    Args:
        patch: DAS patch. Dim order (time, distance) or (distance, time) both supported.
        method: "BayesShrink" or "VisuShrink".
        wavelet: Wavelet family used for the decomposition.
        mode: Thresholding type, "soft" or "hard".
        sigma: Noise standard deviation. If None it is estimated from the
            data's finest wavelet coefficients (only used by VisuShrink;
            BayesShrink always estimates per subband unless sigma is given).
        sigma_scale: Multiplier applied to the (estimated or given) sigma for
            VisuShrink, mirroring the sigma/2, sigma/4 variants.
        wavelet_levels: Number of decomposition levels (None = automatic).
        plot: If True, show original / denoised / removed-noise waterfalls
            with a second row overlaying one channel's before/after trace.
        plot_channel: Channel index for the trace row. Defaults to the middle
            channel when None. Giving it triggers the figure even if
            ``plot=False``.
        cmap: Waterfall colormap.

    Returns:
        Patch with denoised data (same shape and coords as input).
    """
    from skimage.restoration import denoise_wavelet, estimate_sigma

    if method not in ("BayesShrink", "VisuShrink"):
        raise ValueError(
            f"unknown method {method!r}; expected 'BayesShrink' or 'VisuShrink'"
        )

    # raw h5 data may be integer counts; wavelet thresholding needs floats
    if np.issubdtype(patch.data.dtype, np.floating):
        data = patch.data
    else:
        data = patch.data.astype(np.float32)

    if method == "VisuShrink":
        sigma_used = sigma if sigma is not None else float(estimate_sigma(data))
        sigma_used *= sigma_scale
        log.info(f"VisuShrink denoising with sigma={sigma_used:.4g}")
    else:
        sigma_used = sigma  # None -> BayesShrink estimates per subband
        log.info("BayesShrink denoising (per-subband adaptive threshold)")

    denoised = denoise_wavelet(
        data,
        wavelet=wavelet,
        method=method,
        mode=mode,
        sigma=sigma_used,
        wavelet_levels=wavelet_levels,
        channel_axis=None,
        rescale_sigma=True,
    )

    result = patch.new(data=denoised.astype(data.dtype))

    if plot or plot_channel is not None:
        _plot_denoise(patch, result, method, plot_channel, cmap=cmap)

    return result


def _plot_denoise(
    before: dc.Patch,
    after: dc.Patch,
    method: str,
    channel: int | None = None,
    cmap: str = "RdBu_r",
) -> None:
    """Original / denoised / removed-noise waterfalls with a second full-width
    row overlaying one channel's before/after trace (mid channel if None)."""
    dist_axis = before.dims.index("distance")
    data_b = before.data if dist_axis == 0 else before.data.T
    data_a = after.data if dist_axis == 0 else after.data.T
    dist = before.coords.get_array("distance")
    if channel is None:
        channel = len(dist) // 2
    time = before.coords.get_array("time")
    t_s = (time - time[0]) / np.timedelta64(1, "s")
    vmax = np.percentile(np.abs(data_b), 99)
    extent = [t_s[0], t_s[-1], float(dist[-1]), float(dist[0])]

    fig = plt.figure(figsize=(10, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])

    axes = []
    for i in range(3):
        ax = fig.add_subplot(
            gs[0, i],
            sharex=axes[0] if axes else None,
            sharey=axes[0] if axes else None,
        )
        axes.append(ax)

    panels = [
        (data_b, "Original"),
        (data_a, f"Denoised ({method})"),
        (data_b - data_a, "Removed noise"),
    ]
    for ax, (d, label) in zip(axes, panels):
        im = ax.imshow(
            d,
            aspect="auto",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            extent=extent,
            interpolation="nearest",
        )
        ax.axhline(float(dist[channel]), color="red", lw=0.8, ls="--")
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("Distance (m)", fontsize=7)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cbar.set_label("Amplitude", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.yaxis.get_offset_text().set_fontsize(6)

    if channel is not None:
        idx = [slice(None), slice(None)]
        idx[dist_axis] = channel
        ax_tr = fig.add_subplot(gs[1, :])
        ax_tr.plot(t_s, before.data[tuple(idx)], lw=0.4, color="black",
                   label="Original")
        ax_tr.plot(t_s, after.data[tuple(idx)], lw=0.5, color="red",
                   label=f"Denoised ({method})")
        ax_tr.set_xlim(t_s[0], t_s[-1])
        ax_tr.set_title(f"Ch {channel} ({dist[channel]:.0f} m)", fontsize=8)
        ax_tr.set_xlabel("Time (s)", fontsize=7)
        ax_tr.set_ylabel("Amplitude", fontsize=7)
        ax_tr.legend(fontsize=6, loc="upper right")
        ax_tr.tick_params(labelsize=6)
        ax_tr.yaxis.get_offset_text().set_fontsize(6)

    plt.show()
