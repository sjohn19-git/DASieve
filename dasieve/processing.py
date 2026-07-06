import csv
import logging

import numpy as np
import matplotlib.pyplot as plt
from dascore.utils.patch import get_dim_sampling_rate
from dascore.units import m
import dascore as dc

log = logging.getLogger(__name__)


def load_survey(survey_path: str) -> dict:
    """Load a well-survey CSV into a dict mapping channel index → (x, y, z).

    CSV must have a header row with columns: channel, x_m, y_m, z_m.
    Units are meters.
    """
    survey = {}
    with open(survey_path, newline="") as f:
        for row in csv.DictReader(f):
            ch = int(row["channel"])
            survey[ch] = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
    log.info(f"Loaded {len(survey)} channel entries from {survey_path}")
    return survey


def attach_geometry(patch: dc.Patch, survey: dict) -> dc.Patch:
    """Attach X, Y, Z coordinates from a well survey to a DASCore Patch.

    Each channel index (0-based position in the distance array) is looked up
    in the survey dict. Channels absent from the survey receive NaN.
    """
    n_ch = patch.channel_count
    x_arr = np.full(n_ch, np.nan)
    y_arr = np.full(n_ch, np.nan)
    z_arr = np.full(n_ch, np.nan)

    for ch_idx, (x, y, z) in survey.items():
        if 0 <= ch_idx < n_ch:
            x_arr[ch_idx] = x
            y_arr[ch_idx] = y
            z_arr[ch_idx] = z
        else:
            log.warning(f"Survey channel {ch_idx} out of range [0, {n_ch-1}] — skipped")

    log.info(f"Geometry attached: {int(np.sum(~np.isnan(x_arr)))}/{n_ch} channels assigned")

    return patch.update_coords(
        x=("distance", x_arr),
        y=("distance", y_arr),
        z=("distance", z_arr),
    )


def to_strain_rate(patch: dc.Patch) -> dc.Patch:
    """Convert raw DAS counts to strain rate (m/m/s).

    This is an amplitude/unit calibration -- it scales the raw interrogator
    counts to physical strain rate using the gauge length and sampling rate
    """
    fs = float(get_dim_sampling_rate(patch, "time"))
    gl = patch.attrs.gauge_length * m
    scale_factor = 1 / 8192
    norm = scale_factor * (116 * fs) / gl
    return (patch * norm * 1e-9).set_units("m/m/s")


def cmd_remove(
    patch: dc.Patch,
    dim: str = "distance",
    window: float | None = None,
    samples: bool = False,
    method: str = "median",
    plot: bool = False,
) -> dc.Patch:
    """Remove common-mode noise by block-wise median subtraction along ``dim``.

    The dimension is split into non-overlapping blocks of size ``window``.
    Within each block, the median across all channels/samples is computed
    (one scalar per position in the perpendicular dimension) and subtracted
    from every trace in that block. If ``window`` is None, the whole dimension
    is treated as a single block.

    Args:
        patch: DAS patch with dims ("time", "distance").
        dim: Dimension to split into blocks, either "distance" or "time".
        window: Block size in physical units (metres for distance, seconds for
            time), or in samples if ``samples=True``. None uses the full dim.
        samples: If True, treat ``window`` as a sample/channel count.
        plot: If True, show the qc.plot_patch diagnostic of the result.

    Returns:
        Patch with block-wise common mode removed (same shape and coords as input).
    """
    if method not in ("median", "stack"):
        raise ValueError(
            f"unknown method {method!r}; expected 'median' or 'stack'"
        )

    n = len(patch.coords.get_array(dim))
    dim_axis = patch.dims.index(dim)

    if window is None:
        block_size = n
    elif samples:
        block_size = max(1, int(window))
    else:
        fs = float(get_dim_sampling_rate(patch, dim))
        block_size = max(1, round(window * fs))

    # raw h5 data may be integer counts; the subtracted common mode is float,
    # so work in float regardless of call order relative to to_strain_rate
    if np.issubdtype(patch.data.dtype, np.floating):
        data = patch.data.copy()
    else:
        data = patch.data.astype(np.float32)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        idx = [slice(None), slice(None)]
        idx[dim_axis] = slice(start, end)

        if method == "median":
            common_mode = np.median(
                data[tuple(idx)], axis=dim_axis, keepdims=True
            )
        else:  # "stack"
            other_dim = next(d for d in patch.dims if d != dim)
            sub_patch = patch.new(
                data=data[tuple(idx)],
                coords={
                    dim: patch.coords.get_array(dim)[start:end],
                    other_dim: patch.coords.get_array(other_dim),
                },
            )
            mean_patch = sub_patch.phase_weighted_stack(
                stack_dim=dim, power=0, dim_reduce="squeeze"
            )
            common_mode = np.expand_dims(mean_patch.data, axis=dim_axis)

        data[tuple(idx)] -= common_mode

    result = patch.new(data=data)

    if plot:
        from .qc import plot_patch

        plot_patch(result, show=True)

    return result


def decimate(
    patch: dc.Patch,
    target_fs: float | None = None,
    target_dx: float | None = None,
    plot: bool = False,
    lateral_stacking: bool = True,
    pws_power: float = 2.0,
    plot_channel: int | None = None,
) -> dc.Patch:
    """Decimate a DAS patch in time, space, or both.

    Args:
        patch: DAS patch. Dim order (time, distance) or (distance, time) both supported.
        target_fs: Target sampling rate in Hz. If provided, decimates in time.
        target_dx: Target channel spacing in metres. If provided, decimates in space.
        plot: If True, visualise decimation.
        lateral_stacking: If True (default), average groups of channels before
            spatial decimation instead of applying the FIR anti-alias filter.
        pws_power: Phase-weighted stack exponent used when lateral_stacking=True.
        plot_channel: Channel index to highlight in plots. Defaults to the middle
            channel when None.
    """
    if target_fs is None and target_dx is None:
        raise ValueError("At least one of target_fs or target_dx must be provided")

    result = patch
    stages = (
        [(patch, f"Original ({patch.data.shape[patch.dims.index('distance')]} ch)")]
        if plot
        else None
    )

    if target_fs is not None:
        current_fs = float(get_dim_sampling_rate(result, "time"))
        factor = round(current_fs / target_fs)
        if factor < 1:
            raise ValueError(
                f"target_fs ({target_fs} Hz) exceeds current rate ({current_fs} Hz)"
            )
        result = result.decimate(time=factor, filter_type="fir")
        if plot:
            stages.append((result, f"Temporal Decimation ×{factor}"))

    if target_dx is not None:
        dist = result.coords.get_array("distance")
        current_dx = float(np.median(np.diff(dist)))
        factor = round(target_dx / current_dx)
        if factor < 1:
            raise ValueError(
                f"target_dx ({target_dx} m) is smaller than current spacing ({current_dx} m)"
            )
        if lateral_stacking:
            result = _lateral_stack(result, factor, pws_power=pws_power)
        else:
            result = result.decimate(distance=factor, filter_type="fir")
        if plot:
            method = f"PWS, Power= {pws_power}" if lateral_stacking else "FIR"
            stages.append(
                (
                    result,
                    f"Spatial Decimation ×{factor} {method}",
                )
            )

    if plot:
        _plot_decimation(stages, plot_channel)

    return result


def _non_dim_coords(patch: dc.Patch) -> dict:
    """Return {name: dims_tuple} for every coord that is not itself a dim
    (e.g. x/y/z from attach_geometry, or any metadata coord from the file)."""
    dim_map = getattr(patch.coords, "dim_map", None)
    if dim_map is None:  # very old dascore fallback: known geometry names only
        return {n: ("distance",) for n in ("x", "y", "z")
                if n in getattr(patch.coords, "coord_map", {})}
    return {n: tuple(d) for n, d in dim_map.items() if n not in patch.dims}


def _grouped_distance_coords(patch: dc.Patch, n_trim: int, n_groups: int, factor: int) -> dict:
    """Propagate every non-dim coord through lateral stacking so no metadata
    from the source file (or attach_geometry) is dropped.

    Coords attached to ``distance`` are reduced per channel group: numeric
    coords are nanmean-averaged (all-NaN groups stay NaN); non-numeric coords
    take the first value of each group. Coords attached to ``time`` pass
    through unchanged. Anything else (multi-dim coords) cannot be reduced
    meaningfully and is skipped with a warning."""
    import warnings

    out = {}
    for name, dims in _non_dim_coords(patch).items():
        arr = np.asarray(patch.coords.get_array(name))
        if dims == ("distance",):
            if np.issubdtype(arr.dtype, np.number):
                with warnings.catch_warnings():
                    # all-NaN groups (channels absent from the survey) stay NaN
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    grouped = np.nanmean(
                        arr[:n_trim].astype(float).reshape(n_groups, factor), axis=1
                    )
            else:
                grouped = arr[:n_trim:factor]  # first value of each group
            out[name] = ("distance", grouped)
        elif dims == ("time",):
            out[name] = ("time", arr)
        else:
            warnings.warn(
                f"coord {name!r} (dims={dims}) cannot be carried through "
                f"lateral stacking and was dropped.",
                stacklevel=2,
            )
    return out


def _lateral_stack(patch: dc.Patch, factor: int, pws_power: float = 2.0) -> dc.Patch:
    """Phase-weighted stack groups of *factor* channels, returning a spatially decimated patch."""
    dist = patch.coords.get_array("distance")
    nch = len(dist)
    n_trim = (nch // factor) * factor
    n_groups = n_trim // factor
    dist_axis = patch.dims.index("distance")

    new_dist = dist[:n_trim].reshape(n_groups, factor).mean(axis=1)
    geom_coords = _grouped_distance_coords(patch, n_trim, n_groups, factor)
    cols = []

    for i in range(n_groups):
        idx = [slice(None), slice(None)]
        idx[dist_axis] = slice(i * factor, (i + 1) * factor)
        sub_data = patch.data[tuple(idx)]
        sub_dist = dist[i * factor : (i + 1) * factor]
        sub = patch.new(
            data=sub_data,
            coords={"distance": sub_dist, "time": patch.coords.get_array("time")},
        )
        result = sub.phase_weighted_stack(
            stack_dim="distance", power=pws_power, dim_reduce="squeeze"
        )
        cols.append(result.data)

    stacked = np.stack(cols, axis=dist_axis)
    return patch.new(
        data=stacked,
        coords={
            "distance": new_dist,
            "time": patch.coords.get_array("time"),
            **geom_coords,  # keep x/y/z geometry (group-averaged)
        },
        dims=patch.dims,
    )


def _plot_decimation(
    stages: list[tuple[dc.Patch, str]],
    channel: int | None = None,
) -> None:
    """Waterfall + extracted trace for each decimation stage (n×2 grid)."""
    n = len(stages)
    colors = ["black", "red", "tab:blue"]

    patch0 = stages[0][0]
    data0 = patch0.data
    dist_axis0 = patch0.dims.index("distance")
    nch = data0.shape[dist_axis0]
    ch = nch // 2 if channel is None else channel
    orig_dist = patch0.coords.get_array("distance")
    t0 = patch0.coords.get_array("time")[0]
    vmax = np.percentile(np.abs(data0), 99)

    fig, axes = plt.subplots(
        n,
        2,
        figsize=(7.5, 2.7 * n),
        gridspec_kw={"width_ratios": [3, 1]},
        constrained_layout=True,
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    im = None
    for row, (patch, label) in enumerate(stages):
        d = patch.data
        dist_axis = patch.dims.index("distance")
        time = patch.coords.get_array("time")
        t_s = (time - t0) / np.timedelta64(1, "s")
        dist = patch.coords.get_array("distance")
        ch_row = int(np.argmin(np.abs(dist - float(orig_dist[ch]))))

        # imshow expects (distance, time) layout
        plot_data = d if dist_axis == 0 else d.T

        ax_im = axes[row, 0]
        ax_tr = axes[row, 1]

        im = ax_im.imshow(
            plot_data,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[t_s[0], t_s[-1], float(dist[-1]), float(dist[0])],
            interpolation="nearest",
        )
        ax_im.axhline(float(dist[ch_row]), color="red", lw=0.8, ls="--")
        ax_im.set_title(label, fontsize=8)
        ax_im.set_ylabel("Distance (m)", fontsize=7)
        ax_im.tick_params(labelsize=6)
        if row == n - 1:
            ax_im.set_xlabel("Time (s)", fontsize=7)

        tr_idx = [slice(None), slice(None)]
        tr_idx[dist_axis] = ch_row
        ax_tr.plot(t_s, d[tuple(tr_idx)], lw=0.3, color=colors[row])
        ax_tr.set_title(f"Ch {ch_row} ({dist[ch_row]:.0f} m)", fontsize=8)
        ax_tr.set_ylabel("Amplitude", fontsize=7)
        ax_tr.tick_params(labelsize=6)
        ax_tr.yaxis.get_offset_text().set_fontsize(6)
        if row == n - 1:
            ax_tr.set_xlabel("Time (s)", fontsize=7)

    cax = axes[-1, 0].inset_axes([0.02, 0.88, 0.35, 0.05])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Amplitude", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.xaxis.get_offset_text().set_fontsize(6)
    plt.show()
