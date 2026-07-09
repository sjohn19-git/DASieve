"""Windowed event detection on DAS picks.

Scans pick onset times with a sliding detection window and emits
non-overlapping association windows wherever enough of the fiber picked at
the same time. The windows gate association: only picks inside an emitted
window are handed to an associator (see ``detector=`` / ``windows=`` on
:meth:`dasieve.associator.BaseAssociator.run`).

Two trigger mechanisms, selected by ``method``:

* ``"count"`` -- the window triggers when at least ``min_channels`` distinct
  channels (unique ``distance`` values) have a pick inside it.
* ``"vote"``  -- the fiber is split into ``n_segments`` equal channel-count
  segments; a segment votes True when at least ``seg_min_channels`` of its
  channels picked inside the window; the window triggers when at least
  ``min_votes`` segments vote True.

Counting is phase agnostic (a pick is a pick) and per distinct channel, so a
single noisy channel firing repeatedly cannot satisfy a threshold by itself.

Detection windows advance by ``stride`` (default ``window / 2``) and overlap,
so a phase moveout straddling a window boundary still triggers at the next
position. When a window starting at t0 triggers, the association window
``[t0, t0 + look_ahead]`` is emitted and scanning resumes at
``t0 + look_ahead`` -- emitted windows never overlap, so no event is
associated twice. ``look_ahead`` is the user's choice: it should cover the
expected S-minus-P delay plus moveout for the events of interest.

Typical use::

    from dasieve import detection

    det = detection.EventDetector(
        method="vote", window=0.5, look_ahead=2.0,
        n_segments=8, seg_min_channels=4, min_votes=5,
        channels=patch.coords.get_array("distance"),
    )
    windows = det.detect(picks=df_picks, plot=True, patch=patch)

    # hand the windows (or the detector itself) to an associator:
    catalog, assignments = assoc.run(picks=df_picks, windows=windows)
    catalog, assignments = assoc.run(picks=df_picks, detector=det)

``detect`` can also pull picks straight from the store with the same filters
as :func:`dasieve.store.select_pick_ids`::

    windows = det.detect(pick_method="phasenetdas", file_name=".../ev.h5",
                         min_score=0.3)
"""

import warnings

import numpy as np
import pandas as pd

from .store import DEFAULT_DB_PATH, load_picks_by_ids, select_pick_ids

#: Columns of the windows DataFrame returned by :meth:`EventDetector.detect`.
WINDOW_COLUMNS = ["t_start", "t_end", "n_picks", "n_channels", "n_votes"]


class EventDetector:
    """Sliding-window pick-density detector (see module docstring).

    Parameters
    ----------
    method : "count" or "vote"
        Trigger mechanism.
    window : float
        Detection window length in seconds.
    stride : float, optional
        Detection window advance in seconds; default ``window / 2``. Strides
        larger than ``window`` leave unscanned gaps (warned).
    look_ahead : float
        Length in seconds of the emitted association window; required.
    min_channels : int
        ("count") distinct channels needed inside the window to trigger.
    n_segments : int
        ("vote") number of equal channel-count segments the fiber is split
        into.
    seg_min_channels : int
        ("vote") distinct channels a segment needs to vote True.
    min_votes : int
        ("vote") votes needed for the window to trigger; required for
        ``method="vote"``.
    channels : array-like, optional
        Full set of channel positions (e.g.
        ``patch.coords.get_array("distance")``). With it, segment boundaries
        cover the whole fiber and ``seg_min_channels`` is validated against
        the smallest segment at construction; without it, segments are
        derived from the channels present in the picks at detect time.
    """

    def __init__(
        self,
        method="count",
        window=0.5,
        stride=None,
        look_ahead=None,
        min_channels=20,
        n_segments=8,
        seg_min_channels=2,
        min_votes=None,
        channels=None,
    ):
        if method not in ("count", "vote"):
            raise ValueError(f"method must be 'count' or 'vote', got {method!r}")
        if window <= 0:
            raise ValueError("window must be > 0 seconds")
        if stride is None:
            stride = window / 2
        if stride <= 0:
            raise ValueError("stride must be > 0 seconds")
        if stride > window:
            warnings.warn(
                f"stride ({stride} s) > window ({window} s): parts of the "
                "record are never inside any detection window",
                stacklevel=2,
            )
        if look_ahead is None or look_ahead <= 0:
            raise ValueError(
                "look_ahead (seconds) is required and must be > 0; it sets "
                "the emitted association window length"
            )

        if method == "count":
            if min_channels < 1:
                raise ValueError("min_channels must be >= 1")
        else:
            if n_segments < 1:
                raise ValueError("n_segments must be >= 1")
            if seg_min_channels < 1:
                raise ValueError("seg_min_channels must be >= 1")
            if min_votes is None:
                raise ValueError("min_votes is required for method='vote'")
            if not 1 <= min_votes <= n_segments:
                raise ValueError(
                    f"min_votes must be in [1, n_segments={n_segments}], "
                    f"got {min_votes}"
                )

        self.method = method
        self.window = float(window)
        self.stride = float(stride)
        self.look_ahead = float(look_ahead)
        self.min_channels = int(min_channels)
        self.n_segments = int(n_segments)
        self.seg_min_channels = int(seg_min_channels)
        self.min_votes = None if min_votes is None else int(min_votes)

        # segment layout from the full channel set, if provided (validated
        # here so an impossible seg_min_channels fails fast, not silently)
        self._segments = None
        if channels is not None:
            self._segments = self._build_segments(np.asarray(channels, float))

    # ------------------------------------------------------------------
    # segments
    # ------------------------------------------------------------------
    def _build_segments(self, channel_values):
        """Split channels into ``n_segments`` equal channel-count segments.

        Returns (inner_bounds, seg_ranges, seg_sizes): ``inner_bounds`` are
        the n_segments-1 distance cuts for ``np.digitize``; ``seg_ranges``
        the (lo, hi) distance of each segment (for plotting); ``seg_sizes``
        the channel count per segment.
        """
        uniq = np.unique(channel_values[np.isfinite(channel_values)])
        if len(uniq) < self.n_segments:
            raise ValueError(
                f"only {len(uniq)} distinct channels for "
                f"n_segments={self.n_segments}"
            )
        splits = np.array_split(uniq, self.n_segments)
        sizes = [len(s) for s in splits]
        if self.method == "vote" and self.seg_min_channels > min(sizes):
            raise ValueError(
                f"seg_min_channels={self.seg_min_channels} can never be met: "
                f"the smallest segment has only {min(sizes)} channels"
            )
        inner_bounds = np.array(
            [(splits[i][-1] + splits[i + 1][0]) / 2.0
             for i in range(self.n_segments - 1)]
        )
        seg_ranges = [(float(s[0]), float(s[-1])) for s in splits]
        return inner_bounds, seg_ranges, sizes

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def _votes(self, dists, seg_idx):
        """Votes (segments with >= seg_min_channels distinct channels)."""
        votes = np.zeros(self.n_segments, dtype=bool)
        for s in range(self.n_segments):
            n_ch = len(np.unique(dists[seg_idx == s]))
            votes[s] = n_ch >= self.seg_min_channels
        return votes

    def _evaluate(self, dists, seg_idx):
        """(triggered, n_channels, n_votes) for the picks in one window."""
        n_channels = len(np.unique(dists))
        if self.method == "count":
            return n_channels >= self.min_channels, n_channels, np.nan
        n_votes = int(self._votes(dists, seg_idx).sum())
        return n_votes >= self.min_votes, n_channels, n_votes

    def detect(
        self,
        picks=None,
        db_path=DEFAULT_DB_PATH,
        *,
        pick_method=None,
        pick_starttime=None,
        pick_endtime=None,
        phase=None,
        file_name=None,
        min_score=None,
        plot=False,
        patch=None,
    ):
        """Scan picks and emit non-overlapping association windows.

        Parameters
        ----------
        picks : pandas.DataFrame, optional
            Picks in the pickers' output schema (needs ``onset_time`` and
            ``distance``). When omitted, picks are selected from the store
            with the remaining filters (same semantics as
            :func:`dasieve.store.select_pick_ids`; ``pick_method`` /
            ``pick_starttime`` / ``pick_endtime`` map to its ``method`` /
            ``time_start`` / ``time_end``).
        plot : bool
            Draw the detector diagnostics: picks (over the patch waterfall if
            ``patch`` is given) with every emitted window shaded, and the
            trigger metric vs. its threshold in a lower panel. For
            ``method="vote"`` each triggered window carries green/red
            brackets per segment (green = voted True).
        patch : dascore.Patch, optional
            Only used as the imshow background of the plot.

        Returns
        -------
        pandas.DataFrame with :data:`WINDOW_COLUMNS`: one row per emitted
        window -- ``t_start`` / ``t_end`` (association window, timestamps),
        ``n_picks`` / ``n_channels`` in the *detection* window that
        triggered, and ``n_votes`` (NaN for ``method="count"``).
        """
        if picks is None:
            pick_ids = select_pick_ids(
                db_path,
                method=pick_method,
                time_start=pick_starttime,
                time_end=pick_endtime,
                phase=phase,
                file_name=file_name,
                min_score=min_score,
            )
            picks = load_picks_by_ids(pick_ids, db_path)
            print(f"EventDetector.detect: {len(picks)} picks selected from store")

        empty = pd.DataFrame(columns=WINDOW_COLUMNS)
        if not len(picks):
            print("EventDetector.detect: no picks -> no windows")
            return empty

        times = pd.to_datetime(picks["onset_time"])
        ref = times.min()
        t_s = (times - ref).dt.total_seconds().to_numpy()
        dists = pd.to_numeric(picks["distance"], errors="coerce").to_numpy()

        segments = self._segments
        if self.method == "vote":
            if segments is None:
                segments = self._build_segments(dists)
            seg_idx = np.digitize(dists, segments[0])
        else:
            seg_idx = np.zeros(len(dists), dtype=int)

        rows = []
        cursor = 0.0
        duration = float(t_s.max())
        while cursor <= duration:
            mask = (t_s >= cursor) & (t_s < cursor + self.window)
            triggered, n_channels, n_votes = self._evaluate(
                dists[mask], seg_idx[mask]
            )
            if triggered:
                rows.append(
                    {
                        "t_start": ref + pd.Timedelta(seconds=cursor),
                        "t_end": ref + pd.Timedelta(seconds=cursor
                                                    + self.look_ahead),
                        "n_picks": int(mask.sum()),
                        "n_channels": int(n_channels),
                        "n_votes": n_votes,
                    }
                )
                cursor += self.look_ahead  # emitted windows never overlap
            else:
                cursor += self.stride

        windows = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
        print(
            f"EventDetector({self.method}): {len(windows)} window(s) from "
            f"{len(picks)} picks over {duration:.2f} s"
        )

        if plot:
            self._plot(t_s, dists, seg_idx, segments, windows, ref, patch)
        return windows

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------
    def _plot(self, t_s, dists, seg_idx, segments, windows, ref, patch):
        """Detector diagnostics: picks + emitted windows | metric trace."""
        import matplotlib.pyplot as plt

        duration = float(t_s.max())
        w_starts = ((windows["t_start"] - ref).dt.total_seconds().to_numpy()
                    if len(windows) else np.array([]))

        fig, (ax, ax_m) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # --- top: picks over the data, emitted windows shaded
        if patch is not None:
            p_time = patch.coords.get_array("time")
            p_dist = patch.coords.get_array("distance")
            pt = (p_time - np.datetime64(ref)) / np.timedelta64(1, "s")
            data2d = np.moveaxis(
                patch.data,
                [patch.dims.index("distance"), patch.dims.index("time")],
                [0, 1],
            )
            vmax = np.percentile(np.abs(data2d), 99)
            ax.imshow(data2d, aspect="auto", cmap="gray",
                      extent=(pt[0], pt[-1], p_dist[-1], p_dist[0]),
                      vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.scatter(t_s, dists, marker="|", s=25, linewidths=0.8,
                   color="darkblue", label="picks", zorder=3)

        for k, t0 in enumerate(w_starts):
            # look-ahead (association) window, light; detection window, darker
            ax.axvspan(t0, t0 + self.look_ahead, color="gold", alpha=0.20,
                       zorder=2, label="association window" if k == 0 else None)
            ax.axvspan(t0, t0 + self.window, color="orange", alpha=0.30,
                       zorder=2, label="detection window" if k == 0 else None)

        if self.method == "vote" and segments is not None:
            _, seg_ranges, _ = segments
            for lo, _hi in seg_ranges[1:]:
                ax.axhline(lo, color="0.5", lw=0.6, ls="--", zorder=2)
            # green/red bracket per segment on each triggered window
            tick = max(duration, 1e-3) * 0.008
            for t0 in w_starts:
                mask = (t_s >= t0) & (t_s < t0 + self.window)
                votes = self._votes(dists[mask], seg_idx[mask])
                for s, (lo, hi) in enumerate(seg_ranges):
                    c = "limegreen" if votes[s] else "red"
                    t1 = t0 + self.window
                    ax.plot([t0 + tick, t0, t0, t0 + tick],
                            [lo, lo, hi, hi], color=c, lw=1.8, zorder=4)
                    ax.plot([t1 - tick, t1, t1, t1 - tick],
                            [lo, lo, hi, hi], color=c, lw=1.8, zorder=4)

        ax.set_ylabel("Distance (m)")
        ax.set_title(
            f"EventDetector({self.method}): {len(windows)} window(s); "
            + ("green/red bracket = segment vote"
               if self.method == "vote" else
               f"trigger at >= {self.min_channels} channels")
        )
        if patch is None:
            ax.invert_yaxis()
        ax.legend(loc="upper right", fontsize=8)

        # --- bottom: trigger metric on a uniform stride grid vs threshold
        grid = np.arange(0.0, duration + self.stride, self.stride)
        metric = np.empty(len(grid))
        for i, g in enumerate(grid):
            m = (t_s >= g) & (t_s < g + self.window)
            if self.method == "count":
                metric[i] = len(np.unique(dists[m]))
            else:
                metric[i] = self._votes(dists[m], seg_idx[m]).sum()
        thr = self.min_channels if self.method == "count" else self.min_votes
        label = ("distinct channels" if self.method == "count"
                 else "segment votes")
        ax_m.step(grid, metric, where="post", color="darkblue", lw=1.0,
                  label=label)
        ax_m.axhline(thr, color="crimson", lw=1.0, ls="--",
                     label=f"threshold ({thr})")
        for t0 in w_starts:
            ax_m.axvspan(t0, t0 + self.look_ahead, color="gold", alpha=0.20)
        ax_m.set_xlabel(f"Time (s) since {ref}")
        ax_m.set_ylabel(label)
        ax_m.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.show()
        return fig
