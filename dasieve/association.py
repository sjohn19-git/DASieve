"""Phase association for DAS picks.

Takes P/S picks from the SQLite store (:mod:`dasieve.store`), treats each
DAS channel (unique ``distance``) as a station using the x/y/z geometry the
pickers persisted, and clusters the picks into events. Results are written to
two tables in the same store database (default ``~/DASieve/dasieve.sqlite``):

* ``events``      -- one row per associated event (origin time, location, ...)
* ``assignments`` -- one row per (event, pick) link, carrying the ``pick_id``
                     back into the ``picks`` table.

Each associator is a class; its config schema is its own (GaMMA's knobs are
not shared by other associators), so config lives on the associator instance
rather than as free-floating module functions. Only :class:`GammaAssociator`
is implemented so far.

Every picker emits P/S phase labels ("phasenet", "*_sb", "ar"; STA/LTA
labels each trigger onset as a P pick), so any picker's output can be
associated.

Typical use::

    from dasieve import association
    assoc = association.GammaAssociator.from_preset("default", dbscan_eps=3.0)
    assoc.update_config(**{"x(km)": (4258.0, 4263.0)})   # optional edits
    catalog_df, assignments_df = assoc.run(
        pick_method="phasenetdas", file_name=".../event.h5", min_score=0.3,
    )   # saves to events/assignments automatically (db_save=True),
        # replacing previous results for the same (file_name, method)

Picks can also be supplied directly as a DataFrame (e.g. straight from a
picker, without going through the store). This path never touches the
database -- the events/assignments tables are only returned::

    df = sieve.picking.trigger_picker(patch, db_save=False)
    catalog_df, assignments_df = assoc.run(picks=df)

or the one-call convenience wrapper::

    catalog_df, assignments_df = association.run_association(
        method="gamma", pick_method="phasenetdas", dbscan_eps=3.0,
    )

GaMMA itself is imported lazily inside :meth:`GammaAssociator._associate`, so
importing ``dasieve`` does not require it to be installed.
"""

import copy
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .store import (
    DEFAULT_DB_PATH,
    load_picks,
    save_associations,
)


# ---------------------------------------------------------------------------
# Base associator
# ---------------------------------------------------------------------------
class BaseAssociator(ABC):
    """Common machinery shared by every associator.

    The generic part -- selecting picks from the store and persisting the
    results -- lives here; subclasses supply the associator-specific config
    schema (:meth:`from_preset`) and the actual clustering
    (:meth:`_associate`). ``name`` labels the associator and is written to the
    ``method`` column of the events/assignments tables.
    """

    #: Short associator id, e.g. "gamma"; set by each subclass.
    name: str = ""

    def __init__(self, config):
        self.config = config

    @classmethod
    @abstractmethod
    def from_preset(cls, flag="default", **overrides):
        """Build an associator from a named preset, with optional overrides."""

    def update_config(self, **changes):
        """Edit this associator's config in place; returns ``self`` so calls
        can be chained (``assoc.update_config(...).run(...)``)."""
        self.config.update(changes)
        return self

    @abstractmethod
    def _associate(self, picks_df):
        """Associate a store picks DataFrame into events.

        Returns ``(catalog_df, assignments_df)`` where ``assignments_df`` has
        a ``pick_id`` column mapping each row back to ``picks.id``."""
        raise NotImplementedError

    def run(
        self,
        db_path=DEFAULT_DB_PATH,
        *,
        picks=None,
        detector=None,
        windows=None,
        pick_method=None,
        pick_starttime=None,
        pick_endtime=None,
        phase=None,
        file_name=None,
        min_score=None,
        db_save=True,
        plot=False,
        patch=None,
    ):
        """Associate picks -- from a DataFrame, or selected from the store.

        When ``picks`` is given, that DataFrame is associated directly: the
        database is not touched at all (no selection, no save; the pick
        filters and ``db_save`` are ignored) and the events/assignments
        tables are only returned.

        Otherwise selection is delegated to
        :func:`dasieve.store.select_pick_ids`; the number of selected picks is
        printed before association runs. When ``db_save`` is True (default)
        results are written to the events/assignments tables via
        :func:`dasieve.store.save_associations`, keyed on
        (file_name, ``self.name``) -- re-running on the same data replaces the
        previous results instead of duplicating them.

        Either way, association can be gated by detection windows: pass
        ``detector`` (an :class:`dasieve.detection.EventDetector`, run here on
        the same picks) or precomputed ``windows``. Association then runs once
        per window on the picks inside it; picks outside every window are
        never associated. ``event_index`` stays unique across windows.

        Parameters
        ----------
        db_path : str
            Catalog database holding the picks.
        picks : pandas.DataFrame, optional
            Picks to associate directly, in the pickers' output schema (must
            include ``onset_time``, ``score``, ``phase``, ``distance``,
            ``x``, ``y``, ``z``), e.g. the DataFrame returned by
            :func:`dasieve.picking.trigger_picker` or
            :func:`dasieve.picking.phasenet_das_picker`. If there is no ``id``
            column, row positions are used, so ``assignments_df["pick_id"]``
            indexes back into ``picks`` rows.
        detector : dasieve.detection.EventDetector, optional
            Run on the picks to produce detection windows that gate the
            association. Mutually exclusive with ``windows``.
        windows : pandas.DataFrame or iterable of (start, end), optional
            Precomputed association windows (as returned by
            :meth:`EventDetector.detect`, or any (t_start, t_end) pairs).
        pick_method : str or list of str, optional
            Which picker's picks to associate (the ``picks.method`` column,
            e.g. "phasenetdas") -- NOT the associator.
        pick_starttime, pick_endtime : str or datetime, optional
            Inclusive ``onset_time`` window.
        phase, file_name, min_score : optional
            Further filters, forwarded to
            :func:`dasieve.store.select_pick_ids`. ``file_name`` doubles as the
            save key: results are stored under (file_name, ``self.name``), with
            file_name "all" when no file filter is set.
        db_save : bool
            If True (default), save results to the events/assignments tables.
            Only applies when picks come from the store.
        plot : bool
            Draw :func:`plot_association` for the results: event locations on
            the cable geometry next to the picks-on-data view colored by
            event. Geometry comes from the picks' x/y/z, so it adapts to the
            cable at hand.
        patch : dascore.Patch, optional
            Only used when ``plot=True``, as the waterfall background.

        Returns
        -------
        (catalog_df, assignments_df)
            ``catalog_df`` -- the event catalog (time, x(km), y(km), z(km),
            gamma_score, sigma_time, magnitude, number_picks, ...).
            ``assignments_df`` -- ``pick_index``, ``event_index``,
            ``gamma_score``, plus ``pick_id`` mapping each row back to
            ``picks.id`` in the store database (or to the row position in
            ``picks`` when a DataFrame was supplied without an ``id`` column).
        """
        if detector is not None and windows is not None:
            raise ValueError("pass either detector= or windows=, not both")

        if picks is not None:
            picks_df = picks.reset_index(drop=True)
            if "id" not in picks_df.columns:
                picks_df = picks_df.assign(id=np.arange(len(picks_df)))
            from_store = False
            print(
                f"{self.name}.run: {len(picks_df)} picks supplied as a "
                "DataFrame (no database read/write)"
            )
        else:
            picks_df = load_picks(
                db_path,
                method=pick_method,
                time_start=pick_starttime,
                time_end=pick_endtime,
                phase=phase,
                file_name=file_name,
                min_score=min_score,
            )
            print(
                f"{self.name}.run: {len(picks_df)} picks selected "
                f"(pick_method={pick_method!r}, "
                f"window={pick_starttime!r}..{pick_endtime!r})"
            )
            from_store = True

        if detector is not None:
            windows = detector.detect(picks_df)

        if windows is None:
            catalog_df, assignments_df = self._associate(picks_df)
        else:
            catalog_df, assignments_df = self._associate_windowed(
                picks_df, windows
            )

        if from_store and db_save:
            save_key = file_name if file_name is not None else "all"
            n_ev, n_as = save_associations(
                catalog_df, assignments_df, db_path,
                file_name=save_key, method=self.name,
            )
            print(
                f"{self.name}.run: saved {n_ev} events / {n_as} assignments "
                f"(file_name={save_key!r}, method={self.name!r})"
            )

        if plot:
            picks_ev = picks_df.merge(
                assignments_df[["pick_id", "event_index"]],
                left_on="id", right_on="pick_id", how="left",
            )
            plot_association(picks_ev, catalog_df, patch=patch,
                             title=self.name)

        return catalog_df, assignments_df

    def _associate_windowed(self, picks_df, windows):
        """Associate once per detection window; concatenate the results.

        ``windows`` is a DataFrame with ``t_start`` / ``t_end`` columns (as
        emitted by :meth:`EventDetector.detect`) or an iterable of
        (start, end) pairs. Picks outside every window are never associated.
        Each window's ``event_index`` is offset by a running counter so
        indices stay unique across windows.
        """
        if not isinstance(windows, pd.DataFrame):
            windows = pd.DataFrame(list(windows), columns=["t_start", "t_end"])
        times = pd.to_datetime(picks_df["onset_time"])

        catalogs, assigns = [], []
        offset = 0
        for _, w in windows.iterrows():
            t_lo, t_hi = pd.Timestamp(w["t_start"]), pd.Timestamp(w["t_end"])
            sub = picks_df[(times >= t_lo) & (times < t_hi)]
            if sub.empty:
                continue
            cat, asg = self._associate(sub.reset_index(drop=True))
            if len(cat):
                cat = cat.copy()
                asg = asg.copy()
                cat["event_index"] = cat["event_index"] + offset
                asg["event_index"] = asg["event_index"] + offset
                offset = int(cat["event_index"].max()) + 1
            catalogs.append(cat)
            assigns.append(asg)

        catalog_df = (
            pd.concat(catalogs, ignore_index=True) if catalogs
            else pd.DataFrame()
        )
        assignments_df = (
            pd.concat(assigns, ignore_index=True) if assigns
            else pd.DataFrame(
                columns=["pick_index", "event_index", "gamma_score", "pick_id"]
            )
        )
        print(
            f"{self.name}: {len(windows)} window(s) -> {len(catalog_df)} "
            f"events / {len(assignments_df)} assignments"
        )
        return catalog_df, assignments_df


# ---------------------------------------------------------------------------
# GaMMA associator
# ---------------------------------------------------------------------------
class GammaAssociator(BaseAssociator):
    """GaMMA (Gaussian-mixture) phase associator.

    Config presets hold GaMMA's non-geometric knobs only; the spatial search
    bounds ("x(km)", "y(km)", "z(km)", "bfgs_bounds") are derived from the
    station geometry in :meth:`build_inputs` unless supplied explicitly via
    :meth:`from_preset` / :meth:`update_config`."""

    name = "gamma"

    #: Named config presets (keys passed as ``flag`` to :meth:`from_preset`).
    PRESETS = {
        "default": {
            "dims": ["x(km)", "y(km)", "z(km)"],
            "use_dbscan": True,
            "use_amplitude": False,
            "vel": {"p": 6.0, "s": 6.0 / 1.75},
            "method": "BGMM",
            "dbscan_eps": 25.0,          # seconds
            "dbscan_min_samples": 3,
            "min_picks_per_eq": 5,
            "max_sigma11": 2.0,
            "max_sigma22": 1.0,
            "max_sigma12": 1.0,
        },
    }

    _OVERSAMPLE_BY_METHOD = {"BGMM": 4, "GMM": 1}
    # Padding (km) added around the station extent when auto-deriving the
    # spatial search bounds.
    _BOUNDS_PAD_KM = 1.0
    # prob for picks without a score (e.g. AR picks)
    _DEFAULT_PROB = 0.5

    @classmethod
    def from_preset(cls, flag="default", **overrides):
        """Build a GaMMA associator from a preset, with optional edits.

        ``overrides`` both selects a preset and edits it in one call, e.g.
        ``GammaAssociator.from_preset("default", dbscan_eps=10,
        vel={"p": 5.5, "s": 3.2})``. Spatial bounds ("x(km)", "y(km)",
        "z(km)", "bfgs_bounds") may be passed here; otherwise they are
        auto-derived from station geometry in :meth:`build_inputs`.
        ``oversample_factor`` is derived from ``method`` (BGMM -> 4, GMM -> 1)
        unless explicitly overridden.
        """
        if flag not in cls.PRESETS:
            raise KeyError(
                f"unknown config preset {flag!r}; available presets: "
                f"{sorted(cls.PRESETS)}"
            )
        config = copy.deepcopy(cls.PRESETS[flag])
        config.update(overrides)
        if "oversample_factor" not in config:
            method = config.get("method", "BGMM")
            config["oversample_factor"] = cls._OVERSAMPLE_BY_METHOD.get(method, 1)
        return cls(config)

    def update_config(self, **changes):
        """Edit the GaMMA config in place; returns ``self`` for chaining.

        If ``method`` changes and ``oversample_factor`` is not explicitly
        given, the factor is re-derived (BGMM -> 4, GMM -> 1)."""
        self.config.update(changes)
        if "method" in changes and "oversample_factor" not in changes:
            self.config["oversample_factor"] = self._OVERSAMPLE_BY_METHOD.get(
                self.config["method"], 1
            )
        return self

    def build_inputs(self, picks_df):
        """Build GaMMA's ``pick_df`` / ``station_df`` from store picks.

        Each unique ``distance`` is treated as a station whose id is the
        distance value (as string) and whose position is the pick's x/y/z
        geometry (meters -> km; no lat/lon projection, the survey is already in
        local meters). Stations with NaN geometry are dropped with a warning.

        If this associator's spatial bounds are unset, "x(km)", "y(km)",
        "z(km)" are auto-filled from the station extent (padded by
        ``_BOUNDS_PAD_KM``) and ``bfgs_bounds`` is derived from them. The
        config is modified in place.

        Parameters
        ----------
        picks_df : pandas.DataFrame
            Picks as returned by :func:`dasieve.store.load_picks_by_ids`
            (must include ``id``, ``onset_time``, ``score``, ``phase``,
            ``distance``, ``x``, ``y``, ``z``).

        Returns
        -------
        (pick_df, station_df) : the two DataFrames GaMMA's ``association``
        expects. ``pick_df`` keeps the store pick ``id`` so assignments can be
        mapped back to the ``picks`` table.
        """
        config = self.config
        picks_df = picks_df.reset_index(drop=True)

        # association needs station positions: keep only picks that carry x/y/z
        # (channels outside the survey have NaN geometry and cannot associate)
        geom_ok = picks_df[["x", "y", "z"]].notna().all(axis=1)
        if (~geom_ok).any():
            print(
                f"build_inputs: {int((~geom_ok).sum())} of {len(picks_df)} "
                f"picks have no x/y/z geometry -> dropped; associating "
                f"{int(geom_ok.sum())} picks"
            )
            picks_df = picks_df[geom_ok].reset_index(drop=True)

        pick_df = pd.DataFrame(
            {
                "id": picks_df["distance"].astype(str),
                "timestamp": pd.to_datetime(picks_df["onset_time"]),
                "prob": pd.to_numeric(picks_df["score"], errors="coerce").fillna(
                    self._DEFAULT_PROB
                ),
                "type": picks_df["phase"].astype(str).str.lower(),
            }
        )
        # keep the DB pick id (not passed to GaMMA's columns, but rides along
        # so assignments' pick_index can be mapped back to the store)
        pick_df["pick_id"] = (
            picks_df["id"].to_numpy() if "id" in picks_df else np.arange(len(picks_df))
        )

        if not pick_df["type"].isin(["p", "s"]).any() and len(pick_df):
            warnings.warn(
                "no P/S picks in the selection (phases: "
                f"{sorted(pick_df['type'].unique())}); GaMMA association is "
                "intended for P/S pickers (phasenet, *_sb). Results will likely "
                "be empty/meaningless.",
                stacklevel=2,
            )

        # one station per unique distance, geometry from the first pick/station
        stations = (
            picks_df.groupby("distance", as_index=False)
            .first()[["distance", "x", "y", "z"]]
            .sort_values("distance", ignore_index=True)
        )
        nan_mask = stations[["x", "y", "z"]].isna().any(axis=1)
        if nan_mask.any():
            warnings.warn(
                f"dropping {int(nan_mask.sum())} station(s) with NaN geometry "
                "(channels absent from the survey?); their picks will not "
                "associate.",
                stacklevel=2,
            )
            stations = stations[~nan_mask].reset_index(drop=True)

        station_df = pd.DataFrame(
            {
                "id": stations["distance"].astype(str),
                "x(km)": stations["x"] / 1000.0,
                "y(km)": stations["y"] / 1000.0,
                "z(km)": stations["z"] / 1000.0,
            }
        )

        if len(station_df):
            for dim in ("x(km)", "y(km)", "z(km)"):
                if config.get(dim) is None:
                    lo = float(station_df[dim].min()) - self._BOUNDS_PAD_KM
                    hi = float(station_df[dim].max()) + self._BOUNDS_PAD_KM
                    config[dim] = (lo, hi)
            if config.get("bfgs_bounds") is None:
                config["bfgs_bounds"] = (
                    (config["x(km)"][0] - 1, config["x(km)"][1] + 1),
                    (config["y(km)"][0] - 1, config["y(km)"][1] + 1),
                    (0, config["z(km)"][1] + 1),
                    (None, None),  # origin time
                )

        return pick_df, station_df

    def _associate(self, picks_df):
        from gamma.utils import association  # lazy: GaMMA optional at import

        pick_df, station_df = self.build_inputs(picks_df)

        catalogs, assignments = association(
            pick_df, station_df, self.config, method=self.config["method"]
        )

        catalog_df = pd.DataFrame(catalogs)
        # normalize across GaMMA versions: newer releases emit num_* names
        catalog_df = catalog_df.rename(
            columns={
                "num_picks": "number_picks",
                "num_p_picks": "number_p_picks",
                "num_s_picks": "number_s_picks",
            }
        )
        # without amplitudes GaMMA fills magnitude with the 999 placeholder
        if not self.config.get("use_amplitude") and "magnitude" in catalog_df.columns:
            catalog_df["magnitude"] = np.nan

        assignments_df = pd.DataFrame(
            assignments, columns=["pick_index", "event_index", "gamma_score"]
        )
        if len(assignments_df):
            idx = assignments_df["pick_index"].astype(int).to_numpy()
            assignments_df["pick_id"] = pick_df["pick_id"].iloc[idx].to_numpy()
        else:
            assignments_df["pick_id"] = pd.Series(dtype="int64")

        return catalog_df, assignments_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_association(picks_ev, catalog_df, patch=None, title=""):
    """Event locations on the cable geometry | picks-on-data, event-colored.

    Left: the fiber in 3D with one star per event (one color per event).
    Right: picks over time/distance -- light gray = unassociated, colored =
    its event (colors match the stars); drawn over the patch waterfall when
    ``patch`` is given, otherwise on a plain scatter.

    The fiber line comes from the patch's x/y/z coords only, so it adapts to
    whatever cable the patch carries; when no patch (or a patch without
    geometry) is given, the 3D panel shows just the event locations.
    Coordinate convention (from :func:`dasieve.processing.load_survey`):
    x = northing, y = easting, z = depth positive down -- plotted easting
    horizontal, northing vertical, depth axis inverted.

    Parameters
    ----------
    picks_ev : pandas.DataFrame
        Picks with an ``event_index`` column (NaN = unassociated), i.e. the
        associator's picks joined to its assignments on pick id.
    catalog_df : pandas.DataFrame
        Event catalog with ``event_index``, ``x(km)``, ``y(km)``, ``z(km)``.
    patch : dascore.Patch, optional
        Waterfall background (and fiber geometry, if it carries x/y/z).
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    picks_ev = picks_ev.copy()
    times = pd.to_datetime(picks_ev["onset_time"])

    # fiber geometry (m -> km) from the patch only; without it the 3D panel
    # shows event locations alone
    geom = None
    if patch is not None and {"x", "y", "z"} <= set(patch.coords.coord_map):
        arrs = [np.asarray(patch.coords.get_array(c), float)
                for c in ("x", "y", "z")]
        if np.isfinite(arrs[0]).any():
            geom = arrs

    event_ids = sorted(picks_ev["event_index"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    ev_color = {ev: cmap(k % 10) for k, ev in enumerate(event_ids)}

    fig = plt.figure(figsize=(15, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_w = fig.add_subplot(1, 2, 2)

    # --- left: fiber (if the patch has geometry) + event locations
    east_parts, north_parts, dep_parts = [], [], []
    if geom is not None:
        f_north, f_east, f_dep = (g / 1000.0 for g in geom)
        ax3d.plot(f_east, f_north, f_dep, color="0.4", lw=1.5, label="fiber")
        east_parts.append(f_east)
        north_parts.append(f_north)
        dep_parts.append(f_dep)
    for ev in event_ids:
        row = catalog_df[catalog_df["event_index"] == ev].iloc[0]
        ax3d.scatter(row["y(km)"], row["x(km)"], row["z(km)"], s=120,
                     marker="*", color=ev_color[ev], edgecolor="k",
                     linewidths=0.5, label=f"event {int(ev)}")
    if len(catalog_df):
        east_parts.append(catalog_df["y(km)"].to_numpy(float))
        north_parts.append(catalog_df["x(km)"].to_numpy(float))
        dep_parts.append(catalog_df["z(km)"].to_numpy(float))

    # per-cable extent with true spatial proportions
    if east_parts:
        lims = [
            (np.nanmin(np.concatenate(v)) - 0.1,
             np.nanmax(np.concatenate(v)) + 0.1)
            for v in (east_parts, north_parts, dep_parts)
        ]
        ax3d.set_xlim(lims[0])
        ax3d.set_ylim(lims[1])
        ax3d.set_zlim(lims[2])
        spans = [max(hi - lo, 1e-3) for lo, hi in lims]
        ax3d.set_box_aspect(spans)
        # tick count follows each axis's drawn length; full values (no offset)
        for axis, span in zip((ax3d.xaxis, ax3d.yaxis, ax3d.zaxis), spans):
            nbins = int(np.clip(round(6 * np.sqrt(span / max(spans))), 3, 5))
            axis.set_major_locator(MaxNLocator(nbins))
            fmt = axis.get_major_formatter()
            if isinstance(fmt, ScalarFormatter):
                fmt.set_useOffset(False)
    ax3d.invert_zaxis()  # z is depth: shallowest on top
    ax3d.tick_params(labelsize=7, pad=0)
    ax3d.set_xlabel("Easting (km)", fontsize=8, labelpad=4)
    ax3d.set_ylabel("Northing (km)", fontsize=8, labelpad=4)
    ax3d.set_zlabel("depth (km)", fontsize=8, labelpad=2)
    ax3d.set_title(f"{title}: event locations" if title else "event locations",
                   fontsize=10)
    ax3d.legend(fontsize=7, loc="upper left")

    # --- right: picks on data
    if patch is not None:
        p_time = patch.coords.get_array("time")
        p_dist = patch.coords.get_array("distance")
        t0 = pd.Timestamp(p_time[0])
        pt = (p_time - p_time[0]) / np.timedelta64(1, "s")
        data2d = np.moveaxis(
            patch.data,
            [patch.dims.index("distance"), patch.dims.index("time")], [0, 1],
        )
        vmax = np.percentile(np.abs(data2d), 99)
        ax_w.imshow(data2d, aspect="auto", cmap="gray",
                    extent=(pt[0], pt[-1], p_dist[-1], p_dist[0]),
                    vmin=-vmax, vmax=vmax, interpolation="nearest")
    else:
        t0 = times.min()
        ax_w.invert_yaxis()
    picks_ev["t_s"] = (times - t0).dt.total_seconds()

    un = picks_ev[picks_ev["event_index"].isna()]
    if len(un):
        ax_w.scatter(un["t_s"], un["distance"], marker="|", s=25,
                     linewidths=0.8, color="0.75",
                     label=f"unassociated ({len(un)})", zorder=3)
    asc = picks_ev.dropna(subset=["event_index"])
    if len(asc):
        ax_w.scatter(asc["t_s"], asc["distance"], marker="|", s=40,
                     linewidths=1.2, zorder=4,
                     c=[ev_color[e] for e in asc["event_index"]])
    for ev in event_ids:
        n = int((asc["event_index"] == ev).sum())
        ax_w.scatter([], [], marker="|", s=40, linewidths=1.2,
                     color=ev_color[ev], label=f"event {int(ev)} ({n})")
    ax_w.set_xlabel("Time (s)")
    ax_w.set_ylabel("Distance (m)")
    ax_w.set_title(
        (f"{title}: " if title else "")
        + "picks on data (gray = unassociated)"
    )
    ax_w.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Registry + convenience wrapper
# ---------------------------------------------------------------------------
#: method id -> associator class. New associators register here so callers can
#: select one by name without importing the class directly.
ASSOCIATORS = {GammaAssociator.name: GammaAssociator}


def get_associator(method="gamma", flag="default", **overrides):
    """Build an associator by name from a preset (see each class's PRESETS)."""
    try:
        cls = ASSOCIATORS[method]
    except KeyError:
        raise NotImplementedError(
            f"associator method {method!r} is not implemented; "
            f"available: {sorted(ASSOCIATORS)}"
        )
    return cls.from_preset(flag, **overrides)


def run_association(
    db_path=DEFAULT_DB_PATH,
    *,
    config_flag="default",
    save=True,
    method="gamma",
    picks=None,
    detector=None,
    windows=None,
    pick_method=None,
    pick_starttime=None,
    pick_endtime=None,
    phase=None,
    file_name=None,
    min_score=None,
    plot=False,
    patch=None,
    **config_overrides,
):
    """Build an associator, select picks, associate, and (optionally) save in
    one call.

    ``method`` selects the associator (see :data:`ASSOCIATORS`); ``config_flag``
    and ``**config_overrides`` are forwarded to its ``from_preset``; ``picks``
    (a picker DataFrame, associated directly with no database read/write),
    ``detector`` / ``windows`` (detection-window gating) and the remaining
    pick filters are forwarded to :meth:`BaseAssociator.run`.

    Returns ``(catalog_df, assignments_df)`` -- see :meth:`BaseAssociator.run`.
    """
    assoc = get_associator(method, config_flag, **config_overrides)
    return assoc.run(
        db_path,
        picks=picks,
        detector=detector,
        windows=windows,
        pick_method=pick_method,
        pick_starttime=pick_starttime,
        pick_endtime=pick_endtime,
        phase=phase,
        file_name=file_name,
        min_score=min_score,
        db_save=save,
        plot=plot,
        patch=patch,
    )
