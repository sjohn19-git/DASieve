"""Phase association for DAS picks.

Takes P/S picks from the SQLite store (:mod:`dasieve.store`), treats each
DAS channel (unique ``distance``) as a station using the x/y/z geometry the
pickers persisted, and clusters the picks into origins. Results are written
to three tables in the same store database (default
``~/DASieve/dasieve.sqlite``):

* ``origins``      -- one row per located hypocentre (origin time, location)
* ``associations`` -- one row per (pick, origin) link, carrying the
                      ``pick_id`` back into the ``picks`` table
* ``events``       -- one per origin, pointing at it as the preferred origin

Unlike the picks table, none of these carries a cable / time-window / method
run key: an origin is identified by the picks it was built from, reached
through ``associations``. Re-running an associator therefore replaces its
previous results by superseding the associations on the picks it just ran
over -- see :func:`dasieve.store.save_associations`.

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
    origins_df, associations_df = assoc.run(
        pick_method="phasenetdas", cable_id="16BConst", min_probability=0.3,
    )   # saves origins/associations/events automatically (db_save=True),
        # superseding this associator's previous results on these picks

Picks can also be supplied directly as a DataFrame (e.g. straight from a
picker, without going through the store). This path never touches the
database -- the origins/associations tables are only returned::

    df = sieve.picking.trigger_picker(patch, db_save=False)
    origins_df, associations_df = assoc.run(picks=df)

GaMMA itself is imported lazily inside :meth:`GammaAssociator._associate`, so
importing ``dasieve`` does not require it to be installed.
"""

import copy
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .store import (
    ASSOCIATION_COLUMNS,
    DEFAULT_DB_PATH,
    ORIGIN_COLUMNS,
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
    ``association_method`` column of the associations table.
    """

    #: Short associator id, e.g. "gamma"; set by each subclass.
    name: str = ""

    def __init__(self, config):
        # copied, not aliased: update_config writes to self.config, so a
        # caller-supplied dict must not be mutated behind their back
        self.config = copy.deepcopy(config)

    @classmethod
    @abstractmethod
    def from_preset(cls, flag="default", **overrides):
        """Build an associator from a named preset, with optional overrides."""

    @contextmanager
    def _run_scope(self, picks_df):
        """Per-run setup over the *whole* pick set, before any windowing.

        Subclasses override this to derive run-wide state once -- e.g. GaMMA's
        spatial search bounds, which must cover the full cable rather than
        whichever channels happened to pick in the first detection window.
        The base implementation does nothing."""
        yield

    def update_config(self, **changes):
        """Edit this associator's config in place; returns ``self`` so calls
        can be chained (``assoc.update_config(...).run(...)``)."""
        self.config.update(changes)
        return self

    @abstractmethod
    def _associate(self, picks_df):
        """Associate a picks DataFrame into origins.

        Returns ``(origins_df, associations_df)`` with
        :data:`dasieve.store.ORIGIN_COLUMNS` and
        :data:`dasieve.store.ASSOCIATION_COLUMNS`; ``associations_df`` has a
        ``pick_id`` column mapping each row back to ``picks.id`` and an
        ``origin_index`` linking it to a row of ``origins_df``."""
        raise NotImplementedError

    def run(
        self,
        db_path=DEFAULT_DB_PATH,
        *,
        picks=None,
        windows=None,
        pick_method=None,
        phase=None,
        cable_id=None,
        time_window=None,
        min_probability=None,
        db_save=True,
        plot=False,
        patch=None,
        cmap="gray",
    ):
        """Associate picks -- from a DataFrame, or selected from the store.

        When ``picks`` is given, that DataFrame is associated directly: the
        database is not touched at all (no selection, no save; the pick
        filters and ``db_save`` are ignored) and the origins/associations
        tables are only returned.

        Otherwise selection is delegated to
        :func:`dasieve.store.select_pick_ids`; the number of selected picks is
        printed before association runs. When ``db_save`` is True (default)
        results are written via :func:`dasieve.store.save_associations`, which
        first supersedes this associator's existing associations on the
        selected picks and cascades that to their origins and events. So
        re-running the same associator over the same picks leaves one set of
        origins, while another associator's results on those picks are left
        alone.

        Either way, association can be gated by detection windows: pass
        ``windows`` (e.g. from :meth:`dasieve.detection.EventDetector.detect`).
        Association then runs once per window on the picks inside it; picks
        outside every window are never associated. ``origin_index`` stays
        unique across windows.

        Parameters
        ----------
        db_path : str
            Catalog database holding the picks.
        picks : pandas.DataFrame, optional
            Picks to associate directly, in the pickers' output schema
            (:data:`dasieve.picking.PICK_COLUMNS`), e.g. the DataFrame
            returned by :func:`dasieve.picking.trigger_picker` or
            :func:`dasieve.picking.phasenet_das_picker`. If there is no ``id``
            column, row positions are used, so ``associations_df["pick_id"]``
            indexes back into ``picks`` rows.
        windows : pandas.DataFrame or iterable of (start, end), optional
            Precomputed association windows: a ``t_start``/``t_end`` DataFrame
            as returned by :meth:`EventDetector.detect`, or an iterable of
            (start, end) pairs. This is the only time knob on ``run()`` -- it
            both restricts which picks are associated and segments them, since
            each window is associated independently. To pre-filter picks by
            their own onset time instead, select them with
            :func:`dasieve.store.load_picks` and pass ``picks=``.
        pick_method : str or list of str, optional
            Which picker's picks to associate (the ``picks.pick_method``
            column, e.g. "phasenetdas") -- NOT the associator.
        phase, cable_id, time_window, min_probability : optional
            Further filters, forwarded to
            :func:`dasieve.store.select_pick_ids`. ``cable_id`` may name one
            fiber or a list of them, and ``time_window`` one
            ``(file_starttime, file_endtime)`` window or a list of them; picks
            from all of them are associated together, as one run.

            The selection also scopes the replace: everything this associator
            previously wrote on the selected picks is superseded, whether or
            not this run re-associates each one.
        db_save : bool
            If True (default), save results to the store. Only applies when
            picks come from the store.
        plot : bool
            Draw :func:`plot_association` for the results: origin locations on
            the cable geometry next to the picks-on-data view colored by
            origin. Geometry comes from the picks' x/y/z, so it adapts to the
            cable at hand.
        patch : dascore.Patch, optional
            Only used when ``plot=True``, as the waterfall background.
        cmap : str
            Waterfall colormap for the plot background.

        Returns
        -------
        (origins_df, associations_df)
            ``origins_df`` -- :data:`dasieve.store.ORIGIN_COLUMNS`:
            ``origin_index``, ``origin_method``, ``origin_time``, ``x``,
            ``y``, ``z`` (meters), ``number_picks``.
            ``associations_df`` -- :data:`dasieve.store.ASSOCIATION_COLUMNS`:
            ``pick_id`` (back into ``picks.id``, or the row position in
            ``picks`` when a DataFrame was supplied without an ``id``),
            ``origin_index``, ``association_method``, ``probability``.
        """
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
                pick_method=pick_method,
                phase=phase,
                cable_id=cable_id,
                time_window=time_window,
                min_probability=min_probability,
            )
            print(
                f"{self.name}.run: {len(picks_df)} picks selected "
                f"(pick_method={pick_method!r}, cable_id={cable_id!r}, "
                f"time_window={time_window!r})"
            )
            from_store = True

        # run-wide setup (e.g. GaMMA's search bounds) is derived here, from
        # the full pick set, so windowing cannot narrow it
        with self._run_scope(picks_df):
            if windows is None:
                origins_df, associations_df = self._associate(picks_df)
            else:
                origins_df, associations_df = self._associate_windowed(
                    picks_df, windows
                )

        if from_store and db_save:
            # the replace scope is the whole selection, not just the picks
            # that ended up associated: a rerun associating fewer picks must
            # still supersede what the previous run built from them
            n_or, n_as, n_ev = save_associations(
                origins_df, associations_df, db_path,
                association_method=self.name,
                pick_ids=picks_df["id"].tolist(),
            )
            print(
                f"{self.name}.run: saved {n_or} origins / {n_as} associations "
                f"/ {n_ev} events (association_method={self.name!r})"
            )

        if plot:
            picks_ev = picks_df.merge(
                associations_df[["pick_id", "origin_index"]],
                left_on="id", right_on="pick_id", how="left",
            )
            plot_association(picks_ev, origins_df, patch=patch,
                             title=self.name, cmap=cmap)

        return origins_df, associations_df

    def _associate_windowed(self, picks_df, windows):
        """Associate once per detection window; concatenate the results.

        ``windows`` is a DataFrame with ``t_start`` / ``t_end`` columns (as
        emitted by :meth:`EventDetector.detect`) or an iterable of
        (start, end) pairs. Picks outside every window are never associated.
        Each window's ``origin_index`` is offset by a running counter so
        indices stay unique across windows.
        """
        if not isinstance(windows, pd.DataFrame):
            windows = pd.DataFrame(list(windows), columns=["t_start", "t_end"])
        times = pd.to_datetime(picks_df["onset_time"])

        origins, assocs = [], []
        offset = 0
        for _, w in windows.iterrows():
            t_lo, t_hi = pd.Timestamp(w["t_start"]), pd.Timestamp(w["t_end"])
            sub = picks_df[(times >= t_lo) & (times < t_hi)]
            if sub.empty:
                continue
            org, asc = self._associate(sub.reset_index(drop=True))
            if len(org):
                org = org.copy()
                asc = asc.copy()
                org["origin_index"] = org["origin_index"] + offset
                asc["origin_index"] = asc["origin_index"] + offset
                offset = int(org["origin_index"].max()) + 1
            origins.append(org)
            assocs.append(asc)

        origins_df = (
            pd.concat(origins, ignore_index=True) if origins
            else pd.DataFrame(columns=ORIGIN_COLUMNS)
        )
        associations_df = (
            pd.concat(assocs, ignore_index=True) if assocs
            else pd.DataFrame(columns=ASSOCIATION_COLUMNS)
        )
        print(
            f"{self.name}: {len(windows)} window(s) -> {len(origins_df)} "
            f"origins / {len(associations_df)} associations"
        )
        return origins_df, associations_df


# ---------------------------------------------------------------------------
# GaMMA associator
# ---------------------------------------------------------------------------
class GammaAssociator(BaseAssociator):
    """GaMMA (Gaussian-mixture) phase associator.

    Config presets hold GaMMA's non-geometric knobs only; the spatial search
    bounds ("x(km)", "y(km)", "z(km)", "bfgs_bounds") are derived from the
    station geometry unless supplied explicitly via :meth:`from_preset` /
    :meth:`update_config`.

    Derivation happens once per :meth:`run`, over that run's whole pick set
    (:meth:`_run_scope`), and the result is *not* written back into ``config``
    -- so every detection window shares one box covering the cable, and a
    later run on different data (another cable, another window set) derives
    its own. The box actually used is readable afterwards as ``last_bounds``.
    Bounds the user supplies are never overwritten."""

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

    #: Spatial bounds derived for the run in progress (see :meth:`_run_scope`);
    #: None outside a run. ``last_bounds`` keeps the most recently derived set
    #: for inspection -- the derivation itself never writes to ``config``.
    _derived_bounds = None
    last_bounds = None

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
        "z(km)", "bfgs_bounds") may be passed here, and are then used as given
        on every run; otherwise they are auto-derived from station geometry
        once per run (see :meth:`_run_scope`).
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

    def _station_frame(self, picks_df):
        """One station per unique ``distance``, positioned in km.

        Each unique ``distance`` is a station whose id is the distance value
        (as string) and whose position is the pick's x/y/z geometry (meters ->
        km; no lat/lon projection, the survey is already in local meters).
        Picks without x/y/z cannot place a station and are dropped -- channels
        outside the survey carry NaN geometry.

        Shared by :meth:`build_inputs` and :meth:`_derive_bounds` so the
        bounds are always derived from exactly the stations that will be
        associated."""
        geom_ok = picks_df[["x", "y", "z"]].notna().all(axis=1)
        stations = (
            picks_df[geom_ok]
            .groupby("distance", as_index=False)
            .first()[["distance", "x", "y", "z"]]
            .sort_values("distance", ignore_index=True)
        )
        return pd.DataFrame(
            {
                "id": stations["distance"].astype(str),
                "x(km)": stations["x"] / 1000.0,
                "y(km)": stations["y"] / 1000.0,
                "z(km)": stations["z"] / 1000.0,
            }
        )

    def _derive_bounds(self, station_df):
        """Spatial search bounds from the station extent, as a plain dict.

        Only fills dims the *user* left unset. The guard reads ``self.config``,
        which the derivation never writes to, so ``None`` unambiguously means
        "not supplied" -- rather than "not computed yet" """
        if not len(station_df):
            return {}
        out = {}
        for dim in ("x(km)", "y(km)", "z(km)"):
            if self.config.get(dim) is None:
                out[dim] = (
                    float(station_df[dim].min()) - self._BOUNDS_PAD_KM,
                    float(station_df[dim].max()) + self._BOUNDS_PAD_KM,
                )
        if self.config.get("bfgs_bounds") is None:
            x = out.get("x(km)", self.config.get("x(km)"))
            y = out.get("y(km)", self.config.get("y(km)"))
            z = out.get("z(km)", self.config.get("z(km)"))
            if x is not None and y is not None and z is not None:
                out["bfgs_bounds"] = (
                    (x[0] - 1, x[1] + 1),
                    (y[0] - 1, y[1] + 1),
                    (0, z[1] + 1),
                    (None, None),  # origin time
                )
        return out

    def _effective_config(self, station_df):
        """User config plus this run's derived bounds, as a fresh dict.

        Prefers the bounds derived once per run by :meth:`_run_scope`; when
        called outside a run (a direct :meth:`build_inputs` / :meth:`_associate`)
        it falls back to deriving from the stations at hand."""
        bounds = self._derived_bounds
        if bounds is None:
            bounds = self._derive_bounds(station_df)
        return {**self.config, **bounds}

    @contextmanager
    def _run_scope(self, picks_df):
        """Derive the spatial search bounds once, from the whole pick set.

        Every window in the run then shares one box covering the whole cable,
        instead of each window re-deriving from its own picks -- which let
        whichever channels picked in the first window constrain all the later
        ones, pinning their events to that box's edge. The bounds are dropped
        on exit (kept only as ``last_bounds`` for inspection) so the next run
        re-derives for its own data."""
        self._derived_bounds = self._derive_bounds(self._station_frame(picks_df))
        try:
            yield
        finally:
            self.last_bounds = self._derived_bounds
            self._derived_bounds = None

    def build_inputs(self, picks_df):
        """Build GaMMA's ``pick_df`` / ``station_df`` from store picks.

        Each unique ``distance`` becomes a station (see
        :meth:`_station_frame`); picks with NaN geometry are dropped.

        This does *not* touch the spatial search bounds -- those are resolved
        separately by :meth:`_derive_bounds` / :meth:`_effective_config`, and
        this associator's ``config`` is never modified.

        Parameters
        ----------
        picks_df : pandas.DataFrame
            Picks as returned by :func:`dasieve.store.load_picks_by_ids`
            (must include ``id``, ``onset_time``, ``probability``,
            ``phase``, ``distance``, ``x``, ``y``, ``z``).

        Returns
        -------
        (pick_df, station_df) : the two DataFrames GaMMA's ``association``
        expects. ``pick_df`` keeps the store pick ``id`` so associations can
        be mapped back to the ``picks`` table.
        """
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
                "prob": pd.to_numeric(
                    picks_df["probability"], errors="coerce"
                ).fillna(self._DEFAULT_PROB),
                "type": picks_df["phase"].astype(str).str.lower(),
            }
        )
        # keep the DB pick id (not passed to GaMMA's columns, but rides along
        # so associations' pick_index can be mapped back to the store)
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

        # one station per unique distance. The search bounds are resolved
        # separately (_derive_bounds / _effective_config), so nothing here
        # writes to self.config.
        station_df = self._station_frame(picks_df)

        return pick_df, station_df

    def _associate(self, picks_df):
        from gamma.utils import association  # lazy: GaMMA optional at import

        pick_df, station_df = self.build_inputs(picks_df)
        # user config + the run's derived bounds; self.config stays untouched
        config = self._effective_config(station_df)

        catalogs, assignments = association(
            pick_df, station_df, config, method=config["method"]
        )

        catalog_df = pd.DataFrame(catalogs)
        if not len(catalog_df):
            return (pd.DataFrame(columns=ORIGIN_COLUMNS),
                    pd.DataFrame(columns=ASSOCIATION_COLUMNS))

        # normalize across GaMMA versions: newer releases emit num_* names
        catalog_df = catalog_df.rename(
            columns={"num_picks": "number_picks", "event_index": "origin_index"}
        )
        # GaMMA locates in km; everything else in dasieve (picks' x/y/z, the
        # store, the plots) is in meters -- convert on the way out
        for src, dst in (("x(km)", "x"), ("y(km)", "y"), ("z(km)", "z")):
            if src in catalog_df.columns:
                catalog_df[dst] = catalog_df.pop(src) * 1000.0

        origins_df = pd.DataFrame(
            {
                "origin_index": catalog_df["origin_index"],
                "origin_method": self.name,
                "origin_time": pd.to_datetime(catalog_df.get("time"),
                                              errors="coerce"),
                "x": catalog_df.get("x"),
                "y": catalog_df.get("y"),
                "z": catalog_df.get("z"),
                "number_picks": catalog_df.get("number_picks"),
            }
        )
        # Without amplitudes GaMMA fills magnitude with a 999 placeholder, so
        # it is only a real magnitude when use_amplitude is on. It rides along
        # as an extra column for the event row; NaN otherwise.
        origins_df["magnitude"] = (
            catalog_df.get("magnitude") if self.config.get("use_amplitude")
            else np.nan
        )

        # GaMMA returns bare (pick_index, event_index, score) tuples. NOTE the
        # third element is a per-pick mixture likelihood *density* (unbounded,
        # unit-dependent), not a normalized probability -- it is stored in the
        # associations.probability column as-is, so compare it only against
        # other values from the same run.
        raw = pd.DataFrame(
            assignments, columns=["pick_index", "origin_index", "probability"]
        )
        associations_df = pd.DataFrame(columns=ASSOCIATION_COLUMNS)
        if len(raw):
            idx = raw["pick_index"].astype(int).to_numpy()
            associations_df = pd.DataFrame(
                {
                    "pick_id": pick_df["pick_id"].iloc[idx].to_numpy(),
                    "origin_index": raw["origin_index"].to_numpy(),
                    "association_method": self.name,
                    "probability": raw["probability"].to_numpy(),
                }
            )

        return origins_df, associations_df










# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_association(picks_ev, origins_df, patch=None, title="", cmap="gray"):
    """Origin locations on the cable geometry | picks-on-data, origin-colored.

    Left: the fiber in 3D with one star per origin (one color per origin).
    Right: picks over time/distance -- light gray = unassociated, colored =
    its origin (colors match the stars); drawn over the patch waterfall when
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
        Picks with an ``origin_index`` column (NaN = unassociated), i.e. the
        associator's picks joined to its associations on pick id.
    origins_df : pandas.DataFrame
        Origins with ``origin_index`` and ``x``, ``y``, ``z`` (meters).
    patch : dascore.Patch, optional
        Waterfall background (and fiber geometry, if it carries x/y/z).
    cmap : str
        Waterfall colormap for the background.
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

    event_ids = sorted(picks_ev["origin_index"].dropna().unique())
    ev_cmap = plt.get_cmap("tab10")
    ev_color = {ev: ev_cmap(k % 10) for k, ev in enumerate(event_ids)}

    fig = plt.figure(figsize=(15, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_w = fig.add_subplot(1, 2, 2)

    # --- left: fiber (if the patch has geometry) + event locations
    east_parts, north_parts, dep_parts = [], [], []
    if geom is not None:
        f_north, f_east, f_dep = geom
        ax3d.plot(f_east, f_north, f_dep, color="0.4", lw=1.5, label="fiber")
        east_parts.append(f_east)
        north_parts.append(f_north)
        dep_parts.append(f_dep)
    for ev in event_ids:
        row = origins_df[origins_df["origin_index"] == ev].iloc[0]
        ax3d.scatter(row["y"], row["x"], row["z"], s=120,
                     marker="*", color=ev_color[ev], edgecolor="k",
                     linewidths=0.5, label=f"origin {int(ev)}")
    if len(origins_df):
        east_parts.append(origins_df["y"].to_numpy(float))
        north_parts.append(origins_df["x"].to_numpy(float))
        dep_parts.append(origins_df["z"].to_numpy(float))

    # per-cable extent with true spatial proportions
    if east_parts:
        lims = [
            (np.nanmin(np.concatenate(v)) - 100.0,
             np.nanmax(np.concatenate(v)) + 100.0)
            for v in (east_parts, north_parts, dep_parts)
        ]
        ax3d.set_xlim(lims[0])
        ax3d.set_ylim(lims[1])
        ax3d.set_zlim(lims[2])
        spans = [max(hi - lo, 1.0) for lo, hi in lims]
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
    ax3d.set_xlabel("Easting (m)", fontsize=8, labelpad=4)
    ax3d.set_ylabel("Northing (m)", fontsize=8, labelpad=4)
    ax3d.set_zlabel("depth (m)", fontsize=8, labelpad=2)
    ax3d.set_title(f"{title}: origin locations" if title else "origin locations",
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
        ax_w.imshow(data2d, aspect="auto", cmap=cmap,
                    extent=(pt[0], pt[-1], p_dist[-1], p_dist[0]),
                    vmin=-vmax, vmax=vmax, interpolation="nearest")
    else:
        t0 = times.min()
        ax_w.invert_yaxis()
    picks_ev["t_s"] = (times - t0).dt.total_seconds()

    un = picks_ev[picks_ev["origin_index"].isna()]
    if len(un):
        ax_w.scatter(un["t_s"], un["distance"], marker="|", s=25,
                     linewidths=0.8, color="0.75",
                     label=f"unassociated ({len(un)})", zorder=3)
    asc = picks_ev.dropna(subset=["origin_index"])
    if len(asc):
        ax_w.scatter(asc["t_s"], asc["distance"], marker="|", s=40,
                     linewidths=1.2, zorder=4,
                     c=[ev_color[e] for e in asc["origin_index"]])
    for ev in event_ids:
        n = int((asc["origin_index"] == ev).sum())
        ax_w.scatter([], [], marker="|", s=40, linewidths=1.2,
                     color=ev_color[ev], label=f"origin {int(ev)} ({n})")
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