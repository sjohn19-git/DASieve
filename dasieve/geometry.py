"""
geometry.py

Reads iDAS HDF5 files, attaches geometry to each channel, and writes output
patches in DASDAE format.

Mirrors the geometry-assignment step in Michal/realTimeDASnextGen.m:
  1. Read the FULL file (all channels) first.
  2. Assign X, Y, Z coordinates to every channel via patch.update_coords().
     - "survey" mode: load a CSV mapping channel index → (x, y, z) from a
       well-survey file (same as passing chanNum/chanx/chany/chanz in MATLAB).
     - "default" mode: assume a vertical borehole at the origin —
       x=0, y=0, z=optical_distance along fiber.
  3. Optionally subset to a channel window [ch_start, ch_end] AFTER geometry
     is embedded (so the spatial labels stay consistent with the channel slice).
  4. Write the geometry-annotated patch to disk.

Usage:
    # Manual mode — process specific files
    python geometry_ingest.py manual /path/to/output file1.h5 file2.h5

    # Batch mode — process all .h5 files in a directory
    python geometry_ingest.py batch /path/to/raw /path/to/output

    # Watch mode — monitor directory for new .h5 files continuously
    python geometry_ingest.py watch /path/to/raw /path/to/output

Survey CSV format (--survey flag):
    channel,x_m,y_m,z_m
    0,0.0,0.0,0.0
    1,0.0,0.0,1.0
    ...

Authors: Sebin John, 2025
"""

import os
import sys
import time
import glob
import logging
import argparse
from pathlib import Path

import numpy as np

try:
    import dascore as dc
    from dascore.core import get_coord
except ImportError:
    sys.exit("dascore is required: pip install dascore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Survey loader
# ---------------------------------------------------------------------------

def load_survey(survey_path: str) -> dict:
    """
    Load a well-survey CSV into a dict mapping channel index → (x, y, z).

    Expected CSV columns: channel, x_m, y_m, z_m
    (header row required; units are meters).

    Returns
    -------
    dict : {channel_int: (x_float, y_float, z_float)}
    """
    import csv
    survey = {}
    with open(survey_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ch = int(row["channel"])
            survey[ch] = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
    log.info(f"Loaded survey with {len(survey)} channel entries from {survey_path}")
    return survey


# ---------------------------------------------------------------------------
# Geometry assignment
# ---------------------------------------------------------------------------

def assign_geometry(patch: "dc.Patch", survey: dict | None = None) -> "dc.Patch":
    """
    Attach X, Y, Z spatial coordinates to every channel of a DASCore Patch.

    Mirrors the geometry-assignment block in realTimeDASnextGen.m (lines 287-323):

      - If `survey` is provided (dict of {channel_index: (x, y, z)}):
          Map each fiber channel (indexed by its position in the distance array)
          to its survey (x, y, z). Channels absent from the survey get NaN.
      - If no survey: assume a simple vertical borehole at the origin.
          x = 0, y = 0, z = optical_distance_along_fiber (in meters).

    Parameters
    ----------
    patch : dc.Patch
        Input patch. Must have a 'distance' dimension (standard for iDAS data).
    survey : dict or None
        {channel_index (int): (x_m, y_m, z_m)} mapping from load_survey().
        If None, the default vertical-borehole geometry is used.

    Returns
    -------
    dc.Patch
        New patch with 'x', 'y', 'z' non-dimensional coordinates attached to
        the 'distance' dimension.
    """
    dist_arr = patch.get_array("distance")  # shape: (n_channels,)
    n_ch = len(dist_arr)

    if survey is not None:
        # --- Survey mode: map channel index → (x, y, z) ---
        # Channel index here is the position in the distance array (0-based),
        # matching MATLAB's chanNum which refers to the original channel number.
        x_arr = np.full(n_ch, np.nan)
        y_arr = np.full(n_ch, np.nan)
        z_arr = np.full(n_ch, np.nan)

        for ch_idx, (x, y, z) in survey.items():
            if 0 <= ch_idx < n_ch:
                x_arr[ch_idx] = x
                y_arr[ch_idx] = y
                z_arr[ch_idx] = z
            else:
                log.warning(
                    f"Survey channel {ch_idx} is outside patch range [0, {n_ch-1}] — skipped"
                )

        n_assigned = int(np.sum(~np.isnan(x_arr)))
        log.info(f"Survey geometry: {n_assigned}/{n_ch} channels assigned")

    else:
        # --- Default mode: simple vertical borehole at origin ---
        # Matches MATLAB fallback: fchanX=0, fchanY=0, fchanZ=channel_index
        # We use the optical distance along fiber as Z (meters), x=y=0.
        x_arr = np.zeros(n_ch)
        y_arr = np.zeros(n_ch)
        z_arr = dist_arr.astype(float)  # along-fiber distance = depth proxy
        log.info(f"Default geometry: vertical borehole, z = optical distance (0 to {z_arr[-1]:.1f} m)")

    # Attach as non-dimensional coords associated with the 'distance' dimension.
    # This is the DASCore equivalent of writing to DSI trace headers 35, 37, 39.
    patch = patch.update_coords(
        x=("distance", x_arr),
        y=("distance", y_arr),
        z=("distance", z_arr),
    )

    return patch


# ---------------------------------------------------------------------------
# Channel subsetting (after geometry assignment)
# ---------------------------------------------------------------------------

def subset_channels(
    patch: "dc.Patch",
    ch_start: int | None = None,
    ch_end: int | None = None,
) -> "dc.Patch":
    """
    Subset the patch to a channel window [ch_start, ch_end] (0-based, inclusive).

    Must be called AFTER assign_geometry() so that X/Y/Z coords are already
    embedded and will be correctly sliced along with the data.
    Mirrors MATLAB's subsetDSI(fullDSI, tr1, tr2, ...) call in nextGen.

    Parameters
    ----------
    patch : dc.Patch
        Geometry-annotated patch.
    ch_start : int or None
        First channel index to keep (0-based). None = start of array.
    ch_end : int or None
        Last channel index to keep (0-based, inclusive). None = end of array.

    Returns
    -------
    dc.Patch
        Subset patch with geometry coords intact.
    """
    if ch_start is None and ch_end is None:
        return patch  # nothing to do

    # Convert channel indices to sample-based select
    # DASCore select with samples=True treats indices as positions (0-based)
    s = ch_start if ch_start is not None else 0
    e = ch_end   if ch_end   is not None else len(patch.get_array("distance")) - 1

    patch = patch.select(distance=(s, e), samples=True)
    log.info(f"Subset to channels [{s}, {e}] → {patch.channel_count} channels remain")
    return patch


# ---------------------------------------------------------------------------
# Core processing: read → assign geometry → subset → write
# ---------------------------------------------------------------------------

def process_file(
    h5_path: str,
    output_dir: str,
    survey: dict | None = None,
    ch_start: int | None = None,
    ch_end: int | None = None,
    out_format: str = "dasdae",
) -> str | None:
    """
    Read one iDAS HDF5 file, assign geometry, subset channels, and write output.

    Parameters
    ----------
    h5_path : str
        Path to the iDAS HDF5 file.
    output_dir : str
        Directory to write the output file.
    survey : dict or None
        Channel→(x,y,z) mapping. None = default vertical-borehole geometry.
    ch_start : int or None
        First channel to keep (0-based). None = keep all.
    ch_end : int or None
        Last channel to keep (0-based, inclusive). None = keep all.
    out_format : str
        DASCore output format. Default is 'dasdae'.

    Returns
    -------
    str or None
        Path to the written output file, or None on failure.
    """
    h5_path = str(h5_path)
    stem = Path(h5_path).stem

    try:
        # --- Step 1: Read FULL file (all channels) ---
        # Mirrors MATLAB: fullDSI = localDashdf2dsi(file1)  [no tr1/tr2 yet]
        log.info(f"Reading: {h5_path}")
        spool = dc.spool(h5_path)
        patches = list(spool)
        if not patches:
            log.warning(f"No patches found in {h5_path} — skipping")
            return None
        patch = patches[0]
        log.info(
            f"  {patch.channel_count} channels, "
            f"{patch.seconds:.2f}s, "
            f"distance [{patch.coords.min('distance'):.1f}, "
            f"{patch.coords.max('distance'):.1f}] m"
        )

        # --- Step 2: Assign geometry to ALL channels ---
        # Must happen before subsetting — mirrors MATLAB geometry block.
        patch = assign_geometry(patch, survey=survey)

        # --- Step 3: Subset channels (with geometry already embedded) ---
        # Mirrors: fullDSI2 = subsetDSI(fullDSI, tr1, tr2, ...)
        patch = subset_channels(patch, ch_start=ch_start, ch_end=ch_end)

        # --- Step 4: Write output ---
        os.makedirs(output_dir, exist_ok=True)
        out_ext  = ".h5" if out_format == "dasdae" else f".{out_format}"
        out_path = os.path.join(output_dir, stem + "_geom" + out_ext)

        dc.write(patch, out_path, file_format=out_format)
        log.info(f"  Written: {out_path}")
        return out_path

    except Exception as exc:
        log.error(f"Failed processing {h5_path}: {exc}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_manual(args):
    """Process a list of explicitly specified files."""
    survey = load_survey(args.survey) if args.survey else None
    for h5_path in args.files:
        process_file(
            h5_path,
            args.output_dir,
            survey=survey,
            ch_start=args.ch_start,
            ch_end=args.ch_end,
        )


def run_batch(args):
    """Process all .h5 files found in raw_dir."""
    survey = load_survey(args.survey) if args.survey else None
    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.h5")))
    if not files:
        log.warning(f"No .h5 files found in {args.raw_dir}")
        return
    log.info(f"Batch mode: {len(files)} files to process")
    for h5_path in files:
        process_file(
            h5_path,
            args.output_dir,
            survey=survey,
            ch_start=args.ch_start,
            ch_end=args.ch_end,
        )


def run_watch(args):
    """Watch raw_dir for new .h5 files and process them as they arrive."""
    from dasieve.watcher import watch_directory

    survey = load_survey(args.survey) if args.survey else None
    log.info(f"Watch mode: monitoring {args.raw_dir}")

    def callback(filepath: str):
        process_file(
            filepath,
            args.output_dir,
            survey=survey,
            ch_start=args.ch_start,
            ch_end=args.ch_end,
        )

    watch_directory(args.raw_dir, callback, poll_interval=args.poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach geometry to iDAS HDF5 files using DASCore."
    )
    parser.add_argument(
        "--survey",
        metavar="CSV",
        help="Well-survey CSV file (columns: channel,x_m,y_m,z_m). "
             "If omitted, default vertical-borehole geometry is used.",
    )
    parser.add_argument(
        "--ch-start",
        type=int,
        default=None,
        metavar="N",
        help="First channel to keep (0-based). Default: keep all.",
    )
    parser.add_argument(
        "--ch-end",
        type=int,
        default=None,
        metavar="N",
        help="Last channel to keep (0-based, inclusive). Default: keep all.",
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # manual
    p_manual = sub.add_parser("manual", help="Process specific files.")
    p_manual.add_argument("output_dir", help="Output directory.")
    p_manual.add_argument("files", nargs="+", help="HDF5 files to process.")
    p_manual.set_defaults(func=run_manual)

    # batch
    p_batch = sub.add_parser("batch", help="Process all .h5 files in a directory.")
    p_batch.add_argument("raw_dir", help="Directory with raw HDF5 files.")
    p_batch.add_argument("output_dir", help="Output directory.")
    p_batch.set_defaults(func=run_batch)

    # watch
    p_watch = sub.add_parser("watch", help="Watch directory for new files.")
    p_watch.add_argument("raw_dir", help="Directory to watch.")
    p_watch.add_argument("output_dir", help="Output directory.")
    p_watch.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between directory polls. Default: 1.0",
    )
    p_watch.set_defaults(func=run_watch)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
