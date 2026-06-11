"""
geometry.py

Attaches well-survey geometry (x, y, z) to DASCore patches read from iDAS HDF5 files.

Survey CSV format (columns: channel, x_m, y_m, z_m):
    channel,x_m,y_m,z_m
    0,0.0,0.0,0.0
    1,0.5,0.0,1.2
    ...

Authors: Sebin John, 2025
"""

import csv
import logging

import numpy as np

import dascore as dc

log = logging.getLogger(__name__)


def load_survey(survey_path: str) -> dict:
    """
    Load a well-survey CSV into a dict mapping channel index → (x, y, z).

    CSV must have a header row with columns: channel, x_m, y_m, z_m
    Units are meters.

    Returns
    -------
    dict : {channel_int: (x_float, y_float, z_float)}
    """
    survey = {}
    with open(survey_path, newline="") as f:
        for row in csv.DictReader(f):
            ch = int(row["channel"])
            survey[ch] = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
    log.info(f"Loaded {len(survey)} channel entries from {survey_path}")
    return survey


def attach_geometry(patch: dc.Patch, survey: dict) -> dc.Patch:
    """
    Attach X, Y, Z coordinates from a well survey to a DASCore Patch.

    Each channel index (0-based position in the distance array) is looked up
    in the survey dict. Channels absent from the survey receive NaN.

    Parameters
    ----------
    patch : dc.Patch
        Input patch with a 'distance' dimension.
    survey : dict
        {channel_index: (x_m, y_m, z_m)} from load_survey().

    Returns
    -------
    dc.Patch
        Patch with 'x', 'y', 'z' coordinates attached to the distance dimension.
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
