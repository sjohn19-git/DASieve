"""
Test for geometry_ingest.process_file — no survey, no channel selection.

Reads a single local HDF5 file, assigns default vertical-borehole geometry,
and writes the output to ~/Downloads.

Run:
    python test_watcher.py
"""

import sys
import os

# Allow running without installing the package
sys.path.insert(0, os.path.expanduser("~/DASieve"))

import logging
from dasieve.geometry import process_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

if __name__ == "__main__":
    h5_path    = os.path.expanduser("~/Downloads/16B_Commissioning_UTC_20240201_235934.512.h5")
    output_dir = os.path.expanduser("~/Downloads")

    print(f"Input : {h5_path}")
    print(f"Output: {output_dir}")

    out = process_file(
        h5_path=h5_path,
        output_dir=output_dir,
        survey=None,      # default vertical-borehole geometry (x=0, y=0, z=distance)
        ch_start=None,    # keep all channels
        ch_end=None,
    )

    if out:
        print(f"\nSuccess — output written to:\n  {out}")
    else:
        print("\nFailed — check logs above.")
