"""
watcher.py
----------
Directory file watcher for DASsieve.
Monitors a directory for new HDF5 (.h5) files and triggers the
processing pipeline when one is detected.
"""

import os
import time
import logging

logger = logging.getLogger(__name__)


def watch_directory(raw_dat_dir: str, callback, poll_interval: float = 1.0):
    """
    Poll a directory for new .h5 files and call `callback` for each one.

    Parameters
    ----------
    raw_dat_dir : str
        Path to the directory where the iDAS writes HDF5 files.
    callback : callable
        Function to call with the full file path when a new .h5 is detected.
        Signature: callback(filepath: str)
    poll_interval : float
        Seconds between directory scans. Default is 1.0.
    """
    raw_dat_dir = os.path.expanduser(raw_dat_dir)
    logger.info(f"Watching directory: {raw_dat_dir}")
    current_files = set(os.listdir(raw_dat_dir))

    while True:
        try:
            new_listing = set(os.listdir(raw_dat_dir))
            new_files   = new_listing - current_files

            for filename in sorted(new_files):
                if filename.endswith(".h5"):
                    filepath = os.path.join(raw_dat_dir, filename)
                    logger.info(f"New file detected: {filename}")
                    callback(filepath)

            current_files = new_listing

        except Exception as e:
            logger.error(f"Watcher error: {e}")

        time.sleep(poll_interval)
