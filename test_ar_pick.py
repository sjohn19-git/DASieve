"""
AR-AIC picker test on the 2018-01-23 M7.9 Gulf of Alaska earthquake
recorded at IU.COLA (College/Fairbanks, AK) — ~990 km regional distance.

ar_pick runs on raw (detrend-only) data so its internal bandpass is not
corrupted by pre-filtering. The highpass at 1 Hz is applied separately,
only for plotting.

Run:
    python test_ar_pick.py
"""

import copy
import os

import numpy as np
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.trigger import ar_pick

# ---------------------------------------------------------------------------
# ar_pick parameters  — tuned for regional distances (~1000 km), 1–10 Hz band
# ---------------------------------------------------------------------------
AR_PARAMS = dict(
    f1=1,  # lower corner for ar_pick's internal bandpass (Hz)
    f2=7.0,  # upper corner (Hz)
    lta_p=5.0,  # LTA window for P picker (s)
    sta_p=0.5,  # STA window for P picker (s)
    lta_s=5.0,  # LTA window for S picker (s)
    sta_s=0.5,  # STA window for S picker (s)
    m_p=2,
    m_s=8,
    l_p=0.5,  # pick refinement window for P (s)
    l_s=0.5,  # pick refinement window for S (s)
)

# ---------------------------------------------------------------------------
# Event + station
# ---------------------------------------------------------------------------
EVENT = {
    "name": "2018 Gulf of Alaska M7.9",
    "origin": UTCDateTime("2018-01-23T09:31:42"),
    "lat": 56.046,
    "lon": -149.073,
    "depth_km": 25.0,
}

# IU.COLA (College/Fairbanks, AK) — GSN vault, ~990 km from epicentre.
STATION = {
    "network": "IU",
    "station": "COLA",
    "location": "00",
}

HIGHPASS_HZ = 1.0  # applied only for plotting
T_START = 100.0  # seconds after origin to start window (P at ~990 km ≈ 124 s)
T_END = 380.0  # seconds after origin to end window (S at ~990 km ≈ 220 s)

CLIENT_NAME = "IRIS"


def fetch_zne(client, t1, t2):
    """Download all BH? channels, detrend only (no filter — ar_pick filters internally).

    Returns (st_raw, tr_z, tr_n, tr_e).  st_raw holds the detrended traces;
    tr_z/n/e are references into it."""
    st = client.get_waveforms(
        STATION["network"],
        STATION["station"],
        STATION["location"],
        "BH?",
        t1,
        t2,
    )
    st.merge(fill_value="interpolate")
    st.detrend("demean")
    st.detrend("linear")
    print(f"  channels retrieved: {[tr.stats.channel for tr in st]}")

    def pick(suffixes):
        for s in suffixes:
            sel = st.select(channel=f"BH{s}")
            if sel:
                return sel[0]
        raise RuntimeError(f"No BH channel with suffix in {suffixes} found.")

    tr_z = pick(["Z"])
    tr_n = pick(["N", "1"])
    tr_e = pick(["E", "2"])
    return st, tr_z, tr_n, tr_e


def main():
    client = Client(CLIENT_NAME)

    # station coords for distance label
    inv = client.get_stations(
        network=STATION["network"],
        station=STATION["station"],
        starttime=EVENT["origin"],
        endtime=EVENT["origin"] + 60,
        level="station",
    )
    sta = inv[0][0]
    dist_km = (
        gps2dist_azimuth(EVENT["lat"], EVENT["lon"], sta.latitude, sta.longitude)[0]
        / 1000.0
    )
    print(
        f"{EVENT['name']}  ->  {STATION['network']}.{STATION['station']}  ({dist_km:.0f} km)"
    )

    t1 = EVENT["origin"] + T_START
    t2 = EVENT["origin"] + T_END
    st_raw, tr_z, tr_n, tr_e = fetch_zne(client, t1, t2)

    fs = tr_z.stats.sampling_rate
    print(
        f"  {tr_z.stats.channel}/{tr_n.stats.channel}/{tr_e.stats.channel}  "
        f"@ {fs:.0f} Hz, {tr_z.stats.npts} samples"
    )

    # --- ar_pick on raw detrended data ---
    p = AR_PARAMS

    def to_f64(tr):
        return np.ascontiguousarray(tr.data, dtype=np.float64)

    p_pick, s_pick = ar_pick(
        to_f64(tr_z),
        to_f64(tr_n),
        to_f64(tr_e),
        fs,
        p["f1"],
        p["f2"],
        p["lta_p"],
        p["sta_p"],
        p["lta_s"],
        p["sta_s"],
        p["m_p"],
        p["m_s"],
        p["l_p"],
        p["l_s"],
    )

    print(f"\np_pick={p_pick:.3f}s   s_pick={s_pick:.3f}s  (seconds from trace start)")

    # --- highpass-filter copies for plotting only ---
    st_plot = copy.deepcopy(st_raw)
    st_plot.filter("highpass", freq=HIGHPASS_HZ, corners=4, zerophase=True)

    def plot_tr(suffix_list):
        for s in suffix_list:
            sel = st_plot.select(channel=f"BH{s}")
            if sel:
                return sel[0]
        return None

    tr_z_p = plot_tr(["Z"])
    tr_n_p = plot_tr(["N", "1"])
    tr_e_p = plot_tr(["E", "2"])

    # --- plot ---
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    for ax, tr in zip(axes, [tr_z_p, tr_n_p, tr_e_p]):
        t = tr.times()
        ax.plot(t, tr.data, color="black", linewidth=0.5)
        if p_pick > 0:
            ax.axvline(p_pick, color="blue", lw=1.5, label=f"P pick ({p_pick:.1f}s)")
        if s_pick > 0:
            ax.axvline(s_pick, color="red", lw=1.5, label=f"S pick ({s_pick:.1f}s)")
        ax.set_ylabel(f"{tr.stats.channel}\nCounts")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time since trace start (s)")
    axes[0].set_title(
        f"{EVENT['name']}  @  {STATION['network']}.{STATION['station']}  "
        f"({dist_km:.0f} km)   plot: hp {HIGHPASS_HZ} Hz   ar_pick: {p['f1']}–{p['f2']} Hz"
    )
    plt.tight_layout()

    out = os.path.expanduser("~/DASieve/test_ar_pick_result.png")
    fig.savefig(out, dpi=120)
    print(f"saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
