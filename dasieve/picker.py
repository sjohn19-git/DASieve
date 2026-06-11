import pandas as pd
from obspy.signal.trigger import classic_sta_lta, trigger_onset, plot_trigger
from dascore.utils.io import patch_to_obspy


def trigger_picker(
    patch, sta=0.05, lta=1.0, thr_on=3.0, thr_off=0.3, plot=False, plot_channel=None
):
    stream = patch_to_obspy(patch)
    dist_vals = patch.coords.get_array("distance")
    time_vals = patch.coords.get_array("time")

    n_traces = len(stream)

    rows = []
    for i, trace in enumerate(stream):
        df = trace.stats.sampling_rate
        nsta = int(sta * df)
        nlta = int(lta * df)

        cft = classic_sta_lta(trace.data, nsta, nlta)
        for on, off in trigger_onset(cft, thr_on, thr_off):
            rows.append(
                (
                    dist_vals[i],
                    int(on),
                    time_vals[on],
                    cft[on],
                    int(off),
                    time_vals[off],
                    cft[off],
                )
            )

    if plot:
        ch_idx = plot_channel if plot_channel is not None else n_traces // 2
        df_plot = stream[ch_idx].stats.sampling_rate
        nsta_p = int(sta * df_plot)
        nlta_p = int(lta * df_plot)
        cft_plot = classic_sta_lta(stream[ch_idx].data, nsta_p, nlta_p)
        plot_trigger(stream[ch_idx], cft_plot, thr_on, thr_off)

    return pd.DataFrame(
        rows,
        columns=[
            "distance",
            "onset_sample",
            "onset_time",
            "cft_at_onset",
            "off_sample",
            "off_time",
        ],
    )
