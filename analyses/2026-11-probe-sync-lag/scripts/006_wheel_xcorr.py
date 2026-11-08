"""Spike-clock offset from the cross-correlation of population firing rate with wheel speed (wheel is on the nidq clock).

The wheel speed signal is aperiodic, so the lag of the cross-correlation peak is unambiguous, unlike the sync square wave.
Aligned insertions are expected to peak near 0 (neural activity leads movement by tens of ms).
Spikes: local bwm_ephys 1.2.1 shards, or ONE pykilosort spikes.times for insertions outside the release.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from one.api import ONE
from scipy.signal import correlate

from ibl_ai_agent.data_locations import resolve_dataset_dir
from ibl_ai_agent.datasets.bwm_ephys import load_spike_shard

HERE = Path(__file__).parent
BWM_DIR = resolve_dataset_dir("bwm_ephys")
DT, MAX_LAG, SMOOTH = 0.01, 3.0, 0.05  # bin (s), max lag (s), boxcar smoothing of wheel speed (s)
PIDS = {
    "c893c0a3-5597-49cf-baa1-60efdfdef542": "NYU-12 2020-01-21 probe01 (suspect)",
    "f05b5941-68bd-4bf5-980d-721959bd4443": "NYU-12 2020-01-21 probe00 (sibling)",
    "f9d8aacd-b2a0-49f2-bd71-c2f5aadcfdd1": "UCLA034 2022-01-25 probe00 (suspect)",
    "a94a0023-d461-49fa-8cc6-032237039c27": "UCLA034 2022-01-25 probe01 (sibling, aligned)",
    "7a620688-66cb-44d3-b79b-ccac1c8ba23e": "MFD_07 2023-08-31 probe00 (+0.5 s control)",
    "3d3d5a5e-df26-43ee-80b6-2d72d85668a5": "UCLA015 2022-03-31 probe00 (aligned control)",
}
one = ONE()


def spike_times(pid, eid, pname):
    shard = BWM_DIR.joinpath("spikes", pid)
    if shard.exists():
        return np.asarray(load_spike_shard(shard)["spike_times_seconds"]), "bwm_ephys 1.2.1 (good units)"
    dsets = one.list_datasets(eid)
    col = sorted({str(Path(d).parent) for d in dsets if d.startswith(f"alf/{pname}/") and d.endswith("spikes.times.npy")})
    col = [c for c in col if "pykilosort" in c] or col
    return np.sort(one.load_dataset(eid, "spikes.times.npy", collection=col[0])), f"ONE {col[0]} (all units)"


def z(x):
    return (x - x.mean()) / x.std()


rows, curves = [], {}
nlag = int(MAX_LAG / DT)
for pid, label in PIDS.items():
    eid, pname = one.pid2eid(pid)
    st, src = spike_times(pid, eid, pname)
    wheel = one.load_object(eid, "wheel", collection="alf")
    t0, t1 = max(wheel.timestamps[0], st[0]), min(wheel.timestamps[-1], st[-1])
    edges = np.arange(t0, t1, DT)
    rate = np.histogram(st, edges)[0].astype(float)
    pos = np.interp(edges, wheel.timestamps, wheel.position)
    speed = np.abs(np.convolve(np.diff(pos), np.ones(int(SMOOTH / DT)) / (SMOOTH / DT), mode="same"))
    # positive lag: spikes come after wheel movement, i.e. the spike clock is late
    lags = np.arange(-nlag, nlag + 1) * DT

    def xcorr(idx):
        c = correlate(z(rate[idx]), z(speed[idx]), mode="full", method="fft") / idx.size
        return c[idx.size - 1 - nlag: idx.size + nlag]

    xc = xcorr(np.arange(rate.size))
    peak = lags[np.argmax(xc)]
    # bootstrap the peak lag over 10 contiguous session blocks
    xcb = np.array([xcorr(b) for b in np.array_split(np.arange(rate.size), 10)])
    rng = np.random.default_rng(0)
    boot = [lags[np.argmax(xcb[rng.integers(0, 10, 10)].mean(0))] for _ in range(500)]
    rows.append(dict(pid=pid, label=label, spikes=src, n_spikes=st.size, peak_lag_s=peak, ci_lo=np.percentile(boot, 2.5),
                     ci_hi=np.percentile(boot, 97.5), peak_r=xc.max()))
    curves[label] = (lags, xc)
df = pd.DataFrame(rows)
df.to_csv(HERE.joinpath("006_wheel_xcorr.csv"), index=False)
pd.set_option("display.width", 250)
print(df.drop(columns="pid").round(3).to_string())

sns.set_theme(context="notebook", style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))
for label, (lags, xc) in curves.items():
    ax.plot(lags, xc / xc.max(), label=label)
for x in [-1, -0.5, 0, 0.5]:
    ax.axvline(x, color="k", lw=0.5, ls=":")
ax.set(xlabel="lag (s); > 0 = spikes late relative to wheel", ylabel="normalised xcorr (rate vs wheel speed)", xlim=(-2, 2))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE.joinpath("006_wheel_xcorr.png"), dpi=150)
