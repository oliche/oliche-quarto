"""Wheel-speed cross-correlation offset for every sync-scan candidate with a local bwm_ephys shard, plus aligned controls.

Candidates (from 004_signature_summary.csv): edge-count difference != 0, |start lead| > 0.1 s, behavioural flag (003 QC),
or SpikeGLX 20201103 + base station firmware 3.2.186. Controls: 30 random insertions with none of these.
Offset = xcorr peak lag minus the median peak lag of the controls (neural activity leads movement).
Per-insertion results are checkpointed in artifacts/007_wheel_xcorr/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from one.api import ONE
from scipy.signal import correlate

from ibl_ai_agent.data_locations import resolve_dataset_dir
from ibl_ai_agent.datasets.bwm_ephys import load_spike_shard

HERE = Path(__file__).parent
ART = HERE.parent.joinpath("artifacts", "007_wheel_xcorr")
ART.mkdir(parents=True, exist_ok=True)
BWM_DIR = resolve_dataset_dir("bwm_ephys")
DT, MAX_LAG, SMOOTH = 0.01, 3.0, 0.05
NLAG = int(MAX_LAG / DT)
one = ONE()

sig = pd.read_csv(HERE.joinpath("004_signature_summary.csv"))
sig["has_shard"] = sig.pid.map(lambda p: BWM_DIR.joinpath("spikes", p).exists())
sig["candidate"] = (sig.dn != 0) | (sig.start_lead_s.abs() > 0.1) | (sig.flag == True) | ((sig.appVersion == 20201103) & (sig.imDatBsc_fw == "3.2.186"))  # noqa: E712
cand = sig[sig.candidate & sig.has_shard]
ctrl = sig[~sig.candidate & sig.has_shard].sample(30, random_state=0)
todo = pd.concat([cand.assign(role="candidate"), ctrl.assign(role="control")])
print(f"{len(cand)} candidates with shards ({sig.candidate.sum()} total), {len(ctrl)} controls")


def z(x):
    return (x - x.mean()) / x.std()


def wheel_xcorr(pid, eid):
    st = np.asarray(load_spike_shard(BWM_DIR.joinpath("spikes", pid))["spike_times_seconds"])
    wheel = one.load_object(eid, "wheel", collection="alf")
    edges = np.arange(max(wheel.timestamps[0], st[0]), min(wheel.timestamps[-1], st[-1]), DT)
    rate = np.histogram(st, edges)[0].astype(float)
    speed = np.abs(np.convolve(np.diff(np.interp(edges, wheel.timestamps, wheel.position)), np.ones(int(SMOOTH / DT)) / (SMOOTH / DT), mode="same"))
    lags = np.arange(-NLAG, NLAG + 1) * DT

    def xc(idx):
        return (correlate(z(rate[idx]), z(speed[idx]), mode="full", method="fft") / idx.size)[idx.size - 1 - NLAG: idx.size + NLAG]

    full = xc(np.arange(rate.size))
    xcb = np.array([xc(b) for b in np.array_split(np.arange(rate.size), 10)])
    rng = np.random.default_rng(0)
    boot = [lags[np.argmax(xcb[rng.integers(0, 10, 10)].mean(0))] for _ in range(500)]
    srt = np.sort(full)[::-1]
    return dict(peak_lag_s=float(lags[np.argmax(full)]), ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                peak_r=float(full.max()), peak_snr=float((full.max() - np.median(full)) / (np.percentile(full, 75) - np.percentile(full, 25) + 1e-12)))


rows = []
for r in todo.itertuples():
    f = ART.joinpath(f"{r.pid}.json")
    if not f.exists():
        try:
            res = wheel_xcorr(r.pid, r.eid)
        except Exception as e:  # noqa: BLE001
            res = dict(error=repr(e)[:200])
        f.write_text(json.dumps(res))
    rows.append(dict(pid=r.pid, role=r.role) | json.loads(f.read_text()))
res = todo.merge(pd.DataFrame(rows), on=["pid", "role"])
base = res[res.role == "control"].peak_lag_s.median()
res["offset_s"] = res.peak_lag_s - base
res["ci_width_s"] = res.ci_hi - res.ci_lo
res.to_csv(HERE.joinpath("007_wheel_xcorr_candidates.csv"), index=False)
pd.set_option("display.width", 250, "display.max_rows", 200)
print(f"control baseline peak lag {base:.3f} s; control offsets:", res[res.role == "control"].offset_s.describe().round(3).to_dict())
cols = ["subject", "date", "probe", "dn", "start_lead_s", "end_lead_s", "transient_s", "offset_s", "ci_lo", "ci_hi", "peak_snr"]
print(res[res.role == "candidate"].sort_values("offset_s")[cols].round(3).to_string())
