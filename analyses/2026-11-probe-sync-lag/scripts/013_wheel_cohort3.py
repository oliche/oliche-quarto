"""Wheel-speed cross-correlation offset for the cohort-3 recordings (criteria 1 and 2 fail, regular and complete edges,
SpikeGLX < 20221012) that are not in the BWM release.

The integer number of half-periods k = round(dT / h) is known from fileTimeSecs, but not whether the missing edges are at the
start (spikes shifted by k*h, pair from the last edge) or at the end (truncation, pair from the first edge, no shift).
The wheel (nidq clock) settles it: offset ~ k*h -> shift, offset ~ 0 -> truncation.

Spikes: ONE `spikes.times` (pykilosort collection preferred, all units). The lag window is widened to |k| h + 2 s.
Offset = peak lag minus the control baseline peak lag of 007 (neural activity leads movement by tens of ms).
Per-insertion results are checkpointed in artifacts/013_wheel/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from one.api import ONE
from scipy.signal import correlate

HERE = Path(__file__).parent
ROOT = HERE.parent
ART = ROOT.joinpath("artifacts", "013_wheel")
ART.mkdir(parents=True, exist_ok=True)
SRC = Path.home().joinpath("Documents", "ibl-ai-agent", "projects", "spike-clock-offset-investigation", "exploratory-analyses")
DT, SMOOTH, H = 0.01, 0.05, 0.5
one = ONE()


def z(x):
    """Z-score."""
    return (x - x.mean()) / x.std()


def spike_times(pid):
    """All-unit spike times of an insertion, pykilosort collection preferred."""
    eid, pname = one.pid2eid(pid)
    dsets = one.list_datasets(eid)
    col = sorted({str(Path(d).parent) for d in dsets if d.startswith(f"alf/{pname}/") and d.endswith("spikes.times.npy")})
    col = [c for c in col if "pykilosort" in c] or col
    if not col:
        raise FileNotFoundError("no spikes.times")
    return eid, np.sort(one.load_dataset(eid, "spikes.times.npy", collection=col[0])), col[0]


def wheel_xcorr(pid, max_lag):
    """Peak lag of the cross-correlation of population rate with wheel speed, with a block bootstrap CI.

    Parameters
    ----------
    pid : str
        Probe insertion id.
    max_lag : float
        Half-width of the lag window (s).

    Returns
    -------
    dict
        peak_lag_s, ci_lo, ci_hi, peak_r, peak_snr, collection.
    """
    eid, st, col = spike_times(pid)
    wheel = one.load_object(eid, "wheel", collection="alf")
    edges = np.arange(max(wheel.timestamps[0], st[0]), min(wheel.timestamps[-1], st[-1]), DT)
    rate = np.histogram(st, edges)[0].astype(float)
    speed = np.abs(np.convolve(np.diff(np.interp(edges, wheel.timestamps, wheel.position)), np.ones(int(SMOOTH / DT)) / (SMOOTH / DT), mode="same"))
    nlag = int(max_lag / DT)
    lags = np.arange(-nlag, nlag + 1) * DT

    def xc(idx):
        return (correlate(z(rate[idx]), z(speed[idx]), mode="full", method="fft") / idx.size)[idx.size - 1 - nlag: idx.size + nlag]

    full = xc(np.arange(rate.size))
    xcb = np.array([xc(b) for b in np.array_split(np.arange(rate.size), 10)])
    rng = np.random.default_rng(0)
    boot = [lags[np.argmax(xcb[rng.integers(0, 10, 10)].mean(0))] for _ in range(500)]
    return {"peak_lag_s": float(lags[np.argmax(full)]), "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
            "peak_r": float(full.max()), "peak_snr": float((full.max() - np.median(full)) / (np.percentile(full, 75) - np.percentile(full, 25) + 1e-12)),
            "collection": col}


if __name__ == "__main__":
    d = pd.read_parquet(sys.argv[1])  # cohorts table, one row per recording, with the census pid per probe folder
    allp = pd.read_parquet(ROOT.joinpath("artifacts", "010", "010_spikeglx_census.pqt"))
    allp["rec"] = allp.probe.str.replace(r"[a-d]$", "", regex=True)
    c3 = d[d.cohort.str.startswith("3") & ~d.is_bwm]
    todo = c3[["subject", "date", "number", "rec", "dT_s"]].merge(allp[["subject", "date", "number", "rec", "probe", "pid"]])
    todo = todo[todo.pid.notna() & todo.pid.ne("")]
    ctrl = pd.read_csv(SRC.joinpath("007_wheel_xcorr_candidates.csv"))
    base = ctrl[ctrl.role == "control"].peak_lag_s.median()
    rows = []
    for r in todo.itertuples():
        f = ART.joinpath(f"{r.pid}.json")
        if not f.exists():
            try:
                res = wheel_xcorr(r.pid, abs(round(r.dT_s / H)) * H + 2)
            except Exception as e:  # noqa: BLE001
                res = {"error": repr(e)[:200]}
            f.write_text(json.dumps(res))
        rows.append({"pid": r.pid} | json.loads(f.read_text()))
        print(r.subject, r.date, r.probe, rows[-1].get("peak_lag_s", rows[-1].get("error")), flush=True)
    res = todo.merge(pd.DataFrame(rows), on="pid")
    res["offset_s"] = res.peak_lag_s - base
    res["rule_s"] = (res.dT_s / H).round() * H
    res.to_csv(ROOT.joinpath("artifacts", "013_wheel_cohort3.csv"), index=False)
