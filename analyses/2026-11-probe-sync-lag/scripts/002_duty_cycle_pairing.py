"""Disambiguate probe<->nidq edge pairing with the sync square-wave duty cycle.

For a correct pairing, each probe edge maps to the nidq edge of the same physical transition, so the per-cycle
high-minus-low durations (jitter included) correlate across streams. A one-edge shift breaks that correlation
(or anti-correlates it). Uses files cached by 001.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from one.api import ONE

one = ONE()
scan = pd.read_csv(Path(__file__).parent.joinpath("001_sync_signature_scan.csv")).dropna(subset=["n_nidq"])


def edges(eid, raw, suffix, wiring_glob):
    t, c, p = (np.load(one.load_dataset(eid, f"{raw}/_spikeglx_sync.{a}{suffix}.npy", download_only=True)) for a in ["times", "channels", "polarities"])
    wir = json.loads(next(Path(one.eid2path(eid)).joinpath(raw).glob(wiring_glob)).read_text())
    ch = int([k for k, v in wir["SYNC_WIRING_DIGITAL"].items() if v == "imec_sync"][0].split(".")[-1])
    return t[c == ch], p[c == ch]


rows = []
for _, r in scan.iterrows():
    pname = one.pid2eid(r.pid)[1]
    tn, pn = edges(r.eid, "raw_ephys_data", "", "*nidq.wiring.json")
    tp, pp = edges(r.eid, f"raw_ephys_data/{pname}", f".{pname}", "*.wiring.json")
    row = dict(pid=r.pid, subject=r.subject, date=r.date, flag=r.flag)
    for lag in [-2, -1, 0, 1, 2]:
        a, b = (tn, tp[lag:]) if lag >= 0 else (tn[-lag:], tp)
        n = min(a.size, b.size)
        da, db = np.diff(a[:n]), np.diff(b[:n])
        row[f"corr_lag{lag}"] = np.corrcoef(da - np.median(da), db - np.median(db))[0, 1]
        pa, pb = (pn, pp[lag:]) if lag >= 0 else (pn[-lag:], pp)
        row[f"polmatch_lag{lag}"] = np.mean(pa[:n] == pb[:n])
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).with_suffix(".csv"), index=False)
pd.set_option("display.width", 300)
print(df.round(3).to_string())

# %% duty-cycle asymmetry: mean duration after rising minus after falling edge (µs), in each stream's own clock.
# Same sign on both streams => same polarity view; opposite sign => the probe sees the sync wave inverted.
rows = []
for _, r in scan.iterrows():
    pname = one.pid2eid(r.pid)[1]
    out = dict(pid=r.pid, subject=r.subject, date=r.date, flag=r.flag)
    for name, (t, p) in dict(nidq=edges(r.eid, "raw_ephys_data", "", "*nidq.wiring.json"),
                             probe=edges(r.eid, f"raw_ephys_data/{pname}", f".{pname}", "*.wiring.json")).items():
        d = np.diff(t)
        out[f"{name}_high_minus_low_us"] = (d[p[:-1] == 1].mean() - d[p[:-1] == -1].mean()) * 1e6
        out[f"{name}_sem_us"] = d.std() / np.sqrt(d.size) * 1e6 * 2
    rows.append(out)
duty = pd.DataFrame(rows)
duty.to_csv(Path(__file__).parent.joinpath("002_duty_cycle.csv"), index=False)
print(duty.round(2).to_string())
