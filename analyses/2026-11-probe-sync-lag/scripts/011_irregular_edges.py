"""Classify the sync-edge anomalies of every irregular recording in the SpikeGLX census (010).

Runs on popeye next to the 010 outputs (reads `010_spikeglx_census.pqt` and the cached `010_edges/*.npz`).

Definitions, per stream (nidq or probe), on the imec sync line, with half-period h = syncSourcePeriod / 2 (0.5 s):
- interval: time between two consecutive edges; nominally h.
- irregular stream: at least one interval > 1.5 h (0.75 s) or < 0.5 h (0.25 s).
- irregular recording: the nidq or the probe stream is irregular.
Anomaly labels, per stream, not exclusive:
- dropout: an interval > 10 h (5 s), the sync is absent for a while.
- rate_halving: a run of >= 4 consecutive intervals within [1.6 h, 2.4 h], every other edge missing.
- missing: other intervals > 1.5 h, isolated lost edges.
- duplicate: an interval < 0.1 h (50 ms), bounced or double-triggered edge.
- mistimed: an interval within [0.1 h, 0.5 h), one displaced edge.
Onset position: time of the first anomalous interval as a fraction of the stream's edge span.

Output: `011_irregular_edges.pqt`, one row per irregular recording (NP2.4 shanks merged, first shank kept).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EDGES = HERE.joinpath("010_edges")
LABELS = ["dropout", "rate_halving", "missing", "duplicate", "mistimed"]


def max_run(mask):
    """Length of the longest run of True in a boolean array."""
    if not mask.any():
        return 0
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def run_members(mask, min_len):
    """Boolean mask of the elements belonging to runs of True of length >= min_len."""
    out = np.zeros_like(mask)
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a >= min_len:
            out[a:b] = True
    return out


def classify_stream(t, p, h, kind):
    """Anomaly counts, labels and onset of one stream's sync edges."""
    d = np.diff(t)
    big = d > 10 * h
    dbl = (d > 1.6 * h) & (d < 2.4 * h)
    halving = run_members(dbl, 4)
    long = d > 1.5 * h
    missing = long & ~big & ~halving
    dup = d < 0.1 * h
    mist = (d >= 0.1 * h) & (d < 0.5 * h)
    anomalous = long | (d < 0.5 * h)
    n = {"dropout": big.sum(), "rate_halving": halving.sum(), "missing": missing.sum(), "duplicate": dup.sum(), "mistimed": mist.sum()}
    out = {f"{kind}_n_{k}": int(v) for k, v in n.items()}
    out[f"{kind}_labels"] = "+".join(k for k in LABELS if n[k]) or "regular"
    out[f"{kind}_max_halving_run"] = max_run(dbl)
    out[f"{kind}_longest_gap_s"] = float(d.max()) if d.size else np.nan
    # edges lost (long intervals) and gained (short intervals), in half-periods
    out[f"{kind}_edges_lost"] = int(np.round(d[long] / h).sum() - long.sum())
    out[f"{kind}_edges_extra"] = int((d < 0.5 * h).sum())
    if anomalous.any():
        i0, i1 = np.flatnonzero(anomalous)[[0, -1]]
        span = t[-1] - t[0]
        out |= {f"{kind}_onset_s": float(t[i0]), f"{kind}_onset_frac": float((t[i0] - t[0]) / span),
                f"{kind}_last_anomaly_frac": float((t[i1 + 1] - t[0]) / span)}
    return out


def position(f0, f1):
    """Where the anomalies sit in the recording, from the first and last anomaly fractions."""
    if np.isnan(f0):
        return ""
    if f0 > 0.95:
        return "end"
    if f1 < 0.05:
        return "start"
    return "middle" if f1 - f0 < 0.5 else "throughout"


if __name__ == "__main__":
    d = pd.read_parquet(HERE.joinpath("010_spikeglx_census.pqt"))
    d["rec"] = d.probe.str.replace(r"[a-d]$", "", regex=True)
    d = d.drop_duplicates(["subject", "date", "number", "rec"])
    d = d[d.nidq_sync_status.eq("ok") & d.probe_sync_status.eq("ok") & d.nidq_n_edges.gt(0) & d.probe_n_edges.gt(0)]
    gaps = ["nidq_n_long_gaps", "probe_n_long_gaps", "nidq_n_short_gaps", "probe_n_short_gaps"]
    irr = d[d[gaps].sum(axis=1) > 0].copy()
    irr["half_period_s"] = pd.to_numeric(irr["probe.syncSourcePeriod"], errors="coerce").fillna(1) / 2
    rows = []
    for r in irr.itertuples():
        ses = f"{r.lab}_{r.subject}_{r.date}_{r.number:03d}"
        h = r.half_period_s
        out = {"index": r.Index}
        for kind, label in [("nidq", f"{ses}_nidq"), ("probe", f"{ses}_{r.probe}")]:
            e = np.load(EDGES.joinpath(f"{label}.npz"))
            out |= classify_stream(e["times"], e["polarities"], h, kind)
        rows.append(out)
    c = pd.DataFrame(rows).set_index("index")
    irr = irr.join(c)
    irr["streams"] = np.select([(irr.nidq_labels != "regular") & (irr.probe_labels != "regular"), irr.nidq_labels != "regular"],
                               ["both", "nidq"], "probe")
    for k in ["nidq", "probe"]:
        irr[f"{k}_position"] = [position(a, b) for a, b in zip(irr[f"{k}_onset_frac"], irr[f"{k}_last_anomaly_frac"])]
    irr["consistent"] = (irr.dT_s - 0.5 * irr.dn_edges).abs() < 0.25
    irr.to_parquet(HERE.joinpath("011_irregular_edges.pqt"))
    print(len(irr), "irregular recordings")
    print(irr.streams.value_counts().to_string())
