"""Label every recording of the SpikeGLX census with its sync issue and the inferred correction.

Labels (one row per recording, NP2.4 shanks merged):
- `no issue`: all five good-probe criteria pass.
- `missing edges at start`: criteria 1 (same fileTimeSecs) and 2 (same edge count) fail, dn = k = round(dT / h) != 0,
  edges regular and complete, and SpikeGLX < 20221012. Correction: pair the edges from the last one, i.e. drop the first |k|
  edges of the longer stream; spikes are offset by shift_s = k * h (sign of the wheel offset).
- `unresolved: <reason>`: everything else, named after the first failing criterion (no sync, durations differ without
  matching edges, same duration but edge counts differ, irregular edges, sync stops before file end), plus the single
  SpikeGLX >= 20221012 case where firstSample says the files start together (end truncation per firstSample, KM_041).
Writes `artifacts/010/012_good_probe.pqt` (adds issue, k, shift_s, bwm, wheel_offset_s) and the sheet export
`artifacts/010/014_sync_issues.csv` (every recording that is not `no issue`).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
ART = ROOT.joinpath("artifacts", "010")
SRC = Path.home().joinpath("Documents", "ibl-ai-agent", "projects", "spike-clock-offset-investigation", "exploratory-analyses")
H = 0.5
KEY = ["subject", "date", "number", "rec"]

a = pd.read_parquet(ART.joinpath("010_spikeglx_census.pqt"))
a["rec"] = a.probe.str.replace(r"[a-d]$", "", regex=True)
bwm = set(pd.read_csv(SRC.parents[1].joinpath("repeated-site-decoding-power", "artifacts", "003_spike_alignment_qc.csv")).pid)
a["bwm"] = a.pid.isin(bwm)
per_rec = a.groupby(KEY).agg(bwm=("bwm", "any"), pids=("pid", lambda s: ";".join(sorted({p for p in s if isinstance(p, str) and p}))))
d = a.drop_duplicates(KEY).drop(columns=["bwm"]).merge(per_rec.reset_index(), on=KEY)
g = pd.read_parquet(ART.joinpath("012_good_probe.pqt"))[KEY + ["lab", "pid", "good", "first_failure"]]
d = d.merge(g[KEY + ["first_failure"]], on=KEY)

d["k"] = (d.dT_s / H).round()
regular = d[["nidq_n_long_gaps", "probe_n_long_gaps", "nidq_n_short_gaps", "probe_n_short_gaps"]].sum(axis=1) == 0
complete = (d.nidq_tail_s < H + 0.01) & (d.probe_tail_s < H + 0.01)
pattern = d.first_failure.str.startswith("1 ") & d.dn_edges.eq(d.k) & regular & complete
start = pattern & ~d.firstSample_reliable
REASON = {"0": "no sync", "1": "durations differ without matching edges", "2": "same duration, edge counts differ",
          "3": "irregular edges", "5": "sync stops before file end"}
d["issue"] = np.select([d.first_failure.eq("pass"), start], ["no issue", "missing edges at start"], "unresolved: " + d.first_failure.str[0].map(REASON))
d.loc[pattern & d.firstSample_reliable, "issue"] = "unresolved: end truncation per firstSample"
d["shift_s"] = np.where(start, d.k * H, np.nan)
d.loc[~start, "k"] = np.nan

# wheel offsets: BWM (008) and other projects (013), keyed by pid
w = pd.concat([pd.read_csv(SRC.joinpath("008_damage_report.csv"))[["pid", "offset_wheel_s"]],
               pd.read_csv(ROOT.joinpath("artifacts", "013_wheel_cohort3.csv"))[["pid", "offset_s"]].rename(columns={"offset_s": "offset_wheel_s"})])
w = w.dropna().drop_duplicates("pid").set_index("pid").offset_wheel_s
d["wheel_offset_s"] = d.pids.str.split(";").map(lambda ps: next((w[p] for p in ps if p in w.index), np.nan))

out = g.drop(columns=["first_failure"]).merge(d[KEY + ["first_failure", "issue", "k", "shift_s", "bwm", "wheel_offset_s"]], on=KEY)
out.to_parquet(ART.joinpath("012_good_probe.pqt"))

cols = {"lab": "lab", "subject": "subject", "date": "date", "number": "number", "rec": "probe", "bwm": "BWM", "issue": "issue",
        "k": "k (half-periods)", "shift_s": "shift (s)", "wheel_offset_s": "wheel offset (s)", "dT_s": "dT fileTimeSecs (s)",
        "dn_edges": "dn edges", "start_lead_s": "start lead firstSample (s)", "end_lead_s": "end lead (s)",
        "probe.appVersion": "SpikeGLX", "probe.imDatBs_fw": "base station fw", "probe.imDatBsc_fw": "connect card fw", "pids": "pid"}
s = d[d.issue.ne("no issue")][list(cols)].rename(columns=cols).sort_values(["issue", "lab", "subject", "date", "probe"])
s["BWM"] = np.where(s.BWM, "BWM", "")
for c in ["k (half-periods)", "dn edges", "number"]:
    s[c] = s[c].astype("Int64")
s.round(3).to_csv(ART.joinpath("014_sync_issues.csv"), index=False)
print(out.issue.value_counts().to_string())
print(pd.crosstab(out.issue, out.bwm).to_string())
