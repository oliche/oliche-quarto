"""Build the tables and figures of this page from the investigation artifacts.

Source: ibl-ai-agent/projects/spike-clock-offset-investigation (scripts copied in ./scripts):
003 sync-front scan (BWM + churchlandlab_ucla + census candidates), 005/009 metadata census of all 3B probes on Alyx,
006/007 wheel-speed cross-correlation, 008 consolidated sync-rule damage report.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from one.api import ONE

HERE = Path(__file__).parent
SRC = Path.home().joinpath("Documents", "ibl-ai-agent", "projects", "spike-clock-offset-investigation", "exploratory-analyses")
FIG = HERE.joinpath("figures")
ALYX = "https://alyx.internationalbrainlab.org/admin/experiments/probeinsertion"
MAX_SHIFT = 20  # half-periods; larger boundary offsets mean nidq/probe files from different runs, reported separately

cen = pd.read_parquet(SRC.joinpath("009_meta_rule_census.pqt"))
dmg = pd.read_csv(SRC.joinpath("008_damage_report.csv"))
wx = pd.read_csv(SRC.joinpath("007_wheel_xcorr_candidates.csv"))
bwm = set(pd.read_csv(SRC.parents[1].joinpath("repeated-site-decoding-power", "artifacts", "003_spike_alignment_qc.csv")).pid)

# per-probe classification from metadata, all 3B probes
large = (cen.rule_class != "aligned") & (cen.boundary.abs() > MAX_SHIFT)
cen["category"] = np.select(
    [cen.rule_class == "no meta", large & (cen.start_lead_s.abs() < 0.5), large, cen.rule_class == "ambiguous", cen.rule_class == "shifted"],
    ["no meta", "truncated file (>10 s end mismatch)", "inconsistent metadata (>10 s start mismatch)", "ambiguous", "shifted"], "aligned")
cen["mechanism"] = np.where((cen.appVersion == "20201103") & (cen.imDatBsc_fw == "3.2.186"), "A: SpikeGLX 20201103 + BSC fw 3.2.186",
                            np.where(cen.category == "aligned", "", "B: other"))
flagged = cen[~cen.category.isin(["aligned", "no meta"])].copy()

# pids, session refs
one = ONE()
pid_map = dict(zip(zip(dmg.eid, dmg.probe), dmg.pid))
missing = [(e, p) for e, p in zip(flagged.eid, flagged.probe) if (e, p) not in pid_map]
for e, p in missing:
    ins = one.alyx.rest("insertions", "list", session=e, name=p)
    pid_map[(e, p)] = ins[0]["id"] if ins else None
flagged["pid"] = [pid_map.get((e, p)) for e, p in zip(flagged.eid, flagged.probe)]
flagged["session"] = flagged.subject + "/" + flagged.date + "/" + [f"{one.eid2ref(e)['sequence']:03d}" for e in flagged.eid]
flagged["in_bwm_release"] = flagged.pid.isin(bwm)
flagged["bwm_project"] = flagged.projects.str.contains("brainwide")
flagged["offset_s"] = np.where(flagged.category == "shifted", flagged.boundary.round() * 0.5, np.nan)
w = dmg[["pid", "offset_wheel_s", "wheel_confident", "dn"]]
flagged = flagged.merge(w, on="pid", how="left")
flagged["evidence"] = np.select([flagged.wheel_confident == True, flagged.dn.notna()], ["meta + sync fronts + wheel", "meta + sync fronts"], "meta")  # noqa: E712
flagged["link"] = ALYX + "/" + flagged.pid.astype(str) + "/change/"
cols = ["category", "mechanism", "offset_s", "offset_wheel_s", "evidence", "lab", "session", "probe", "pid", "eid", "in_bwm_release", "bwm_project",
        "boundary", "start_lead_s", "end_lead_s", "dn", "appVersion", "imDatBs_fw", "imDatBsc_fw", "link"]
flagged = flagged[cols].sort_values(["category", "offset_s", "lab", "session"]).round(3)
flagged.to_csv(HERE.joinpath("flagged_probes.csv"), index=False)

# summary by acquisition combination
summ = pd.crosstab(cen.appVersion.fillna("?") + " | " + cen.imDatBs_fw.fillna("?") + " / " + cen.imDatBsc_fw.fillna("?"), cen.category)
summ.index.name = "SpikeGLX | base station fw"
summ.reset_index().to_csv(HERE.joinpath("summary_by_version.csv"), index=False)

# wheel validation table
wv = dmg[dmg.offset_wheel_s.notna()][["subject", "date", "probe", "pid", "dn", "offset_rule_s", "offset_wheel_s", "peak_snr", "wheel_confident", "confidence"]]
wv = wv.merge(wx[["pid", "role"]], on="pid").round(3).sort_values("offset_rule_s")
wv.to_csv(HERE.joinpath("wheel_validation.csv"), index=False)

# figures
sns.set_theme(context="notebook", style="whitegrid")
fig, ax = plt.subplots(figsize=(7, 5))
c = dmg[dmg.wheel_confident]
sns.scatterplot(data=c, x="offset_rule_s", y="offset_wheel_s", s=70, ax=ax, color="tab:blue", label="sync-front rule")
sns.scatterplot(x=c.lag_float.round() * 0.5, y=c.offset_wheel_s, s=30, ax=ax, color="tab:orange", marker="X", label="firstSample start only")
ax.plot([-3.3, 1], [-3.3, 1], "k:", lw=1)
ax.set(xlabel="predicted spike offset (s)", ylabel="wheel xcorr offset (s)", title=f"Behavioural validation (n={len(c)} confident)")
fig.tight_layout()
fig.savefig(FIG.joinpath("wheel_vs_rule.png"), dpi=150)

fig, ax = plt.subplots(figsize=(7, 5))
d = dmg.dropna(subset=["boundary_halfperiods"])
d = d[d.boundary_halfperiods.abs() < 20]
sns.scatterplot(data=d, x="dn", y="boundary_halfperiods", hue=d.offset_rule_s != 0, palette={True: "tab:red", False: "tab:gray"}, s=40, ax=ax)
ax.plot([-15, 2], [-15, 2], "k:", lw=1)
ax.set(xlabel="edge-count difference n_probe - n_nidq (sync fronts)", ylabel="(start lead - end lead) / 0.5 s (metadata)", title=f"Edge deficit at the file boundaries (n={len(d)})")
ax.legend(title="shifted")
fig.tight_layout()
fig.savefig(FIG.joinpath("boundary_vs_edges.png"), dpi=150)

ch = cen[(cen.lab == "churchlandlab_ucla") & cen.date.between("2023-06-01", "2023-12-31")].copy()
ch["date"] = pd.to_datetime(ch.date)
ch["combo"] = ch.appVersion + " | " + ch.imDatBsc_fw
fig, ax = plt.subplots(figsize=(10, 3.5))
sns.stripplot(data=ch, x="date", y="combo", hue="category", palette={"aligned": "tab:gray", "shifted": "tab:red", "ambiguous": "tab:orange"},
              native_scale=True, jitter=0.25, size=4, ax=ax)
ax.set(xlabel="", ylabel="SpikeGLX | BSC firmware", title="churchlandlab_ucla probes, 2023")
fig.tight_layout()
fig.savefig(FIG.joinpath("churchland_timeline.png"), dpi=150)

for src in ["006_wheel_xcorr.png"]:
    FIG.joinpath(src).write_bytes(SRC.joinpath(src).read_bytes())
print(flagged.category.value_counts().to_string())
print(flagged[flagged.category == "shifted"].groupby(["mechanism", "offset_s"]).size().to_string())
