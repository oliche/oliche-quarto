"""Session table for the UCLA rig (mechanism A): SpikeGLX 20201103 + base station firmware 2.0.169 / 3.2.186.

Reads the metadata census (009) and the flagged-probe table; adds session details from Alyx.
Writes ucla_rig_sessions.csv (one row per session) and ucla_rig_probes.csv (one row per probe).
"""
from pathlib import Path

import pandas as pd
from one.api import ONE

HERE = Path(__file__).parent
SRC = Path.home().joinpath("Documents", "ibl-ai-agent", "projects", "spike-clock-offset-investigation", "exploratory-analyses")
ADMIN = "https://alyx.internationalbrainlab.org/admin"
one = ONE()

cen = pd.read_parquet(SRC.joinpath("009_meta_rule_census.pqt"))
flagged = pd.read_csv(HERE.joinpath("flagged_probes.csv"))
rig = cen[(cen.lab == "churchlandlab_ucla") & (cen.appVersion == "20201103") & (cen.imDatBsc_fw == "3.2.186")].copy()
rig = rig.drop_duplicates(["eid", "probe"])
rig = rig.merge(flagged[["eid", "probe", "pid", "offset_s", "offset_wheel_s", "in_bwm_release"]], on=["eid", "probe"], how="left")

rows = []
for eid in rig.eid.unique():
    ses = one.alyx.rest("sessions", "read", id=eid)
    ref = one.eid2ref(eid)
    p = rig[rig.eid == eid].sort_values("probe")
    rows.append(dict(
        date=str(ref["date"]), subject=ref["subject"], number=f"{ref['sequence']:03d}", eid=eid,
        projects=", ".join(ses.get("projects") or []), task_protocol=ses.get("task_protocol"),
        n_probes=len(p), probes=", ".join(p.probe), offset_s=", ".join(sorted({f"{v:+.1f}" for v in p.offset_s})),
        wheel_offset_s=", ".join(f"{v:+.2f}" for v in p.offset_wheel_s.dropna()),
        pids=", ".join(p.pid.fillna("(no insertion)")), in_bwm_release=bool(p.in_bwm_release.fillna(False).any()),
        base_station_sn=", ".join(sorted(set(p.imDatBsc_sn))), alyx=f"{ADMIN}/actions/session/{eid}/change/"))
ses = pd.DataFrame(rows).sort_values(["date", "subject", "number"])
ses.to_csv(HERE.joinpath("ucla_rig_sessions.csv"), index=False)
rig[["eid", "probe", "pid", "subject", "date", "offset_s", "offset_wheel_s", "in_bwm_release", "start_lead_s", "end_lead_s", "boundary",
     "imDatBs_fw", "imDatBsc_fw", "imDatBsc_sn", "imDatHs_fw", "imDatHs_sn", "imDatFx_hw", "imDatPrb_pn"]].sort_values(["date", "probe"]).to_csv(
    HERE.joinpath("ucla_rig_probes.csv"), index=False)
pd.set_option("display.width", 300, "display.max_rows", 100, "display.max_colwidth", 60)
print(ses.drop(columns=["alyx", "pids"]).to_string())
print(len(ses), "sessions,", ses.n_probes.sum(), "probes")

# markdown pid table for the report, viz links only for BWM-release insertions (the viz site serves dset=bwm only)
VIZ = "https://viz.internationalbrainlab.org/app?spikesorting=ss_2024-05-06&dset=bwm&pid={pid}&tid=0&cid=0&qc=0#trialviewer"
pr = rig[rig.pid.notna()].merge(ses[["eid", "number", "projects"]], on="eid", suffixes=("_census", ""))
pr["bwm"] = pr.in_bwm_release.eq(True)  # BWM 2023 release insertions are listed first
pr = pr.sort_values(["bwm", "date", "subject", "probe"], ascending=[False, True, True, True])
lines = ["| BWM release | date | subject | number | probe | pid | eid | project | shift (s) | viz |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in pr.itertuples():
    pid_link = f"[{r.pid}]({ADMIN}/experiments/probeinsertion/{r.pid}/change/)"
    eid_link = f"[{r.eid}]({ADMIN}/actions/session/{r.eid}/change/)"
    viz = f"[trials]({VIZ.format(pid=r.pid)})" if r.bwm else ""
    lines.append(f"| {'**BWM**' if r.bwm else ''} | {r.date} | {r.subject} | {r.number} | {r.probe} | {pid_link} | {eid_link} | {r.projects} | {r.offset_s:+.1f} | {viz} |")
HERE.joinpath("ucla_rig_pids.md").write_text("\n".join(lines) + "\n")
print(len(pr), "pids written")
