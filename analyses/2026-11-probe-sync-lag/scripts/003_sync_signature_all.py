"""Sync-edge signature + SpikeGLX/hardware versions for all 699 BWM insertions and extra churchlandlab_ucla insertions.

Per insertion (small files only: _spikeglx_sync.* for nidq and probe, ap/nidq meta, wiring):
- imec_sync edge counts on nidq/probe, SpikeGLX start/end lead of the probe file vs the nidq file (firstSample),
- implied edge lag (probe edge index matching nidq edge 0) and implied spike offset (lag * half-period) under the
  ibllib version3B index-0 pairing,
- probe duty-cycle asymmetry (high minus low half-period, µs),
- SpikeGLX appVersion and imec hardware/firmware fields.
Results are checkpointed per insertion in artifacts/003_sync_signature/ so reruns only fetch new insertions.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from one.api import ONE

HERE = Path(__file__).parent
ART = HERE.parent.joinpath("artifacts", "003_sync_signature")
ART.mkdir(parents=True, exist_ok=True)
QC = HERE.parents[1].joinpath("repeated-site-decoding-power", "artifacts", "003_spike_alignment_qc.csv")
META_KEYS = ["appVersion", "imDatBs_fw", "imDatBsc_fw", "imDatBsc_hw", "imDatBsc_sn", "imDatHs_fw", "imDatHs_sn", "imDatFx_hw",
             "imDatPrb_pn", "imDatPrb_port", "imDatPrb_slot", "imDatApi", "syncImInputSlot", "syncSourceIdx", "imSampRate", "typeThis"]
DEFAULT_IMEC_SYNC = {"nidq": 3, "probe": 6}  # ibllib 3B defaults when no wiring.json
one = ONE()


def read_meta(path):
    return dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)


def imec_sync_channel(dsets, eid, meta_rel, kind):
    wir = meta_rel.replace(".ap.meta", ".wiring.json").replace(".nidq.meta", ".nidq.wiring.json")
    if wir in dsets:
        w = json.loads(Path(one.load_dataset(eid, wir, download_only=True)).read_text())
        ch = [k for k, v in w["SYNC_WIRING_DIGITAL"].items() if v == "imec_sync"]
        return int(ch[0].split(".")[-1]), "wiring"
    return DEFAULT_IMEC_SYNC[kind], "default"


def signature(pid, eid, pname):
    dsets = list(one.list_datasets(eid))
    out = dict(pid=pid, eid=eid, probe=pname)
    nmeta_rel = [d for d in dsets if d.startswith("raw_ephys_data/_spikeglx") and d.endswith(".nidq.meta")]
    pmeta_rel = [d for d in dsets if d.startswith(f"raw_ephys_data/{pname}/") and d.endswith(".ap.meta")]
    if not pmeta_rel:
        return out | dict(status="no probe ap.meta")
    pmeta = read_meta(one.load_dataset(eid, pmeta_rel[0], download_only=True))
    out |= {k: pmeta.get(k) for k in META_KEYS}
    out["probe_pn_or_type"] = pmeta.get("imDatPrb_pn", pmeta.get("imProbeOpt"))
    if not nmeta_rel:
        return out | dict(status="no nidq (3A?)")
    nmeta = read_meta(one.load_dataset(eid, nmeta_rel[0], download_only=True))
    out["nidq_appVersion"] = nmeta.get("appVersion")
    try:
        sync = {}
        for kind, raw, suffix, meta_rel in [("nidq", "raw_ephys_data", "", nmeta_rel[0]), ("probe", f"raw_ephys_data/{pname}", f".{pname}", pmeta_rel[0])]:
            t, c, p = (np.load(one.load_dataset(eid, f"{raw}/_spikeglx_sync.{a}{suffix}.npy", download_only=True)) for a in ["times", "channels", "polarities"])
            ch, src = imec_sync_channel(dsets, eid, meta_rel, kind)
            sync[kind] = (t[c == ch], p[c == ch])
            out[f"{kind}_sync_ch_source"] = src
    except Exception as e:  # noqa: BLE001
        return out | dict(status=f"sync load error: {e!r}"[:200])
    (tn, pn), (tp, pp) = sync["nidq"], sync["probe"]
    if tn.size < 10 or tp.size < 10:
        return out | dict(status="too few sync edges", n_nidq=tn.size, n_probe=tp.size)
    half = np.median(np.diff(tn))
    t0n = float(nmeta["firstSample"]) / float(nmeta["niSampRate"])
    t0p = float(pmeta["firstSample"]) / float(pmeta["imSampRate"])
    start_lead = t0n - t0p  # > 0: probe file started before the nidq file
    end_lead = (t0n + float(nmeta["fileTimeSecs"])) - (t0p + float(pmeta["fileTimeSecs"]))
    lag_f = (tn[0] + start_lead - tp[0]) / half
    d = np.diff(tp)
    out |= dict(status="ok", n_nidq=tn.size, n_probe=tp.size, dn=tp.size - tn.size, half_period_s=half,
                pol0_nidq=pn[0], pol0_probe=pp[0], t0_nidq=tn[0], t0_probe=tp[0], start_lead_s=start_lead, end_lead_s=end_lead,
                lag_float=lag_f, implied_lag_edges=int(np.round(lag_f)), implied_offset_s=int(np.round(lag_f)) * half,
                probe_duty_asym_us=(d[pp[:-1] == 1].mean() - d[pp[:-1] == -1].mean()) * 1e6,
                start_lead_raw_s=t0n - t0p)
    return out


def run(row):
    f = ART.joinpath(f"{row.pid}.json")
    if f.exists():
        return json.loads(f.read_text())
    try:
        pname = row.probe if isinstance(row.probe, str) else one.pid2eid(row.pid)[1]
        res = signature(row.pid, row.eid, pname)
    except Exception as e:  # noqa: BLE001
        res = dict(pid=row.pid, eid=row.eid, status=f"error: {e!r}"[:200])
    res["set"] = row.set
    f.write_text(json.dumps(res, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    return res


# insertion list: 699 BWM + churchlandlab_ucla BWM-project insertions since 2022-06 + 2 U19_chronic_NP per month
qc = pd.read_csv(QC)[["pid", "eid", "lab", "subject", "date", "transient_s", "prominence", "flag"]]
qc["set"], qc["probe"] = "bwm", None
ch = pd.read_csv(HERE.parent.joinpath("artifacts", "003_churchland_insertions_since_2022-06.csv"))
ch = ch[~ch.pid.isin(qc.pid)]
extra_bwm = ch[ch.projects.str.contains("brainwide")].assign(set="churchland_bwm_project")
u19 = ch[ch.projects == "U19_chronic_NP"].groupby("month").apply(lambda x: x.sample(min(2, len(x)), random_state=0), include_groups=False)
u19 = u19.reset_index(level=0).assign(set="churchland_u19_sample")
extra = pd.concat([extra_bwm, u19]).rename(columns={"name": "probe"}).assign(lab="churchlandlab_ucla")
# metadata census candidates (005): start lead > 0.1 s, or SpikeGLX 20201103 with base station firmware 2.0.169/3.2.186
census = pd.read_parquet(HERE.joinpath("005_meta_census.pqt"))
cand = census[(census.start_lead_s.abs() > 0.1) | ((census.appVersion == "20201103") & (census.imDatBsc_fw == "3.2.186"))]
cand_file = HERE.parent.joinpath("artifacts", "003_census_candidates.csv")
if not cand_file.exists():
    pids = [next((i["id"] for i in one.alyx.rest("insertions", "list", session=r.eid, name=r.probe)), None) for r in cand.itertuples()]
    cand.assign(pid=pids)[["pid", "eid", "lab", "subject", "date", "probe"]].dropna(subset=["pid"]).to_csv(cand_file, index=False)
cand = pd.read_csv(cand_file).assign(set="census_candidate")
todo = pd.concat([qc, extra[["pid", "eid", "lab", "subject", "date", "probe", "set"]]], ignore_index=True)
todo = pd.concat([todo, cand[~cand.pid.isin(todo.pid)]], ignore_index=True)
if len(sys.argv) > 1 and int(sys.argv[1]) > 0:
    todo = todo.head(int(sys.argv[1]))
print(f"{len(todo)} insertions")
with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as ex:
    res = list(ex.map(run, todo.itertuples()))
df = todo.drop(columns=["probe"]).merge(pd.DataFrame(res).drop(columns=["eid", "set"]), on="pid", how="left")
df.to_csv(HERE.joinpath("003_sync_signature_all.csv"), index=False)
print(df.status.value_counts().to_string())
