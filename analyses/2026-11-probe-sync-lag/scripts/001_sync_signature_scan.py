"""Sync-edge signature for flagged insertions and aligned controls (small raw sync files + meta only).

Per probe: imec_sync edge counts on nidq and probe, first-edge polarities, the start offset of the probe file relative
to the nidq file from SpikeGLX `firstSample`, and the lag (in edges) implied by that offset.
ibllib version3B pairs edges from index 0; a non-zero implied lag shifts spikes by lag * 0.5 s.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from one.api import ONE

QC = Path(__file__).parents[2].joinpath("repeated-site-decoding-power", "artifacts", "003_spike_alignment_qc.csv")
one = ONE()
qc = pd.read_csv(QC)
controls = qc[(~qc.flag) & (qc.lab == "churchlandlab_ucla") & (qc.prominence > 10)].head(4)
others = qc[(~qc.flag) & (qc.lab != "churchlandlab_ucla") & (qc.prominence > 20)].sample(4, random_state=0)
sel = pd.concat([qc[qc.flag], controls, others])


def meta(path):
    return dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)


def edges(raw, suffix, wiring):
    t, c, p = (np.load(one.load_dataset(EID, f"{raw}/_spikeglx_sync.{a}{suffix}.npy", download_only=True)) for a in ["times", "channels", "polarities"])
    ch = int([k for k, v in wiring["SYNC_WIRING_DIGITAL"].items() if v == "imec_sync"][0].split(".")[-1])
    return t[c == ch], p[c == ch]


rows = []
for _, r in sel.iterrows():
    EID = r.eid
    pname = one.pid2eid(r.pid)[1]
    dsets = one.list_datasets(EID)
    try:
        nmeta_f = [d for d in dsets if d.startswith("raw_ephys_data/_spikeglx") and d.endswith("nidq.meta")][0]
        pmeta_f = [d for d in dsets if d.startswith(f"raw_ephys_data/{pname}/") and d.endswith("ap.meta")][0]
        nmeta, pmeta = (meta(one.load_dataset(EID, f, download_only=True)) for f in [nmeta_f, pmeta_f])
        nwir = json.loads(Path(one.load_dataset(EID, nmeta_f.replace(".meta", ".wiring.json"), download_only=True)).read_text())
        pwir = json.loads(Path(one.load_dataset(EID, pmeta_f.replace(".ap.meta", ".wiring.json"), download_only=True)).read_text())
        tn, pn = edges("raw_ephys_data", "", nwir)
        tp, pp = edges(f"raw_ephys_data/{pname}", f".{pname}", pwir)
    except Exception as e:  # noqa: BLE001
        rows.append(dict(pid=r.pid, eid=EID, subject=r.subject, date=r.date, flag=r.flag, error=repr(e)[:120]))
        continue
    t0n = float(nmeta["firstSample"]) / float(nmeta["niSampRate"])
    t0p = float(pmeta["firstSample"]) / float(pmeta["imSampRate"])
    start_lead = t0n - t0p  # > 0: probe file started before nidq file
    # implied lag: probe edge index matching nidq edge 0, from file start offsets
    lag = int(np.round((tn[0] + start_lead - tp[0]) / 0.5))
    rows.append(dict(pid=r.pid, eid=EID, subject=r.subject, date=r.date, flag=r.flag, transient_s=r.transient_s,
                     n_nidq=tn.size, n_probe=tp.size, pol0_nidq=pn[0], pol0_probe=pp[0], t0_nidq=tn[0], t0_probe=tp[0],
                     probe_start_lead_s=start_lead, implied_lag_edges=lag, implied_offset_s=lag * 0.5,
                     app=pmeta.get("appVersion"), fw=pmeta.get("imDatBsc_fw"), probe_pn=pmeta.get("imDatPrb_pn")))
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).with_suffix(".csv"), index=False)
pd.set_option("display.width", 300)
print(df.drop(columns=["eid"]).round(3).to_string())
