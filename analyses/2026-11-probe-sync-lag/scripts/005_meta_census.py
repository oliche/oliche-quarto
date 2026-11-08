"""Metadata-only census of probe vs nidq file start offsets for every 3B session on Alyx.

For each probe ap.meta with a nidq.meta in the same session: lead = firstSample_nidq / niSampRate -
firstSample_probe / imSampRate (s, > 0 when the probe file started first). The legacy ibllib version3B pairing
assumes lead ~ 0; |lead| near a multiple of the 0.5 s sync half-period means spikes shifted by that amount.
Meta files are fetched directly from the data server (32 threads) and cached in artifacts/005_meta/.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from one.api import ONE

HERE = Path(__file__).parent
CACHE = HERE.parent.joinpath("artifacts", "005_meta")
CACHE.mkdir(parents=True, exist_ok=True)
KEYS = ["appVersion", "typeThis", "firstSample", "fileTimeSecs", "imSampRate", "niSampRate", "imDatBs_fw", "imDatBsc_fw", "imDatBsc_hw",
        "imDatBsc_sn", "imDatHs_fw", "imDatHs_sn", "imDatFx_hw", "imDatPrb_pn", "imDatPrb_type", "imDatPrb_port", "imDatPrb_slot", "syncImInputSlot",
        "imDatApi", "imSvyMaxBnk"]
one = ONE()
http = requests.Session()
http.auth = (one.alyx._par.HTTP_DATA_SERVER_LOGIN, one.alyx._par.HTTP_DATA_SERVER_PWD)

dsets = []
for suffix in ["nidq.meta", "ap.meta"]:
    dsets += one.alyx.rest("datasets", "list", django=f"name__endswith,{suffix},default_dataset,True")
recs = []
for d in dsets:
    url = next((fr["data_url"] for fr in d["file_records"] if fr["exists"] and fr["data_url"]), None)
    recs.append(dict(did=d["url"].split("/")[-1], eid=d["session"].split("/")[-1], collection=d["collection"], name=d["name"], url=url))
recs = pd.DataFrame(recs)
print(f"{len(recs)} meta datasets, {recs.url.isna().sum()} without a data url")


def fetch(row):
    f = CACHE.joinpath(f"{row.did}.meta")
    if not f.exists():
        if not isinstance(row.url, str):
            return {}
        for attempt in range(3):
            try:
                r = http.get(row.url, timeout=60)
                break
            except requests.RequestException as e:
                err = repr(e)[:100]
        else:
            return dict(fetch_error=err)
        if r.status_code == 200:
            f.write_bytes(r.content)
        else:  # private S3 bucket: fall back on ONE, which holds the S3 credentials
            try:
                f.write_bytes(Path(one.load_dataset(row.eid, f"{row.collection}/{row.name}", download_only=True)).read_bytes())
            except Exception as e:  # noqa: BLE001
                return dict(fetch_error=f"{r.status_code} then ONE {e!r}"[:120])
    md = dict(line.split("=", 1) for line in f.read_text().splitlines() if "=" in line)
    return {k: md.get(k) for k in KEYS}


with ThreadPoolExecutor(32) as ex:
    meta = pd.DataFrame(list(ex.map(fetch, recs.itertuples())))
recs = pd.concat([recs, meta], axis=1)
recs.to_parquet(HERE.parent.joinpath("artifacts", "005_meta_records.pqt"))

nidq = recs[recs.name.str.endswith("nidq.meta") & (recs.collection == "raw_ephys_data")].drop_duplicates("eid")
probes = recs[recs.name.str.endswith("ap.meta")].copy()
df = probes.merge(nidq[["eid", "firstSample", "niSampRate", "fileTimeSecs", "appVersion"]], on="eid", suffixes=("", "_nidq"))
for c in ["firstSample", "imSampRate", "fileTimeSecs", "firstSample_nidq", "niSampRate_nidq", "fileTimeSecs_nidq"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["start_lead_s"] = df.firstSample_nidq / df.niSampRate_nidq - df.firstSample / df.imSampRate
df["end_lead_s"] = df.start_lead_s + df.fileTimeSecs_nidq - df.fileTimeSecs
df["lead_halfperiods"] = df.start_lead_s / 0.5
df["misaligned"] = df.start_lead_s.abs() > 0.1
df["probe"] = df.collection.str.split("/").str[1]
# session metadata for context
eids = list(df.eid.unique())
ses = {s["id"]: s for i in range(0, len(eids), 100) for s in one.alyx.rest("sessions", "list", django=f"id__in,{eids[i:i + 100]}")}
df["lab"] = df.eid.map(lambda e: ses.get(e, {}).get("lab"))
df["subject"] = df.eid.map(lambda e: ses.get(e, {}).get("subject"))
df["date"] = df.eid.map(lambda e: str(ses.get(e, {}).get("start_time", ""))[:10])
df["projects"] = df.eid.map(lambda e: ",".join(ses.get(e, {}).get("projects") or []))
df.drop(columns=["url"]).to_parquet(HERE.joinpath("005_meta_census.pqt"))
print(df.misaligned.value_counts(dropna=False).to_string())
