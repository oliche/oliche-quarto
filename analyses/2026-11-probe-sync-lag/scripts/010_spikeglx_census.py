"""SpikeGLX census of every probe recorded with a nidq file (3B, NP2, Quad-base...) on the Flatiron/popeye mount.

Runs on the popeye head node: reads only metadata and the small `_spikeglx_sync.*` arrays, never the raw binaries.

Stages (each cached, delete the file to recompute):
1. `010_paths.pqt`: filesystem walk of /mnt/sdceph/users/ibl/data/<lab>/Subjects/<subject>/<date>/<number>/raw_ephys_data,
   one row per probe ap.meta in a session that also has a nidq.meta, with the full paths of the meta, raw binary,
   sync arrays and wiring files of both streams.
2. `010_insertions.pqt`: Alyx probe insertions per lab, to attach the pid (matched on subject/date/number/probe name).
3. `010_spikeglx_census.pqt` / `.csv`: one row per probe. Raw meta fields verbatim as strings (`nidq.<key>`,
   `probe.<key>`), imec sync edge statistics per stream, derived timing columns. Columns are defined in
   `../census_columns.md`.
4. `010_meta_raw.pqt`: every non-table (`~`) key of every meta file read, long format (path, key, value).
5. `010_edges/<lab>_<subject>_<date>_<number>_<nidq|probeXX>.npz`: imec sync edge times and polarities per stream.

Usage: python 010_spikeglx_census.py [n_threads]
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = Path("/mnt/sdceph/users/ibl/data")
N_THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
DEFAULT_IMEC_SYNC = {"nidq": 3, "probe": 6}  # ibllib 3B defaults when no wiring.json
# meta keys copied verbatim into the census table, per stream
RAW_KEYS_COMMON = ["appVersion", "typeThis", "fileCreateTime", "firstSample", "fileTimeSecs", "fileSizeBytes", "nSavedChans",
                   "syncSourceIdx", "syncSourcePeriod"]
RAW_KEYS = {"nidq": RAW_KEYS_COMMON + ["niSampRate", "syncNiChan", "syncNiChanType"],
            "probe": RAW_KEYS_COMMON + ["imSampRate", "imDatApi", "imDatBs_fw", "imDatBsc_fw", "imDatBsc_hw", "imDatHs_fw", "imDatPrb_pn",
                                        "imDatPrb_sn", "imDatPrb_type", "imDatPrb_slot", "imDatPrb_port", "syncImInputSlot"]}
EDGES = HERE.joinpath("010_edges")
FIRSTSAMPLE_FIX = "20221012"  # SpikeGLX version from which firstSample is reliable
UUID = re.compile(r"\.[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?=\.[^.]+$)")


def strip_uuid(name):
    """Remove the Flatiron dataset uuid from a file name: `a.b.<uuid>.npy` -> `a.b.npy`."""
    return UUID.sub("", name)


def find_one(folder_files, pattern):
    """Full paths of the files of a folder whose uuid-stripped name matches a regex, as a `;`-joined string."""
    hits = sorted(str(f) for f in folder_files if re.fullmatch(pattern, strip_uuid(f.name)))
    return ";".join(hits) if hits else None


def stream_paths(files, kind):
    """Paths of the meta, raw binary, sync arrays and wiring of one stream (nidq or probe) given its folder listing."""
    ext = "nidq" if kind == "nidq" else "ap"
    out = {f"{kind}_meta": find_one(files, rf".*\.{ext}\.meta"), f"{kind}_raw": find_one(files, rf".*\.{ext}\.(cbin|bin)"),
           f"{kind}_wiring": find_one(files, rf".*{'nidq.' if kind == 'nidq' else ''}wiring\.json")}
    for a in ["times", "channels", "polarities"]:
        out[f"{kind}_sync_{a}"] = find_one(files, rf"_spikeglx_sync\.{a}(\.probe\w+)?\.npy")
    return out


def walk_session(sdir):
    """All probes of one session folder that has a nidq meta; empty list otherwise."""
    raw = sdir.joinpath("raw_ephys_data")
    if not raw.is_dir():
        return []
    entries = list(raw.iterdir())
    top = [f for f in entries if f.is_file()]
    if not any(strip_uuid(f.name).endswith(".nidq.meta") for f in top):
        return []
    lab, _, subject, date, number = sdir.parts[-5:]
    nidq = stream_paths(top, "nidq")
    rows = []
    for pdir in sorted(d for d in entries if d.is_dir() and d.name.startswith("probe")):
        pfiles = [f for f in pdir.iterdir() if f.is_file()]
        if not any(strip_uuid(f.name).endswith(".ap.meta") for f in pfiles):
            continue
        rows.append({"lab": lab, "subject": subject, "date": date, "number": int(number), "probe": pdir.name} | nidq | stream_paths(pfiles, "probe"))
    return rows


def list_sessions(lab_dir):
    """Session folders <lab>/Subjects/<subject>/<date>/<number>."""
    return [n for s in lab_dir.joinpath("Subjects").iterdir() if s.is_dir()
            for d in s.iterdir() if d.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name)
            for n in d.iterdir() if n.is_dir() and n.name.isdigit()]


def with_path_suffix(df):
    """Name the path columns `<stream>_<file>_path` (the first cached version of 010_paths.pqt had no suffix)."""
    return df.rename(columns=lambda c: f"{c}_path" if c.startswith(("nidq_", "probe_")) and not c.endswith("_path") else c)


def build_paths():
    f = HERE.joinpath("010_paths.pqt")
    if f.exists():
        return with_path_suffix(pd.read_parquet(f))
    labs = [d for d in ROOT.iterdir() if d.joinpath("Subjects").is_dir() and d.name != "aggregates"]
    with ThreadPoolExecutor(N_THREADS) as ex:
        sessions = [s for ss in ex.map(list_sessions, labs) for s in ss]
        print(f"{len(sessions)} session folders in {len(labs)} labs")
        df = pd.DataFrame([r for rr in ex.map(walk_session, sessions) for r in rr])
    df.to_parquet(f)
    return with_path_suffix(df)


def build_insertions(labs):
    f = HERE.joinpath("010_insertions.pqt")
    if f.exists():
        return pd.read_parquet(f)
    from one.api import ONE
    one = ONE(base_url="https://alyx.internationalbrainlab.org", silent=True)
    recs = []
    for lab in labs:
        for i in one.alyx.rest("insertions", "list", django=f"session__lab__name,{lab}", no_cache=True):
            si = i["session_info"]
            recs.append({"pid": i["id"], "eid": i["session"], "subject": si["subject"], "date": si["start_time"][:10],
                             "number": int(si["number"]), "probe": i["name"], "projects": ",".join(si.get("projects") or [])})
        print(lab, len(recs))
    df = pd.DataFrame(recs)
    df.to_parquet(f)
    return df


def has(x):
    """True for a non-empty path string (missing paths are None or NaN after a parquet round trip)."""
    return isinstance(x, str) and len(x) > 0


def read_meta(path):
    """All `key=value` lines of a SpikeGLX meta file, values as strings."""
    return dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)


def sync_channel(wiring, kind):
    if has(wiring):
        w = json.loads(Path(wiring.split(";")[0]).read_text())
        ch = [k for k, v in w.get("SYNC_WIRING_DIGITAL", {}).items() if v == "imec_sync"]
        if ch:
            return int(ch[0].split(".")[-1]), "wiring"
    return DEFAULT_IMEC_SYNC[kind], "default"


def edge_stats(t, p, kind):
    """Counts, first/last edge and interval anomalies of one stream's imec sync edges (half-period 0.5 s)."""
    if t.size == 0:
        return {}
    d = np.diff(t)
    return {f"{kind}_pol_first": int(p[0]), f"{kind}_pol_last": int(p[-1]), f"{kind}_t_first_edge_s": t[0], f"{kind}_t_last_edge_s": t[-1],
            f"{kind}_median_half_period_s": np.median(d) if d.size else np.nan, f"{kind}_max_gap_s": d.max() if d.size else np.nan,
            f"{kind}_n_long_gaps": int((d > 0.75).sum()), f"{kind}_n_short_gaps": int((d < 0.25).sum()),
            f"{kind}_n_same_polarity": int((np.diff(p) == 0).sum())}


def stream_features(row, kind, label):
    """Raw meta fields, full meta dump and imec sync edges of one stream; edges are cached to 010_edges/<label>.npz."""
    meta = getattr(row, f"{kind}_meta_path")
    md = read_meta(meta.split(";")[0])
    out = {f"{kind}_n_meta_files": len(meta.split(";"))} | {f"{kind}.{k}": md.get(k) for k in RAW_KEYS[kind]}
    raw = [(meta.split(";")[0], k, v) for k, v in md.items() if not k.startswith("~")]
    if not all(has(getattr(row, f"{kind}_sync_{a}_path")) for a in ["times", "channels", "polarities"]):
        return out | {f"{kind}_sync_status": "no sync arrays"}, raw
    t, c, p = (np.load(getattr(row, f"{kind}_sync_{a}_path").split(";")[0]) for a in ["times", "channels", "polarities"])
    ch, src = sync_channel(getattr(row, f"{kind}_wiring_path"), kind)
    t, p = t[c == ch], p[c == ch]
    f = EDGES.joinpath(f"{label}.npz")
    if not f.exists():
        np.savez(f, times=t, polarities=p.astype(np.int8))
    out |= {f"{kind}_sync_channel": ch, f"{kind}_sync_channel_source": src, f"{kind}_n_edges": t.size, f"{kind}_sync_status": "ok"}
    return out | edge_stats(t, p, kind), raw


def features(row):
    out, raw = {}, []
    ses = f"{row.lab}_{row.subject}_{row.date}_{row.number:03d}"
    for kind, label in [("nidq", f"{ses}_nidq"), ("probe", f"{ses}_{row.probe}")]:
        try:
            o, r = stream_features(row, kind, label)
            out |= o
            raw += r
        except Exception as e:  # noqa: BLE001
            out[f"{kind}_error"] = repr(e)[:200]
    return out, raw


def derive(df):
    """Timing columns derived from the raw meta strings and the sync edges."""
    for k, fs in [("nidq", "niSampRate"), ("probe", "imSampRate")]:
        num = {c: pd.to_numeric(df[f"{k}.{c}"], errors="coerce") for c in ["firstSample", "fileTimeSecs", fs]}
        df[f"{k}_t0_s"] = num["firstSample"] / num[fs]
        df[f"{k}_end_s"] = df[f"{k}_t0_s"] + num["fileTimeSecs"]
        # time from the last sync edge to the end of the file: > 0.5 s means the sync stopped before the file did
        df[f"{k}_tail_s"] = num["fileTimeSecs"] - df[f"{k}_t_last_edge_s"]
    df["start_lead_s"] = df.nidq_t0_s - df.probe_t0_s
    df["end_lead_s"] = df.nidq_end_s - df.probe_end_s
    df["dT_s"] = pd.to_numeric(df["probe.fileTimeSecs"], errors="coerce") - pd.to_numeric(df["nidq.fileTimeSecs"], errors="coerce")
    df["dn_edges"] = df.probe_n_edges - df.nidq_n_edges
    df["firstSample_reliable"] = (df["nidq.appVersion"].fillna("") >= FIRSTSAMPLE_FIX) & (df["probe.appVersion"].fillna("") >= FIRSTSAMPLE_FIX)
    tol = np.where(df.firstSample_reliable, 0.25, 5.0)
    df["truncated"] = df.end_lead_s.abs() > tol
    df["truncated_stream"] = np.select([~df.truncated, df.end_lead_s > 0], [None, "probe"], "nidq")
    df["has_raw"] = df.nidq_raw_path.notna() & df.probe_raw_path.notna()
    return df


if __name__ == "__main__":
    EDGES.mkdir(exist_ok=True)
    paths = build_paths()
    print(f"{len(paths)} probes with a nidq file in {paths.groupby(['lab', 'subject', 'date', 'number']).ngroups} sessions")
    ins = build_insertions(sorted(paths.lab.unique()))
    with ThreadPoolExecutor(N_THREADS) as ex:
        res = list(ex.map(features, paths.itertuples()))
    feats = pd.DataFrame([r[0] for r in res])
    raw = pd.DataFrame([x for r in res for x in r[1]], columns=["path", "key", "value"]).drop_duplicates(["path", "key"])
    raw.to_parquet(HERE.joinpath("010_meta_raw.pqt"))
    df = paths.merge(ins.drop_duplicates(["subject", "date", "number", "probe"]), on=["subject", "date", "number", "probe"], how="left")
    df = derive(pd.concat([df, feats], axis=1))
    df.to_parquet(HERE.joinpath("010_spikeglx_census.pqt"))
    df.to_csv(HERE.joinpath("010_spikeglx_census.csv"), index=False)
    pd.set_option("display.width", 250)
    n_err = sum(df[c].notna().sum() for c in ["nidq_error", "probe_error"] if c in df)
    print(f"{df.pid.notna().sum()} with a pid, {df.has_raw.sum()} with both raw files, {n_err} read errors, {raw.path.nunique()} meta files dumped")
    print(pd.crosstab(df["probe.appVersion"].fillna("?"), df.truncated).to_string())
