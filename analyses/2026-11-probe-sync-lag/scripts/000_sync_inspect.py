"""Inspect probe vs nidq sync for one flagged MFD session: pulse trains, the probe->session fit, and residuals."""
import json
from pathlib import Path

import numpy as np
from one.api import ONE

EID = "15b69921-d471-4ded-8814-2adad954bcd8"  # MFD_07 2023-08-31 001, probe00 = 7a620688-66cb-44d3-b79b-ccac1c8ba23e
one = ONE()
files = [
    "raw_ephys_data/_spikeglx_sync.times.npy", "raw_ephys_data/_spikeglx_sync.channels.npy", "raw_ephys_data/_spikeglx_sync.polarities.npy",
    "raw_ephys_data/_spikeglx_ephysData_g0_t0.nidq.meta", "raw_ephys_data/_spikeglx_ephysData_g0_t0.nidq.wiring.json",
    "raw_ephys_data/probe00/_spikeglx_sync.times.probe00.npy", "raw_ephys_data/probe00/_spikeglx_sync.channels.probe00.npy",
    "raw_ephys_data/probe00/_spikeglx_sync.polarities.probe00.npy", "raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.meta",
    "raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.wiring.json", "raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.sync.npy",
    "raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.timestamps.npy",
]
paths = {f: one.load_dataset(EID, f, download_only=True) for f in files}
for f in files:
    print(f, "->", paths[f])
nwir = json.loads(Path(paths[files[4]]).read_text())
pwir = json.loads(Path(paths[files[9]]).read_text())
print("nidq wiring", nwir)
print("probe wiring", pwir)
for f in [files[3], files[8]]:
    meta = Path(paths[f]).read_text().splitlines()
    print(f, [m for m in meta if any(k in m for k in ["SampRate", "fileTimeSecs", "firstSample", "appVersion", "imDatPrb_type", "typeThis", "syncSource", "syncSourcePeriod", "fileCreateTime", "imDatBsc_fw", "imDatPrb_pn"])])
nt, nc, npol = (np.load(paths[f]) for f in files[:3])
pt, pc, ppol = (np.load(paths[f]) for f in files[5:8])
print("nidq channels", np.unique(nc, return_counts=True))
print("probe channels", np.unique(pc, return_counts=True))
sync_ch_n = nwir["SYNC_WIRING_DIGITAL"]
imec_n = [int(k.split(".")[-1]) for k, v in sync_ch_n.items() if v == "imec_sync"][0]
imec_p = [int(k.split(".")[-1]) for k, v in pwir["SYNC_WIRING_DIGITAL"].items() if v == "imec_sync"][0]
tn = nt[(nc == imec_n)]
tp = pt[(pc == imec_p)]
print(f"nidq imec_sync ch {imec_n}: n={tn.size} first={tn[:6]} last={tn[-3:]}")
print(f"probe imec_sync ch {imec_p}: n={tp.size} first={tp[:6]} last={tp[-3:]}")
print("nidq dt", np.unique(np.round(np.diff(tn), 3), return_counts=True))
print("probe dt", np.unique(np.round(np.diff(tp), 3), return_counts=True))
s = np.load(paths[files[10]])
ts = np.load(paths[files[11]])
print("sync.npy", s.shape, s[:3], s[-3:])
print("timestamps.npy", ts.shape, ts[:3], ts[-3:])
np.savez(Path(__file__).parents[1].joinpath("artifacts", "000_mfd07_sync.npz"), tn=tn, tp=tp, sync=s, ts=ts)
