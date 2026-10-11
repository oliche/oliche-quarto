"""
Behaviour decoding of lfpack v05 on the 11 benchmark insertions, and of the shipped v04 archives as its reference.

The v04 points of the decoding figure come from the codec re-implementation run on the June Cadzow checkpoints
(`lf_resampled_car_cadzow.npy`, June preprocessing).  v05 goes through the production pipeline
(`lfpack.compress_bin_to_h5`: 0.5 Hz high-pass, saturation muting, Cadzow NP1 full grid + shrinkage with the bad
channels as missing), so the shipped v04 archives (same production pipeline, v04 Cadzow) are decoded as well to
compare like with like.

Phases (`python decode_v05.py <phase>`):
    v04     decode the three tiers of the merged v04 archives (DATADISK), no recompression
    v05     per insertion: download the raw LF band, compress the three v04 tiers with the v05 Cadzow (bad-channel
            labels taken from the v04 archive, as the production run reads its precomputed labels), decode the
            uncompressed v05 Cadzow output and the three files, then delete the raw file and the checkpoint

Each insertion runs in a fresh subprocess (`worker` phase).  Rows are appended to `<TODAY>_decoding_v05.csv`; finished
(pid, tier) pairs are skipped on restart.  Decoding is `shared_basis_sweep.decode_all` on the 4-channel binned signal,
with the clean epochs of `shared_basis_sweep.behaviour`, as for the other tiers of the figure.
"""

# %%
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import lfpack
import numpy as np
import pandas as pd

import shared_basis_sweep as sbs

TODAY = '2026-10-11'
OUT_DIR = Path(__file__).parent
CSV = OUT_DIR.joinpath(f'{TODAY}_decoding_v05.csv')
PYTHON = '/Users/olivier/PycharmProjects/ephys-atlas/.venv/bin/python'
V04_DIR = Path.home().joinpath('Documents', 'datadisk', 'lfp-processing', 'lfpack', 'v04')
V05_DIR = sbs.SCRATCH.joinpath('v05_tiers')
TIERS = {  # tier: (v04 archive, lfpack parameters)
    'small': ('lf_compressed_v04_a14_small_all.h5', dict(epsilon=100.0, alpha=14.0, basis_size=32)),
    'default': ('lf_compressed_v04_a07_default_all.h5', dict(epsilon=100.0, alpha=7.0, basis_size=32)),
    'fine': ('lf_compressed_v04_a2p5_fine_all.h5', dict(epsilon=100.0, alpha=2.5, basis_size=32)),
}
N_JOBS = 8


def done_keys():
    return set() if not CSV.exists() else set(zip(*pd.read_csv(CSV)[['pid', 'tier']].values.T))


def storage_bytes(h5_file, recording):
    """Summed HDF5 storage of the datasets of one recording (comparable between merged and single files)."""
    n = []
    with h5py.File(h5_file, 'r') as f:
        f[recording].visititems(lambda _, o: n.append(o.id.get_storage_size()) if isinstance(o, h5py.Dataset) else None)
    return int(sum(n))


def decode_row(pid, tier, recon, nbytes, one):
    """Decode a (ns, 96) binned signal and append one row to the CSV."""
    wheel, categ = sbs.decode_all(recon, sbs.behaviour(pid, one))
    row = dict(pid=pid, tier=tier, wheel=wheel['r2'], choice=categ['choice']['balanced_accuracy'],
               feedback=categ['feedback']['balanced_accuracy'], bytes=nbytes)
    sbs._append(CSV, row)
    print(f"{pid[:8]} {tier}: R² {row['wheel']:.4f}, choice {row['choice']:.3f}, feedback {row['feedback']:.3f}, "
          f"{nbytes / 1e6:.1f} MB", flush=True)


def read_binned(h5_file, recording=None):
    """Whole recording decompressed and summed over 4 adjacent channels, (ns, nc / 4)."""
    reader = lfpack.LFPackReader(h5_file, recording=recording, bin_channels=4)
    return reader.read(slice(0, reader.ns), sync=False)


# %% Workers
def worker_v04(pid):
    from one.api import ONE

    keys, one = done_keys(), ONE()
    for tier, (archive, _) in TIERS.items():
        if (pid, f'v04 {tier} (archive)') in keys:
            continue
        h5 = V04_DIR.joinpath(archive)
        decode_row(pid, f'v04 {tier} (archive)', read_binned(h5, recording=pid), storage_bytes(h5, pid), one)


def worker_v05(pid):
    from brainbox.io.one import SpikeSortingLoader
    from one.api import ONE

    keys, one = done_keys(), ONE()
    tiers = ['v05 uncompressed'] + [f'v05 {t}' for t in TIERS]
    if all((pid, t) in keys for t in tiers):
        return
    V05_DIR.mkdir(exist_ok=True)
    files = {t: V05_DIR.joinpath(f'{pid}_v05_a{p["alpha"]:g}_{t}.h5') for t, (_, p) in TIERS.items()}
    checkpoint = V05_DIR.joinpath(f'{pid}_cadzow_v05.npy')
    labels = lfpack.LFPackReader(V04_DIR.joinpath(TIERS['small'][0]), recording=pid).channels_full['labels']
    ssl = SpikeSortingLoader(pid=pid, one=one)
    sr = None
    for tier, (_, params) in TIERS.items():
        if files[tier].exists():
            continue
        sr = sr or ssl.raw_electrophysiology(band='lf', stream=False)
        t0 = time.perf_counter()
        tmp = files[tier].with_suffix('.h5tmp')
        lfpack.compress_bin_to_h5(sr.file_bin, tmp, recording=pid, q=sbs.Q, cadzow_checkpoint_file=checkpoint,
                                  cadzow_kwargs={}, channel_labels=labels, n_jobs=N_JOBS, **params)
        tmp.rename(files[tier])
        print(f'{pid[:8]} v05 {tier}: compressed in {time.perf_counter() - t0:.0f} s', flush=True)
    if (pid, 'v05 uncompressed') not in keys:
        x = np.load(checkpoint, mmap_mode='r')
        recon = np.concatenate([np.asarray(x[i:i + 65536], np.float32).reshape(-1, x.shape[1] // 4, 4).sum(axis=2)
                                for i in range(0, x.shape[0], 65536)])
        decode_row(pid, 'v05 uncompressed', recon, checkpoint.stat().st_size, one)
        del recon
    for tier in TIERS:
        if (pid, f'v05 {tier}') not in keys:
            decode_row(pid, f'v05 {tier}', read_binned(files[tier]), storage_bytes(files[tier], pid), one)
    # free the disk for the next insertion: the raw LF band (ONE cache) and the Cadzow checkpoint
    checkpoint.unlink(missing_ok=True)
    if sr is not None:
        Path(sr.file_bin).unlink()


def run_worker(phase, pid):
    subprocess.run([PYTHON, __file__, f'worker_{phase}', pid], check=True, cwd=OUT_DIR)


if __name__ == '__main__':
    phase = sys.argv[1]
    if phase.startswith('worker_'):
        {'worker_v04': worker_v04, 'worker_v05': worker_v05}[phase](sys.argv[2])
    else:
        for pid in (sys.argv[2:] or sbs.PIDS):
            run_worker(phase, pid)
