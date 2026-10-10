"""
lfpack compression of the 11 benchmark chunks before (lfpack v04 Cadzow) and after (`cadzow_denoiser_np1`, the
lfpack#22 routing for NP1) the Cadzow change, for the three lfpack v04 tiers.

Both inputs are the production no-Cadzow chunk worker output (`_chunk_orig.npy`, 32.8 s). Before is denoised with the
v04 settings then re-interpolated at the bad channels (`groupby_y=True`), as the v04 chunk worker; after is denoised
with the bad channels as missing (`bad_channels`, the worker's `bad_channels_as_missing=True`) and not re-interpolated. Each is written as a
time-first float32 .npy and compressed with `lfpack.compress_to_h5`; size is the HDF5 file size, SNR is the
reconstruction against its own (denoised) input, in voltage and in CSD. The after input is also compressed at
lower alphas, so its size at the SNR of before can be interpolated for every tier (`size_at_equal_snr`).
"""
# %%
import sys
import tempfile
from pathlib import Path

# the Cadzow helpers (synthetic field, cached chunks, plots) live in the Cadzow seams analysis; every script of this
# folder imports this module first
sys.path.insert(0, str(Path(__file__).parents[1].joinpath('2026-10-cadzow-seams')))

import h5py
import lfpack
import neuropixel
import numpy as np
import pandas as pd
from ibldsp import cadzow
from ibldsp.voltage import current_source_density

from cadzow_stagger import ARCHIVE_FINE  # noqa: E402

CACHE_DIR = Path.home().joinpath('scratch', 'lfp', 'csd_lfpack_v04')
H = neuropixel.trace_header(version=1)
HXY = {'x': H['x'], 'y': H['y']}
FS = 250.0
TIERS = {'small': 14.0, 'default': 7.0, 'fine': 2.5}  # alpha, lfpack v04 (epsilon=100, basis_size=32)
EXTRA = {'alpha 1.5': 1.5, 'alpha 1.0': 1.0}  # after only: brackets the fine-tier SNR of before


def snr_db(ref, est):
    """Reconstruction SNR [dB]: ref power over error power."""
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - est) ** 2))


def compress_roundtrip(wav, alpha, tmp, epsilon=100.0, basis_size=32):
    """
    Compress (nc, ns) LFP with lfpack and read it back.

    Returns
    -------
    size : int
        HDF5 file size [bytes].
    rec : ndarray (nc, ns)
    """
    npy, h5 = Path(tmp).joinpath('in.npy'), Path(tmp).joinpath('out.h5')
    np.save(npy, wav.T.astype(np.float32))
    h5.unlink(missing_ok=True)
    lfpack.compress_to_h5(npy, h5, "rec", h=HXY, epsilon=epsilon, alpha=alpha, fs=FS, basis_size=basis_size, n_jobs=1)
    rec, _ = lfpack.LFPackReader(h5).read(nsel=slice(0, wav.shape[1]))
    return h5.stat().st_size, np.asarray(rec, dtype=np.float64).T


def size_at_equal_snr(df):
    """
    Size of after relative to before at the reconstruction SNR of before, per insertion and tier.

    log(size) of after is interpolated linearly in SNR between its compression settings.

    Returns
    -------
    pandas.DataFrame
        pid x tier.
    """
    out = []
    for pid, d in df.groupby('pid'):
        a, b = d[d.method == 'after'].sort_values('snr_db'), d[d.method == 'before'].set_index('tier')
        for tier in TIERS:
            s = np.interp(b.loc[tier, 'snr_db'], a.snr_db, np.log(a.size_kb), left=np.nan, right=np.nan)
            out.append({'pid': pid[:8], 'tier': tier, 'ratio': np.exp(s) / b.loc[tier, 'size_kb']})
    return pd.DataFrame(out).pivot(index='pid', columns='tier', values='ratio')[list(TIERS)]


# %%
if __name__ == '__main__':
    pids = sorted(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy'))
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for pid in pids:
            with h5py.File(ARCHIVE_FINE, 'r') as f:
                labels = f[pid]['00/meta'].attrs['labels'][:]
            orig = np.load(CACHE_DIR.joinpath(f'{pid}_chunk_orig.npy')).T.astype(np.float64)
            file_after = CACHE_DIR.joinpath(f'{pid}_exp_np1_missing.npy')
            if not file_after.exists():
                bad = np.isin(labels, (1, 2))
                np.save(file_after, cadzow.cadzow_denoiser_np1(orig, h=HXY, fs=FS, bad_channels=bad, n_jobs=8))
            inputs = {'before': np.load(CACHE_DIR.joinpath(f'{pid}_exp_production.npy')).astype(np.float64),
                      'after': np.load(file_after).astype(np.float64)}
            for method, wav in inputs.items():
                csd = current_source_density(wav, H)
                for tier, alpha in {**TIERS, **(EXTRA if method == 'after' else {})}.items():
                    size, rec = compress_roundtrip(wav, alpha, tmp)
                    rows.append({'pid': pid, 'method': method, 'tier': tier, 'size_kb': size / 1024,
                                 'cr': wav.size * 4 / size, 'snr_db': snr_db(wav, rec),
                                 'csd_snr_db': snr_db(csd, current_source_density(rec, H)),
                                 'rms_kept': np.std(wav) / np.std(orig)})
            print(pid[:8], pd.DataFrame(rows[-8:])[['method', 'tier', 'cr', 'snr_db']].round(1).values.tolist(),
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent.joinpath('2026-10-10_cadzow_compression.csv'), index=False)
    piv = df.pivot_table(index='pid', columns=['tier', 'method'], values='size_kb')
    ratio = pd.DataFrame({t: piv[(t, 'after')] / piv[(t, 'before')] for t in TIERS})
    print('size after / before, per tier\n', ratio.describe().loc[['mean', '50%', 'min', 'max']].round(3))
    eq = size_at_equal_snr(df)
    eq.to_csv(Path(__file__).parent.joinpath('2026-10-10_cadzow_compression_equal_snr.csv'))
    print('size after / before at the SNR of before\n', eq.median().round(3))
    print(df.groupby(['tier', 'method'])[['cr', 'snr_db', 'csd_snr_db', 'rms_kept']].median().round(2))
