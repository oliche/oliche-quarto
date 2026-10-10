"""
LFP, bipolar and CSD of the 11 benchmark chunks: uncompressed (production worker output, no Cadzow), the uncompressed
v05 Cadzow output (`cadzow_denoiser_np1`), lfpack v04 fine (v04 Cadzow) and lfpack v05 fine, with the brain regions on
the left. The uncompressed Cadzow column shows what the compression removes from the v05 input. v05 uses the bad
channels as missing (`bad_channels`, no interpolation), as `compress_bin_to_h5` with the lfpack#22 defaults.

Both compressed columns go through the same codec settings (fine tier: epsilon 100, alpha 2.5, shared basis of 32),
each chunk compressed on its own (`cadzow_compression.compress_roundtrip`). The v04 archive itself was compressed
over the whole recording and differs from this per-chunk encoding (22.2 against 26.6 dB on 1a276285), so it is not
used: no v05 archive exists to compare it with. Reconstructions are cached next to the production chunks.
"""
# %%
import tempfile
from pathlib import Path

import h5py
import numpy as np

from cadzow_compression import CACHE_DIR, compress_roundtrip, snr_db
from cadzow_cycle_spin import csd_rms
from cadzow_shrink_real import plot_methods
from cadzow_stagger import ARCHIVE_FINE

TITLES = {'orig': 'uncompressed (no Cadzow)', 'cadzow': 'uncompressed v05 Cadzow', 'v04': 'lfpack v04 fine',
          'v05': 'lfpack v05 fine'}
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
INPUTS = {'v04': 'production', 'v05': 'np1_missing'}  # cached Cadzow outputs (`_exp_{suffix}.npy`)

# %%
if __name__ == '__main__':
    pids = sorted(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy'))
    with tempfile.TemporaryDirectory() as tmp:
        for pid in pids:
            with h5py.File(ARCHIVE_FINE, 'r') as f:
                labels = f[pid]['00/meta'].attrs['labels'][:]
            ys = {'orig': np.load(CACHE_DIR.joinpath(f'{pid}_chunk_orig.npy')).T.astype(np.float64),
                  'cadzow': np.load(CACHE_DIR.joinpath(f'{pid}_exp_np1_missing.npy')).astype(np.float64)}
            snr = {}
            for version, suffix in INPUTS.items():
                den = np.load(CACHE_DIR.joinpath(f'{pid}_exp_{suffix}.npy')).astype(np.float64)
                file = CACHE_DIR.joinpath(f'{pid}_exp_{version}_fine.npy')
                if not file.exists():
                    np.save(file, compress_roundtrip(den, 2.5, tmp)[1].astype(np.float32))
                ys[version] = np.load(file).astype(np.float64)
                snr[version] = snr_db(den, ys[version])
            r0 = csd_rms(ys['orig'])
            kept = {k: csd_rms(y) / r0 for k, y in ys.items() if k != 'orig'}
            titles = {**TITLES, **{k: f'{TITLES[k]} (SNR {snr[k]:.1f} dB vs its Cadzow input)' for k in snr}}
            plot_methods(pid, ys, kept, FIGURE_DIR.joinpath(f'2026-10-10_cadzow_lfpack_fine_{pid[:8]}.webp'),
                         titles=titles, seam_step={'cadzow': 32, 'v04': 32, 'v05': 32}, labels=labels, ref='v04',
                         suptitle=f'{pid}, 4-7 s into the chunk. Stars: Cadzow window boundaries; triangles: dead '
                                  f'(black) and noisy (red) channels')
            print(pid[:8], {k: round(v, 3) for k, v in kept.items()}, {k: round(v, 1) for k, v in snr.items()},
                  flush=True)
