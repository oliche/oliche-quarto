"""
Bad channels handled inside Cadzow (missing grid positions, `cadzow_fill.cadzow_full_grid(bad=...)`) on the cached
production chunks of the 11 benchmark insertions, against production and against the same shrinkage with the
production interpolation of bad channels before (in the chunk worker) and after Cadzow.

Bad channels are labels 1 (dead) and 2 (noisy) of the v04 archive. With bad channels as missing, their data is
discarded, so the interpolation already applied by the chunk worker does not matter, and nothing is interpolated
after Cadzow.
"""
# %%
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cadzow_cycle_spin import csd_rms
from cadzow_shrink import cadzow_shrink, calibrate_scale
from cadzow_shrink_real import HXY, plot_methods
from cadzow_stagger import ARCHIVE_FINE, CACHE_DIR, CADZOW_KWARGS, FS

FIGURE_DIR = Path(__file__).parent.joinpath('figures')
TITLES = {'orig': 'no Cadzow', 'production': 'production', 'shrink_window': 'shrinkage, bad channels interpolated',
          'shrink_missing': 'shrinkage, bad channels as missing'}
SEAM_STEP = {'production': 32, 'shrink_window': 32, 'shrink_missing': 32}

# %%
if __name__ == '__main__':
    scale = calibrate_scale(HXY, 'window')
    rows = []
    for pid in sorted(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy')):
        with h5py.File(ARCHIVE_FINE, 'r') as f:
            labels = f[pid]['00/meta'].attrs['labels'][:]
        orig = np.load(CACHE_DIR.joinpath(f'{pid}_chunk_orig.npy')).T.astype(np.float64)
        ys = {'orig': orig, **{k: np.load(CACHE_DIR.joinpath(f'{pid}_exp_{k}.npy'))
                               for k in ('production', 'shrink_window')}}
        file = CACHE_DIR.joinpath(f'{pid}_exp_shrink_missing.npy')
        if not file.exists():
            np.save(file, cadzow_shrink(orig, HXY, mode='window', scale=scale, bad=np.isin(labels, (1, 2)), fs=FS,
                                        **CADZOW_KWARGS, n_jobs=8))
        ys['shrink_missing'] = np.load(file)
        r0 = csd_rms(orig)
        kept = {k: csd_rms(y) / r0 for k, y in ys.items() if k != 'orig'}
        rows.append({'pid': pid, 'n_bad': int(np.isin(labels, (1, 2)).sum()), **{f'csd_kept_{k}': v for k, v in kept.items()}})
        plot_methods(pid, ys, kept, FIGURE_DIR.joinpath(f'2026-10-10_cadzow_bad_{pid[:8]}.webp'), titles=TITLES,
                     seam_step=SEAM_STEP, labels=labels)
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent.joinpath('2026-10-10_cadzow_bad_real.csv'), index=False)
    print(df.round(3).to_string())
