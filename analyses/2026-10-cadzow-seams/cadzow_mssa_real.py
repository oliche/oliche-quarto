"""
Column-stacked Hankel (MSSA) Cadzow on the cached production chunks of the 11 benchmark insertions.

MSSA settings use a fixed rank and no PPCA (the adaptive rank and PPCA add window-to-window disagreement on the
synthetic field). Compared with production (nswx=64, ovx=32, adaptive rank, PPCA) and ovx=48; every run is followed
by the production re-interpolation of bad channels. Timing is single-threaded, as in the lfpack worker.
"""
# %%
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ibldsp.voltage import current_source_density, interpolate_bad_channels

from cadzow_cycle_spin import csd_rms
from cadzow_remedies import run
from cadzow_stagger import ARCHIVE_FINE, CACHE_DIR, CADZOW_KWARGS, FS, H

FIGURE_DIR = Path(__file__).parent.joinpath('figures')
FIXED = {**CADZOW_KWARGS, 'gap_threshold': None, 'ppca_k': None}
# name: (trajectory variant, kwargs, seam period [channels])
MSSA = {
    'mssa_r3_w64': ('mssa', {**FIXED, 'rank': 3}, 32),
    'mssa_r5_w64': ('mssa', FIXED, 32),
    'mssa_r5_w128': ('mssa', {**FIXED, 'nswx': 128, 'ovx': 64}, 64),
}
TITLES = {'production': 'production (2-D Hankel, nswx=64, ovx=32)', 'ovx48': '2-D Hankel, ovx=48',
          'mssa_r3_w64': 'MSSA rank 3, nswx=64', 'mssa_r5_w64': 'MSSA rank 5, nswx=64',
          'mssa_r5_w128': 'MSSA rank 5, nswx=128'}
SEAM_STEP = {'production': 32, 'ovx48': 16, **{k: v[2] for k, v in MSSA.items()}}

# %%
if __name__ == '__main__':
    pids = sorted(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy'))
    rows = []
    for pid in pids:
        with h5py.File(ARCHIVE_FINE, 'r') as f:
            labels = f[pid]['00/meta'].attrs['labels'][:]
        orig = np.load(CACHE_DIR.joinpath(f'{pid}_chunk_orig.npy')).T.astype(np.float64)
        ys = {k: np.load(CACHE_DIR.joinpath(f'{pid}_exp_{k}.npy')) for k in ('production', 'ovx48')}
        timing = {}
        for k, (variant, kwargs, _) in MSSA.items():
            t = time.time()
            y = run(orig, variant, **{**kwargs, 'n_jobs': 1})
            timing[f'time_{k}'] = time.time() - t
            ys[k] = interpolate_bad_channels(y, labels, H['x'], H['y'], groupby_y=True)
        r0 = csd_rms(orig)
        rows.append({'pid': pid, **{f'csd_kept_{k}': csd_rms(y) / r0 for k, y in ys.items()}, **timing})
        # CSD of the first snippet (4-7 s into the chunk), production colour scale, as the cycle-spinning figures
        sl = slice(int(4 * FS), int(7 * FS))
        csd = {k: current_source_density(y.astype(np.float64), H)[:, sl] for k, y in ys.items()}
        v = np.percentile(np.abs(csd['production']), 99)
        fig, axes = plt.subplots(1, len(csd), figsize=(24, 8), sharey=True, layout='constrained')
        for ax, (k, d) in zip(axes, csd.items()):
            im = ax.imshow(d, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v, origin='lower', interpolation='none')
            ax.set_title(f'{TITLES[k]}, CSD kept {rows[-1][f"csd_kept_{k}"]:.3f}', fontsize=10)
            ax.set_xticks([])
            ax.grid(False)
            seams = np.arange(SEAM_STEP[k], 384, SEAM_STEP[k]) - 0.5
            ax.plot(np.full(seams.size, -0.012), seams, ls='none', marker='*', ms=7, color='k',
                    transform=ax.get_yaxis_transform(), clip_on=False)
        axes[0].set_ylabel('channel (*: window seams of each method)')
        fig.colorbar(im, ax=axes, location='bottom', shrink=0.3, aspect=40, label='A m$^{-3}$')
        fig.suptitle(pid, fontsize=9)
        fig.savefig(FIGURE_DIR.joinpath(f'2026-10-09_cadzow_mssa_{pid[:8]}.webp'), dpi=90, pil_kwargs={'quality': 80})
        plt.close(fig)
        print(pid[:8], {k: round(v, 2) for k, v in timing.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent.joinpath('2026-10-09_cadzow_mssa_real.csv'), index=False)
    print(df.filter(like='csd_kept').median().round(3))
    print(df.filter(like='time').median().round(2))
