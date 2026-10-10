"""
Window dependence of Cadzow on the real chunks: the CSD difference between two runs whose window grids are offset by
16 channels (`cadzow_cycle_spin.cadzow_grid_offset`), for the v04 settings and the v05 entry point
(`cadzow_denoiser_np1`).

A window-independent denoiser gives the same output for both grids. Where the fits depend on the window, the
difference peaks at the boundaries of either grid (every 16 channels); its seam ratio (period 16) is the real-data
counterpart of the synthetic seam ratio. Both runs get the production re-interpolation of bad channels.

Figure: per-channel profiles for one insertion, CSD RMS of uncompressed and fine-compressed v04 / v05, the codec CSD
error and the window dependence, with the window boundaries.
"""
# %%
import sys
from pathlib import Path

import addcopyfighandler  # noqa: F401
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ibldsp import cadzow
from ibldsp.voltage import current_source_density, interpolate_bad_channels

from cadzow_compression import CACHE_DIR
from cadzow_cycle_spin import cadzow_grid_offset
from cadzow_stagger import ARCHIVE_FINE, CADZOW_KWARGS, H
from cadzow_synthetic_remedies import seam_ratio

sns.set_theme(context='notebook')
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
KWARGS = {'v04': {**CADZOW_KWARGS, 'n_jobs': 8}, 'v05': {**cadzow.LFP_KWARGS_NP1, 'n_jobs': 8}}
EXP = {'v04': 'production', 'v05': 'np1'}  # cached offset-0 outputs


def csd(wav):
    """CSD (384, ns) of LFP (384, ns) [V]."""
    return current_source_density(np.asarray(wav, dtype=np.float64), H)


def rms(x):
    """Per-channel RMS over time."""
    return np.sqrt(np.mean(x ** 2, axis=1))


def window_dependence(pid, version, labels):
    """
    Per-channel RMS of CSD(grid 0) - CSD(grid offset by 16), relative to the median per-channel CSD RMS of grid 0.

    The offset run is cached as `{pid}_exp_{version}_off16.npy`.
    """
    orig = np.load(CACHE_DIR.joinpath(f'{pid}_chunk_orig.npy')).T.astype(np.float64)
    file = CACHE_DIR.joinpath(f'{pid}_exp_{version}_off16.npy')
    if not file.exists():
        y = cadzow_grid_offset(orig, 16, **KWARGS[version])
        np.save(file, interpolate_bad_channels(y, labels, H['x'], H['y'], groupby_y=True))
    c0 = csd(np.load(CACHE_DIR.joinpath(f'{pid}_exp_{EXP[version]}.npy')))
    return rms(c0 - csd(np.load(file))) / np.median(rms(c0))


# %%
if __name__ == '__main__':
    pids = sorted(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy'))
    rows, prof = [], {}
    for pid in pids:
        with h5py.File(ARCHIVE_FINE, 'r') as f:
            labels = f[pid]['00/meta'].attrs['labels'][:]
        row = {'pid': pid[:8]}
        for v in KWARGS:
            d = prof[(pid, v)] = window_dependence(pid, v, labels)
            row[f'dependence {v}'], row[f'seams {v}'] = np.median(d), seam_ratio(d, period=16)
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows).set_index('pid')
    df.loc['median'] = df.median()
    print(df.round(2).to_string())
    df.to_csv(Path(__file__).parent.joinpath('2026-10-10_cadzow_window_dependence.csv'))

    # profiles of one insertion
    pid = next(p for p in pids if p.startswith(sys.argv[1] if len(sys.argv) > 1 else '749cb2b7'))
    c = {k: csd(np.load(CACHE_DIR.joinpath(f'{pid}_exp_{s}.npy')))
         for k, s in (('v04', 'production'), ('v05', 'np1'), ('v04 fine', 'v04_fine'), ('v05 fine', 'v05_fine'))}
    ch = np.arange(384)
    fig, axs = plt.subplots(1, 3, figsize=(18, 9), sharey=True, layout='constrained')
    for k, x in c.items():
        axs[0].plot(rms(x), ch, label=k, lw=1, ls='-' if 'fine' in k else '--')
    axs[0].set(xlabel='CSD RMS [A m$^{-3}$]', ylabel='channel', title='CSD RMS per channel')
    for v in ('v04', 'v05'):
        axs[1].plot(rms(c[f'{v} fine'] - c[v]) / np.median(rms(c[v])), ch, label=v, lw=1)
        axs[2].plot(prof[(pid, v)], ch, label=f'{v}, seams {seam_ratio(prof[(pid, v)], 16):.2f}', lw=1)
    axs[1].set(xlabel='relative CSD error', title='codec error: fine - uncompressed')
    axs[2].set(xlabel='relative CSD difference', title='window dependence: grid 0 - grid 16')
    for ax in axs:
        for b in range(32, 384, 32):
            ax.axhline(b - 0.5, color='k', lw=0.5, alpha=0.4)
        for b in range(16, 384, 32):
            ax.axhline(b - 0.5, color='k', lw=0.5, alpha=0.4, ls=':')
        ax.legend(fontsize=9)
    axs[0].set_ylim(0, 383)
    fig.suptitle(f'{pid}: lines, window boundaries of grid 0 (solid) and grid 16 (dotted)')
    fig.savefig(FIGURE_DIR.joinpath(f'2026-10-10_cadzow_window_dependence_{pid[:8]}.webp'), dpi=80,
                pil_kwargs={'quality': 80})
