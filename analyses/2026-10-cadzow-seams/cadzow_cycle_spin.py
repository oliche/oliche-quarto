"""
Cycle-spinning Cadzow: average the lfpack Cadzow denoiser over 4 offsets of its spatial window grid.

Each run puts the window seams (every 32 channels) at a different depth (offsets 0, 8, 16, 24 channels); the runs are
averaged with constant equal weights, so the blend adds no depth-varying term to the CSD and each run's seams are
diluted 4 times. The shifted runs mirror whole rows at the probe tip so every run covers every channel.

Compared on the cached production chunks against the production parameters and against ovx=48 (more overlap).
"""
# %%
import time
from pathlib import Path

import addcopyfighandler  # noqa: F401
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ibldsp import cadzow
from ibldsp.voltage import current_source_density, interpolate_bad_channels

from cadzow_stagger import ARCHIVE_FINE, CACHE_DIR, CADZOW_KWARGS, FS, STEP, H

sns.set_theme(context='notebook')
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
SPIN_OFFSETS = (0, 8, 16, 24)  # seam positions mod 32; multiples of 4 keep the NP1 column pattern
# seam period per method [channels]: production windows every 32, ovx=48 every 16, cycle spinning one run per 8
SEAM_STEP = {'production': STEP, 'cyclespin': STEP // len(SPIN_OFFSETS), 'ovx48': CADZOW_KWARGS['nswx'] - 48}
SEAM_MS = {'production': 7, 'cyclespin': 4, 'ovx48': 7}  # cycle-spinning seams each carry 1/4 of the weight


def cadzow_grid_offset(wav, offset, h=H, **kwargs):
    """
    Cadzow-denoise with the window seams at channels offset + k * STEP, using a mirrored pad at the probe tip.

    The pad is (STEP - offset) % STEP channels: whole rows 1, 2, ... are mirrored below row 0 (same column pattern,
    since the NP1 pattern repeats every 2 rows), the padded array is denoised, and the pad is dropped.

    Parameters
    ----------
    wav : numpy.ndarray
        LFP (n_channels, n_samples) [V].
    offset : int
        Seam offset in channels, a multiple of 4.
    h : dict
        Probe geometry with 'x', 'y' [um] and 'row'.
    **kwargs
        Forwarded to `ibldsp.cadzow.cadzow_denoiser`.

    Returns
    -------
    numpy.ndarray
        Denoised LFP (n_channels, n_samples).
    """
    npad = (STEP - offset) % STEP
    if npad == 0:
        return cadzow.cadzow_denoiser(wav, h={k: h[k] for k in 'xy'}, fs=FS, **kwargs)
    # channels of rows npad/2, ..., 2, 1 (in that order) mirrored below row 0
    ipad = np.concatenate([np.where(h['row'] == r)[0] for r in range(npad // 2, 0, -1)])
    dy = np.median(np.diff(np.unique(h['y'])))
    hp = {'x': np.r_[h['x'][ipad], h['x']],
          'y': np.r_[h['y'][0] - np.repeat(np.arange(npad // 2, 0, -1), 2) * dy, h['y']]}
    out = cadzow.cadzow_denoiser(np.r_[wav[ipad], wav], h=hp, fs=FS, **kwargs)
    return out[npad:]


def cadzow_cycle_spin(wav, labels, h=H, offsets=SPIN_OFFSETS, runs=None, **kwargs):
    """
    Average Cadzow over several window-grid offsets with equal weights, then re-interpolate bad channels.

    Parameters
    ----------
    wav : numpy.ndarray
        LFP (n_channels, n_samples) [V].
    labels : numpy.ndarray
        Bad-channel labels (0 = good).
    h : dict
        Probe geometry with 'x', 'y' [um] and 'row'.
    offsets : tuple of int
        Seam offsets in channels.
    runs : dict or None
        Precomputed runs {offset: array}, e.g. the production run for offset 0.
    **kwargs
        Forwarded to `ibldsp.cadzow.cadzow_denoiser`.

    Returns
    -------
    numpy.ndarray
        Denoised LFP (n_channels, n_samples).
    dict
        Wall time per computed offset [s].
    """
    runs, timing = dict(runs or {}), {}
    for o in offsets:
        if o not in runs:
            t = time.time()
            runs[o] = cadzow_grid_offset(wav, o, h=h, **kwargs)
            timing[o] = time.time() - t
    out = np.mean([runs[o] for o in offsets], axis=0)
    return interpolate_bad_channels(out, labels, h['x'], h['y'], groupby_y=True), timing


def csd_rms(y):
    """Median over channels of the per-channel CSD RMS [A m-3]."""
    return np.median(np.sqrt(np.mean(current_source_density(y.astype(np.float64), H) ** 2, axis=1)))


# %%
if __name__ == '__main__':
    pids = sorted(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy'))
    rows = []
    for pid in pids:
        with h5py.File(ARCHIVE_FINE, 'r') as f:
            labels = f[pid]['00/meta'].attrs['labels'][:]
        orig = np.load(CACHE_DIR.joinpath(f'{pid}_chunk_orig.npy')).T.astype(np.float64)
        files = {k: CACHE_DIR.joinpath(f'{pid}_exp_{k}.npy') for k in ('production', 'cyclespin', 'ovx48')}
        timing = {}
        if not files['cyclespin'].exists():
            y, timing = cadzow_cycle_spin(orig, labels, runs={0: np.load(files['production'])},
                                          **CADZOW_KWARGS, n_jobs=8)
            np.save(files['cyclespin'], y)
        if not files['ovx48'].exists():
            t = time.time()
            y = cadzow.cadzow_denoiser(orig, h=H, fs=FS, **{**CADZOW_KWARGS, 'ovx': 48}, n_jobs=8)
            timing['ovx48'] = time.time() - t
            np.save(files['ovx48'], interpolate_bad_channels(y, labels, H['x'], H['y'], groupby_y=True))
        ys = {k: np.load(f) for k, f in files.items()}
        r0 = csd_rms(orig)
        rows.append({'pid': pid, **{f'csd_kept_{k}': csd_rms(y) / r0 for k, y in ys.items()},
                     **{f'time_{k}': v for k, v in timing.items()}})
        # comparison figure: CSD of the first snippet (4-7 s into the chunk), production colour scale
        sl = slice(int(4 * FS), int(7 * FS))
        csd = {k: current_source_density(y.astype(np.float64), H)[:, sl] for k, y in ys.items()}
        v = np.percentile(np.abs(csd['production']), 99)
        fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True, layout='constrained')
        titles = {'production': 'production (nswx=64, ovx=32)', 'cyclespin': 'cycle spinning (4 offsets)',
                  'ovx48': 'ovx=48'}
        for ax, (k, d) in zip(axes, csd.items()):
            im = ax.imshow(d, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v, origin='lower', interpolation='none')
            ax.set_title(f'{titles[k]}, CSD kept {rows[-1][f"csd_kept_{k}"]:.3f}', fontsize=10)
            ax.set_xticks([])
            ax.grid(False)  # no seaborn grid lines over the data
            # stars just left of each panel at that method's own window seams, off the data
            seams = np.arange(SEAM_STEP[k], 384, SEAM_STEP[k]) - 0.5
            ax.plot(np.full(seams.size, -0.012), seams, ls='none', marker='*', ms=SEAM_MS[k], color='k',
                    transform=ax.get_yaxis_transform(), clip_on=False)
        axes[0].set_ylabel('channel (*: window seams of each method)')
        fig.colorbar(im, ax=axes, location='bottom', shrink=0.4, aspect=40, label='A m$^{-3}$')
        fig.suptitle(pid, fontsize=9)
        fig.savefig(FIGURE_DIR.joinpath(f'2026-10-09_cadzow_cyclespin_{pid[:8]}.webp'), dpi=90, pil_kwargs={'quality': 80})
        plt.close(fig)
        print(pid[:8], {k: round(v, 1) for k, v in timing.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent.joinpath('2026-10-09_cadzow_cyclespin.csv'), index=False)
    print(df.filter(like='csd_kept').median().round(3))
    print(df.filter(like='time').median().round(1))
