"""
Full-grid Cadzow with Gavish-Donoho shrinkage (`cadzow_shrink.py`) on the cached production chunks of the 11
benchmark insertions, against production and ovx=48.

Every run is followed by the production re-interpolation of bad channels. Production is re-run single-threaded for
timing, since the lfpack worker calls Cadzow with n_jobs=1. Outputs are cached next to the production chunks.

Figure per insertion: rows CSD, bipolar and voltage of the first snippet (4-7 s into the chunk); columns no Cadzow
(the production worker output without Cadzow, own colour scale) then each Cadzow method (shared colour scale per row,
from production).
"""
# %%
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainbox.ephys_plots import plot_brain_regions
from iblatlas.regions import BrainRegions
from ibldsp import cadzow
from ibldsp.voltage import current_source_density, interpolate_bad_channels

from cadzow_cycle_spin import csd_rms
from cadzow_shrink import cadzow_shrink, calibrate_scale
from cadzow_stagger import ARCHIVE_FINE, CACHE_DIR, CADZOW_KWARGS, FS, H

FIGURE_DIR = Path(__file__).parent.joinpath('figures')
HXY = {'x': H['x'], 'y': H['y']}
ATLAS_CACHE = Path(__file__).parents[1].joinpath('2026-10-csd', 'channels_atlas_ids.csv')
REGIONS = BrainRegions()
TITLES = {'orig': 'no Cadzow', 'production': 'production (nswx=64, ovx=32)', 'ovx48': 'production, ovx=48',
          'shrink_window': 'full grid + shrinkage, noise per window',
          'shrink_chunk': 'full grid + shrinkage, noise per chunk'}
SEAM_STEP = {'production': 32, 'ovx48': 16, 'shrink_window': 32, 'shrink_chunk': 32}


def bipolar(lfp, h):
    """
    Bipolar re-referencing: difference between consecutive sites of the same probe column (copy of
    `../2026-10-csd/plot_cadzow_csd.py`).

    Parameters
    ----------
    lfp : numpy.ndarray
        LFP (n_channels, n_samples) [V].
    h : dict
        Trace header with 'col' and 'row' arrays.

    Returns
    -------
    numpy.ndarray
        Bipolar signal (n_channels, n_samples) [V]; site k holds lfp[k + 1] - lfp[k] along its column, the
        last site of each column repeats its neighbour.
    """
    out = np.zeros(lfp.shape, dtype=np.float64)
    for col in np.unique(h['col']):
        itr = np.where(h['col'] == col)[0]
        itr = itr[np.argsort(h['row'][itr])]
        out[itr[:-1]] = np.diff(lfp[itr].astype(np.float64), axis=0)
        out[itr[-1]] = out[itr[-2]]
    return out


TRANSFORMS = {  # row: (transform, unit)
    'CSD': (lambda x: current_source_density(x.astype(np.float64), H), 'A m$^{-3}$'),
    'bipolar': (lambda x: bipolar(x, H) * 1e6, 'µV'),
    'voltage': (lambda x: x * 1e6, 'µV'),
}


def load_atlas_ids(pid):
    """
    Allen atlas id per channel (raw channel order), from the cache of the CSD page
    (`../2026-10-csd/channels_atlas_ids.csv`, all 11 benchmark insertions).

    Parameters
    ----------
    pid : str

    Returns
    -------
    ndarray (384,) of int
    """
    df = pd.read_csv(ATLAS_CACHE)
    return df.loc[df['pid'] == pid].sort_values('channel')['atlas_id'].values.astype(int)


def plot_methods(pid, ys, kept, file, titles=TITLES, seam_step=SEAM_STEP, labels=None, ref='production',
                 suptitle=None):
    """
    CSD, bipolar and voltage rows of the first snippet (4-7 s into the chunk), one column per method, with the
    brain regions of the insertion in a column on the left.

    One colour scale and colorbar per row, from ``ref`` (the no-Cadzow column saturates in CSD and bipolar). Stars
    inside the left edge of each panel mark that method's window seams; triangles on the anatomy column mark dead
    (black) and noisy (red) channels.

    Parameters
    ----------
    pid : str
    ys : dict
        Method name -> LFP (384, ns) [V]; must hold ``ref``.
    kept : dict
        Method name -> fraction of CSD RMS kept, shown in the titles.
    file : pathlib.Path
        Output WebP.
    titles, seam_step : dict
        Column titles and window-seam period [channels] per method (no stars if absent).
    labels : ndarray, optional
        Channel labels (1 dead, 2 noisy).
    ref : str
        Method whose 99th percentile sets the colour scale of each row.  Default 'production'.
    suptitle : str, optional
        Figure title; default names the snippet and the seam stars.
    """
    sl = slice(int(4 * FS), int(7 * FS))
    atlas_ids = load_atlas_ids(pid)
    fig, axes = plt.subplots(len(TRANSFORMS), len(ys) + 1, figsize=(26, 14.6), layout='constrained',
                             gridspec_kw={'width_ratios': [0.22] + [1] * len(ys)})
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.005, hspace=0.005)
    for irow, (kind, (fcn, unit)) in enumerate(TRANSFORMS.items()):
        plot_brain_regions(atlas_ids, channel_depths=np.arange(atlas_ids.size), brain_regions=REGIONS,
                           ax=axes[irow, 0], label='left')
        axes[irow, 0].set_ylim(-0.5, atlas_ids.size - 0.5)
        axes[irow, 0].tick_params(axis='y', labelsize=7)
        axes[irow, 0].set_title(kind if irow else f'{kind}\nregions', fontsize=10)
        if labels is not None:
            for lab, col in ((1, 'k'), (2, 'crimson')):
                ich = np.where(labels == lab)[0]
                axes[irow, 0].plot(np.full(ich.size, 0.93), ich, ls='none', marker='<', ms=5, color=col,
                                   transform=axes[irow, 0].get_yaxis_transform(), clip_on=False, in_layout=False)
            axes[irow, 0].set_ylim(-0.5, atlas_ids.size - 0.5)
        d = {k: fcn(y)[:, sl] for k, y in ys.items()}
        vmax = np.percentile(np.abs(d[ref]), 99)
        for ax, (k, x) in zip(axes[irow, 1:], d.items()):
            im = ax.imshow(x, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower', interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if irow == 0:
                ax.set_title(f'{titles[k]}' + (f', CSD kept {kept[k]:.3f}' if k in kept else ''), fontsize=10)
            if k in seam_step:
                seams = np.arange(seam_step[k], 384, seam_step[k]) - 0.5
                ax.plot(np.full(seams.size, 0.008), seams, ls='none', marker='*', ms=6, color='k',
                        transform=ax.get_yaxis_transform())
        fig.colorbar(im, ax=axes[irow, :], location='right', shrink=0.95, aspect=25, pad=0.005,
                     label=f'{kind} [{unit}]')
    marks = '' if labels is None else '; triangles: dead (black) and noisy (red) channels'
    fig.suptitle(suptitle or f'{pid}, 4-7 s into the chunk. Stars: window seams of each method{marks}', fontsize=11)
    fig.savefig(file, dpi=80, pil_kwargs={'quality': 80})
    plt.close(fig)

# %%
if __name__ == '__main__':
    scale = {m: calibrate_scale(HXY, m) for m in ('window', 'chunk')}
    pids = sorted(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy'))
    rows = []
    for pid in pids:
        with h5py.File(ARCHIVE_FINE, 'r') as f:
            labels = f[pid]['00/meta'].attrs['labels'][:]
        orig = np.load(CACHE_DIR.joinpath(f'{pid}_chunk_orig.npy')).T.astype(np.float64)
        ys = {'orig': orig, **{k: np.load(CACHE_DIR.joinpath(f'{pid}_exp_{k}.npy')) for k in ('production', 'ovx48')}}
        timing = {}
        file_time = CACHE_DIR.joinpath(f'{pid}_exp_shrink_timing.npy')
        if not file_time.exists():
            t = time.time()
            cadzow.cadzow_denoiser(orig, h=HXY, fs=FS, **CADZOW_KWARGS, n_jobs=1)
            timing['time_production'] = time.time() - t
        for mode in ('window', 'chunk'):
            file = CACHE_DIR.joinpath(f'{pid}_exp_shrink_{mode}.npy')
            if not file.exists():
                t = time.time()
                y = cadzow_shrink(orig, HXY, mode=mode, scale=scale[mode], fs=FS, **CADZOW_KWARGS, n_jobs=1)
                timing[f'time_shrink_{mode}'] = time.time() - t
                np.save(file, interpolate_bad_channels(y, labels, H['x'], H['y'], groupby_y=True))
            ys[f'shrink_{mode}'] = np.load(file)
        if timing:
            np.save(file_time, timing)
        timing = np.load(file_time, allow_pickle=True).item()
        r0 = csd_rms(orig)
        rows.append({'pid': pid, **{f'csd_kept_{k}': csd_rms(y) / r0 for k, y in ys.items() if k != 'orig'},
                     **timing})
        kept = {k: rows[-1][f'csd_kept_{k}'] for k in ys if k != 'orig'}
        plot_methods(pid, ys, kept, FIGURE_DIR.joinpath(f'2026-10-09_cadzow_shrink_{pid[:8]}.webp'))
        print(pid[:8], {k: round(v, 1) for k, v in timing.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent.joinpath('2026-10-09_cadzow_shrink_real.csv'), index=False)
    print(df.filter(like='csd_kept').median().round(3))
    print(df.filter(like='time').median().round(1))
