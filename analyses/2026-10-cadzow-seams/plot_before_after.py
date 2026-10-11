"""
Before / after figures of the Cadzow seams fix for the 11 benchmark insertions, one 3 s snippet each (4-7 s into the
production chunk at mid-recording).

Rows: LFP (CAR voltage, 250 Hz), bipolar (BPR) and CSD. Columns: no Cadzow, Cadzow with ibl-neuropixel 1.13 and the lfpack v04 settings
(window seams, bad channels interpolated before and after) and Cadzow with the ibl-neuropixel 1.14 LFP defaults
(`LFP_KWARGS_NP1`: full grid + Gavish-Donoho shrinkage, bad channels estimated by Cadzow), then the brain regions.

All three are outputs of the production chunk worker, cached by `../2026-10-csd/plot_cadzow_csd.py`.
"""
# %%
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainbox.ephys_plots import plot_brain_regions
from ibldsp.voltage import current_source_density

from cadzow_shrink_real import REGIONS, bipolar, load_atlas_ids
from cadzow_stagger import ARCHIVE_FINE, CACHE_DIR, H

sns.set_theme(context='notebook')

FIGURE_DIR = Path(__file__).parent.joinpath('figures')
FS = 250.0  # Hz, decimated LFP
T_SNIPPET = (4.0, 7.0)  # s, within the production chunk
PRCTILE = 99  # colour scale: percentile of |after| per row
NSWX_STEP = 32  # channels between window seams of the v04 settings (nswx=64, ovx=32)
COLUMNS = {'orig': 'no Cadzow', 'cadzow': 'Cadzow before (ibl-neuropixel 1.13)',
           'cadzow_v05': 'Cadzow after (ibl-neuropixel 1.14)'}
ROWS = {  # row: (transform, unit)
    'LFP': (lambda x: x * 1e6, 'µV'),
    'BPR': (lambda x: bipolar(x, H) * 1e6, 'µV'),
    'CSD': (lambda x: current_source_density(x.astype(np.float64), H), 'A m$^{-3}$'),
}


def plot_before_after(pid, ys, atlas_ids, labels, t0, file):
    """
    Plot one snippet: rows LFP, BPR and CSD; columns no Cadzow, Cadzow before and after the fix, brain regions.

    One colour scale per row, ±PRCTILE percentile of the "after" column. Ticks on the inner edge of the Cadzow
    columns mark the v04 window seams; triangles on the region column mark dead (black) and noisy (red) channels.

    Parameters
    ----------
    pid : str
        Probe insertion id.
    ys : dict
        COLUMNS key -> LFP snippet (384, ns) [V].
    atlas_ids : numpy.ndarray
        Allen atlas id per channel.
    labels : numpy.ndarray
        Channel labels (1 dead, 2 noisy).
    t0 : float
        Snippet start in the recording [s].
    file : pathlib.Path
        Output WebP.
    """
    nc, ns = ys['orig'].shape
    extent = [t0, t0 + ns / FS, -0.5, nc - 0.5]
    fig, axes = plt.subplots(len(ROWS), len(COLUMNS) + 2, figsize=(19.2, 12), layout='constrained',
                             gridspec_kw={'width_ratios': [1] * len(COLUMNS) + [0.04, 0.16]})
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.01, hspace=0.01)
    seams = np.arange(NSWX_STEP, nc, NSWX_STEP) - 0.5
    for irow, (kind, (fcn, unit)) in enumerate(ROWS.items()):
        d = {k: fcn(y) for k, y in ys.items()}
        vmax = np.percentile(np.abs(d['cadzow_v05']), PRCTILE)
        for icol, (k, x) in enumerate(d.items()):
            ax = axes[irow, icol]
            im = ax.imshow(x, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower',
                           interpolation='none', extent=extent)
            ax.grid(False)
            if k != 'orig':
                ax.plot(np.full(seams.size, 0), seams, ls='none', marker=0, ms=8, mew=1.5, color='k',
                        transform=ax.get_yaxis_transform(), clip_on=False)
            if irow == 0:
                ax.set_title(COLUMNS[k], fontsize=13)
            if icol == 0:
                ax.set_ylabel(f'{kind}\nchannel', fontsize=13)
            else:
                ax.set_yticklabels([])
            if irow == len(ROWS) - 1:
                ax.set_xlabel('time [s]')
            else:
                ax.set_xticklabels([])
        fig.colorbar(im, cax=axes[irow, len(COLUMNS)], label=f'{kind} [{unit}]')
        ax = axes[irow, -1]
        plot_brain_regions(atlas_ids, channel_depths=np.arange(nc), brain_regions=REGIONS, ax=ax, label='right')
        for lab, col in ((1, 'k'), (2, 'crimson')):
            ich = np.where(labels == lab)[0]
            ax.plot(np.full(ich.size, 0.0), ich, ls='none', marker='>', ms=5, color=col,
                    transform=ax.get_yaxis_transform(), clip_on=False, in_layout=False)
        ax.set_ylim(-0.5, nc - 0.5)
        ax.tick_params(axis='y', labelsize=8)
        if irow == 0:
            ax.set_title('regions', fontsize=13)
    fig.suptitle(f'{pid}, t = {t0:.0f}-{t0 + ns / FS:.0f} s. Ticks: Cadzow window seams; '
                 'triangles: dead (black) and noisy (red) channels', fontsize=12)
    fig.savefig(file, dpi=100, pil_kwargs={'quality': 80})
    plt.close(fig)


# %%
if __name__ == '__main__':
    snippets = pd.read_csv(Path(__file__).parents[1].joinpath('2026-10-csd', '2026-10-10_snippets.csv'))
    sl = slice(int(T_SNIPPET[0] * FS), int(T_SNIPPET[1] * FS))
    for pid in sorted(f.name[:36] for f in CACHE_DIR.glob('*_chunk_cadzow_v05.npy')):
        with h5py.File(ARCHIVE_FINE, 'r') as f:
            labels = f[pid]['00/meta'].attrs['labels'][:]
        ys = {k: np.load(CACHE_DIR.joinpath(f'{pid}_chunk_{k}.npy')).T[:, sl] for k in COLUMNS}
        t0 = snippets.query('pid == @pid and snippet == 0')['t'].item()
        plot_before_after(pid, ys, load_atlas_ids(pid), labels, t0,
                          FIGURE_DIR.joinpath(f'2026-10-11_cadzow_before_after_{pid[:8]}.webp'))
        print(pid[:8], flush=True)
