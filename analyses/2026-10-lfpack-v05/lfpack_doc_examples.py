"""
Compression examples for the lfpack documentation: LFP of benchmark chunks uncompressed, after lfpack at the three
tiers (v1.1 Cadzow input, `cadzow_denoiser_np1` with the bad channels as missing) and the difference with the
uncompressed signal, 4-7 s into the 32.8 s chunk.

Uncompressed is the Cadzow output the codec receives (so the difference is the codec alone). One voltage colour scale
per figure, from the uncompressed 99th percentile, shared by the difference. Chunks are compressed on their own
(`cadzow_compression.compress_roundtrip`); reconstructions are cached next to the production chunks.
"""
# %%
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1].joinpath('2026-10-cadzow-seams')))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainbox.ephys_plots import plot_brain_regions

from cadzow_compression import CACHE_DIR, TIERS, compress_roundtrip, snr_db
from cadzow_shrink_real import REGIONS, load_atlas_ids

FIGURE_DIR = Path.home().joinpath('PycharmProjects', 'ephys-atlas', 'packages', 'lfpack', 'docs', 'figures')
FS = 250.0
SNIPPET = slice(int(4 * FS), int(7 * FS))
PIDS = ['dab512bd', '6638cfb3', 'da8dfec1', 'e8f9fba4']  # docs examples; empty: only print metrics of all chunks


def load_roundtrips(pid, tmp):
    """
    Uncompressed input and its lfpack reconstruction at each tier (cached `.npy`, (384, ns) float64 [V]).

    Returns
    -------
    dict
        'uncompressed' and tier name -> (384, ns); and a dataframe of size [kB], compression ratio and SNR [dB].
    """
    ys = {'uncompressed': np.load(CACHE_DIR.joinpath(f'{pid}_exp_np1_missing.npy')).astype(np.float64)}
    rows = []
    for tier, alpha in TIERS.items():
        file = CACHE_DIR.joinpath(f'{pid}_exp_v05_{tier}.npy')
        size, rec = compress_roundtrip(ys['uncompressed'], alpha, tmp)
        if not file.exists():
            np.save(file, rec.astype(np.float32))
        ys[tier] = np.load(file).astype(np.float64)
        rows.append({'pid': pid[:8], 'tier': tier, 'cr': ys['uncompressed'].size * 4 / size,
                     'snr_db': snr_db(ys['uncompressed'], ys[tier])})
    return ys, pd.DataFrame(rows)


def plot_examples(pid, ys, stats, file):
    """
    Figure: top row uncompressed then each tier; bottom row the difference uncompressed - lfpack of each tier.
    Brain regions on the left.
    """
    sns.set_theme(context='notebook', style='white')
    tiers = list(TIERS)
    atlas_ids = load_atlas_ids(pid)
    fig, axes = plt.subplots(2, 5, figsize=(20, 9), layout='constrained',
                             gridspec_kw={'width_ratios': [0.2] + [1] * 4})
    vmax = np.percentile(np.abs(ys['uncompressed'][:, SNIPPET]), 99) * 1e6
    for irow in range(2):
        plot_brain_regions(atlas_ids, channel_depths=np.arange(atlas_ids.size), brain_regions=REGIONS,
                           ax=axes[irow, 0], label='left')
        axes[irow, 0].set_ylim(-0.5, atlas_ids.size - 0.5)
        axes[irow, 0].tick_params(axis='y', labelsize=7)
    axes[0, 0].set_title('regions', fontsize=10)
    axes[1, 1].axis('off')
    panels = [(0, 1, ys['uncompressed'], 'uncompressed')]
    panels += [(0, 2 + i, ys[t], f"lfpack {t} (α={TIERS[t]:g}), {s.cr:.0f}x, SNR {s.snr_db:.1f} dB")
               for i, (t, s) in enumerate(stats.set_index('tier').loc[tiers].iterrows())]
    panels += [(1, 2 + i, ys['uncompressed'] - ys[t], f'difference, {t}') for i, t in enumerate(tiers)]
    for irow, icol, y, title in panels:
        ax = axes[irow, icol]
        im = ax.imshow(y[:, SNIPPET] * 1e6, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower',
                       interpolation='none', extent=(4, 7, -0.5, 383.5))
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('time [s]')
    fig.colorbar(im, ax=axes[:, 1:], location='right', shrink=0.9, aspect=30, label='LFP [µV]')
    fig.suptitle(f'{pid}', fontsize=11)
    fig.savefig(file, dpi=80, pil_kwargs={'quality': 80})
    plt.close(fig)


# %%
if __name__ == '__main__':
    pids = sorted(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy'))
    all_stats = []
    with tempfile.TemporaryDirectory() as tmp:
        for pid in pids:
            if PIDS and pid[:8] not in PIDS:
                continue
            ys, stats = load_roundtrips(pid, tmp)
            all_stats.append(stats)
            if PIDS:
                plot_examples(pid, ys, stats, FIGURE_DIR.joinpath(f'compression_example_{pid[:8]}.webp'))
    print(pd.concat(all_stats).pivot(index='pid', columns='tier', values=['cr', 'snr_db']).round(1))
