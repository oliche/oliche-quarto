"""
Where the horizontal lines of the lfpack fine tier come from in the codec: CSD error of each lossy stage, on one
insertion, for the v04 and v05 Cadzow inputs.

Stages: spatial SVD truncation only (epsilon 100, alpha 0), wavelet-packet thresholding only (epsilon 1e-6, every
component kept, alpha 2.5), and the fine tier (epsilon 100, alpha 2.5). Line energy is the share of the CSD error
power below 2 Hz in time (structure that persists over time at one depth, i.e. horizontal), against the same share in
the uncompressed CSD.
"""
# %%
import sys
import tempfile
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import seaborn as sns
from ibldsp.voltage import current_source_density

from cadzow_compression import CACHE_DIR, FS, H, compress_roundtrip, snr_db

sns.set_theme(context='notebook')
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
STAGES = {'spatial SVD only (ε 100, α 0)': (100.0, 0.0), 'WP only (all components, α 2.5)': (1e-6, 2.5),
          'fine (ε 100, α 2.5)': (100.0, 2.5)}
INPUTS = {'v04': 'production', 'v05': 'np1_missing'}


def csd(wav):
    """CSD (384, ns) of LFP (384, ns) [V]."""
    return current_source_density(np.asarray(wav, dtype=np.float64), H)


def low_share(x, fmax=2.0):
    """Share of the power of x (nc, ns) below fmax [Hz] in time."""
    f, p = scipy.signal.welch(x, fs=FS, nperseg=2048, axis=-1)
    p = p.sum(0)
    return p[f < fmax].sum() / p.sum()


# %%
if __name__ == '__main__':
    pid = next(f.name[:36] for f in CACHE_DIR.glob('*_chunk_orig.npy') if f.name.startswith(sys.argv[1]))
    rows, imgs = [], {}
    sl = slice(int(4 * FS), int(7 * FS))
    with tempfile.TemporaryDirectory() as tmp:
        for v, suffix in INPUTS.items():
            wav = np.load(CACHE_DIR.joinpath(f'{pid}_exp_{suffix}.npy')).astype(np.float64)
            c0 = csd(wav)
            imgs[(v, 'uncompressed')] = c0
            rows.append({'input': v, 'stage': 'uncompressed', 'low_share': low_share(c0)})
            for stage, (eps, alpha) in STAGES.items():
                size, rec = compress_roundtrip(wav, alpha, tmp, epsilon=eps)
                e = csd(rec) - c0
                imgs[(v, stage)] = e
                rows.append({'input': v, 'stage': stage, 'size_kb': size / 1024, 'snr_db': snr_db(wav, rec),
                             'csd_snr_db': snr_db(c0, c0 + e), 'low_share': low_share(e)})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string())
    df.to_csv(Path(__file__).parent.joinpath(f'2026-10-10_cadzow_codec_lines_{pid[:8]}.csv'), index=False)

    cols = ['uncompressed', *STAGES]
    fig, axs = plt.subplots(2, len(cols), figsize=(24, 12), sharex=True, sharey=True, layout='constrained')
    for i, v in enumerate(INPUTS):
        vmax = np.percentile(np.abs(imgs[(v, 'uncompressed')][:, sl]), 99)
        for ax, k in zip(axs[i], cols):
            im = ax.imshow(imgs[(v, k)][:, sl], aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower',
                           interpolation='none')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_title(f'{v} {"CSD" if k == "uncompressed" else "CSD error, " + k}', fontsize=10)
        axs[i, 0].set_ylabel('channel')
        fig.colorbar(im, ax=axs[i], location='right', shrink=0.9, aspect=30, label='CSD [A m$^{-3}$]')
    fig.suptitle(f'{pid}, 4-7 s: CSD of the uncompressed Cadzow output and CSD error of each codec stage')
    fig.savefig(FIGURE_DIR.joinpath(f'2026-10-10_cadzow_codec_lines_{pid[:8]}.webp'), dpi=80,
                pil_kwargs={'quality': 80})
