"""
Smooth adaptive rank (Gavish-Donoho shrinkage, `cadzow_shrink.py`) against production on the calibrated synthetic
field, at several white-noise levels around the calibrated 3 %.

Methods (all but production on the full grid, all with PPCA):
- production: 2-D Hankel, hard adaptive rank (gap 2.0, max 5), PPCA.
- full grid, fixed rank 2: best fixed rank so far.
- shrink per window / per chunk: Gavish-Donoho shrinkage, noise from each window's median singular value or per
  frequency bin from the whole chunk, capped at rank 5.
"""
# %%
import time
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ibldsp import cadzow
from ibldsp.voltage import current_source_density

from cadzow_cycle_spin import csd_rms
from cadzow_fill import cadzow_full_grid
from cadzow_shrink import cadzow_shrink, calibrate_scale
from cadzow_synthetic import FS, PRODUCTION, H, csd_error, synthetic_lfp_v2
from cadzow_synthetic_remedies import HXY, seam_ratio

sns.set_theme(context='notebook')
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
NOISES = (0, 0.015, 0.03, 0.06)
SHOW = 0.03  # noise level of the image figure

# %%
if __name__ == '__main__':
    scale = {m: calibrate_scale(HXY, m) for m in ('window', 'chunk')}
    print('noise scale', scale)
    methods = {
        'production': lambda w: cadzow.cadzow_denoiser(w, h=HXY, fs=FS, **PRODUCTION, n_jobs=8),
        'full grid, fixed rank 2': lambda w: cadzow_full_grid(
            w, HXY, fs=FS, **{**PRODUCTION, 'gap_threshold': None, 'rank': 2}, n_jobs=8),
        'shrink, noise per window': lambda w: cadzow_shrink(
            w, HXY, mode='window', scale=scale['window'], fs=FS, **PRODUCTION, n_jobs=8),
        'shrink, noise per chunk': lambda w: cadzow_shrink(
            w, HXY, mode='chunk', scale=scale['chunk'], fs=FS, **PRODUCTION, n_jobs=8),
    }
    rng = np.random.default_rng(42)
    truth = synthetic_lfp_v2(rng)
    csd_true = current_source_density(truth, H)
    white = rng.standard_normal(truth.shape) * 100e-6
    rows, csds = [], {}
    for noise in NOISES:
        wav = truth + white * noise
        for km, fcn in methods.items():
            t0 = time.time()
            out = fcn(wav)
            dt = time.time() - t0
            e = csd_error(out, csd_true)
            if noise == SHOW:
                csds[km] = current_source_density(out.astype(np.float64), H)
            rows.append({'noise': noise, 'method': km, 'csd_err_median': np.median(e), 'seam_ratio': seam_ratio(e),
                         'csd_kept': csd_rms(out) / csd_rms(wav), 'time_s': dt})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent.joinpath('2026-10-09_cadzow_synthetic_shrink.csv'), index=False)
    print(df.pivot(index='method', columns='noise', values=['csd_err_median', 'seam_ratio']).round(2).to_string())

    # CSD images at the calibrated noise level, true CSD colour scale
    sl = slice(int(4 * FS), int(7 * FS))
    v = np.percentile(np.abs(csd_true[:, sl]), 99)
    panels = {'true CSD': csd_true, **csds}
    fig, axs = plt.subplots(1, len(panels), figsize=(20, 8), sharey=True, layout='constrained')
    for ax, (k, c) in zip(axs, panels.items()):
        im = ax.imshow(c[:, sl], aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v, origin='lower', interpolation='none')
        ax.grid(False)
        ax.set_xticks([])
        r = df[(df.noise == SHOW) & (df.method == k)]
        ax.set_title(k if r.empty else f'{k}\nerror {r.csd_err_median.iloc[0]:.2f}, seams {r.seam_ratio.iloc[0]:.2f}',
                     fontsize=10)
        seams = np.arange(32, 384, 32) - 0.5
        ax.plot(np.full(seams.size, -0.015), seams, ls='none', marker='*', ms=6, color='k',
                transform=ax.get_yaxis_transform(), clip_on=False)
    axs[0].set_ylabel(f'noise {SHOW:.0%}, channel (*: 32-channel window seams)')
    fig.colorbar(im, ax=axs, location='bottom', shrink=0.3, aspect=40, label='A m$^{-3}$')
    fig.savefig(FIGURE_DIR.joinpath('2026-10-09_cadzow_synthetic_shrink_csd.webp'), dpi=90, pil_kwargs={'quality': 80})
