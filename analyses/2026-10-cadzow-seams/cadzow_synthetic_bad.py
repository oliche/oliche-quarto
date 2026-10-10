"""
Bad channels on the calibrated synthetic field: interpolation before and after Cadzow against bad channels handled
inside Cadzow as missing grid positions (`cadzow_fill.cadzow_full_grid(bad=...)`).

The synthetic field (`synthetic_lfp_v2`, 3 % white noise) gets the bad-channel layout of e8f9fba4: dead channels
(label 1) carry only 1 % noise, noisy channels (label 2) get 10x the background noise. Pipelines:

- production: interpolate bad channels, production Cadzow, re-interpolate (`groupby_y=True`), as the lfpack worker.
- shrink, interpolated: the same with full-grid Gavish-Donoho shrinkage (noise per window).
- shrink, bad as missing: shrinkage with the bad channels discarded and estimated by Cadzow, no interpolation.
- shrink, no bad channels: shrinkage on the uncorrupted field, the reference.
"""
# %%
from pathlib import Path

import addcopyfighandler  # noqa: F401
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ibldsp import cadzow
from ibldsp.voltage import current_source_density, interpolate_bad_channels

from cadzow_shrink import cadzow_shrink, calibrate_scale
from cadzow_stagger import ARCHIVE_FINE
from cadzow_synthetic import FS, PRODUCTION, H, csd_error, synthetic_lfp_v2
from cadzow_synthetic_remedies import HXY, seam_ratio

sns.set_theme(context='notebook')
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
PID = 'e8f9fba4-d151-4b00-bee7-447f0f3e752c'
NOISE = 0.03

# %%
if __name__ == '__main__':
    with h5py.File(ARCHIVE_FINE, 'r') as f:
        labels = f[PID]['00/meta'].attrs['labels'][:]
    bad = np.isin(labels, (1, 2))
    rng = np.random.default_rng(42)
    truth = synthetic_lfp_v2(rng)
    csd_true = current_source_density(truth, H)
    clean = truth + rng.standard_normal(truth.shape) * NOISE * 100e-6
    corrupt = clean.copy()
    corrupt[labels == 1] = rng.standard_normal(((labels == 1).sum(), truth.shape[1])) * 0.01 * 100e-6
    corrupt[labels == 2] += rng.standard_normal(((labels == 2).sum(), truth.shape[1])) * 10 * NOISE * 100e-6

    scale = calibrate_scale(HXY, 'window')
    interp = lambda w, **kw: interpolate_bad_channels(w, labels, H['x'], H['y'], **kw)
    shrink = lambda w, **kw: cadzow_shrink(w, HXY, mode='window', scale=scale, fs=FS, **PRODUCTION, n_jobs=8, **kw)
    outs = {
        'production': interp(cadzow.cadzow_denoiser(interp(corrupt), h=HXY, fs=FS, **PRODUCTION, n_jobs=8),
                             groupby_y=True),
        'shrink, interpolated': interp(shrink(interp(corrupt)), groupby_y=True),
        'shrink, bad as missing': shrink(corrupt, bad=bad),
        'shrink, no bad channels': shrink(clean),
    }
    # channels whose CSD involves a bad channel: same column (every 4 channels), up to 2 rows away
    near = np.zeros(384, dtype=bool)
    for b in np.where(bad)[0]:
        near[[k for k in (b - 8, b - 4, b, b + 4, b + 8) if 0 <= k < 384]] = True
    rows, csds = [], {}
    for k, out in outs.items():
        e = csd_error(out, csd_true)
        csds[k] = current_source_density(out.astype(np.float64), H)
        rows.append({'method': k, 'csd_err_near_bad': np.median(e[near]), 'csd_err_elsewhere': np.median(e[~near]),
                     'seam_ratio': seam_ratio(e)})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string())
    df.to_csv(Path(__file__).parent.joinpath('2026-10-10_cadzow_synthetic_bad.csv'), index=False)

    # CSD images (3 s), true CSD colour scale; ticks mark dead (black) and noisy (red) channels
    sl = slice(int(4 * FS), int(7 * FS))
    v = np.percentile(np.abs(csd_true[:, sl]), 99)
    panels = {'true CSD': csd_true, **csds}
    fig, axs = plt.subplots(1, len(panels), figsize=(20, 8), sharey=True, layout='constrained')
    for ax, (k, c) in zip(axs, panels.items()):
        im = ax.imshow(c[:, sl], aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v, origin='lower', interpolation='none')
        ax.grid(False)
        ax.set_xticks([])
        r = df[df.method == k]
        ax.set_title(k if r.empty else f'{k}\nerror near bad {r.csd_err_near_bad.iloc[0]:.2f}, '
                     f'elsewhere {r.csd_err_elsewhere.iloc[0]:.2f}', fontsize=10)
        for lab, col in ((1, 'k'), (2, 'crimson')):
            ich = np.where(labels == lab)[0]
            ax.plot(np.full(ich.size, -0.015), ich, ls='none', marker='>', ms=5, color=col,
                    transform=ax.get_yaxis_transform(), clip_on=False)
    axs[0].set_ylabel('channel (black: dead, red: noisy; layout of e8f9fba4)')
    fig.colorbar(im, ax=axs, location='bottom', shrink=0.3, aspect=40, label='A m$^{-3}$')
    fig.savefig(FIGURE_DIR.joinpath('2026-10-10_cadzow_synthetic_bad_csd.webp'), dpi=90, pil_kwargs={'quality': 80})
