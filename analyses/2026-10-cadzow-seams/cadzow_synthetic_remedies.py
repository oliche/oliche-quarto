"""
Seam remedies on the calibrated synthetic field (`synthetic_lfp_v2`, 3 % white noise, see
`cadzow_synthetic_calibration.py`), against the true CSD.

Methods: production, fixed rank (no adaptive gap, no PPCA) with ranks 5 and 6, and Cadzow on the full grid with
interpolated virtual channels (`cadzow_fill.py`), with and without one imputation pass. Each runs on the noise-free
and the noisy field: the noise-free run isolates the algorithmic bias (seams), the noisy run the denoising.
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
from cadzow_synthetic import FS, PRODUCTION, H, csd_error, synthetic_lfp_v2

sns.set_theme(context='notebook')
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
NOISE = 0.03
FIXED = {**PRODUCTION, 'gap_threshold': None, 'ppca_k': None}
HXY = {'x': H['x'], 'y': H['y']}
METHODS = {
    'production': lambda w: cadzow.cadzow_denoiser(w, h=HXY, fs=FS, **PRODUCTION, n_jobs=8),
    'fixed rank 5': lambda w: cadzow.cadzow_denoiser(w, h=HXY, fs=FS, **FIXED, n_jobs=8),
    'fixed rank 6': lambda w: cadzow.cadzow_denoiser(w, h=HXY, fs=FS, **{**FIXED, 'rank': 6}, n_jobs=8),
    'full grid, production rank + PPCA': lambda w: cadzow_full_grid(w, HXY, fs=FS, **PRODUCTION, n_jobs=8),
    'full grid, fixed rank 5': lambda w: cadzow_full_grid(w, HXY, fs=FS, **FIXED, n_jobs=8),
    'full grid, fixed rank 5, 1 imputation': lambda w: cadzow_full_grid(w, HXY, n_impute=1, fs=FS, **FIXED, n_jobs=8),
}


def seam_ratio(err, period=32):
    """Mean per-channel error within 2 channels of an interior window boundary over the mean elsewhere."""
    near = np.zeros(err.size, dtype=bool)
    for b in range(period, err.size, period):
        near[b - 2:b + 3] = True
    return err[near].mean() / err[~near].mean()


# %%
if __name__ == '__main__':
    rng = np.random.default_rng(42)
    truth = synthetic_lfp_v2(rng)
    csd_true = current_source_density(truth, H)
    inputs = {'noise-free': truth, f'noise {NOISE:.0%}': truth + rng.standard_normal(truth.shape) * NOISE * 100e-6}
    rows, errors, csds = [], {}, {}
    for kin, wav in inputs.items():
        errors[kin] = {'no Cadzow': csd_error(wav, csd_true)}
        for km, fcn in METHODS.items():
            t0 = time.time()
            out = fcn(wav)
            dt = time.time() - t0
            errors[kin][km] = e = csd_error(out, csd_true)
            csds[(kin, km)] = current_source_density(out.astype(np.float64), H)
            rows.append({'input': kin, 'method': km, 'csd_err_median': np.median(e), 'seam_ratio': seam_ratio(e),
                         'csd_kept': csd_rms(out) / csd_rms(wav), 'time_s': dt})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print(df.round(3).to_string())
    df.to_csv(Path(__file__).parent.joinpath('2026-10-09_cadzow_synthetic_remedies.csv'), index=False)

    # per-channel CSD error profiles
    fig, axs = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    for ax, (kin, err) in zip(axs, errors.items()):
        for km, e in err.items():
            ax.plot(e, np.arange(e.size), lw=1, label=km, color='k' if km == 'no Cadzow' else None)
        for s in range(32, 384, 32):
            ax.axhline(s, color='grey', lw=0.5, ls=':')
        ax.set(title=f'{kin}: CSD error per channel', xlabel='RMS(CSD error) / median RMS(true CSD)', xscale='log')
    axs[0].set(ylabel='channel (dotted: production window boundaries)')
    axs[0].invert_yaxis()
    axs[1].legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR.joinpath('2026-10-09_cadzow_synthetic_remedies_error.webp'), dpi=110,
                pil_kwargs={'quality': 80})

    # CSD images (3 s), true CSD colour scale, one row per input
    sl = slice(int(4 * FS), int(7 * FS))
    v = np.percentile(np.abs(csd_true[:, sl]), 99)
    fig, axs = plt.subplots(2, len(METHODS) + 1, figsize=(24, 13), sharey=True, layout='constrained')
    for i, kin in enumerate(inputs):
        panels = {'true CSD': csd_true, **{km: csds[(kin, km)] for km in METHODS}}
        for ax, (k, c) in zip(axs[i], panels.items()):
            im = ax.imshow(c[:, sl], aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v, origin='lower',
                           interpolation='none')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_title(k if i == 0 else '', fontsize=9)
            seams = np.arange(32, 384, 32) - 0.5
            ax.plot(np.full(seams.size, -0.015), seams, ls='none', marker='*', ms=6, color='k',
                    transform=ax.get_yaxis_transform(), clip_on=False)
        axs[i, 0].set_ylabel(f'{kin}\nchannel (*: 32-channel window seams)')
    fig.colorbar(im, ax=axs, location='bottom', shrink=0.3, aspect=40, label='A m$^{-3}$')
    fig.savefig(FIGURE_DIR.joinpath('2026-10-09_cadzow_synthetic_remedies_csd.webp'), dpi=90,
                pil_kwargs={'quality': 80})
