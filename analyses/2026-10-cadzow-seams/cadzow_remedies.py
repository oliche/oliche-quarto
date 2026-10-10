"""
Remedies for the half-empty NP1 trajectory matrix of `ibldsp.cadzow`, tested on the noise-free synthetic field.

On NP1 each depth row holds 2 of the 4 lateral positions, so the 2-D block-Hankel trajectory of a 64-channel window
(51 x 32) has 50 % zero entries that the SVD treats as data. Two alternatives, swapped into `cadzow_denoiser`:

- MSSA: one Hankel matrix per probe column (16 channels at 40 µm in a 64-channel window, all filled), the 4 column
  blocks stacked side by side (9 x 32) so the rank-r model is shared by all columns. No empty entries.
- Imputation: the original trajectory, but the empty entries are filled with the current rank-r estimate and the
  rank reduction is repeated (Oropeza & Sacchi 2011, MSSA reconstruction). Costs one SVD per iteration.
"""
# %%
import contextlib
import time
from pathlib import Path
from unittest import mock

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ibldsp import cadzow
from ibldsp.voltage import current_source_density

from cadzow_synthetic import FS, PRODUCTION, H, csd_error, synthetic_lfp

sns.set_theme(context='notebook')
FIGURE_DIR = Path(__file__).parent.joinpath('figures')
N_IMPUTE = 5  # rank reductions of the imputation variant


def trajectory_mssa(x, y, dtype=np.complex128):
    """
    Column-stacked Hankel trajectory: one Hankel block per lateral position, blocks side by side.

    Drop-in for `ibldsp.cadzow.trajectory` (same return signature); every entry of the matrix holds a channel.

    Parameters
    ----------
    x, y : ndarray (nc,)
        Channel coordinates [µm] of the window.
    dtype : numpy.dtype

    Returns
    -------
    T : ndarray (nrows, ncols)
        Empty trajectory matrix.
    it : tuple of ndarray
        Row and column index of each entry.
    ic : ndarray
        Channel index of each entry.
    trcount : ndarray (nc,)
        Number of entries per channel.
    """
    rows, cols, ic, col0 = [], [], [], 0
    blocks = []
    for xc in np.unique(x):
        ichan = np.where(x == xc)[0]
        blocks.append((ichan[np.argsort(y[ichan])], cadzow.traj_matrix_indices(ichan.size)))
    nrows = min(b[1].shape[0] for b in blocks)  # columns with fewer channels (probe end) set the row count
    for ichan, itraj in blocks:
        itraj = itraj[:nrows]
        r, c = np.meshgrid(np.arange(itraj.shape[0]), np.arange(itraj.shape[1]), indexing='ij')
        rows.append(r.ravel())
        cols.append(c.ravel() + col0)
        ic.append(ichan[itraj.ravel()])
        col0 += itraj.shape[1]
    it = (np.concatenate(rows), np.concatenate(cols))
    ic = np.concatenate(ic)
    return np.zeros((nrows, col0), dtype=dtype), it, ic, np.bincount(ic, minlength=x.size)


def process_window_impute(WAV_sl, it, ic, T_shape, scatter, r, imax, niter, gap_threshold, ppca_k, shrink_scale=None):
    """
    `ibldsp.cadzow._process_window` with the empty trajectory entries imputed by the rank-r estimate.

    The observed entries are reset to the data after each rank reduction; the empty ones keep the estimate. PPCA and
    ``niter`` are ignored (single pass, ``N_IMPUTE`` rank reductions).
    """
    WAV_ = WAV_sl.copy()
    obs = np.zeros(T_shape, dtype=bool)
    obs[it] = True
    T_data = np.zeros((imax, *T_shape), dtype=complex)
    T_data[:, it[0], it[1]] = WAV_[ic, :imax].T
    T_batch = T_data.copy()
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for _ in range(N_IMPUTE):
            U, s, Vh = cadzow._safe_svd(T_batch)
            cadzow._apply_rank_threshold(s, r, gap_threshold)
            T_batch_ = (U * s[:, np.newaxis, :]) @ Vh
            T_batch = np.where(obs, T_data, T_batch_)
    WAV_[:, :imax] = (T_batch_[:, it[0], it[1]] @ scatter).T
    return WAV_


def run(wav, variant, **kwargs):
    """Run `cadzow_denoiser` with the trajectory variant ('hankel2d', 'mssa' or 'impute')."""
    patches = {'mssa': mock.patch.object(cadzow, 'trajectory', trajectory_mssa),
               'impute': mock.patch.object(cadzow, '_process_window', process_window_impute)}
    with patches.get(variant, contextlib.nullcontext()):
        return cadzow.cadzow_denoiser(wav, h=H, fs=FS, **{'n_jobs': 8, **kwargs})


# %%
if __name__ == '__main__':
    # real no-Cadzow chunks have a per-channel CSD/voltage RMS ratio matching ~13 % white noise on this synthetic field
    # (this includes their real CSD, so it is an upper bound): test 5 % and 10 %
    rng = np.random.default_rng(42)
    truth = synthetic_lfp(rng)
    csd_true = current_source_density(truth, H)
    inputs = {f'noise {n:.0%}': truth + rng.standard_normal(truth.shape) * n * 100e-6 for n in (0, 0.05, 0.1)}
    base = {**PRODUCTION, 'gap_threshold': None, 'ppca_k': None}
    methods = {
        'production': ('hankel2d', PRODUCTION),
        '2-D Hankel, rank 5': ('hankel2d', base),
        '2-D Hankel, rank 6': ('hankel2d', {**base, 'rank': 6}),
        f'2-D Hankel + imputation x{N_IMPUTE}, rank 5': ('impute', base),
        'MSSA, rank 3': ('mssa', {**base, 'rank': 3}),
        'MSSA, rank 5': ('mssa', base),
        'MSSA, production adaptive rank + PPCA': ('mssa', PRODUCTION),
        'MSSA nswx=128, rank 5': ('mssa', {**base, 'nswx': 128, 'ovx': 64}),
    }
    near = np.zeros(384, dtype=bool)  # channels within 2 of an interior 32-channel window boundary
    for b in range(32, 384, 32):
        near[b - 2:b + 3] = True
    errors, rows = {}, []
    for kin, wav in inputs.items():
        errors[kin] = {'no Cadzow': csd_error(wav, csd_true)}
        for km, (variant, kwargs) in methods.items():
            t0 = time.time()
            out = run(wav, variant, **kwargs)
            dt = time.time() - t0
            errors[kin][km] = e = csd_error(out, csd_true)
            rows.append({'input': kin, 'method': km, 'csd_err_median': np.median(e),
                         'csd_err_p95': np.percentile(e, 95), 'seam_ratio': e[near].mean() / e[~near].mean(),
                         'time_s': dt})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string())
    df.to_csv(Path(__file__).parent.joinpath('2026-10-09_cadzow_remedies.csv'), index=False)

    fig, axs = plt.subplots(1, len(inputs), figsize=(18, 9), sharey=True)
    for ax, (kin, err) in zip(axs, errors.items()):
        for km, e in err.items():
            ax.plot(e, np.arange(e.size), lw=1, label=km, color='k' if km == 'no Cadzow' else None)
        for s in range(32, 384, 32):
            ax.axhline(s, color='grey', lw=0.5, ls=':')
        ax.set(title=f'{kin}: CSD error per channel', xlabel='RMS(CSD error) / median RMS(true CSD)', xscale='log')
    axs[0].set(ylabel='channel (dotted: production window boundaries)')
    axs[0].invert_yaxis()
    axs[-1].legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR.joinpath('2026-10-09_cadzow_remedies_error.webp'), dpi=110, pil_kwargs={'quality': 80})
