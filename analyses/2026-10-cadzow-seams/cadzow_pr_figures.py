"""
Figures of the ibl-neuropixel pull requests, made with the `ibldsp.cadzow` API (not the prototypes of this folder) on
the calibrated synthetic field (`synthetic_lfp_v2`, 3 % white noise).

- PR 1, `fill_grid` + `shrinkage`: true CSD, lfpack v04 production settings, full grid + Gavish-Donoho shrinkage, on
  the noise-free and the noisy field.
- PR 2, `bad_channels`: the noisy field with the bad-channel layout of e8f9fba4; production with interpolation before
  and after Cadzow, the PR 1 settings with the same interpolation, and the PR 1 settings with the bad channels as
  missing grid positions.

Usage: python cadzow_pr_figures.py {pr1,pr2}; figures are written to ibl-neuropixel/docs.
"""
# %%
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ibldsp import cadzow
from ibldsp.voltage import current_source_density, interpolate_bad_channels

from cadzow_stagger import ARCHIVE_FINE
from cadzow_synthetic import FS, PRODUCTION, H, csd_error, synthetic_lfp_v2
from cadzow_synthetic_remedies import HXY, seam_ratio

sns.set_theme(context='notebook')
DOCS = Path.home().joinpath('PycharmProjects', 'ephys-atlas', 'packages', 'ibl-neuropixel', 'docs')
NEW = {**PRODUCTION, 'gap_threshold': None, 'fill_grid': True, 'shrinkage': 2.25}
PID = 'e8f9fba4-d151-4b00-bee7-447f0f3e752c'  # bad-channel layout of the PR 2 figure


def plot_csd_panels(rows, csd_true, file, marks=None):
    """
    Grid of CSD images (3 s), true CSD colour scale.

    Parameters
    ----------
    rows : dict
        Row label -> {panel title: CSD (384, ns)}.
    csd_true : ndarray (384, ns)
    file : pathlib.Path
    marks : dict, optional
        Colour -> channel indices marked left of each panel.
    """
    sl = slice(int(4 * FS), int(7 * FS))
    v = np.percentile(np.abs(csd_true[:, sl]), 99)
    ncol = max(len(r) for r in rows.values())
    fig, axs = plt.subplots(len(rows), ncol, figsize=(4.2 * ncol, 6.5 * len(rows)), sharey=True, squeeze=False,
                            layout='constrained')
    for irow, (label, panels) in enumerate(rows.items()):
        for ax, (title, c) in zip(axs[irow], panels.items()):
            im = ax.imshow(c[:, sl], aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v, origin='lower',
                           interpolation='none')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_title(title, fontsize=10)
            for col, ich in (marks or {}).items():
                ax.plot(np.full(len(ich), -0.015), ich, ls='none', marker='>', ms=4, color=col,
                        transform=ax.get_yaxis_transform(), clip_on=False)
        axs[irow, 0].set_ylabel(f'{label}\nchannel')
    fig.colorbar(im, ax=axs, location='right', shrink=0.6, aspect=30, label='CSD [A m$^{-3}$]')
    fig.savefig(file, dpi=70, pil_kwargs={'quality': 80})
    print(file)


def title(name, out, csd_true):
    """Panel title with the CSD error against the truth and the seam ratio."""
    e = csd_error(out, csd_true)
    return f'{name}\nCSD error {np.median(e):.2f}, seams {seam_ratio(e):.2f}'


# %%
if __name__ == '__main__':
    rng = np.random.default_rng(42)
    truth = synthetic_lfp_v2(rng)
    csd_true = current_source_density(truth, H)
    noisy = truth + rng.standard_normal(truth.shape) * 0.03 * 100e-6
    csd = lambda w: current_source_density(w.astype(np.float64), H)
    den = lambda w, **kw: cadzow.cadzow_denoiser(w, h=HXY, fs=FS, n_jobs=8, **kw)
    if sys.argv[1] == 'pr1':
        rows = {}
        for label, wav in (('noise-free', truth), ('3 % white noise', noisy)):
            prod, new = den(wav, **PRODUCTION), den(wav, **NEW)
            rows[label] = {'true CSD': csd_true,
                           title('lfpack v04 (gap_threshold=2, ppca_k=2)', prod, csd_true): csd(prod),
                           title('fill_grid + shrinkage (ppca_k=2)', new, csd_true): csd(new)}
        plot_csd_panels(rows, csd_true, DOCS.joinpath('cadzow_fill_grid_shrinkage.webp'))
    elif sys.argv[1] == 'pr2':
        with h5py.File(ARCHIVE_FINE, 'r') as f:
            labels = f[PID]['00/meta'].attrs['labels'][:]
        bad = np.isin(labels, (1, 2))
        corrupt = noisy.copy()
        corrupt[labels == 1] = rng.standard_normal(((labels == 1).sum(), truth.shape[1])) * 0.01 * 100e-6
        corrupt[labels == 2] += rng.standard_normal(((labels == 2).sum(), truth.shape[1])) * 0.3 * 100e-6
        interp = lambda w, **kw: interpolate_bad_channels(w, labels, H['x'], H['y'], **kw)
        outs = {'lfpack v04, interpolated before/after': interp(den(interp(corrupt), **PRODUCTION), groupby_y=True),
                'PR 1 settings, interpolated before/after': interp(den(interp(corrupt), **NEW), groupby_y=True),
                'PR 1 settings, bad_channels': den(corrupt, **NEW, bad_channels=bad)}
        near = np.zeros(384, dtype=bool)  # channels whose CSD involves a bad channel (same column, <= 2 rows)
        for i in np.where(bad)[0]:
            near[[k for k in range(i - 8, i + 9, 4) if 0 <= k < 384]] = True
        errs = {k: csd_error(o, csd_true) for k, o in outs.items()}
        rows = {'3 % noise, e8f9fba4 bad channels': {
            'true CSD': csd_true,
            **{f'{k}\nCSD error near bad {np.median(errs[k][near]):.2f}, elsewhere {np.median(errs[k][~near]):.2f}':
               csd(o) for k, o in outs.items()}}}
        plot_csd_panels(rows, csd_true, DOCS.joinpath('cadzow_bad_channels.webp'),
                        marks={'k': np.where(labels == 1)[0], 'crimson': np.where(labels == 2)[0]})
