"""Global PCA basis for the ephys-atlas encoding volume: a whitened, data-driven alternative to
hand-picking feature families, kept unchanged from the original neighbour-buffer-framed work (see
`PLAN.md`) since the fit itself doesn't depend on how regions are later compared.

Fit once on all in-mask, non-excluded voxels (z-scored per feature), keep the top-k orthogonal
components explaining 95% of variance. `pairwise_mahalanobis.py` projects the volume onto this
basis and combines per-PC Cohen's d between region pairs in quadrature — a proper Mahalanobis
distance, since the PCs are orthogonal so squaring-and-summing doesn't double-count the ~20
mutually-correlated LFP power/CSD/residual bands the way a naive sum over the raw 41 features would.
"""

from pathlib import Path

import numpy as np

from distinctiveness import EXCLUDE_ACRONYMS, no_coverage_mask


def in_mask_valid(ba, vol=None):
    """Boolean volume (aligned to `ba.label`'s axes, i.e. the same axes as the loaded encoding
    volume): in the atlas brain mask, not void/root/void_fluid, and (if `vol` is given) not an
    unfilled "no probe coverage" voxel (see `distinctiveness.no_coverage_mask`) — ~47k such voxels
    sit inside otherwise-valid regions and would otherwise bias the global PCA fit as extreme,
    non-physiological outliers.
    """
    br = ba.regions
    excl_idx = np.where(np.isin(br.acronym, list(EXCLUDE_ACRONYMS)))[0]
    mask = ba.mask() & ~np.isin(ba.label, excl_idx)
    if vol is not None:
        mask &= ~no_coverage_mask(vol)
    return mask


def fit_global_pca(vol, valid_mask, mean_per_feature, std_per_feature, variance_threshold=0.95):
    """PCA on z-scored in-mask, non-excluded voxels, pooled brainwide.

    z-scoring uses `mean_per_feature`/`std_per_feature` from the encoding-volume npz (computed
    upstream from real recording data), not the volume's own voxel statistics — the volume has no
    NaNs even outside the brain (extrapolated, not real), so an empirical in-volume mean/std would
    still be fine here since we restrict to `valid_mask`, but using the npz's stats keeps this
    consistent with how the volume's features are documented to be normalised elsewhere.

    Returns a dict with `components` (k, n_features), `eigvals` (k,), `explained_ratio` (k,),
    `k`, and `all_explained_ratio` (n_features,) for a scree plot.
    """
    x = vol[valid_mask]
    xz = (x - mean_per_feature) / std_per_feature
    xc = xz - xz.mean(0)
    cov = (xc.T @ xc) / (xc.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    explained_ratio = eigvals / eigvals.sum()
    k = int(np.searchsorted(np.cumsum(explained_ratio), variance_threshold) + 1)
    return {
        "components": eigvecs[:, :k].T,  # (k, n_features)
        "eigvals": eigvals[:k],
        "explained_ratio": explained_ratio[:k],
        "k": k,
        "all_components": eigvecs.T,
        "all_explained_ratio": explained_ratio,
    }


def project_onto_pcs(vol, mean_per_feature, std_per_feature, components):
    """z-score `vol` and project onto `components` (k, n_features) -> (..., k) PC-score volume.

    Voxels outside the brain mask get projected too (cheap, vectorised) but are never used
    downstream since every consumer restricts to `compact >= 0` first.
    """
    vz = (vol - mean_per_feature) / std_per_feature
    return vz @ components.T
