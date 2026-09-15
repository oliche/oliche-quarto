"""Per-region sufficient statistics and pairwise distance utilities, generic over any grouping of
the atlas's raw painted positions and over any per-voxel channel space (the raw 41 features or the
PCA-whitened components from `pca_mahalanobis.py`).

Scope note (2026-09-10): this file used to also build a full hierarchy-aggregated stats table
(one row per Allen ontology node, self + descendants pooled) with a distance-buffer neighbour
Cohen's d per node. That machinery is gone (see `PLAN.md`) — `pairwise_mahalanobis.py` only needs
the *leaf* per-raw-position statistics below, pooled across each named region's raw-id "twins"
(see `pool_by_acronym`), and a plain pairwise distance between all such regions (`mahalanobis_matrix`,
`cohens_d_between`/`mahalanobis_between` for one-off pairs) — no neighbour buffer or hierarchy walk
involved.
"""

import numpy as np
import pandas as pd

from distinctiveness import EXCLUDE_ACRONYMS
from pca_mahalanobis import fit_global_pca, in_mask_valid, project_onto_pcs


def direct_paint_stats(vol, feature_names, ba, extra_exclude=None):
    """Per-position (n, mean, var) from voxels *directly* painted with each raw atlas position
    (0..len(br.id)-1 — the same indices `ba.label` itself uses), in one pass over the volume."""
    n_regions = len(ba.regions.id)
    n_features = len(feature_names)
    mask = ba.mask()
    if extra_exclude is not None:
        mask = mask & ~extra_exclude
    flat_labels = ba.label[mask].astype(np.int64)
    x = vol[mask]

    count = np.bincount(flat_labels, minlength=n_regions).astype(np.int64)
    total = np.zeros((n_regions, n_features))
    total_sq = np.zeros((n_regions, n_features))
    np.add.at(total, flat_labels, x)
    np.add.at(total_sq, flat_labels, x ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count[:, None]
        var = total_sq / count[:, None] - mean ** 2
    return count, mean, var


def pool_stats(count, mean, var, idx):
    """Pool (n, mean, var) over positions `idx` into one combined (n, mean, var) — an "online"/
    parallel mean-variance combination (Chan et al.), exact rather than approximate.

    This is deliberately general-purpose, not tied to hierarchy traversal: `idx` can be any array
    of raw positions you like — a hand-picked list of several unrelated regions, a region's raw-id
    "twins" (see `pool_by_acronym`), anything — so you can pool arbitrary groupings on demand and
    feed them to `cohens_d_between`/`mahalanobis_between` without ever touching the encoding volume
    again.

    Uses the parallel-axis / law-of-total-variance identity — the pooled variance is the
    (n-weighted) average of the group variances *plus* the (n-weighted) variance *of* the group
    means around the pooled mean. Averaging the per-group standard deviations directly (skipping
    that second term) would silently ignore any real between-group spread and understate the
    pooled variance whenever the group means actually differ from each other.

    Drops zero-count members before weighting: `mean`/`var` are `NaN` where `count == 0` (see
    `direct_paint_stats`), and `0 * NaN == NaN` in IEEE float arithmetic, so leaving them in would
    silently poison the pooled statistic.
    """
    idx = idx[count[idx] > 0]
    n = count[idx]
    n_total = n.sum()
    if n_total == 0:
        nan = np.full(mean.shape[1], np.nan)
        return 0, nan, nan
    m, v = mean[idx], var[idx]
    mean_total = (n[:, None] * m).sum(0) / n_total
    var_total = (n[:, None] * (v + (m - mean_total) ** 2)).sum(0) / n_total
    return int(n_total), mean_total, var_total


def pool_by_acronym(count, mean, var, ba, min_voxels=0):
    """Pool each named region's directly-painted raw position(s) into one row per acronym.

    The atlas's raw partition (`ba.label`) paints every named region across one or two raw
    positions — a `+id` and, for most regions, a genuinely independently-painted `-id` "twin"
    (confirmed by inspection, e.g. PIR: 46286 vs 46289 voxels each, spanning both hemispheres;
    CB: 3304 vs 3317, one per ML side — not a near-empty duplicate either way). For this analysis
    both hemispheres/twins of a region are considered symmetric, so they're pooled with
    `pool_stats` into a single row rather than shown as two near-identical regions.

    `mean`/`var` may be raw feature-space or PCA-space statistics (anything shaped
    `(n_positions, n_channels)` from `direct_paint_stats`, run once on `vol` or once on a
    `project_onto_pcs`-transformed volume). Regions in `EXCLUDE_ACRONYMS` are dropped; regions with
    fewer than `min_voxels` pooled voxels are dropped (too few for a trustworthy estimate).

    Returns `(df, mean_pooled, var_pooled)`: `df` has one row per kept acronym with `n_voxels`,
    `hexcolor` (the shared Allen atlas colour of both twins) and `order` (the CCF hierarchy
    traversal order, identical between twins — useful for sorting a report table or figure
    anatomically); `mean_pooled`/`var_pooled` are `(len(df), n_channels)`, row-aligned with `df`.
    """
    br = ba.regions
    present = np.where(count > 0)[0]
    present = present[~np.isin(br.acronym[present], list(EXCLUDE_ACRONYMS))]
    acronyms_present = br.acronym[present]
    uniq_acr = np.unique(acronyms_present)

    n_pooled = np.zeros(len(uniq_acr), dtype=np.int64)
    mean_pooled = np.zeros((len(uniq_acr), mean.shape[1]))
    var_pooled = np.zeros((len(uniq_acr), mean.shape[1]))
    hexcolor = np.empty(len(uniq_acr), dtype=object)
    order = np.zeros(len(uniq_acr), dtype=np.int64)
    for i, a in enumerate(uniq_acr):
        idx = present[acronyms_present == a]
        n_pooled[i], mean_pooled[i], var_pooled[i] = pool_stats(count, mean, var, idx)
        hexcolor[i] = br.hexcolor[idx[0]]
        order[i] = br.order[idx[0]]

    df = pd.DataFrame({
        "acronym": uniq_acr, "n_voxels": n_pooled, "hexcolor": hexcolor, "order": order,
    })
    keep = n_pooled >= min_voxels
    df = df[keep].reset_index(drop=True)
    return df, mean_pooled[keep], var_pooled[keep]


def cohens_d_between(count, mean, var, idx_a, idx_b):
    """Feature-wise Cohen's d between two arbitrary groupings of raw positions, each pooled first
    with `pool_stats`. Works identically whether `count`/`mean`/`var` are the raw 41-feature
    arrays (`direct_paint_stats(vol, ...)`) or the PCA ones (`direct_paint_stats(pc_vol, ...)`) —
    pick whichever space you want the comparison expressed in.
    """
    n_a, mean_a, var_a = pool_stats(count, mean, var, idx_a)
    n_b, mean_b, var_b = pool_stats(count, mean, var, idx_b)
    pooled_std = np.sqrt((var_a * n_a + var_b * n_b) / (n_a + n_b))
    return (mean_a - mean_b) / (pooled_std + 1e-9)


def mahalanobis_between(count_pc, mean_pc, var_pc, idx_a, idx_b):
    """Mahalanobis distance (PCA-whitened) between two arbitrary groupings: `sqrt(sum(d**2))` of
    `cohens_d_between` run on PC-space stats. Valid because the PCs are orthogonal, so
    squaring-and-summing doesn't double-count shared variance the way it would for the 41
    correlated raw features.
    """
    d = cohens_d_between(count_pc, mean_pc, var_pc, idx_a, idx_b)
    return float(np.sqrt(np.sum(d ** 2)))


def mahalanobis_matrix(count, mean, var):
    """Full pairwise Mahalanobis-distance matrix between every row of already-pooled (n, mean, var)
    statistics (e.g. `pool_by_acronym`'s output) — a vectorised generalisation of
    `cohens_d_between`/`mahalanobis_between` to every pair at once via broadcasting, instead of a
    Python loop over `n*(n-1)/2` pairs (worth it once `n` reaches the few-hundred range this
    project's qualifying-region lists land in).

    Returns `(mahalanobis, d)`: `mahalanobis` is `(n, n)`, symmetric with a zero diagonal. `d` is
    `(n, n, k)`, the signed per-column Cohen's d of row region relative to column region
    (antisymmetric: `d[j, i] == -d[i, j]`); `mahalanobis = sqrt(sum(d**2, axis=-1))`. Only a true
    Mahalanobis distance if the `k` columns are orthogonal (PCA-space input) — on raw correlated
    features this would double-count shared variance instead.
    """
    n_i = count[:, None, None].astype(float)
    n_j = count[None, :, None].astype(float)
    pooled_var = (var[:, None, :] * n_i + var[None, :, :] * n_j) / (n_i + n_j)
    d = (mean[:, None, :] - mean[None, :, :]) / (np.sqrt(pooled_var) + 1e-9)
    mahalanobis = np.sqrt(np.sum(d ** 2, axis=-1))
    return mahalanobis, d


def fit_and_project_pca(vol, ba, mean_per_feature, std_per_feature, extra_exclude=None,
                         variance_threshold=0.95):
    """Fit the global PCA basis once (see `pca_mahalanobis.fit_global_pca`) and project the whole
    volume onto it. Returns `(pca, pc_vol, pc_names, count_pc, mean_pc, var_pc)` — the last three
    are `direct_paint_stats` run on the projected volume: the *leaf* (per-raw-position) PC-score
    statistics that `pool_by_acronym` pools into one row per named region.
    """
    valid_mask = in_mask_valid(ba, vol)
    pca = fit_global_pca(vol, valid_mask, mean_per_feature, std_per_feature, variance_threshold)
    pc_vol = project_onto_pcs(vol, mean_per_feature, std_per_feature, pca["components"])
    pc_names = [f"PC{i + 1}" for i in range(pca["k"])]
    count_pc, mean_pc, var_pc = direct_paint_stats(pc_vol, pc_names, ba, extra_exclude)
    return pca, pc_vol, pc_names, count_pc, mean_pc, var_pc
