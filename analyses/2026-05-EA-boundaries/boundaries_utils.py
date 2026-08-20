"""Utility functions for ephys-atlas region boundary landmark analysis."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import RobustScaler

_DEFAULT_ROOT = Path.home().joinpath('data', 'ephys-atlas', 'features')
_ALYX_URL = 'https://alyx.internationalbrainlab.org'

EXCLUDE_COLS = frozenset({
    'axial_um', 'lateral_um', 'x', 'y', 'z', 'x_target', 'y_target', 'z_target',
    'acronym', 'atlas_id', 'Allen_id', 'Cosmos_id', 'Beryl_id', 'outside',
    'channel_labels',
    # probe-geometry feature — not a tissue property
    'distance_to_tip_um',
    # aperiodic residuals — redundant given psd_pc0/pc1 capturing broadband power
    'psd_residual_lfp', 'psd_residual_delta', 'psd_residual_theta',
    'psd_residual_alpha', 'psd_residual_beta', 'psd_residual_gamma',
})

# Region IDs treated as pass-through labels during nearest-neighbour interpolation.
# void (0), VS/ventricular systems (73), root (997), fiber tracts (1009), void_fluid (2000).
_NNI_IDS = frozenset({0, 73, 997, 1009, 2000})


def load_features(
    project: str = 'ea_active',
    root_path_features: Path | None = None,
    vintage: str | None = None,
    strict: bool = False,
):
    """Load ephys-atlas channel features from local cache, downloading if needed.

    Parameters
    ----------
    project:
        Alyx project tag (default ``'ea_active'``).
    root_path_features:
        Root directory for cached feature parquets. Defaults to
        ``~/data/ephys-atlas/features``.
    vintage:
        Dataset label (e.g. ``'2026_W21'``). Resolved automatically from Alyx
        when ``None``.
    strict:
        Passed to ``ephysatlas.data.read_features_from_disk``.

    Returns
    -------
    df_features:
        One row per recording channel.
    brain_atlas:
        ``ClassifierAtlas`` instance (reuse to avoid rebuilding it).
    vintage:
        Resolved vintage label string.
    """
    import ephysatlas.anatomy
    import ephysatlas.data
    from one.api import ONE

    if root_path_features is None:
        root_path_features = _DEFAULT_ROOT

    one = ONE(base_url=_ALYX_URL, mode='remote')
    if vintage is None:
        vintage = ephysatlas.data.get_latest_label(one=one, project=project)
    print(f"Using vintage: {vintage}")

    path_features = root_path_features.joinpath(project, vintage, 'agg_full')
    if not path_features.exists():
        ephysatlas.data.download_tables(root_path_features, label=vintage, one=one)

    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    df_features = ephysatlas.data.read_features_from_disk(
        path_features, brain_atlas=brain_atlas, strict=strict
    )
    print(f"Loaded {len(df_features):,} channels, {df_features.shape[1]} columns")
    return df_features, brain_atlas, vintage


def load_or_fit_psd_pca(
    raw_features_path: Path,
    cache_dir: Path,
    vintage: str,
    brain_atlas,
    n_components_psd: int = 2,
    n_components_csd: int = 2,
):
    """Fit EphysPsdPCA on raw features using only volume-compatible columns.

    Excludes ``*_csd_diff1`` features so the exact same scaler+PCA can be
    applied to both the measured channel dataframe and the encoding volume.
    The fitted object is cached as ``psd_pca_{vintage}.pkl``.

    Parameters
    ----------
    raw_features_path:
        Path to the raw feature directory (before PCA), i.e.
        ``~/data/ephys-atlas/features/ea_active/{vintage}/agg_full``.
    cache_dir:
        Directory where the pickle is stored.
    vintage:
        Dataset vintage label (used in the cache filename).
    brain_atlas:
        ``ClassifierAtlas`` instance.
    n_components_psd:
        PSD principal components to retain.
    n_components_csd:
        CSD principal components to retain.

    Returns
    -------
    psd_pca:
        Fitted ``EphysPsdPCA`` instance (``scaler_psd_``, ``pca_psd_``,
        ``scaler_csd_``, ``pca_csd_`` attributes ready for ``transform``).
    """
    import ephysatlas.data
    from ephysatlas.features import EphysPsdPCA

    cache_path = cache_dir.joinpath(f'psd_pca_{vintage}.pkl')
    if cache_path.exists():
        print(f'Loading cached PSD/CSD PCA from {cache_path.name}')
        with open(cache_path, 'rb') as fh:
            return pickle.load(fh)

    print('Fitting PSD/CSD PCA on raw features (volume-compatible: no diff1) …')
    df_raw = ephysatlas.data.read_features_from_disk(
        raw_features_path, brain_atlas=brain_atlas, strict=False
    )
    # Drop diff1 CSD features — not present in the encoding volume
    diff1_cols = [c for c in df_raw.columns if c.endswith('_diff1')]
    df_raw = df_raw.drop(columns=diff1_cols)
    print(f'  Dropped {len(diff1_cols)} diff1 columns: {diff1_cols}')

    psd_pca = EphysPsdPCA(
        n_components_psd=n_components_psd, n_components_csd=n_components_csd
    ).fit(df_raw)

    cache_dir.mkdir(exist_ok=True)
    with open(cache_path, 'wb') as fh:
        pickle.dump(psd_pca, fh)
    print(f'  Cached → {cache_path.name}')
    return psd_pca


def load_or_build_pca_df(
    raw_features_path: Path,
    cache_dir: Path,
    vintage: str,
    brain_atlas,
    psd_pca=None,
) -> pd.DataFrame:
    """Load (or build and cache) the post-PCA channel features dataframe.

    Uses the volume-compatible PCA from :func:`load_or_fit_psd_pca`, so the
    resulting ``psd_pc0/pc1`` and ``csd_pc0/pc1`` columns are on the exact same
    axes as those derived from encoding volume lookups.

    Parameters
    ----------
    raw_features_path:
        Path to the raw feature directory.
    cache_dir:
        Directory for the parquet cache.
    vintage:
        Dataset vintage label.
    brain_atlas:
        ``ClassifierAtlas`` instance.
    psd_pca:
        Pre-fitted ``EphysPsdPCA``.  Loaded via :func:`load_or_fit_psd_pca`
        when ``None``.

    Returns
    -------
    df:
        Channel features dataframe with PSD/CSD columns replaced by PC scores.
    """
    import ephysatlas.data

    cache_path = cache_dir.joinpath(f'df_{vintage}.parquet')
    if cache_path.exists():
        print(f'Loading cached post-PCA dataframe from {cache_path.name}')
        return pd.read_parquet(cache_path)

    if psd_pca is None:
        psd_pca = load_or_fit_psd_pca(raw_features_path, cache_dir, vintage, brain_atlas)

    print('Building post-PCA dataframe …')
    df_raw = ephysatlas.data.read_features_from_disk(
        raw_features_path, brain_atlas=brain_atlas, strict=False
    )
    diff1_cols = [c for c in df_raw.columns if c.endswith('_diff1')]
    df_raw = df_raw.drop(columns=diff1_cols)
    df = psd_pca.transform(df_raw)
    df.to_parquet(cache_path)
    print(f'  Cached → {cache_path.name}')
    return df


def build_volume_feature_df(
    df_measured: pd.DataFrame,
    vol_data,
    brain_atlas,
    psd_pca,
) -> pd.DataFrame:
    """Build a synthetic channel features dataframe from encoding volume lookups.

    For every channel in *df_measured* the encoding volume is sampled at the
    channel's CCF (x, y, z) coordinate.  The resulting raw features are passed
    through *psd_pca* — the **same** fitted object used for the measured data —
    so that ``psd_pc0/pc1`` and ``csd_pc0/pc1`` are on identical axes in both
    the measured and volume figures.  All metadata columns (``axial_um``,
    region labels, coordinates) are preserved from *df_measured* so boundary
    detection and probe sorting work unchanged.

    Parameters
    ----------
    df_measured:
        Measured channel features dataframe (MultiIndex ``pid × channel``).
        Provides coordinates and metadata; feature values are replaced.
    vol_data:
        Loaded ``.npz`` encoding volume (see ``ea-load-encoding-volumes`` skill).
    brain_atlas:
        ``AllenAtlas`` / ``ClassifierAtlas`` instance for coordinate conversion.
    psd_pca:
        Fitted ``EphysPsdPCA`` instance — must be the same object used to
        build the measured feature dataframe.

    Returns
    -------
    df_vol:
        Synthetic feature dataframe with the same index as *df_measured*.
    """
    vol = vol_data['ephys_atlas_vol']       # (nx, ny, nz, n_feats) float16
    feat_names = list(vol_data['feature_names'])

    xyz_m = df_measured[['x', 'y', 'z']].values   # metres, mlapdv
    vox_idx = brain_atlas.bc.xyz2i(xyz_m, mode='clip')   # (n_channels, 3)

    # Vectorised lookup: fancy index into (nx, ny, nz, n_feats)
    ix, iy, iz = vox_idx[:, 0], vox_idx[:, 1], vox_idx[:, 2]
    feat_matrix = vol[ix, iy, iz, :].astype(float)   # (n_channels, n_feats)

    df_vol_raw = pd.DataFrame(feat_matrix, index=df_measured.index, columns=feat_names)

    # Preserve all metadata columns (coordinates, region labels, depth, …)
    for col in EXCLUDE_COLS:
        if col in df_measured.columns:
            df_vol_raw[col] = df_measured[col].values

    return psd_pca.transform(df_vol_raw)


def compute_cosmos_transitions(
    df_features: pd.DataFrame,
    brain_atlas,
) -> pd.DataFrame:
    """Build a directional Cosmos-level boundary transition count matrix.

    Channels are aggregated by depth within each probe (modal Cosmos region),
    then adjacent-depth transitions are detected. Root (id=997) is split into
    fiber tracts (1009), ventricular systems (73), and unlabelled root.

    Parameters
    ----------
    df_features:
        Channel features dataframe (MultiIndex with ``pid`` level).
    brain_atlas:
        ``ClassifierAtlas`` instance with a ``.regions`` attribute.

    Returns
    -------
    count_matrix:
        Square DataFrame with acronym labels; rows = lower/deeper region,
        columns = upper/shallower region.
    """
    regions = brain_atlas.regions
    ft_ids = set(regions.subtree(1009)['id']) - {997}
    vs_ids = set(regions.subtree(73)['id']) - {997}

    allen_int = df_features['Allen_id'].astype(int)
    root_mask = df_features['Cosmos_id'] == 997
    cosmos_refined = df_features['Cosmos_id'].copy()
    cosmos_refined.loc[root_mask & allen_int.isin(ft_ids)] = 1009
    cosmos_refined.loc[root_mask & allen_int.isin(vs_ids)] = 73

    cosmos_ids = sorted(cosmos_refined.unique())
    id2acr = {cid: regions.acronym[regions.id2index(cid)[1]][0][0] for cid in cosmos_ids}
    print("Refined Cosmos regions:", {cid: id2acr[cid] for cid in cosmos_ids})

    df_pid = pd.DataFrame(
        {'axial_um': df_features['axial_um'], 'Cosmos_refined_id': cosmos_refined}
    ).reset_index(level='pid')
    df_depth = (
        df_pid.groupby(['pid', 'axial_um'])['Cosmos_refined_id']
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .sort_values(['pid', 'axial_um'])
    )
    df_depth = apply_depth_nni(df_depth, cr_col='Cosmos_refined_id')

    next_cosmos = df_depth.groupby('pid')['Cosmos_refined_id'].shift(-1)
    mask = next_cosmos.notna() & (next_cosmos != df_depth['Cosmos_refined_id'])
    transitions = pd.DataFrame({
        'from': df_depth.loc[mask, 'Cosmos_refined_id'].astype(int).values,
        'to': next_cosmos[mask].astype(int).values,
    })

    count_matrix = (
        transitions.groupby(['from', 'to']).size()
        .unstack(fill_value=0)
        .reindex(index=cosmos_ids, columns=cosmos_ids, fill_value=0)
    )
    count_matrix.index = [id2acr[i] for i in cosmos_ids]
    count_matrix.columns = [id2acr[i] for i in cosmos_ids]
    return count_matrix


def get_cosmos_refined(df_features: pd.DataFrame, brain_atlas) -> pd.Series:
    """Return Cosmos_refined_id: root (997) split into fiber tracts (1009), VS (73), and remainder.

    Parameters
    ----------
    df_features:
        Channel features dataframe with ``Cosmos_id`` and ``Allen_id`` columns.
    brain_atlas:
        ``ClassifierAtlas`` instance.

    Returns
    -------
    cosmos_refined:
        Series of refined Cosmos IDs, same index as *df_features*.
    """
    regions = brain_atlas.regions
    ft_ids = set(regions.subtree(1009)['id']) - {997}
    vs_ids = set(regions.subtree(73)['id']) - {997}
    allen_int = df_features['Allen_id'].astype(int)
    root_mask = df_features['Cosmos_id'] == 997
    cosmos_refined = df_features['Cosmos_id'].copy()
    cosmos_refined.loc[root_mask & allen_int.isin(ft_ids)] = 1009
    cosmos_refined.loc[root_mask & allen_int.isin(vs_ids)] = 73
    return cosmos_refined


def apply_depth_nni(
    df_depth: pd.DataFrame,
    cr_col: str = 'cr',
    nni_ids: frozenset = _NNI_IDS,
) -> pd.DataFrame:
    """Replace interior void/root/fiber-tracts depth entries with nearest valid neighbour.

    Entries whose ``cr_col`` value is in ``nni_ids`` and that are sandwiched
    between two valid labels within the same probe are replaced by the nearer
    neighbour (tie broken in favour of the forward fill). Entries at probe
    extremities — before the first or after the last valid label — are left
    unchanged so that true surface and endpoint transitions are preserved.

    Parameters
    ----------
    df_depth:
        DataFrame sorted by ``(pid, axial_um)`` with columns ``pid`` and
        ``cr_col``.
    cr_col:
        Column name holding the Cosmos-refined region ID.
    nni_ids:
        Region IDs to treat as invalid / interpolatable.

    Returns
    -------
    df_depth:
        Copy with ``cr_col`` updated in-place for interior entries.
    """
    ids = df_depth[cr_col].values.copy().astype(float)
    pids = df_depth['pid'].values
    bad = np.isin(ids, list(nni_ids))

    df_fill = pd.DataFrame({'pid': pids, 'ids': np.where(bad, np.nan, ids)})
    ids_ff = df_fill.groupby('pid')['ids'].ffill().values
    ids_bf = df_fill.groupby('pid')['ids'].bfill().values

    pos = np.arange(len(ids), dtype=float)
    df_pos = pd.DataFrame({'pid': pids, 'pos': np.where(bad, np.nan, pos)})
    pos_ff = df_pos.groupby('pid')['pos'].ffill().values
    pos_bf = df_pos.groupby('pid')['pos'].bfill().values
    dist_prev = pos - pos_ff
    dist_next = pos_bf - pos

    interior = bad & ~np.isnan(ids_ff) & ~np.isnan(ids_bf)
    use_ff = interior & (dist_prev <= dist_next)
    use_bf = interior & ~use_ff

    new_ids = ids.copy()
    new_ids[use_ff] = ids_ff[use_ff]
    new_ids[use_bf] = ids_bf[use_bf]

    df_out = df_depth.copy()
    df_out[cr_col] = new_ids.astype(int)
    n_interp = int((df_out[cr_col].values != df_depth[cr_col].values).sum())
    print(f"NNI: replaced {n_interp:,} interior depth positions")
    return df_out


def find_boundary_crossings(
    df_features: pd.DataFrame,
    brain_atlas,
    from_acr: str,
    to_acr: str,
) -> list[tuple[str, float]]:
    """Find (pid, transition_depth_um) for each from_acr → to_acr crossing.

    Channels are depth-aggregated within each probe (modal Cosmos_refined),
    nearest-neighbour interpolation is applied to interior void/root entries,
    then adjacent-depth transitions are detected. Only the first crossing per
    probe is returned.

    Parameters
    ----------
    df_features:
        Channel features dataframe (MultiIndex with ``pid`` level).
    brain_atlas:
        ``ClassifierAtlas`` instance.
    from_acr:
        Acronym of the lower/deeper region.
    to_acr:
        Acronym of the upper/shallower region.

    Returns
    -------
    crossings:
        List of ``(pid, transition_depth_um)`` tuples sorted by pid.
    """
    regions = brain_atlas.regions
    cosmos_refined = get_cosmos_refined(df_features, brain_atlas)

    cosmos_ids = sorted(cosmos_refined.unique())
    id2acr = {cid: regions.acronym[regions.id2index(cid)[1]][0][0] for cid in cosmos_ids}
    acr2id = {v: k for k, v in id2acr.items()}
    from_id = acr2id[from_acr]
    to_id = acr2id[to_acr]

    df_work = pd.DataFrame(
        {'axial_um': df_features['axial_um'], 'cr': cosmos_refined}
    ).reset_index(level='pid')
    df_depth = (
        df_work.groupby(['pid', 'axial_um'])['cr']
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .sort_values(['pid', 'axial_um'])
    )
    df_depth = apply_depth_nni(df_depth, cr_col='cr')

    crossings = []
    for pid, grp in df_depth.groupby('pid'):
        grp = grp.sort_values('axial_um').reset_index(drop=True)
        rid = grp['cr'].values
        dep = grp['axial_um'].values
        for i in range(len(rid) - 1):
            if rid[i] == from_id and rid[i + 1] == to_id:
                crossings.append((pid, (dep[i] + dep[i + 1]) / 2.0))
                break
    return crossings


def compute_boundary_feature_stats(
    df_features: pd.DataFrame,
    brain_atlas,
    window_um: float = 200.0,
    min_transitions: int = 16,
    extra_exclude: frozenset | None = None,
) -> pd.DataFrame:
    """Scan every Cosmos boundary for sharp feature transitions (Mann-Whitney U).

    Applies nearest-neighbour interpolation to interior void/root depth entries
    before detecting transitions, so double-hops through pass-through regions
    are collapsed into direct A→B crossings.

    Parameters
    ----------
    df_features:
        Channel features dataframe (MultiIndex with ``pid`` level).
    brain_atlas:
        ``ClassifierAtlas`` instance.
    window_um:
        Window size on each side of the boundary used for feature sampling.
    min_transitions:
        Minimum number of depth-level transitions required to test a pair.
    extra_exclude:
        Additional column names to exclude from the feature scan beyond
        ``EXCLUDE_COLS``.  Defaults to waveform-QC columns that have no
        physiological interpretation at boundaries.

    Returns
    -------
    df_results:
        One row per (boundary, feature), sorted by ``cohens_d`` descending.
        Columns: ``from``, ``to``, ``feature``, ``n_trans``, ``n_probes``,
        ``cohens_d``, ``pval``, ``pval_bonf``.
    """
    from scipy import stats as sp_stats

    if extra_exclude is None:
        extra_exclude = frozenset({
            'spike_count', 'polarity', 'decay_n_peaks',
            'decay_fit_r_squared', 'decay_fit_error',
        })
    skip = EXCLUDE_COLS | extra_exclude
    feature_cols = [c for c in df_features.columns if c not in skip]
    print(f"{len(feature_cols)} features to test")

    regions = brain_atlas.regions
    cosmos_refined = get_cosmos_refined(df_features, brain_atlas)
    cosmos_ids = sorted(cosmos_refined.unique())
    id2acr = {cid: regions.acronym[regions.id2index(cid)[1]][0][0] for cid in cosmos_ids}

    # Depth-aggregate then NNI
    df_work = pd.DataFrame(
        {'axial_um': df_features['axial_um'], 'cr': cosmos_refined}
    ).reset_index(level='pid')
    df_depth = (
        df_work.groupby(['pid', 'axial_um'])['cr']
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .sort_values(['pid', 'axial_um'])
    )
    df_depth = apply_depth_nni(df_depth, cr_col='cr')

    # Transition events from NNI'd depth labels
    next_cr = df_depth.groupby('pid')['cr'].shift(-1)
    mask_t = next_cr.notna() & (next_cr != df_depth['cr'])
    df_trans = df_depth[mask_t].copy()
    df_trans['to_cr'] = next_cr[mask_t].astype(int)
    df_trans = df_trans.rename(columns={'cr': 'from_cr', 'axial_um': 'trans_depth'})
    pair_counts = df_trans.groupby(['from_cr', 'to_cr']).size()

    # Feature data: use raw cosmos_refined (channel-level) for region membership
    df_feat = df_features[['axial_um'] + feature_cols].copy()
    df_feat['cr'] = cosmos_refined
    df_feat_reset = df_feat.reset_index(level='pid')

    results = []
    for (from_c, to_c), n_trans in pair_counts.items():
        if n_trans < min_transitions:
            continue
        if from_c not in id2acr or to_c not in id2acr:
            continue

        pair_ev = df_trans.loc[
            (df_trans['from_cr'] == from_c) & (df_trans['to_cr'] == to_c),
            ['pid', 'trans_depth'],
        ]
        pids = pair_ev['pid'].unique()
        merged = pair_ev.merge(df_feat_reset[df_feat_reset['pid'].isin(pids)], on='pid')

        before = merged[
            (merged['cr'] == from_c) &
            (merged['axial_um'] >= merged['trans_depth'] - window_um) &
            (merged['axial_um'] <= merged['trans_depth'])
        ]
        after = merged[
            (merged['cr'] == to_c) &
            (merged['axial_um'] >= merged['trans_depth']) &
            (merged['axial_um'] <= merged['trans_depth'] + window_um)
        ]

        for feat in feature_cols:
            a = before[feat].dropna().values
            b = after[feat].dropna().values
            if len(a) < 10 or len(b) < 10:
                continue
            _, pval = sp_stats.mannwhitneyu(a, b, alternative='two-sided')
            pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            d = float(abs(np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0
            results.append({
                'from': id2acr[from_c], 'to': id2acr[to_c],
                'feature': feat, 'n_trans': int(n_trans), 'n_probes': int(len(pids)),
                'cohens_d': round(d, 4), 'pval': pval,
            })

    df_results = pd.DataFrame(results)
    df_results['pval_bonf'] = (df_results['pval'] * len(df_results)).clip(upper=1.0)
    df_results = df_results.sort_values('cohens_d', ascending=False).reset_index(drop=True)
    n_land = int(((df_results['cohens_d'] > 0.8) & (df_results['pval_bonf'] < 0.01)).sum())
    print(f"Done: {len(df_results)} (boundary, feature) pairs — {n_land} landmark candidates (d>0.8, p_bonf<0.01)")
    return df_results


# RdBu_r for AP-band / waveform features; PuOr for LFP-band / PSD features.
def _feature_cmap(feat: str) -> str:
    """Return diverging colormap name for a feature based on its signal modality."""
    if any(feat.startswith(p) for p in ('psd_', 'aperiodic_', 'rms_lf')):
        return 'PuOr'
    return 'RdBu_r'

# Fallback RGBA colours for region IDs not in the Allen atlas RGB table.
_REGION_COLORS_RGBA: dict[int, tuple[float, float, float, float]] = {
    0: (0.55, 0.55, 0.55, 1.0),      # void
    997: (0.65, 0.65, 0.65, 1.0),    # root
    2000: (0.60, 0.82, 0.93, 1.0),   # void_fluid
}


def _get_region_rgba(regions, region_id: int) -> tuple[float, float, float, float]:
    """Return (r, g, b, a) for a region id, with fallback for custom IDs."""
    if region_id in _REGION_COLORS_RGBA:
        return _REGION_COLORS_RGBA[region_id]
    try:
        idx = regions.id2index(region_id)[1]
        r, g, b = regions.rgb[idx][0, 0] / 255.0
        return (float(r), float(g), float(b), 1.0)
    except Exception:
        return (0.65, 0.65, 0.65, 1.0)


def _percentile_vlims(
    arr: np.ndarray,
    lo: float = 2.0,
    hi: float = 98.0,
) -> tuple[float, float]:
    """Compute display limits symmetric around the local median."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    center = float(np.median(finite))
    v_lo, v_hi = np.percentile(finite, [lo, hi])
    vabs = max(abs(v_lo - center), abs(v_hi - center))
    return (center - vabs, center + vabs)


def _probe_coord(df_features: pd.DataFrame, pid: str, trans_depth: float, col: str) -> float:
    """Return the value of *col* for the channel nearest to *trans_depth* in probe *pid*."""
    try:
        probe_df = df_features.xs(pid, level='pid')
        idx = (probe_df['axial_um'] - trans_depth).abs().idxmin()
        return float(probe_df.loc[idx, col])
    except (KeyError, ValueError):
        return 0.0


def _compute_rastermap_isort(
    feat_data: np.ndarray,
    feature_list: list[str],
    feature_scaler,
    cache_dir: Path | None,
    from_acr: str,
    to_acr: str,
    pids: list,
) -> np.ndarray:
    """Compute (and optionally cache) rastermap probe sort order.

    Parameters
    ----------
    feat_data:
        Array of shape ``(n_bins, n_probes, n_feats)`` — raw (un-normalised) values.
    feature_list:
        Feature names along axis 2 of *feat_data*.
    feature_scaler:
        ``(RobustScaler, list[str])`` tuple where the list gives the feature
        names the scaler was fit on (same column order). Pass ``None`` to skip
        normalisation.
    cache_dir:
        Directory for ``.npy`` cache files.  ``None`` disables caching.
    from_acr, to_acr:
        Boundary acronyms (used in the cache filename).
    pids:
        Ordered list of probe IDs corresponding to axis 1 of *feat_data* (used
        in the cache key).

    Returns
    -------
    isort:
        Integer index array of length ``n_probes`` giving the rastermap sort order.
    """
    n_bins, n_probes, n_feats = feat_data.shape

    # Stable cache key: hash (pids + features + shape)
    key = (
        f"{from_acr}_{to_acr}"
        f"_{'|'.join(str(p) for p in pids)}"
        f"_{'|'.join(feature_list)}"
        f"_{n_bins}"
    )
    cache_hash = hashlib.md5(key.encode()).hexdigest()[:10]

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir.joinpath(f"rastermap_{from_acr}_{to_acr}_{cache_hash}.npy")
        if cache_path.exists():
            print(f"Rastermap cache hit: {cache_path.name}")
            return np.load(cache_path)

    # Normalise using the pre-fitted scaler (per feature)
    feat_norm = feat_data.copy().astype(float)
    if feature_scaler is not None:
        scaler, scaler_features = feature_scaler
        feat_idx_map = {f: i for i, f in enumerate(scaler_features)}
        for k, feat in enumerate(feature_list):
            if feat in feat_idx_map:
                idx = feat_idx_map[feat]
                scale = float(scaler.scale_[idx])
                if scale > 0:
                    feat_norm[:, :, k] = (feat_data[:, :, k] - scaler.center_[idx]) / scale

    feat_norm = np.nan_to_num(feat_norm, nan=0.0)

    # Rastermap expects (n_samples, n_features) where samples = probes
    X = feat_norm.transpose(1, 0, 2).reshape(n_probes, n_bins * n_feats)

    from rastermap import Rastermap  # lazy import — optional dependency
    n_pcs = max(1, min(64, n_probes - 1, n_bins * n_feats - 1))
    print(f"Running Rastermap on ({n_probes} probes × {n_bins * n_feats} dims) …")
    try:
        model = Rastermap(n_PCs=n_pcs, verbose=False).fit(X)
        isort = model.isort
    except Exception as exc:
        print(f"Rastermap failed ({exc}), falling back to original order")
        return np.arange(n_probes)

    if cache_dir is not None:
        np.save(cache_path, isort)
        print(f"Rastermap cached → {cache_path.name}")

    return isort


def compute_feature_vlims(
    df_features: pd.DataFrame,
    features: list[str] | None = None,
    n_iqr: float = 3.5,
    return_scaler: bool = False,
) -> dict[str, tuple[float, float]] | tuple[dict, object, list[str]]:
    """Compute global display limits for each feature using RobustScaler statistics.

    Fits a :class:`~sklearn.preprocessing.RobustScaler` on the full features
    table (median + IQR) and returns limits as
    ``[median - n_iqr*IQR, median + n_iqr*IQR]`` per feature.  Passing the
    result as ``feature_vlims`` to :func:`plot_boundary_feature_profiles`
    keeps colours comparable across all boundary figures.

    Parameters
    ----------
    df_features:
        Full channel features dataframe.
    features:
        Column names to include. Defaults to all numeric columns not in
        ``EXCLUDE_COLS`` plus ``spike_count``.
    n_iqr:
        Half-range in units of IQR around the median.

    Returns
    -------
    vlims:
        ``{feature: (vmin, vmax)}`` in original (unscaled) units.
    scaler:
        Fitted ``RobustScaler`` (only returned when *return_scaler* is ``True``).
    features:
        Ordered list of feature names the scaler was fit on (only returned when
        *return_scaler* is ``True``).
    """
    if features is None:
        extra = frozenset({'spike_count'})
        features = [c for c in df_features.columns if c not in (EXCLUDE_COLS - extra)]

    X = df_features[features].dropna()
    scaler = RobustScaler()
    scaler.fit(X)

    vlims = {
        feat: (
            float(scaler.center_[i] - n_iqr * scaler.scale_[i]),
            float(scaler.center_[i] + n_iqr * scaler.scale_[i]),
        )
        for i, feat in enumerate(features)
    }
    if return_scaler:
        return vlims, scaler, features
    return vlims


_SORT_LABELS = {
    'depth': 'transition depth',
    'ap': 'AP coordinate (posterior → anterior)',
    'ml': '|ML coordinate| (midline → lateral)',
    'rastermap': 'Rastermap',
}


def plot_boundary_feature_profiles(
    df_features: pd.DataFrame,
    brain_atlas,
    from_acr: str,
    to_acr: str,
    df_results: pd.DataFrame | None = None,
    mandatory_features: tuple[str, ...] = ('rms_ap', 'rms_lf', 'spike_count'),
    n_extra: int = 7,
    window_um: float | None = None,
    depth_bin_um: float = 20.0,
    max_probes: int = 150,
    feature_vlims: dict[str, tuple[float, float]] | None = None,
    sort: str = 'depth',
    feature_scaler=None,
    cache_dir: Path | None = None,
    sort_features: bool = True,
    crossings: list[tuple[str, float]] | None = None,
) -> plt.Figure | None:
    """Plot aligned feature depth profiles for all probes crossing a boundary.

    Layout: one imshow per row — anatomical colours on top, then one feature
    row per selected feature.  X-axis = probes sorted by transition depth,
    Y-axis = depth relative to boundary (negative = from_acr / deeper,
    positive = to_acr / shallower).

    Parameters
    ----------
    df_features:
        Channel features dataframe (MultiIndex with ``pid`` level).
    brain_atlas:
        ``ClassifierAtlas`` instance.
    from_acr:
        Acronym of the lower/deeper region.
    to_acr:
        Acronym of the upper/shallower region.
    df_results:
        Feature stats dataframe from the boundary scan (columns: ``from``,
        ``to``, ``feature``, ``cohens_d``, ``pval_bonf``). Used to pick the
        most informative extra features. If ``None``, only mandatory features
        are plotted.
    mandatory_features:
        Features always shown (first), regardless of statistics.
    n_extra:
        Number of extra features selected from ``df_results`` by Cohen's d.
    window_um:
        Half-window in µm shown above and below the boundary.  ``None``
        (default) auto-detects the full extent of each crossing probe so the
        entire recording is visible, aligned at the boundary.
    depth_bin_um:
        Depth bin size (µm).
    max_probes:
        Cap on the number of probe columns displayed.  When the crossing count
        exceeds this, probes are uniformly subsampled while preserving the
        depth-sorted order.
    feature_vlims:
        Pre-computed global display limits ``{feature: (vmin, vmax)}``, as
        returned by :func:`compute_feature_vlims`.  When provided these take
        precedence over per-boundary percentile limits, keeping the colour
        scale consistent across all figures.
    sort:
        Probe sort order along the X axis.  One of:

        * ``'depth'`` — transition depth (default, shallowest first).
        * ``'ap'``    — AP coordinate (y) at the transition channel.
        * ``'ml'``    — absolute ML coordinate (|x|) at the transition channel.
        * ``'rastermap'`` — Rastermap on the normalised concatenated feature
          matrix ``(n_bins × n_feats, n_probes)ᵀ``.
    feature_scaler:
        ``(RobustScaler, list[str])`` tuple — the fitted scaler and the ordered
        list of feature names it was fit on, as returned by
        ``compute_feature_vlims(..., return_scaler=True)``.  Required for the
        ``'rastermap'`` sort; ignored for others.
    cache_dir:
        Directory for rastermap ``.npy`` cache files.  ``None`` disables
        caching.
    sort_features:
        When ``True`` (default), features are reordered by hierarchical
        clustering of their mean depth profiles.  Set to ``False`` to preserve
        the order of *mandatory_features*, which is required when comparing
        measured and volume figures side-by-side.

    Returns
    -------
    fig:
        Matplotlib figure, or ``None`` if no crossings are found.
    """
    regions = brain_atlas.regions
    if crossings is None:
        crossings = find_boundary_crossings(df_features, brain_atlas, from_acr, to_acr)
    if not crossings:
        print(f"No crossings found for {from_acr} → {to_acr}")
        return None

    # Sort probes before (optional) subsampling
    if sort == 'ap':
        crossings = sorted(crossings, key=lambda t: _probe_coord(df_features, t[0], t[1], 'y'))
    elif sort == 'ml':
        crossings = sorted(crossings, key=lambda t: abs(_probe_coord(df_features, t[0], t[1], 'x')))
    else:  # 'depth' and 'rastermap' both pre-sort by depth for consistent subsampling
        crossings = sorted(crossings, key=lambda t: t[1])

    n_total = len(crossings)
    if n_total > max_probes:
        step = n_total / max_probes
        indices = [int(i * step) for i in range(max_probes)]
        crossings = [crossings[i] for i in indices]
        print(f"Subsampled {n_total} → {len(crossings)} probes for display")
    n_probes = len(crossings)

    # Mandatory features present in df_features + top extras from df_results
    present = set(df_features.columns)
    feature_list = [f for f in mandatory_features if f in present]
    if df_results is not None:
        mask_pair = (df_results['from'] == from_acr) & (df_results['to'] == to_acr)
        mandatory_set = set(mandatory_features)
        top_extra = (
            df_results[mask_pair]
            .sort_values('cohens_d', ascending=False)['feature']
            .tolist()
        )
        n_extras_added = 0
        for feat in top_extra:
            if feat not in mandatory_set and feat in present and n_extras_added < n_extra:
                feature_list.append(feat)
                n_extras_added += 1

    n_feats = len(feature_list)

    # Auto-detect full probe extent if window_um not given
    if window_um is None:
        lo, hi = np.inf, -np.inf
        for pid, td in crossings:
            try:
                depths = df_features.xs(pid, level='pid')['axial_um']
                lo = min(lo, float((depths - td).min()))
                hi = max(hi, float((depths - td).max()))
            except KeyError:
                pass
        window_um = max(abs(lo), abs(hi)) if np.isfinite(lo) else 500.0
        print(f"Auto window: [{-window_um:.0f}, +{window_um:.0f}] µm")

    # Depth grid: bin centres spanning −window_um … +window_um
    depth_centers = np.arange(-window_um + depth_bin_um / 2, window_um, depth_bin_um)
    n_bins = len(depth_centers)
    half = depth_bin_um / 2.0

    cosmos_refined = get_cosmos_refined(df_features, brain_atlas)

    histo_rgba = np.full((n_bins, n_probes, 4), np.nan)
    feat_data = np.full((n_bins, n_probes, n_feats), np.nan)

    for j, (pid, trans_depth) in enumerate(crossings):
        try:
            df_p = df_features.xs(pid, level='pid')
        except KeyError:
            continue
        depth_rel = df_p['axial_um'] - trans_depth
        cr_pid = cosmos_refined.xs(pid, level='pid')

        for i, d in enumerate(depth_centers):
            mask = (depth_rel >= d - half) & (depth_rel < d + half)
            if mask.sum() == 0:
                continue

            modal_id = int(cr_pid[mask].mode().iloc[0])
            histo_rgba[i, j, :] = _get_region_rgba(regions, modal_id)

            for k, feat in enumerate(feature_list):
                vals = df_p.loc[mask, feat].dropna()
                if len(vals) > 0:
                    feat_data[i, j, k] = float(vals.mean())

    # Colour limits: use global limits when provided, else fall back to per-boundary percentiles
    vlims = []
    cmaps = []
    for k, feat in enumerate(feature_list):
        if feature_vlims is not None and feat in feature_vlims:
            vlims.append(feature_vlims[feat])
        else:
            vlims.append(_percentile_vlims(feat_data[:, :, k]))
        cmaps.append(_feature_cmap(feat))

    # Optionally reorder features by hierarchical clustering of mean depth profiles.
    if sort_features and n_feats > 2:
        with np.errstate(all='ignore'):
            mean_profiles = np.nanmean(feat_data, axis=1).T  # (n_feats, n_bins)
        profiles_clean = np.where(np.isnan(mean_profiles), 0.0, mean_profiles)
        try:
            dist = pdist(profiles_clean, metric='correlation')
            dist = np.nan_to_num(dist, nan=1.0)
            order = list(leaves_list(linkage(dist, method='ward')))
            feature_list = [feature_list[i] for i in order]
            feat_data = feat_data[:, :, order]
            cmaps = [cmaps[i] for i in order]
            vlims = [vlims[i] for i in order]
        except Exception:
            pass

    # Rastermap probe sort — applied after feature clustering so feat_data columns are fixed
    if sort == 'rastermap':
        pids_used = [pid for pid, _ in crossings]
        isort = _compute_rastermap_isort(
            feat_data, feature_list, feature_scaler, cache_dir, from_acr, to_acr, pids_used
        )
        feat_data = feat_data[:, isort, :]
        histo_rgba = histo_rgba[:, isort, :]

    # --- Figure ---
    n_rows = 1 + n_feats
    row_heights = [1.5] + [1.0] * n_feats
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(max(n_probes * 0.22 + 1.5, 8), n_rows * 1.3 + 0.6),
        gridspec_kw={'height_ratios': row_heights},
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    extent = [-0.5, n_probes - 0.5, depth_centers[0] - half, depth_centers[-1] + half]

    # Histology row
    ax = axes[0]
    histo_plot = histo_rgba.copy()
    nan_mask = np.isnan(histo_plot[:, :, 0])
    histo_plot[nan_mask] = [0.75, 0.75, 0.75, 1.0]
    ax.imshow(histo_plot, origin='lower', extent=extent, aspect='auto', interpolation='nearest')
    ax.axhline(0, color='k', lw=1.2, ls='--', alpha=0.8)
    ax.set_ylabel('depth (µm)', fontsize=12)
    sort_tag = _SORT_LABELS.get(sort, sort)
    ax.set_title(f'{from_acr} → {to_acr}   ({n_probes} probes, {sort_tag})', fontsize=16)
    ax.set_xticks([])

    # Feature rows
    for k, (feat, cmap_name, (vmin, vmax)) in enumerate(zip(feature_list, cmaps, vlims)):
        ax = axes[k + 1]
        img = np.where(np.isfinite(feat_data[:, :, k]), feat_data[:, :, k], np.nan)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad('lightgrey')
        im = ax.imshow(
            img, origin='lower', extent=extent, aspect='auto',
            cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest',
        )
        ax.axhline(0, color='k', lw=1.0, ls='--', alpha=0.7)
        label = feat.replace('_', '\n')
        ax.set_ylabel(label, fontsize=11, rotation=0, labelpad=36, va='center')
        ax.set_xticks([])
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    axes[-1].set_xlabel(f'probe  (sorted by {sort_tag})', fontsize=12)

    return fig


def plot_landmark_line_profiles(
    df_features: pd.DataFrame,
    brain_atlas,
    from_acr: str,
    to_acr: str,
    df_results: pd.DataFrame | None = None,
    mandatory_features: tuple[str, ...] = ('rms_ap', 'rms_lf', 'spike_count'),
    n_extra: int = 5,
    window_um: float = 200.0,
    depth_bin_um: float = 25.0,
) -> plt.Figure | None:
    """Plot mean ± SEM feature depth profiles aligned to a region boundary.

    Reproduces the style of fig 3b in the repro-ephys paper: depth on the
    y-axis, anatomical colour strip on the far left, one feature column per
    selected feature showing the mean ± SEM across all crossing probes.

    Parameters
    ----------
    df_features:
        Channel features dataframe (MultiIndex with ``pid`` level).
    brain_atlas:
        ``ClassifierAtlas`` instance.
    from_acr:
        Acronym of the lower/deeper region.
    to_acr:
        Acronym of the upper/shallower region.
    df_results:
        Feature stats dataframe (columns: ``from``, ``to``, ``feature``,
        ``cohens_d``, ``pval_bonf``).  Selects top extra features by Cohen's d.
        If ``None``, only mandatory features are shown.
    mandatory_features:
        Features always shown first, regardless of statistics.
    n_extra:
        Number of extra features selected from ``df_results`` by Cohen's d.
    window_um:
        Half-window around the boundary (µm).
    depth_bin_um:
        Depth bin size (µm).

    Returns
    -------
    fig:
        Matplotlib figure, or ``None`` if no crossings were found.
    """
    regions = brain_atlas.regions
    crossings = find_boundary_crossings(df_features, brain_atlas, from_acr, to_acr)
    if not crossings:
        print(f"No crossings found for {from_acr} → {to_acr}")
        return None
    n_probes = len(crossings)

    # Feature list: mandatory first, then top extras by Cohen's d
    present = set(df_features.columns)
    feature_list = [f for f in mandatory_features if f in present]
    if df_results is not None:
        pair_mask = (df_results['from'] == from_acr) & (df_results['to'] == to_acr)
        mandatory_set = set(mandatory_features)
        n_added = 0
        for feat in df_results[pair_mask].sort_values('cohens_d', ascending=False)['feature']:
            if feat not in mandatory_set and feat in present and n_added < n_extra:
                feature_list.append(feat)
                n_added += 1
    if not feature_list:
        print(f"No features to plot for {from_acr} → {to_acr}")
        return None

    # Cohen's d lookup per feature
    d_vals: dict[str, float] = {}
    if df_results is not None:
        pair_mask = (df_results['from'] == from_acr) & (df_results['to'] == to_acr)
        for _, row in df_results[pair_mask].iterrows():
            d_vals[row['feature']] = row['cohens_d']

    # Region colours (RGB floats)
    cosmos_refined = get_cosmos_refined(df_features, brain_atlas)
    cosmos_ids = sorted(cosmos_refined.unique())
    id2acr_loc = {cid: regions.acronym[regions.id2index(cid)[1]][0][0] for cid in cosmos_ids}
    acr2id_loc = {v: k for k, v in id2acr_loc.items()}
    from_id = acr2id_loc.get(from_acr)
    to_id = acr2id_loc.get(to_acr)
    col_from = _get_region_rgba(regions, from_id)[:3] if from_id is not None else (0.65, 0.65, 0.65)
    col_to = _get_region_rgba(regions, to_id)[:3] if to_id is not None else (0.65, 0.65, 0.65)

    def _txt(rgb: tuple) -> str:
        r, g, b = rgb
        return 'white' if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else 'black'

    # Merge all crossing events with channel feature data
    crossing_df = pd.DataFrame(crossings, columns=['pid', 'trans_depth'])
    df_feat_reset = df_features[['axial_um'] + feature_list].reset_index(level='pid')
    merged = crossing_df.merge(df_feat_reset, on='pid')
    merged['rel_depth'] = merged['axial_um'] - merged['trans_depth']
    in_window = merged[merged['rel_depth'].abs() <= window_um].copy()

    # Bin by relative depth
    bin_edges = np.arange(-window_um, window_um + depth_bin_um, depth_bin_um)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    in_window['bin'] = pd.cut(in_window['rel_depth'], bins=bin_edges, labels=bin_centers)
    in_window['bin'] = in_window['bin'].astype(float)
    binned = in_window.groupby('bin')[feature_list].agg(['mean', 'sem'])

    n_feats = len(feature_list)
    fig, axes = plt.subplots(
        1, n_feats + 1,
        figsize=(1.4 + 2.2 * n_feats, 7),
        sharey=True,
        gridspec_kw={'width_ratios': [0.28] + [1.0] * n_feats, 'wspace': 0.06},
        constrained_layout=True,
    )
    fig.suptitle(f'{from_acr}  →  {to_acr}    ({n_probes} probes)', fontsize=11)

    # Anatomical colour strip
    ax0 = axes[0]
    ax0.fill_betweenx([-window_um, 0], 0, 1, color=col_from, alpha=0.9)
    ax0.fill_betweenx([0, window_um], 0, 1, color=col_to, alpha=0.9)
    ax0.axhline(0, color='k', lw=1.2)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(-window_um, window_um)
    ax0.set_xticks([])
    ax0.set_ylabel('depth from boundary (µm)', fontsize=8, labelpad=6)
    ax0.text(0.5, -window_um * 0.5, from_acr, ha='center', va='center',
             fontsize=9, color=_txt(col_from), fontweight='bold', rotation=90)
    ax0.text(0.5, window_um * 0.5, to_acr, ha='center', va='center',
             fontsize=9, color=_txt(col_to), fontweight='bold', rotation=90)
    sns.despine(ax=ax0, bottom=True, left=True, right=True, top=True)

    # Feature line-plot columns
    for ax, feat in zip(axes[1:], feature_list):
        feat_mean = binned[feat]['mean']
        feat_sem = binned[feat]['sem'].fillna(0.0)
        bins_arr = feat_mean.index.values.astype(float)
        mean_arr = feat_mean.values
        sem_arr = feat_sem.values

        for mask, color in [(bins_arr <= 0, col_from), (bins_arr > 0, col_to)]:
            bx = bins_arr[mask]
            mx = mean_arr[mask]
            sx = sem_arr[mask]
            finite = np.isfinite(mx)
            if not finite.any():
                continue
            ax.plot(mx[finite], bx[finite], color=color, lw=2)
            ax.fill_betweenx(bx[finite], mx[finite] - sx[finite], mx[finite] + sx[finite],
                             alpha=0.35, color=color)

        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.65)
        d_label = f'd={d_vals[feat]:.2f}' if feat in d_vals else ''
        ax.set_title(d_label, fontsize=8, pad=3)
        ax.set_xlabel(feat.replace('_', '\n'), fontsize=8)
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        sns.despine(ax=ax, left=True)

    return fig


def plot_transition_graph(
    count_matrix: pd.DataFrame,
    brain_atlas,
    min_count: int = 50,
    vintage: str = '',
    output_fig_path: Path | None = None,
) -> plt.Figure | None:
    """Visualise high-count region transitions as a directed graph.

    Nodes are Cosmos brain regions; a directed edge connects pairs whose
    transition count is ≥ *min_count*.  Node colour comes from the Allen
    atlas.  Edge width and label show the transition count.

    Parameters
    ----------
    count_matrix:
        Square transition matrix as returned by
        :func:`compute_cosmos_transitions`. Rows = from (lower/deeper region),
        columns = to (upper/shallower region).
    brain_atlas:
        ``ClassifierAtlas`` instance with a ``.regions`` attribute.
    min_count:
        Minimum transition count to include an edge. Default 50.
    vintage:
        Dataset vintage label shown in the title.
    output_fig_path:
        Directory to save ``transition_graph_min<N>.png``. No file is written
        if ``None``.

    Returns
    -------
    fig:
        Matplotlib figure, or ``None`` if no edges exceed the threshold.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required: pip install networkx")

    regions = brain_atlas.regions
    acr_flat = np.asarray(regions.acronym).flatten()

    def _acr_to_rgba(acr: str) -> tuple[float, float, float, float]:
        """Return RGBA for an acronym, falling back to grey on miss."""
        matches = np.where(acr_flat == acr)[0]
        if len(matches) == 0:
            return (0.65, 0.65, 0.65, 1.0)
        rid = int(np.asarray(regions.id).flatten()[matches[0]])
        return _get_region_rgba(regions, rid)

    # Collect directed edges above threshold (exclude diagonal)
    edges = [
        (row, col, int(count_matrix.loc[row, col]))
        for row in count_matrix.index
        for col in count_matrix.columns
        if row != col and int(count_matrix.loc[row, col]) >= min_count
    ]
    if not edges:
        print(f"No transitions ≥ {min_count} found.")
        return None

    print(f"{len(edges)} edges ≥ {min_count}:")
    for fr, to, cnt in sorted(edges, key=lambda e: -e[2]):
        print(f"  {fr} → {to}: {cnt}")

    G = nx.DiGraph()
    G.add_nodes_from(sorted({n for e in edges for n in e[:2]}))
    for fr, to, cnt in edges:
        G.add_edge(fr, to, weight=cnt)

    node_colors = [_acr_to_rgba(n) for n in G.nodes()]

    counts = [G[u][v]['weight'] for u, v in G.edges()]
    c_min, c_max = min(counts), max(counts)
    def _width(c):
        return 1.5 if c_max == c_min else 1.5 + 6.5 * (c - c_min) / (c_max - c_min)

    edge_widths = [_width(c) for c in counts]
    edge_labels = {(u, v): str(G[u][v]['weight']) for u, v in G.edges()}

    pos = nx.spring_layout(G, seed=42, k=2.5)

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=2800, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color='#333333', arrows=True, arrowsize=22,
                           connectionstyle='arc3,rad=0.08',
                           min_source_margin=32, min_target_margin=32)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=8, bbox=dict(boxstyle='round,pad=0.2',
                                 fc='white', ec='none', alpha=0.75))
    title = f'Region transitions  (n ≥ {min_count})'
    if vintage:
        title += f'  [{vintage}]'
    ax.set_title(title, fontsize=11, pad=12)
    ax.axis('off')
    fig.tight_layout()

    if output_fig_path is not None:
        out = Path(output_fig_path)
        out.mkdir(exist_ok=True)
        out_file = out.joinpath(f'transition_graph_min{min_count}.png')
        fig.savefig(out_file, dpi=150)
        print(f"Saved {out_file}")

    return fig


def plot_transition_matrix(
    count_matrix: pd.DataFrame,
    vintage: str,
    output_fig_path: Path,
) -> None:
    """Save a Cosmos boundary transition count heatmap.

    Parameters
    ----------
    count_matrix:
        Square DataFrame as returned by :func:`compute_cosmos_transitions`.
    vintage:
        Dataset vintage label shown in the figure title.
    output_fig_path:
        Directory where ``cosmos_transition_matrix.png`` is written.
    """
    plot_matrix = count_matrix.astype(float).values.copy()
    np.fill_diagonal(plot_matrix, np.nan)
    plot_matrix = pd.DataFrame(plot_matrix, index=count_matrix.index, columns=count_matrix.columns)

    off_diag = plot_matrix.values[~np.isnan(plot_matrix.values)]
    vmax = np.percentile(off_diag, 97)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color='#cccccc')

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        plot_matrix,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        linewidths=0.4,
        linecolor='white',
        annot=True,
        fmt='.0f',
        annot_kws={'size': 7},
        cbar_kws={'label': 'transition count', 'shrink': 0.8},
    )
    ax.set_title(f'Cosmos boundary transitions ({vintage})', pad=12)
    ax.set_xlabel('upper region')
    ax.set_ylabel('lower region')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    output_fig_path = Path(output_fig_path)
    output_fig_path.mkdir(exist_ok=True)
    fig.savefig(output_fig_path.joinpath('cosmos_transition_matrix.png'), dpi=150)
    plt.close()
    print("Saved cosmos_transition_matrix.png")