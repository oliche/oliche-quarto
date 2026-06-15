"""Utility functions for ephys-atlas region boundary landmark analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_DEFAULT_ROOT = Path.home().joinpath('data', 'ephys-atlas', 'features')
_ALYX_URL = 'https://alyx.internationalbrainlab.org'

EXCLUDE_COLS = frozenset({
    'axial_um', 'lateral_um', 'x', 'y', 'z', 'x_target', 'y_target', 'z_target',
    'acronym', 'atlas_id', 'Allen_id', 'Cosmos_id', 'Beryl_id', 'outside',
    'channel_labels',
})


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


def compute_pca_features(
    df_features: pd.DataFrame,
    n_pcs: int = 6,
    exclude_cols: frozenset | None = None,
) -> tuple[pd.DataFrame, PCA, StandardScaler, list[str]]:
    """Fit a PCA on normalised ephys-atlas channel features.

    Parameters
    ----------
    df_features:
        Full channel features dataframe as returned by
        ``ephysatlas.data.read_features_from_disk``.
    n_pcs:
        Number of principal components to retain in the output dataframe.
    exclude_cols:
        Columns to treat as metadata (excluded from PCA). Defaults to
        ``EXCLUDE_COLS``.

    Returns
    -------
    df_pca:
        One row per channel (NaN-dropped subset). Contains the metadata columns
        that were present in *df_features* plus ``PC1`` … ``PC<n_pcs>``.
    pca:
        Fitted ``sklearn.decomposition.PCA`` object (full, not truncated).
    scaler:
        Fitted ``sklearn.preprocessing.StandardScaler``.
    feature_cols:
        Ordered list of feature column names used as PCA input.
    """
    if exclude_cols is None:
        exclude_cols = EXCLUDE_COLS

    feature_cols = [c for c in df_features.columns if c not in exclude_cols]

    X_raw = df_features[feature_cols].dropna()
    n_dropped = len(df_features) - len(X_raw)
    print(f"PCA input: {len(X_raw):,} channels ({n_dropped:,} dropped for NaNs), {len(feature_cols)} features")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    pca = PCA()
    pca.fit(X)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    print(f"Variance captured by first {n_pcs} PCs: {cumulative[n_pcs - 1]:.1%}")

    pc_cols = [f'PC{i + 1}' for i in range(n_pcs)]
    meta_cols = [c for c in exclude_cols if c in df_features.columns]
    df_pca = df_features.loc[X_raw.index, meta_cols].copy()
    df_pca[pc_cols] = pca.transform(X)[:, :n_pcs]

    return df_pca, pca, scaler, feature_cols


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


def find_boundary_crossings(
    df_features: pd.DataFrame,
    brain_atlas,
    from_acr: str,
    to_acr: str,
) -> list[tuple[str, float]]:
    """Find (pid, transition_depth_um) for each from_acr → to_acr crossing.

    Channels are depth-aggregated within each probe (modal Cosmos_refined), then
    adjacent-depth transitions are detected. Only the first crossing per probe is
    returned.

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


def plot_boundary_pca_profiles(
    df_pca: pd.DataFrame,
    df_features: pd.DataFrame,
    brain_atlas,
    from_acr: str,
    to_acr: str,
    n_pcs: int = 6,
    window_um: float = 200.0,
    depth_bin_um: float = 20.0,
) -> plt.Figure | None:
    """Plot aligned PC depth profiles for all probes crossing a boundary.

    Layout: 7 rows (histology + PC1–PC6), single shared imshow per row.
    X-axis = probes, Y-axis = depth relative to transition (negative = from_acr,
    positive = to_acr). Probes are sorted by transition depth.

    Parameters
    ----------
    df_pca:
        Dataframe with PC scores and metadata (as returned by
        :func:`compute_pca_features`). Must include ``axial_um`` and
        ``Cosmos_id`` columns plus ``PC1`` … ``PC<n_pcs>``.
    df_features:
        Full channel features dataframe (used for ``Allen_id`` to compute
        Cosmos_refined histology colours).
    brain_atlas:
        ``ClassifierAtlas`` instance.
    from_acr:
        Acronym of the lower/deeper region.
    to_acr:
        Acronym of the upper/shallower region.
    n_pcs:
        Number of PCs to display (rows 1 … n_pcs of the figure).
    window_um:
        Half-window size around the transition (µm).
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

    # Sort probes by transition depth (shallowest first)
    crossings = sorted(crossings, key=lambda t: t[1])
    n_probes = len(crossings)

    # Depth grid: centres every depth_bin_um from -window_um to +window_um
    depth_centers = np.arange(-window_um, window_um + 1, depth_bin_um)
    n_bins = len(depth_centers)
    half = depth_bin_um / 2.0

    # Precompute Cosmos_refined for histology colouring
    cosmos_refined = get_cosmos_refined(df_features, brain_atlas)

    pc_cols = [f'PC{i + 1}' for i in range(n_pcs)]

    # Allocate data arrays
    histo_rgba = np.full((n_bins, n_probes, 4), np.nan)
    pc_data = np.full((n_bins, n_probes, n_pcs), np.nan)

    for j, (pid, trans_depth) in enumerate(crossings):
        # PC data for this probe (subset of df_pca)
        try:
            df_p = df_pca.xs(pid, level='pid')
        except KeyError:
            continue
        depth_rel_p = df_p['axial_um'] - trans_depth

        # Feature data for histology (Cosmos_refined colour)
        try:
            df_f = df_features.xs(pid, level='pid')
        except KeyError:
            df_f = None
        cr_pid = cosmos_refined.xs(pid, level='pid') if df_f is not None else None

        for i, d in enumerate(depth_centers):
            mask_p = (depth_rel_p >= d - half) & (depth_rel_p < d + half)
            if mask_p.sum() > 0:
                for k, pc in enumerate(pc_cols):
                    if pc in df_p.columns:
                        pc_data[i, j, k] = df_p.loc[mask_p, pc].mean()

                if cr_pid is not None:
                    depth_rel_f = df_f['axial_um'] - trans_depth
                    mask_f = (depth_rel_f >= d - half) & (depth_rel_f < d + half)
                    if mask_f.sum() > 0:
                        modal_id = int(cr_pid[mask_f].mode()[0])
                        idx_r = regions.id2index(modal_id)[1]
                        rgb = regions.rgb[idx_r][0, 0] / 255.0
                        histo_rgba[i, j, :3] = rgb
                        histo_rgba[i, j, 3] = 1.0

    # Compute symmetric vmin/vmax per PC using IQR-based clipping (robust to outliers)
    vlims = []
    for k in range(n_pcs):
        vals = pc_data[:, :, k]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            vlims.append((-1, 1))
            continue
        q25, q75 = np.percentile(finite, [25, 75])
        iqr = q75 - q25
        vabs = max(abs(q25 - 1.5 * iqr), abs(q75 + 1.5 * iqr))
        vlims.append((-vabs, vabs))

    # --- Figure ---
    n_rows = 1 + n_pcs  # histology + PCs
    row_heights = [1.5] + [1.0] * n_pcs
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(max(n_probes * 0.18 + 1.5, 8), n_rows * 1.2 + 0.8),
        gridspec_kw={'height_ratios': row_heights},
    )

    extent = [-0.5, n_probes - 0.5, depth_centers[0] - half, depth_centers[-1] + half]

    # Histology row (index 0 = bottom = from_acr, index n_bins-1 = top = to_acr)
    ax = axes[0]
    # Replace NaN in RGBA alpha channel with 0 (transparent → show as gray background)
    histo_plot = histo_rgba.copy()
    nan_mask = np.isnan(histo_plot[:, :, 0])
    histo_plot[nan_mask] = [0.75, 0.75, 0.75, 1.0]
    ax.imshow(histo_plot, origin='lower', extent=extent, aspect='auto', interpolation='nearest')
    ax.axhline(0, color='k', lw=1.2, ls='--', alpha=0.8)
    ax.set_ylabel('depth (µm)', fontsize=8)
    ax.set_title(f'{from_acr} → {to_acr}   ({n_probes} probes)', fontsize=10, pad=6)
    ax.set_xlabel('')
    ax.set_xticklabels([])

    # PC rows
    cmap = plt.get_cmap('RdBu_r')
    cmap_miss = cmap.copy()
    cmap_miss.set_bad('lightgrey')

    for k in range(n_pcs):
        ax = axes[k + 1]
        img = np.where(np.isfinite(pc_data[:, :, k]), pc_data[:, :, k], np.nan)
        vmin, vmax = vlims[k]
        im = ax.imshow(
            img, origin='lower', extent=extent, aspect='auto',
            cmap=cmap_miss, vmin=vmin, vmax=vmax, interpolation='nearest',
        )
        ax.axhline(0, color='k', lw=1.0, ls='--', alpha=0.7)
        ax.set_ylabel(f'PC{k + 1}', fontsize=8, rotation=0, labelpad=24)
        ax.set_xticklabels([])
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    axes[-1].set_xlabel('probe (sorted by transition depth)', fontsize=8)
    axes[-1].set_xticklabels([])

    fig.tight_layout(h_pad=0.3)
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
    print(f"Saved cosmos_transition_matrix.png")