"""Boundary classifier trained on encoding-volume virtual probes.

Generates a regular (AP x ML) grid of synthetic near-vertical probes spanning
the full brain, samples the brainwide encoding volume along each DV column,
detects Cosmos-level region transitions, and trains a gradient-boosted
classifier to identify boundary type from a (mean_below - mean_above) feature
difference vector.

No measured Neuropixel data is used -- training and evaluation are entirely
driven by the encoding volume and the Allen atlas.

Coordinate conventions
----------------------
- ``bc.xyz2i(xyz)`` takes (ML, AP, DV) metres and returns (iML, iAP, iDV).
- Encoding volume ``vol[iML, iAP, iDV, :]`` -- same axis order as bc output.
- Atlas label ``label[iAP, iML, iDV]`` -- AP-first; handled by ``get_labels()``.
- DV column goes from most ventral (axial_um=0) to most dorsal (axial_um=max),
  matching the real-probe convention (tip=deep=low axial_um, surface=high).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
import boundaries_utils as bu

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VINTAGE = '2026_W24'
LOCAL_CACHE_DIR = Path(__file__).parent.joinpath('cache')
FIGURES_DIR = Path(__file__).parent.joinpath('figures')
VOL_PATH = Path.home().joinpath(
    'data', 'ephys-atlas', 'encoding_volumes', 'brainwide_ephys_atlas_25um.npz'
)
RAW_FEATURES_PATH = Path.home().joinpath(
    'data', 'ephys-atlas', 'features', 'ea_active', VINTAGE, 'agg_full'
)

GRID_SPACING_UM = 200.0    # AP and ML grid spacing
WINDOW_UM = 1000.0         # half-window around each boundary crossing
MIN_INSERTIONS = 100       # minimum virtual-probe crossings to qualify a pair
N_FOLDS = 4
N_PERMUTATIONS = 100
RANDOM_STATE = 42

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 20, 'axes.labelsize': 18,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 15,
    'figure.titlesize': 22,
})


# ---------------------------------------------------------------------------
# Step 1 — build virtual probe dataframe
# ---------------------------------------------------------------------------

def build_virtual_probe_df(
    brain_atlas,
    vol_data,
    psd_pca,
    grid_spacing_um: float = GRID_SPACING_UM,
) -> pd.DataFrame:
    """Build a synthetic feature dataframe from a regular virtual probe grid.

    Generates an (AP x ML) grid of near-vertical probes. Each probe is a DV
    column sampled at 25 um resolution (one atlas voxel per step). The
    encoding volume is looked up at every (ML, AP, DV) voxel; PSD/CSD features
    are then projected through the same EphysPsdPCA used for measured data.

    The DV column is ordered ventral-to-dorsal so axial_um=0 is the deepest
    point, matching the real-probe convention (tip=deep).  Inside-brain voxels
    are kept (Cosmos_id != 0); outside voxels are discarded.

    Parameters
    ----------
    brain_atlas:
        ClassifierAtlas instance (25 um resolution).
    vol_data:
        Loaded encoding volume npz (keys: 'ephys_atlas_vol', 'feature_names').
    psd_pca:
        Fitted EphysPsdPCA instance (same as used for measured data).
    grid_spacing_um:
        AP and ML spacing in micrometres.

    Returns
    -------
    df:
        MultiIndex (pid x channel) dataframe with transformed features plus
        axial_um, Cosmos_id, Allen_id, x, y, z columns.
    """
    vol = vol_data['ephys_atlas_vol']           # (456, 528, 320, 41) float16
    feat_names = list(vol_data['feature_names'])
    bc = brain_atlas.bc

    step = grid_spacing_um * 1e-6
    ml_vals = np.arange(bc.xlim[0], bc.xlim[1], step)
    ap_vals = np.arange(bc.ylim[0], bc.ylim[1], -step)  # anterior → posterior

    # DV: ventral to dorsal so axial_um=0 is at the deepest point
    dv_step = abs(bc.dxyz[2])                  # +25e-6 m
    dv_vals = np.arange(bc.zlim[1], bc.zlim[0] + dv_step, dv_step)
    n_dv = len(dv_vals)

    n_ml, n_ap = len(ml_vals), len(ap_vals)
    n_probes = n_ml * n_ap
    print(f'Grid: {n_ml} ML x {n_ap} AP = {n_probes:,} virtual probes, {n_dv} DV steps each')
    print(f'Total voxels before brain mask: {n_probes * n_dv:,}')

    ml_grid, ap_grid = np.meshgrid(ml_vals, ap_vals)   # (n_ap, n_ml)
    ml_flat = ml_grid.flatten()
    ap_flat = ap_grid.flatten()

    # All (ML, AP, DV) coordinates at once: shape (n_probes * n_dv, 3)
    n_total = n_probes * n_dv
    xyz_all = np.empty((n_total, 3), dtype=np.float64)
    xyz_all[:, 0] = np.repeat(ml_flat, n_dv)
    xyz_all[:, 1] = np.repeat(ap_flat, n_dv)
    xyz_all[:, 2] = np.tile(dv_vals, n_probes)

    print('Looking up encoding volume ...')
    vox = bc.xyz2i(xyz_all, mode='clip')                # (n_total, 3): (iML, iAP, iDV)
    feats_raw = vol[vox[:, 0], vox[:, 1], vox[:, 2], :].astype(np.float32)

    print('Looking up atlas regions ...')
    cosmos_ids = brain_atlas.get_labels(xyz_all, mapping='Cosmos', mode='clip')
    allen_ids = brain_atlas.get_labels(xyz_all, mode='clip')

    # axial_um: 0 at ventral floor, increases toward dorsal surface
    axial_um_col = np.tile(np.arange(n_dv) * (dv_step * 1e6), n_probes)
    probe_id_col = np.repeat([f'vp_{i:05d}' for i in range(n_probes)], n_dv)
    channel_col = np.tile(np.arange(n_dv), n_probes)

    # Keep only inside-brain voxels
    inside = cosmos_ids != 0
    print(f'Inside-brain voxels: {inside.sum():,} / {n_total:,}')

    df_raw = pd.DataFrame(feats_raw[inside], columns=feat_names)
    df_raw['axial_um'] = axial_um_col[inside]
    df_raw['Cosmos_id'] = cosmos_ids[inside].astype(int)
    df_raw['Allen_id'] = allen_ids[inside].astype(int)
    df_raw['x'] = xyz_all[inside, 0]
    df_raw['y'] = xyz_all[inside, 1]
    df_raw['z'] = xyz_all[inside, 2]
    df_raw.index = pd.MultiIndex.from_arrays(
        [probe_id_col[inside], channel_col[inside]], names=['pid', 'channel']
    )

    print('Applying PSD/CSD PCA ...')
    df = psd_pca.transform(df_raw)
    print(f'Virtual probe df: {len(df):,} channels, {df.shape[1]} columns')
    return df


# ---------------------------------------------------------------------------
# Step 2 — detect all crossings in one pass
# ---------------------------------------------------------------------------

def precompute_crossings(
    df: pd.DataFrame,
    brain_atlas,
) -> tuple[dict, dict]:
    """Detect all Cosmos boundary crossings across all virtual probes.

    For virtual probes each (pid, axial_um) pair is unique (one voxel per
    depth step), so the groupby-aggregation used for real probes is replaced by
    a direct sort.  NNI is applied once to handle interior void/root/VS gaps.

    Parameters
    ----------
    df:
        Virtual probe feature dataframe (MultiIndex pid x channel).
    brain_atlas:
        ClassifierAtlas instance.

    Returns
    -------
    crossings_dict:
        ``{(from_acr, to_acr): [(pid, trans_depth_um), ...]}``
    pair_counts:
        ``{(from_acr, to_acr): n_crossings}``
    """
    regions = brain_atlas.regions
    cosmos_refined = bu.get_cosmos_refined(df, brain_atlas)
    cosmos_ids_uniq = sorted(cosmos_refined.unique())
    id2acr = {
        cid: regions.acronym[regions.id2index(cid)[1]][0][0]
        for cid in cosmos_ids_uniq
    }

    df_work = pd.DataFrame(
        {'axial_um': df['axial_um'], 'cr': cosmos_refined}
    ).reset_index(level='pid')

    # Virtual probes: each channel has a unique axial_um per probe — skip groupby
    df_depth = df_work.sort_values(['pid', 'axial_um']).reset_index(drop=True)
    df_depth = bu.apply_depth_nni(df_depth, cr_col='cr')

    next_cr = df_depth.groupby('pid')['cr'].shift(-1)
    trans_mask = next_cr.notna() & (next_cr != df_depth['cr'])
    df_trans = df_depth[trans_mask].copy()
    df_trans['to_cr'] = next_cr[trans_mask].astype(int)

    crossings_dict: dict[tuple, list] = {}
    pair_counts: dict[tuple, int] = {}
    for row in df_trans.itertuples():
        from_c, to_c = int(row.cr), int(row.to_cr)
        if from_c not in id2acr or to_c not in id2acr:
            continue
        key = (id2acr[from_c], id2acr[to_c])
        crossings_dict.setdefault(key, []).append((row.pid, float(row.axial_um)))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    return crossings_dict, pair_counts


def get_qualifying_pairs(
    pair_counts: dict,
    min_insertions: int = MIN_INSERTIONS,
) -> list[tuple[str, str]]:
    """Return pairs sorted by count (descending) above the threshold.

    Parameters
    ----------
    pair_counts:
        Output of :func:`precompute_crossings`.
    min_insertions:
        Minimum number of crossings to qualify.

    Returns
    -------
    pairs:
        List of ``(from_acr, to_acr)`` tuples.
    """
    qualifying = sorted(
        [(k, v) for k, v in pair_counts.items() if v >= min_insertions],
        key=lambda x: -x[1],
    )
    print(f'\nQualifying transitions (>= {min_insertions} crossings):')
    for (fr, to), cnt in qualifying:
        print(f'  {fr} -> {to}: {cnt}')
    return [k for k, _ in qualifying]


# ---------------------------------------------------------------------------
# Step 3 — feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    crossings_dict: dict,
    window_um: float = WINDOW_UM,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build classifier input from precomputed virtual-probe crossings.

    For each (probe, boundary) crossing computes a (mean_below - mean_above)
    difference vector over the +-window_um window around the transition depth.
    "Below" is the from-region side (lower axial_um = more ventral/deeper);
    "above" is the to-region side (higher axial_um = more dorsal/shallower).

    Parameters
    ----------
    df:
        Virtual probe feature dataframe (MultiIndex pid x channel).
    pairs:
        Qualifying (from_acr, to_acr) pairs.
    crossings_dict:
        Output of :func:`precompute_crossings`.
    window_um:
        Half-window around each transition depth (um).

    Returns
    -------
    X, y, feature_cols, class_names
    """
    feature_cols = [c for c in df.columns if c not in bu.EXCLUDE_COLS]

    # Pre-cache per-probe numpy arrays for fast window lookups
    print('Pre-caching per-probe feature arrays ...')
    all_pids = df.index.get_level_values('pid').unique()
    probe_cache: dict[str, dict] = {}
    df_feat = df[['axial_um'] + feature_cols]
    for pid in all_pids:
        sub = df_feat.xs(pid, level='pid')
        probe_cache[pid] = {
            'axial_um': sub['axial_um'].values,
            'features': sub[feature_cols].values,   # (n_channels, n_feats)
        }

    rows: list[np.ndarray] = []
    labels: list[str] = []

    for from_acr, to_acr in pairs:
        label = f'{from_acr}_to_{to_acr}'
        crossings = crossings_dict.get((from_acr, to_acr), [])
        print(f'  {label}: {len(crossings)} crossings')

        for pid, trans_depth in crossings:
            p = probe_cache.get(pid)
            if p is None:
                continue
            axial = p['axial_um']
            feats = p['features']

            in_window = (axial >= trans_depth - window_um) & (axial <= trans_depth + window_um)
            below_mask = in_window & (axial <= trans_depth)
            above_mask = in_window & (axial >= trans_depth)
            if below_mask.sum() == 0 or above_mask.sum() == 0:
                continue

            diff = (
                np.nanmean(feats[below_mask], axis=0)
                - np.nanmean(feats[above_mask], axis=0)
            )
            rows.append(np.nan_to_num(diff))
            labels.append(label)

    X = np.array(rows)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X, y, feature_cols, list(le.classes_)


# ---------------------------------------------------------------------------
# Step 4 — training and evaluation
# ---------------------------------------------------------------------------

def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Train a gradient-boosted classifier with stratified 4-fold CV.

    ``class_weight='balanced'`` compensates for unequal crossing counts
    across boundary pairs.  The model is also refit on the full dataset.

    Parameters
    ----------
    X:
        Feature matrix ``(n_samples, n_features)``.
    y:
        Integer class labels.

    Returns
    -------
    dict with model, y_pred, overall_acc, balanced_acc, per_class_acc.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model = HistGradientBoostingClassifier(
        max_iter=300, random_state=RANDOM_STATE, class_weight='balanced'
    )
    y_pred = np.empty_like(y)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
        fold_acc = accuracy_score(y[test_idx], y_pred[test_idx])
        fold_bal = balanced_accuracy_score(y[test_idx], y_pred[test_idx])
        print(f'  Fold {fold + 1}: acc={fold_acc:.3f}  bal_acc={fold_bal:.3f}')

    model.fit(X, y)
    n_classes = len(np.unique(y))
    per_class = {
        int(i): float(accuracy_score(y[y == i], y_pred[y == i]))
        for i in range(n_classes)
    }
    return {
        'model': model,
        'y_pred': y_pred,
        'overall_acc': float(accuracy_score(y, y_pred)),
        'balanced_acc': float(balanced_accuracy_score(y, y_pred)),
        'per_class_acc': per_class,
    }


def compute_chance_level(
    y: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    """Estimate chance-level accuracy by permuting true labels.

    Parameters
    ----------
    y:
        True class labels (held fixed).
    y_pred:
        CV predictions (held fixed).
    n_permutations:
        Number of label shuffles.

    Returns
    -------
    dict with observed, balanced_observed, null_scores, null_mean, null_std, pvalue.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    null_scores = np.array([
        accuracy_score(rng.permutation(y), y_pred) for _ in range(n_permutations)
    ])
    observed = float(accuracy_score(y, y_pred))
    pvalue = float((null_scores >= observed).sum() + 1) / (n_permutations + 1)
    return {
        'observed': observed,
        'balanced_observed': float(balanced_accuracy_score(y, y_pred)),
        'null_scores': null_scores,
        'null_mean': float(null_scores.mean()),
        'null_std': float(null_scores.std()),
        'pvalue': pvalue,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _short(class_names: list[str]) -> list[str]:
    return [n.replace('_to_', '→') for n in class_names]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    acc: float,
    bal_acc: float,
) -> None:
    """Save a row-normalised confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred:
        True and predicted integer labels.
    class_names:
        Class label strings.
    output_dir:
        Destination directory.
    acc:
        Overall CV accuracy (shown in title).
    bal_acc:
        Balanced CV accuracy (shown in title).
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    n = len(class_names)
    labels = _short(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.65 + 2), max(7, n * 0.55 + 2)))
    sns.heatmap(
        cm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax, vmin=0, vmax=1, annot_kws={'size': 11},
    )
    ax.set_xlabel('predicted', fontsize=14)
    ax.set_ylabel('true', fontsize=14)
    ax.set_title(
        f'Volume classifier — GB  (acc={acc:.3f}, bal_acc={bal_acc:.3f})',
        fontsize=15,
    )
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', rotation=0, labelsize=11)
    fig.tight_layout()
    out = output_dir.joinpath('classifier_confusion_vol_gb.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out.name}')


def plot_null_distribution(null: dict, output_dir: Path) -> None:
    """Plot observed vs null accuracy distribution.

    Parameters
    ----------
    null:
        Output of :func:`compute_chance_level`.
    output_dir:
        Destination directory.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null['null_scores'], bins=25, color='steelblue', alpha=0.7,
            label=f'null  mean={null["null_mean"]:.3f}')
    ax.axvline(null['observed'], color='crimson', lw=2,
               label=f'observed={null["observed"]:.3f}  p={null["pvalue"]:.4f}')
    ax.axvline(null['balanced_observed'], color='darkorange', lw=2, ls='--',
               label=f'balanced={null["balanced_observed"]:.3f}')
    ax.set_xlabel('accuracy')
    ax.set_ylabel('count')
    ax.set_title('Volume classifier — permutation test')
    ax.legend(fontsize=11)
    fig.tight_layout()
    out = output_dir.joinpath('classifier_null_vol.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out.name}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full volume-based boundary classifier pipeline."""
    FIGURES_DIR.mkdir(exist_ok=True)
    LOCAL_CACHE_DIR.mkdir(exist_ok=True)

    print('=== Loading atlas and PSD/CSD PCA ===')
    import ephysatlas.anatomy
    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    psd_pca = bu.load_or_fit_psd_pca(RAW_FEATURES_PATH, LOCAL_CACHE_DIR, VINTAGE, brain_atlas)

    print('\n=== Loading encoding volume ===')
    vol_data = np.load(VOL_PATH, allow_pickle=True)
    print(f'  Shape: {vol_data["ephys_atlas_vol"].shape}')

    # --- virtual probe dataframe (cached) ---
    vp_cache = LOCAL_CACHE_DIR.joinpath(f'vp_df_{GRID_SPACING_UM:.0f}um.parquet')
    if vp_cache.exists():
        print(f'\nLoading cached virtual probe df from {vp_cache.name}')
        df = pd.read_parquet(vp_cache)
    else:
        print(f'\n=== Building virtual probe dataframe ({GRID_SPACING_UM:.0f} um grid) ===')
        df = build_virtual_probe_df(brain_atlas, vol_data, psd_pca, GRID_SPACING_UM)
        df.to_parquet(vp_cache)
        print(f'  Cached -> {vp_cache.name}')
    print(f'Virtual probe df: {len(df):,} channels, {df.shape[1]} columns')

    # --- crossing detection (cached) ---
    cross_cache = LOCAL_CACHE_DIR.joinpath(f'vp_crossings_{GRID_SPACING_UM:.0f}um.pkl')
    if cross_cache.exists():
        print(f'\nLoading cached crossings from {cross_cache.name}')
        with cross_cache.open('rb') as fh:
            crossings_dict, pair_counts = pickle.load(fh)
    else:
        print('\n=== Detecting boundary crossings ===')
        crossings_dict, pair_counts = precompute_crossings(df, brain_atlas)
        with cross_cache.open('wb') as fh:
            pickle.dump((crossings_dict, pair_counts), fh)
        print(f'  Cached -> {cross_cache.name}')

    pairs = get_qualifying_pairs(pair_counts, MIN_INSERTIONS)
    print(f'Total qualifying pairs: {len(pairs)}')

    # --- feature matrix (cached) ---
    _key = f'{GRID_SPACING_UM:.0f}um_{WINDOW_UM:.0f}w_{MIN_INSERTIONS}ins'
    feat_cache = LOCAL_CACHE_DIR.joinpath(f'vol_features_{_key}.pkl')
    if feat_cache.exists():
        print(f'\nLoading cached feature matrix from {feat_cache.name}')
        with feat_cache.open('rb') as fh:
            X, y, feature_cols, class_names = pickle.load(fh)
    else:
        print(f'\n=== Building feature matrix (+-{WINDOW_UM:.0f} um window) ===')
        X, y, feature_cols, class_names = build_feature_matrix(
            df, pairs, crossings_dict, WINDOW_UM
        )
        with feat_cache.open('wb') as fh:
            pickle.dump((X, y, feature_cols, class_names), fh)
        print(f'  Cached -> {feat_cache.name}')

    print(f'\nX: {X.shape},  n_classes: {len(class_names)}')
    unique, counts = np.unique(y, return_counts=True)
    for i, cnt in zip(unique, counts):
        print(f'  {class_names[i]}: {cnt} samples')

    print(f'\n=== Training GB classifier ({N_FOLDS}-fold stratified CV, class_weight=balanced) ===')
    results = train_classifier(X, y)
    print(f'\nOverall accuracy:   {results["overall_acc"]:.3f}')
    print(f'Balanced accuracy:  {results["balanced_acc"]:.3f}')

    print(f'\n=== Permutation test (n={N_PERMUTATIONS}) ===')
    null = compute_chance_level(y, results['y_pred'])
    print(
        f'  observed={null["observed"]:.3f}  balanced={null["balanced_observed"]:.3f}  '
        f'null={null["null_mean"]:.3f}+-{null["null_std"]:.3f}  p={null["pvalue"]:.4f}'
    )

    print('\n=== Saving figures ===')
    plot_confusion_matrix(
        y, results['y_pred'], class_names, FIGURES_DIR,
        results['overall_acc'], results['balanced_acc'],
    )
    plot_null_distribution(null, FIGURES_DIR)

    rows = [{'class': 'overall', 'accuracy': results['overall_acc'],
              'balanced_accuracy': results['balanced_acc']}]
    for i, acc in results['per_class_acc'].items():
        rows.append({'class': class_names[i], 'accuracy': acc, 'balanced_accuracy': np.nan})
    out_csv = FIGURES_DIR.joinpath('classifier_accuracy_vol.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'Saved {out_csv.name}')

    print('\n=== Per-class accuracy (sorted) ===')
    for i, acc in sorted(results['per_class_acc'].items(), key=lambda kv: -kv[1]):
        print(f'  {class_names[i]}: {acc:.3f}')


if __name__ == '__main__':
    main()
