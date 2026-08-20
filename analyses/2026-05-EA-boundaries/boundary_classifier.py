"""Boundary classifier: predict Cosmos region transition type from LFP features.

Workflow
--------
1. Load the 26-feature post-PCA dataframe (PSD/CSD reduced to 4 PCA components).
2. Select transitions with >= MIN_INSERTIONS crossings.
3. For each (probe, boundary) crossing build a feature vector:
   (mean_below − mean_above) for each feature within ±WINDOW_UM µm.
4. Train a gradient-boosted classifier (HistGradientBoostingClassifier) with
   stratified 4-fold CV.
5. Report per-class accuracy, confusion matrix, and permutation null distribution.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
import boundaries_utils as bu

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.titlesize': 22,
})

VINTAGE = '2026_W24'
CACHE_DIR = Path.home().joinpath(
    'data', 'ephys-atlas', 'features', 'ea_active', VINTAGE, 'agg_full'
)
FIGURES_DIR = Path(__file__).parent.joinpath('figures')
LOCAL_CACHE_DIR = Path(__file__).parent.joinpath('cache')
WINDOW_UM = 1000.0
MIN_INSERTIONS = 32
N_FOLDS = 4
N_PERMUTATIONS = 100
RANDOM_STATE = 42

# Features shown in depth-profile heatmaps
_DISPLAY_FEATURES = (
    'psd_pc0', 'psd_pc1', 'csd_pc0', 'csd_pc1',
    'rms_ap', 'spike_count', 'aperiodic_offset', 'aperiodic_exponent',
)


def load_data() -> tuple[pd.DataFrame, object]:
    """Load channel features and apply PSD/CSD PCA reduction.

    Returns
    -------
    df:
        One row per channel; 26-column feature set after PCA.
    brain_atlas:
        ClassifierAtlas instance.
    """
    import ephysatlas.anatomy
    import ephysatlas.data
    from ephysatlas.features import psd_pca_dataframe

    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()

    parquet_cache = LOCAL_CACHE_DIR.joinpath(f'df_{VINTAGE}.parquet')
    if parquet_cache.exists():
        print(f'  Loading cached post-PCA dataframe from {parquet_cache.name}')
        df = pd.read_parquet(parquet_cache)
    else:
        df = ephysatlas.data.read_features_from_disk(CACHE_DIR, brain_atlas=brain_atlas, strict=False)
        df = psd_pca_dataframe(df, n_components_psd=2, n_components_csd=2)
        LOCAL_CACHE_DIR.mkdir(exist_ok=True)
        df.to_parquet(parquet_cache)
        print(f'  Cached post-PCA dataframe → {parquet_cache.name}')

    return df, brain_atlas


def get_qualifying_transitions(min_insertions: int = MIN_INSERTIONS) -> list[tuple[str, str]]:
    """Return (from_acr, to_acr) pairs with transition count >= min_insertions.

    Parameters
    ----------
    min_insertions:
        Threshold on the directional probe count in the transition matrix.

    Returns
    -------
    pairs:
        List of (from_acr, to_acr) tuples, ordered as in the matrix.
    """
    count_matrix = pd.read_csv(
        FIGURES_DIR.joinpath(f'cosmos_transition_matrix_{VINTAGE}.csv'), index_col=0
    )
    pairs = [
        (str(fr), str(to))
        for fr in count_matrix.index
        for to in count_matrix.columns
        if fr != to and int(count_matrix.loc[fr, to]) >= min_insertions
    ]
    return pairs


def build_feature_matrix(
    df: pd.DataFrame,
    brain_atlas,
    pairs: list[tuple[str, str]],
    window_um: float = WINDOW_UM,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Build the classifier input from crossing events.

    For each (probe, boundary) crossing computes a ``(mean_below − mean_above)``
    difference vector over the ±window_um window.  "Below" is the from-region
    side (deeper channels), "above" is the to-region side (shallower channels).

    Parameters
    ----------
    df:
        Channel features dataframe after PCA (MultiIndex with ``pid`` level).
    brain_atlas:
        ClassifierAtlas instance.
    pairs:
        Qualifying (from_acr, to_acr) pairs.
    window_um:
        Half-window around each transition depth (µm).

    Returns
    -------
    X:
        Feature matrix of shape ``(n_samples, n_features)``.
    y:
        Integer class labels of shape ``(n_samples,)``.
    feature_cols:
        Ordered list of feature column names.
    class_names:
        Class label strings; ``class_names[i]`` corresponds to ``y == i``.
    """
    feature_cols = [c for c in df.columns if c not in bu.EXCLUDE_COLS]
    df_reset = df[['axial_um'] + feature_cols].reset_index(level='pid')

    rows: list[np.ndarray] = []
    labels: list[str] = []

    for from_acr, to_acr in pairs:
        label = f'{from_acr}_to_{to_acr}'
        crossings = bu.find_boundary_crossings(df, brain_atlas, from_acr, to_acr)
        print(f'{label}: {len(crossings)} crossings')

        for pid, trans_depth in crossings:
            probe = df_reset[df_reset['pid'] == pid]
            window = probe[
                (probe['axial_um'] >= trans_depth - window_um) &
                (probe['axial_um'] <= trans_depth + window_um)
            ]
            below = window[window['axial_um'] <= trans_depth]
            above = window[window['axial_um'] >= trans_depth]
            if len(below) == 0 or len(above) == 0:
                continue

            diff = below[feature_cols].mean().values - above[feature_cols].mean().values
            rows.append(np.nan_to_num(diff))
            labels.append(label)

    X = np.array(rows)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X, y, feature_cols, list(le.classes_)


def train_classifiers(
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Train a gradient-boosted classifier with stratified 4-fold CV.

    The model is also refit on the full dataset after CV.

    Parameters
    ----------
    X:
        Feature matrix ``(n_samples, n_features)``.
    y:
        Integer class labels.

    Returns
    -------
    results:
        Dict with key ``'gb'`` containing: ``model`` (fitted),
        ``y_pred`` (CV predictions), ``overall_acc``,
        ``per_class_acc`` (dict class → accuracy).
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model = HistGradientBoostingClassifier(max_iter=300, random_state=RANDOM_STATE)

    y_pred = np.empty_like(y)
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])

    n_classes = len(np.unique(y))
    per_class = {
        int(i): float(accuracy_score(y[y == i], y_pred[y == i]))
        for i in range(n_classes)
    }
    model.fit(X, y)  # refit on full data
    results = {
        'gb': {
            'model': model,
            'y_pred': y_pred,
            'overall_acc': float(accuracy_score(y, y_pred)),
            'per_class_acc': per_class,
        }
    }
    print(f'GB CV accuracy: {results["gb"]["overall_acc"]:.3f}')
    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _short(class_names: list[str]) -> list[str]:
    return [n.replace('_to_', '→') for n in class_names]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    output_dir: Path,
) -> None:
    """Save a row-normalised confusion matrix heatmap.

    Parameters
    ----------
    y_true:
        True integer labels.
    y_pred:
        Predicted integer labels.
    class_names:
        Class label strings.
    model_name:
        Used in the title and filename.
    output_dir:
        Destination directory.
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
    ax.set_title(f'{model_name.upper()} — normalised confusion matrix', fontsize=15)
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', rotation=0, labelsize=11)
    fig.tight_layout()
    out = output_dir.joinpath(f'classifier_confusion_{model_name}.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out.name}')


def compute_chance_level(
    y: np.ndarray,
    results: dict,
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    """Estimate chance-level accuracy by permuting test-set labels.

    The CV predictions are kept fixed; only the true labels are shuffled
    repeatedly.

    Parameters
    ----------
    y:
        True class labels (same order as used in CV).
    results:
        Output of :func:`train_classifiers` (provides ``y_pred`` per model).
    n_permutations:
        Number of label shuffles.

    Returns
    -------
    null_results:
        Dict keyed by model name, each with ``observed``, ``null_scores``,
        ``null_mean``, ``null_std``, ``pvalue``.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    null_results = {}
    for name, res in results.items():
        y_pred = res['y_pred']
        null_scores = np.array([
            accuracy_score(rng.permutation(y), y_pred)
            for _ in range(n_permutations)
        ])
        observed = res['overall_acc']
        pvalue = float((null_scores >= observed).sum() + 1) / (n_permutations + 1)
        null_results[name] = {
            'observed': observed,
            'null_scores': null_scores,
            'null_mean': float(null_scores.mean()),
            'null_std': float(null_scores.std()),
            'pvalue': pvalue,
        }
        print(
            f'  {name.upper()}: observed={observed:.3f}  '
            f'null={null_scores.mean():.3f}±{null_scores.std():.3f}  '
            f'p={pvalue:.4f}'
        )
    return null_results


def save_results(
    results: dict,
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Save per-class and overall CV accuracy to CSV.

    Parameters
    ----------
    results:
        Output of :func:`train_classifiers`.
    class_names:
        Class label strings.
    output_dir:
        Destination directory.
    """
    rows = []
    for model_name, res in results.items():
        rows.append({'model': model_name, 'class': 'overall', 'accuracy': res['overall_acc']})
        for i, acc in res['per_class_acc'].items():
            rows.append({'model': model_name, 'class': class_names[i], 'accuracy': acc})
    out = output_dir.joinpath('classifier_accuracy.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Saved {out.name}')


def _cosmos_acr2id(brain_atlas) -> dict[str, int]:
    """Build acronym → ID lookup restricted to Cosmos-level regions."""
    cosmos_map = brain_atlas.regions.mappings['Cosmos']
    cosmos_indices = np.unique(cosmos_map)
    acr_arr = np.asarray(brain_atlas.regions.acronym).flatten()
    id_arr = np.asarray(brain_atlas.regions.id).flatten()
    return {str(acr_arr[i]): int(id_arr[i]) for i in cosmos_indices}


def _cosmos_ids_slice(brain_atlas, coord: float, axis: int) -> np.ndarray:
    """Return a Cosmos-ID slice at *coord* (metres) along *axis*."""
    cosmos_map = brain_atlas.regions.mappings['Cosmos']
    id_arr = np.asarray(brain_atlas.regions.id).flatten()
    rindex_raw = brain_atlas.slice(float(coord), axis=axis, volume='rindex')
    return id_arr[cosmos_map[rindex_raw]]


def _best_ap_for_pair(
    brain_atlas, from_id: int, to_id: int,
    n_samples: int = 40, cache: dict | None = None, cache_key: str = '',
) -> float:
    """Return the AP coordinate (m) where both Cosmos regions occupy the most voxels."""
    if cache is not None and cache_key in cache:
        return float(cache[cache_key])

    ap_vals = np.linspace(brain_atlas.bc.ylim[0], brain_atlas.bc.ylim[1], n_samples + 2)[1:-1]
    best_ap = float(np.mean(brain_atlas.bc.ylim))
    best_score = -1
    for ap in ap_vals:
        ids = _cosmos_ids_slice(brain_atlas, ap, axis=1)
        score = int(min((ids == from_id).sum(), (ids == to_id).sum()))
        if score > best_score:
            best_score, best_ap = score, float(ap)

    if cache is not None and cache_key:
        cache[cache_key] = best_ap
    return best_ap


def _best_ml_for_pair(
    brain_atlas, from_id: int, to_id: int,
    n_samples: int = 40, cache: dict | None = None, cache_key: str = '',
) -> float:
    """Return the ML coordinate (m) where both Cosmos regions occupy the most voxels."""
    if cache is not None and cache_key in cache:
        return float(cache[cache_key])

    ml_vals = np.linspace(brain_atlas.bc.xlim[0], brain_atlas.bc.xlim[1], n_samples + 2)[1:-1]
    best_ml = float(np.mean(brain_atlas.bc.xlim))
    best_score = -1
    for ml in ml_vals:
        ids = _cosmos_ids_slice(brain_atlas, ml, axis=0)
        score = int(min((ids == from_id).sum(), (ids == to_id).sum()))
        if score > best_score:
            best_score, best_ml = score, float(ml)

    if cache is not None and cache_key:
        cache[cache_key] = best_ml
    return best_ml


def plot_sagittal_boundary_sections(
    brain_atlas,
    pairs_acc: list[tuple[str, str, float]],
    output_dir: Path,
    coords_cache: dict | None = None,
) -> None:
    """Plot sagittal brain sections with Allen-colour region boundaries.

    Parameters
    ----------
    brain_atlas:
        ClassifierAtlas instance.
    pairs_acc:
        List of ``(from_acr, to_acr, cv_accuracy)`` tuples.
    output_dir:
        Destination directory.
    coords_cache:
        Mutable dict for caching best ML coordinates.
    """
    acr2id = _cosmos_acr2id(brain_atlas)
    ext = brain_atlas.extent(axis=0)

    n = len(pairs_acc)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    ax_flat = np.array(axes).flat

    for ax, (from_acr, to_acr, acc) in zip(ax_flat, pairs_acc):
        from_id = acr2id.get(from_acr)
        to_id = acr2id.get(to_acr)

        if from_id is None or to_id is None:
            ax.set_title(f'{from_acr}→{to_acr}: region not found', fontsize=11)
            continue

        ckey = f'{from_acr}_to_{to_acr}_ml'
        ml_m = _best_ml_for_pair(brain_atlas, from_id, to_id, cache=coords_cache, cache_key=ckey)

        from_hexcol = str(brain_atlas.regions.hexcolor[int(brain_atlas.regions.id2index(from_id)[1][0][0])])
        to_hexcol = str(brain_atlas.regions.hexcolor[int(brain_atlas.regions.id2index(to_id)[1][0][0])])

        brain_atlas.plot_sslice(ml_m, volume='image', ax=ax)

        ids = _cosmos_ids_slice(brain_atlas, ml_m, axis=0)
        nap, ndv = ids.shape
        ap_coords = np.linspace(ext[0], ext[1], nap)
        dv_coords = np.linspace(ext[3], ext[2], ndv)
        AP, DV = np.meshgrid(ap_coords, dv_coords)

        from_mask = (ids == from_id).astype(np.float32)
        to_mask = (ids == to_id).astype(np.float32)

        ax.contour(AP, DV, from_mask.T, levels=[0.5], colors=from_hexcol, linewidths=3)
        ax.contour(AP, DV, to_mask.T, levels=[0.5], colors=to_hexcol, linewidths=3, linestyles='--')
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color=from_hexcol, lw=3, label=from_acr),
                plt.Line2D([0], [0], color=to_hexcol, lw=3, ls='--', label=to_acr),
            ],
            fontsize=11, loc='lower right',
        )
        ax.set_title(f'{from_acr} → {to_acr}  ({acc:.0%})  ML={ml_m * 1e6:.0f} µm', fontsize=13)
        ax.set_xlabel('AP (µm)', fontsize=12)
        ax.set_ylabel('DV (µm)', fontsize=12)
        ax.tick_params(labelsize=11)

    for ax in list(ax_flat)[n:]:
        ax.set_visible(False)

    fig.suptitle('Top transitions — sagittal sections (auto ML)', fontsize=15)
    fig.tight_layout()
    out = output_dir.joinpath('classifier_sagittal_boundaries.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out.name}')


def plot_boundary_sections(
    brain_atlas,
    pairs_acc: list[tuple[str, str, float]],
    output_dir: Path,
    coords_cache: dict | None = None,
) -> None:
    """Plot coronal brain sections with Allen-colour region boundaries.

    Parameters
    ----------
    brain_atlas:
        ClassifierAtlas instance.
    pairs_acc:
        List of ``(from_acr, to_acr, cv_accuracy)`` tuples.
    output_dir:
        Destination directory.
    coords_cache:
        Mutable dict for caching best AP coordinates.
    """
    acr2id = _cosmos_acr2id(brain_atlas)
    ext = brain_atlas.extent(axis=1)

    n = len(pairs_acc)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    ax_flat = np.array(axes).flat

    for ax, (from_acr, to_acr, acc) in zip(ax_flat, pairs_acc):
        from_id = acr2id.get(from_acr)
        to_id = acr2id.get(to_acr)

        if from_id is None or to_id is None:
            ax.set_title(f'{from_acr}→{to_acr}: region not found', fontsize=11)
            continue

        ckey = f'{from_acr}_to_{to_acr}_ap'
        ap_m = _best_ap_for_pair(brain_atlas, from_id, to_id, cache=coords_cache, cache_key=ckey)

        from_hexcol = str(brain_atlas.regions.hexcolor[int(brain_atlas.regions.id2index(from_id)[1][0][0])])
        to_hexcol = str(brain_atlas.regions.hexcolor[int(brain_atlas.regions.id2index(to_id)[1][0][0])])

        brain_atlas.plot_cslice(ap_m, volume='image', ax=ax)

        ids = _cosmos_ids_slice(brain_atlas, ap_m, axis=1)
        nml, ndv = ids.shape
        ml_coords = np.linspace(ext[0], ext[1], nml)
        dv_coords = np.linspace(ext[3], ext[2], ndv)
        ML, DV = np.meshgrid(ml_coords, dv_coords)

        from_mask = (ids == from_id).astype(np.float32)
        to_mask = (ids == to_id).astype(np.float32)

        ax.contour(ML, DV, from_mask.T, levels=[0.5], colors=from_hexcol, linewidths=3)
        ax.contour(ML, DV, to_mask.T, levels=[0.5], colors=to_hexcol, linewidths=3, linestyles='--')
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color=from_hexcol, lw=3, label=from_acr),
                plt.Line2D([0], [0], color=to_hexcol, lw=3, ls='--', label=to_acr),
            ],
            fontsize=11, loc='lower right',
        )
        ax.set_title(f'{from_acr} → {to_acr}  ({acc:.0%})  AP={ap_m * 1e6:.0f} µm', fontsize=13)
        ax.set_xlabel('ML (µm)', fontsize=12)
        ax.set_ylabel('DV (µm)', fontsize=12)
        ax.tick_params(labelsize=11)

    for ax in list(ax_flat)[n:]:
        ax.set_visible(False)

    fig.suptitle('Top transitions — coronal sections (auto AP)', fontsize=15)
    fig.tight_layout()
    out = output_dir.joinpath('classifier_coronal_boundaries.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out.name}')


def main() -> None:
    """Run the full boundary classifier pipeline."""
    FIGURES_DIR.mkdir(exist_ok=True)
    LOCAL_CACHE_DIR.mkdir(exist_ok=True)

    _cache_key = f'{VINTAGE}_{WINDOW_UM:.0f}um_{MIN_INSERTIONS}ins'
    features_cache = LOCAL_CACHE_DIR.joinpath(f'features_{_cache_key}.pkl')
    model_cache = LOCAL_CACHE_DIR.joinpath(f'model_{_cache_key}.pkl')
    coords_cache_path = LOCAL_CACHE_DIR.joinpath(f'best_coords_{VINTAGE}.json')

    print('=== Loading data ===')
    df, brain_atlas = load_data()

    print('\n=== Qualifying transitions ===')
    pairs = get_qualifying_transitions()
    for fr, to in pairs:
        print(f'  {fr} → {to}')
    print(f'Total: {len(pairs)} transitions')

    # Print feature list used for training and display
    feature_cols_check = [c for c in df.columns if c not in bu.EXCLUDE_COLS]
    print(f'\n=== Features used for training ({len(feature_cols_check)}) ===')
    for f in feature_cols_check:
        print(f'  {f}')
    print(f'\n=== Display features for heatmaps ===')
    for f in _DISPLAY_FEATURES:
        print(f'  {f}')

    print(f'\n=== Building feature matrix (±{WINDOW_UM:.0f} µm) ===')
    if features_cache.exists():
        print(f'  Loading cached feature matrix from {features_cache.name}')
        with features_cache.open('rb') as fh:
            X, y, feature_cols, class_names = pickle.load(fh)
    else:
        X, y, feature_cols, class_names = build_feature_matrix(df, brain_atlas, pairs)
        with features_cache.open('wb') as fh:
            pickle.dump((X, y, feature_cols, class_names), fh)
        print(f'  Cached feature matrix → {features_cache.name}')
    print(f'X: {X.shape},  classes: {len(class_names)}')
    unique, counts = np.unique(y, return_counts=True)
    for i, cnt in zip(unique, counts):
        print(f'  {class_names[i]}: {cnt} probes')

    print(f'\n=== Training GB classifier ({N_FOLDS}-fold stratified CV) ===')
    if model_cache.exists():
        print(f'  Loading cached model from {model_cache.name}')
        with model_cache.open('rb') as fh:
            results = pickle.load(fh)
        for name, res in results.items():
            print(f'{name.upper()} CV accuracy: {res["overall_acc"]:.3f}')
    else:
        results = train_classifiers(X, y)
        with model_cache.open('wb') as fh:
            pickle.dump(results, fh)
        print(f'  Cached model → {model_cache.name}')

    print(f'\n=== Permutation test (n={N_PERMUTATIONS}) ===')
    null_results = compute_chance_level(y, results)

    print('\n=== Saving figures ===')
    for name, res in results.items():
        plot_confusion_matrix(y, res['y_pred'], class_names, name, FIGURES_DIR)

    save_results(results, class_names, FIGURES_DIR)

    # Load/save best atlas coordinates
    coords_cache: dict = {}
    if coords_cache_path.exists():
        coords_cache = json.loads(coords_cache_path.read_text())
        print(f'\n  Loaded {len(coords_cache)} cached atlas coordinates')

    # Top 6 transitions by GB per-class accuracy
    gb_acc = results['gb']['per_class_acc']
    top6 = sorted(gb_acc.items(), key=lambda kv: -kv[1])[:6]
    pairs_acc = [
        (class_names[i].split('_to_')[0], class_names[i].split('_to_')[1], acc)
        for i, acc in top6
    ]
    print('\n=== Top 6 transitions ===')
    for from_acr, to_acr, acc in pairs_acc:
        print(f'  {from_acr} → {to_acr}: {acc:.1%}')

    print('\n=== Coronal boundary plots ===')
    plot_boundary_sections(brain_atlas, pairs_acc, FIGURES_DIR, coords_cache=coords_cache)
    print('\n=== Sagittal boundary plots ===')
    plot_sagittal_boundary_sections(brain_atlas, pairs_acc, FIGURES_DIR, coords_cache=coords_cache)

    coords_cache_path.write_text(json.dumps(coords_cache, indent=2))
    print(f'  Saved {len(coords_cache)} atlas coordinates → {coords_cache_path.name}')

    # Heatmap profiles for top 6 using post-PCA features
    print('\n=== Heatmap profiles for top 6 (PCA features) ===')
    heatmap_dir = FIGURES_DIR.joinpath('heatmap_profiles')
    heatmap_dir.mkdir(exist_ok=True)
    for from_acr, to_acr, acc in pairs_acc:
        print(f'  {from_acr} → {to_acr}')
        fig = bu.plot_boundary_feature_profiles(
            df, brain_atlas,
            from_acr, to_acr,
            mandatory_features=_DISPLAY_FEATURES,
            df_results=None,
            window_um=WINDOW_UM,
            max_probes=150,
            sort='depth',
        )
        if fig is not None:
            slug = f'{from_acr}_to_{to_acr}'
            out = heatmap_dir.joinpath(f'heatmap_{slug}.png')
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f'    → {out.name}')

    print('\n=== Summary ===')
    for name, res in results.items():
        nr = null_results[name]
        print(
            f'\n{name.upper()} CV accuracy: {res["overall_acc"]:.3f}  '
            f'(null: {nr["null_mean"]:.3f}±{nr["null_std"]:.3f},  p={nr["pvalue"]:.4f})'
        )
        for i, acc in sorted(res['per_class_acc'].items(), key=lambda kv: -kv[1]):
            print(f'  {class_names[i]}: {acc:.3f}')


if __name__ == '__main__':
    main()