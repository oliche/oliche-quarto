"""Per-class SHAP feature importance for the top volume-discovered landmarks.

Reproduces the Step-6 measured-data figure (boundary_feature_importance.py) on the
encoding-volume virtual-probe grid instead: trains a GB classifier on mean-diff feature
vectors for the top-20 "significant transitions" (tissue-only ranking from
run_volume_landmark_scan.py), then computes per-class SHAP values (TreeExplainer) for the
6 best-classified transitions to see which features the volume data relies on for each.

Outputs
-------
figures/landmark_feature_importance_volume.png
figures/landmark_feature_importance_volume.csv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

sys.path.insert(0, str(Path(__file__).parent))
import boundaries_utils as bu
import ephysatlas.anatomy
from boundary_classifier_volume import build_feature_matrix, precompute_crossings, train_classifier

VOL_LABEL = '2026_W26'
RES_UM = 50
GRID_SPACING_UM = 100.0
WINDOW_UM = 1000.0
N_SIGNIFICANT_TRANSITIONS = 20   # cap, matches the report's "significant transitions" table

CACHE_DIR = Path(__file__).parent.joinpath('cache')
FIGURES_DIR = Path(__file__).parent.joinpath('figures')

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 20, 'axes.labelsize': 18,
    'xtick.labelsize': 15, 'ytick.labelsize': 15,
})


def main() -> None:
    print('=== Loading virtual-probe dataframe ===')
    brain_atlas = ephysatlas.anatomy.ClassifierAtlas(res_um=RES_UM)
    vp_cache = CACHE_DIR.joinpath(f'vp_df_{VOL_LABEL}_{GRID_SPACING_UM:.0f}um.parquet')
    df_volume = pd.read_parquet(vp_cache)
    print(f'  {len(df_volume):,} virtual-probe channels')

    print('\n=== Top significant transitions (tissue-only ranking) ===')
    ranking = pd.read_csv(
        FIGURES_DIR.joinpath(f'volume_landmark_ranking_{VOL_LABEL}_tissue_only.csv')
    ).head(N_SIGNIFICANT_TRANSITIONS)
    pairs = list(ranking[['from', 'to']].itertuples(index=False, name=None))
    print(f'  {len(pairs)} transitions: {pairs}')

    print('\n=== Detecting boundary crossings ===')
    cross_cache = CACHE_DIR.joinpath(f'vp_crossings_{VOL_LABEL}_{GRID_SPACING_UM:.0f}um.pkl')
    if cross_cache.exists():
        print(f'  Loading cached crossings from {cross_cache.name}')
        with cross_cache.open('rb') as fh:
            crossings_dict, pair_counts = pickle.load(fh)
    else:
        crossings_dict, pair_counts = precompute_crossings(df_volume, brain_atlas)
        with cross_cache.open('wb') as fh:
            pickle.dump((crossings_dict, pair_counts), fh)
        print(f'  Cached -> {cross_cache.name}')

    print(f'\n=== Building feature matrix (+-{WINDOW_UM:.0f} um window) ===')
    X, y, feature_cols, class_names = build_feature_matrix(df_volume, pairs, crossings_dict, WINDOW_UM)
    print(f'X: {X.shape}, n_classes: {len(class_names)}')
    unique, counts = np.unique(y, return_counts=True)
    for i, cnt in zip(unique, counts):
        print(f'  {class_names[i]}: {cnt} samples')

    print('\n=== Training GB classifier (4-fold stratified CV) ===')
    results = train_classifier(X, y)
    print(f'Overall accuracy: {results["overall_acc"]:.3f}  Balanced: {results["balanced_acc"]:.3f}')

    gb = results['model']
    per_class_acc = results['per_class_acc']
    top6_idx = sorted(per_class_acc, key=lambda i: -per_class_acc[i])[:6]
    top6_names = [class_names[i] for i in top6_idx]
    top6_labels = [n.replace('_to_', ' → ') for n in top6_names]
    print('\nTop-6 transitions by CV accuracy:')
    for name, i in zip(top6_names, top6_idx):
        print(f'  {name}: {per_class_acc[i]:.1%}')

    print('\n=== Computing SHAP values (TreeExplainer) ===')
    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(X)

    rows = []
    shap_matrix = np.zeros((len(top6_idx), len(feature_cols)))
    for row_i, cls_i in enumerate(top6_idx):
        mask = y == cls_i
        mean_abs = np.abs(shap_values[mask, :, cls_i]).mean(axis=0)
        shap_matrix[row_i] = mean_abs
        for feat, val in zip(feature_cols, mean_abs):
            rows.append({'transition': top6_names[row_i], 'feature': feat, 'mean_abs_shap': val})

    df_shap = pd.DataFrame(rows)
    csv_out = FIGURES_DIR.joinpath('landmark_feature_importance_volume.csv')
    df_shap.to_csv(csv_out, index=False)
    print(f'Saved {csv_out.name}')

    # Feature colour groups (volume feature set: no rms_lf_no_car, has cor_ratio/decay_*)
    _LFP_CSD = {'psd_pc0', 'psd_pc1', 'csd_pc0', 'csd_pc1', 'aperiodic_offset', 'aperiodic_exponent'}
    _SPIKE_ALPHA = {'rms_ap', 'spike_count', 'alpha_mean', 'alpha_std', 'cor_ratio'}

    def _feat_color(feat: str) -> str:
        if feat in _LFP_CSD:
            return '#2ca02c'   # green
        if feat in _SPIKE_ALPHA:
            return '#9467bd'   # purple
        return '#1f77b4'       # blue (waveform)

    top_n = 10
    n_cls = len(top6_idx)
    fig, axes = plt.subplots(1, n_cls, figsize=(n_cls * 3.6, 6), constrained_layout=True)

    for cls_i, label, ax in zip(top6_idx, top6_labels, axes):
        row_i = list(top6_idx).index(cls_i)
        order = np.argsort(shap_matrix[row_i])[::-1][:top_n]
        vals = shap_matrix[row_i][order]
        names = [feature_cols[j].replace('_', ' ') for j in order]
        colors = [_feat_color(feature_cols[j]) for j in order]

        positions = range(top_n - 1, -1, -1)
        ax.barh(list(positions), vals, color=colors, edgecolor='white')
        ax.set_yticks(list(positions))
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('mean |SHAP|', fontsize=12)
        ax.set_title(f'{label}\n({per_class_acc[cls_i]:.0%})', fontsize=13)
        ax.tick_params(axis='x', labelsize=11)
        sns.despine(ax=ax)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color='#2ca02c', label='LFP / CSD'),
        Patch(color='#9467bd', label='RMS AP, spikes, alpha'),
        Patch(color='#1f77b4', label='Waveform shape'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Volume-derived landmark feature importance — top 10 features per boundary (mean |SHAP|)',
                 fontsize=14)
    out = FIGURES_DIR.joinpath('landmark_feature_importance_volume.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out.name}')


if __name__ == '__main__':
    main()
