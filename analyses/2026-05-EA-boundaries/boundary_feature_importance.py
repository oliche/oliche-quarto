"""Per-class SHAP feature importance for the top-6 boundary landmarks.

For each of the top-6 classified transitions, computes the mean |SHAP| value
per feature using only the samples belonging to that class.  This answers:
"which features does the model rely on to recognise *this specific boundary*?"

Outputs
-------
figures/landmark_feature_importance.png
    Single-row figure — one bar-chart panel per transition, showing the
    top-10 features by mean |SHAP| in descending order.
figures/landmark_feature_importance.csv
    Tidy CSV with columns: transition, feature, mean_abs_shap.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

VINTAGE = '2026_W24'
FIGURES_DIR = Path(__file__).parent.joinpath('figures')
LOCAL_CACHE_DIR = Path(__file__).parent.joinpath('cache')

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
})


def main() -> None:
    # ------------------------------------------------------------------ load
    model_cache = LOCAL_CACHE_DIR.joinpath(f'model_{VINTAGE}_1000um_32ins.pkl')
    feat_cache = LOCAL_CACHE_DIR.joinpath(f'features_{VINTAGE}_1000um_32ins.pkl')

    print('Loading model and feature matrix …')
    with model_cache.open('rb') as fh:
        results = pickle.load(fh)
    with feat_cache.open('rb') as fh:
        X, y, feature_cols, class_names = pickle.load(fh)

    gb = results['gb']['model']
    per_class_acc = results['gb']['per_class_acc']

    # Top-6 classes by CV accuracy
    top6_idx = sorted(per_class_acc, key=lambda i: -per_class_acc[i])[:6]
    top6_names = [class_names[i] for i in top6_idx]
    top6_labels = [n.replace('_to_', ' → ') for n in top6_names]

    print(f'Top-6 transitions:')
    for i, (name, acc_i) in enumerate(zip(top6_names, [per_class_acc[i] for i in top6_idx])):
        n_samples = int((y == top6_idx[i]).sum())
        print(f'  {name}: {acc_i:.1%}  ({n_samples} samples)')

    # ------------------------------------------------------------------ SHAP
    print('\nComputing SHAP values (TreeExplainer) …')
    explainer = shap.TreeExplainer(gb)
    # shap_values: (n_samples, n_features, n_classes)
    shap_values = explainer.shap_values(X)

    # Per-class mean |SHAP| — only samples of that class, projection onto that class axis
    rows = []
    shap_matrix = np.zeros((len(top6_idx), len(feature_cols)))
    for row_i, cls_i in enumerate(top6_idx):
        mask = y == cls_i
        mean_abs = np.abs(shap_values[mask, :, cls_i]).mean(axis=0)
        shap_matrix[row_i] = mean_abs
        for feat, val in zip(feature_cols, mean_abs):
            rows.append({'transition': top6_names[row_i], 'feature': feat, 'mean_abs_shap': val})

    df_shap = pd.DataFrame(rows)

    # Save CSV
    csv_out = FIGURES_DIR.joinpath('landmark_feature_importance.csv')
    df_shap.to_csv(csv_out, index=False)
    print(f'Saved {csv_out.name}')

    # Feature colour groups
    _LFP_CSD = {'psd_pc0', 'psd_pc1', 'csd_pc0', 'csd_pc1',
                'rms_lf_no_car', 'aperiodic_offset', 'aperiodic_exponent'}
    _SPIKE_ALPHA = {'rms_ap', 'spike_count', 'alpha_mean', 'alpha_std'}
    # everything else → waveform (blue)

    def _feat_color(feat: str) -> str:
        if feat in _LFP_CSD:
            return '#2ca02c'   # green
        if feat in _SPIKE_ALPHA:
            return '#9467bd'   # purple
        return '#1f77b4'       # blue (waveform)

    # ------------------------------------------------------------------ plot
    top_n = 10
    n_cls = len(top6_idx)
    fig, axes = plt.subplots(1, n_cls, figsize=(n_cls * 3.6, 6), constrained_layout=True)

    for cls_i, label, ax in zip(top6_idx, top6_labels, axes):
        row_i = list(top6_idx).index(cls_i)
        order = np.argsort(shap_matrix[row_i])[::-1][:top_n]
        vals = shap_matrix[row_i][order]
        names = [feature_cols[j].replace('_', ' ') for j in order]
        colors = [_feat_color(feature_cols[j]) for j in order]

        # most important at top → reverse so barh index 0 is at top
        positions = range(top_n - 1, -1, -1)
        ax.barh(list(positions), vals, color=colors, edgecolor='white')
        ax.set_yticks(list(positions))
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('mean |SHAP|', fontsize=12)
        ax.set_title(f'{label}\n({per_class_acc[cls_i]:.0%})', fontsize=13)
        ax.tick_params(axis='x', labelsize=11)
        sns.despine(ax=ax)

    # Shared legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color='#2ca02c', label='LFP / CSD'),
        Patch(color='#9467bd', label='RMS AP, spikes, alpha'),
        Patch(color='#1f77b4', label='Waveform shape'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle('Landmark feature importance — top 10 features per boundary (mean |SHAP|)',
                 fontsize=14)
    out = FIGURES_DIR.joinpath('landmark_feature_importance.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out.name}')

    # Print top-3 features per boundary for quick read
    print('\n=== Top-3 features per landmark ===')
    for row_i, (name, cls_i) in enumerate(zip(top6_names, top6_idx)):
        top3 = np.argsort(shap_matrix[row_i])[::-1][:3]
        feats = ', '.join(f'{feature_cols[j]} ({shap_matrix[row_i, j]:.3f})' for j in top3)
        print(f'  {name}: {feats}')


if __name__ == '__main__':
    main()