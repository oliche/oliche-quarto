# %%
"""
PCA of ephys-atlas channel features.

Investigates how many principal components are needed to capture the variance
in the normalised feature matrix — to inform dimensionality reduction for
boundary-transition displays.
"""
import sys
from pathlib import Path

if (_here := '/Users/olivier/Documents/oliche-quarto/analyses/2026-05-EA-boundaries') not in sys.path:
    sys.path.insert(0, _here)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler  # noqa: F401

from boundaries_utils import compute_pca_features, load_features

sns.set_theme(context='notebook', palette='muted', style='whitegrid')

fig_dir = Path.home().joinpath('Documents', 'figures')
fig_dir.mkdir(exist_ok=True)

# %% Load features
df_features, _, VINTAGE = load_features()

# %% Run PCA
N_PCS = 6
df_pca, pca, scaler, feature_cols = compute_pca_features(df_features, n_pcs=N_PCS)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
n_components = np.arange(1, len(explained) + 1)

thresholds = [0.80, 0.90, 0.95, 0.99]
n_at_threshold = {t: int(np.searchsorted(cumulative, t)) + 1 for t in thresholds}
for t, n in n_at_threshold.items():
    print(f"  {int(t*100):2d}% variance explained by {n} components")

# %% Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: per-component variance (scree)
ax = axes[0]
ax.bar(n_components, explained * 100, color=sns.color_palette('muted')[0], width=0.8)
ax.set_xlabel('Principal component')
ax.set_ylabel('Variance explained (%)')
ax.set_title('Scree plot')
ax.set_xlim(0.5, len(explained) + 0.5)

# Right: cumulative variance
ax = axes[1]
ax.plot(n_components, cumulative * 100, marker='o', ms=4, lw=1.8,
        color=sns.color_palette('muted')[0])
for t, n in n_at_threshold.items():
    ax.axhline(t * 100, color='grey', lw=0.7, ls='--', alpha=0.6)
    ax.axvline(n, color='grey', lw=0.7, ls='--', alpha=0.6)
    ax.text(n + 0.15, t * 100 - 1.5, f'{int(t*100)}% @ PC{n}', fontsize=8, color='dimgrey')
ax.set_xlabel('Number of components')
ax.set_ylabel('Cumulative variance explained (%)')
ax.set_title('Cumulative variance')
ax.set_xlim(0.5, len(explained) + 0.5)
ax.set_ylim(0, 101)

fig.suptitle(
    f'PCA of ephys-atlas features ({VINTAGE}, n={len(df_pca):,} channels, {len(feature_cols)} features)',
    fontsize=10, y=1.02,
)
fig.tight_layout()
fig_name = '2026-06-08_pca_features_variance.png'
fig.savefig(fig_dir.joinpath(fig_name), dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved {fig_name}")

# %% Inspect output
print(f"\nPCA dataframe: {df_pca.shape[0]:,} channels × {df_pca.shape[1]} columns")
print(df_pca.head())
print(f"Variance captured by first {N_PCS} PCs: {cumulative[N_PCS - 1]:.1%}")

