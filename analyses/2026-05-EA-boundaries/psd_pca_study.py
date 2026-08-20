# %%
"""
PCA study of PSD (power spectral density) features in the ephys-atlas.

PSD features measure how much electrical "noise" or activity is present at
different brain-wave frequencies at each recording site. Because nearby
frequency bands tend to rise and fall together, they are highly correlated.
PCA finds the main independent "dimensions" of variation hiding behind those
correlations.

Run as a Jupyter-style percent script (e.g. in VS Code or Spyder).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent))
import boundaries_utils as bu

sns.set_theme(context="talk", style="whitegrid", palette="muted")
figures_dir = Path(__file__).parent.joinpath('figures')
figures_dir.mkdir(exist_ok=True)

# %%  ── 1. Load features ───────────────────────────────────────────────────
df_features, brain_atlas, vintage = bu.load_features()
regions = brain_atlas.regions

# %%  ── 2. Select PSD features ────────────────────────────────────────────
# "PSD features" = every column whose value comes from the LFP power spectrum:
#   psd_<band>          – mean power in a canonical frequency band
#   psd_<band>_csd      – same, after current-source-density transform
#   psd_<band>_csd_diff1 – first spatial derivative of the CSD
#   rms_lf / rms_lf_csd  – root-mean-square of the broadband LFP
#   aperiodic_exponent / aperiodic_offset  – slope and intercept of the
#                          1/f background when fit in log-log space
# Split into two groups:
#   RAW  — one value per frequency band, no spatial transformation
#   CSD  — current-source-density variants (including _csd_diff1)
ALL_PSD = sorted(
    c for c in df_features.columns
    if (c.startswith('psd_') and not c.startswith('psd_residual_'))
    or c in ('rms_lf', 'rms_lf_csd')
)
RAW_FEATURES = [c for c in ALL_PSD if '_csd' not in c]
CSD_FEATURES = [c for c in ALL_PSD if '_csd' in c]
print(f"RAW features ({len(RAW_FEATURES)}): {RAW_FEATURES}")
print(f"CSD features ({len(CSD_FEATURES)}): {CSD_FEATURES}")

X_all = df_features[ALL_PSD].dropna()
print(f"\n{len(X_all):,} channels after NaN-drop")

# %%  ── 3. Region labels and colour map ───────────────────────────────────
cosmos_refined = bu.get_cosmos_refined(df_features.loc[X_all.index], brain_atlas)
unique_ids = sorted(cosmos_refined.unique())

def _id_to_acr(rid):
    try:
        return regions.acronym[regions.id2index(int(rid))[1]][0][0]
    except Exception:
        return str(rid)

id2acr = {rid: _id_to_acr(rid) for rid in unique_ids}
id2color = {rid: bu._get_region_rgba(regions, int(rid)) for rid in unique_ids}

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markersize=8,
           markerfacecolor=id2color[rid], label=id2acr[rid])
    for rid in unique_ids if (cosmos_refined.values == rid).sum() >= 5
]

rng = np.random.default_rng(0)
n_plot = min(40_000, len(X_all))
idx_s = rng.choice(len(X_all), size=n_plot, replace=False)
labels_s = cosmos_refined.values[idx_s]

# %%  ── 4. Fit two separate PCAs ──────────────────────────────────────────
def fit_pca(df, features):
    """Standardise *features* from *df*, fit a full PCA, return (pca, scaler, scores)."""
    X = df[features].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(Xs)
    scores = pca.transform(Xs)
    return pca, scaler, scores

pca_raw, scaler_raw, scores_raw = fit_pca(X_all, RAW_FEATURES)
pca_csd, scaler_csd, scores_csd = fit_pca(X_all, CSD_FEATURES)

for name, pca, feats in [('RAW', pca_raw, RAW_FEATURES), ('CSD', pca_csd, CSD_FEATURES)]:
    evr = pca.explained_variance_ratio_
    cumevr = np.cumsum(evr)
    n90 = int(np.searchsorted(cumevr, 0.90)) + 1
    n95 = int(np.searchsorted(cumevr, 0.95)) + 1
    print(f"\n{name} PCA ({len(feats)} features):")
    print(f"  PC1 explains {evr[0]*100:.1f}%,  cumulative: "
          + ", ".join(f"PC{k+1}={c*100:.0f}%" for k, c in enumerate(cumevr[:5])))
    print(f"  → {n90} PCs for 90%,  {n95} PCs for 95%")

# %%  ── 5. Side-by-side scree plots ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for col, (name, pca, feats, color) in enumerate([
    ('RAW', pca_raw, RAW_FEATURES, 'steelblue'),
    ('CSD', pca_csd, CSD_FEATURES, 'darkorange'),
]):
    evr = pca.explained_variance_ratio_
    cumevr = np.cumsum(evr)
    n_feats = len(feats)
    n90 = int(np.searchsorted(cumevr, 0.90)) + 1
    n95 = int(np.searchsorted(cumevr, 0.95)) + 1
    xs = range(1, n_feats + 1)

    # top row: scree bar
    ax = axes[0, col]
    ax.bar(xs, evr * 100, color=color, alpha=0.75)
    ax.plot(xs, evr * 100, 'o-', color='k', ms=4, lw=1)
    ax.set_xticks(xs)
    ax.set_xlabel('PC')
    ax.set_ylabel('Variance explained (%)')
    ax.set_title(f'{name} PCA — scree plot  ({n_feats} features)', fontsize=11)

    # bottom row: cumulative
    ax2 = axes[1, col]
    ax2.plot(xs, cumevr * 100, 'o-', color=color, ms=5)
    ax2.axhline(90, color='gray', ls='--', lw=1)
    ax2.axhline(95, color='gray', ls=':', lw=1)
    ax2.axvline(n90, color='blue', ls='--', lw=1, label=f'{n90} PCs → 90%')
    ax2.axvline(n95, color='red',  ls='--', lw=1, label=f'{n95} PCs → 95%')
    ax2.set_xticks(xs)
    ax2.set_ylim(0, 102)
    ax2.set_xlabel('Number of PCs kept')
    ax2.set_ylabel('Cumulative variance (%)')
    ax2.set_title(f'{name} PCA — cumulative variance', fontsize=11)
    ax2.legend(fontsize=9)

fig.suptitle(f'Separate PCAs: RAW bands vs CSD features  [{vintage}]', fontsize=12)
fig.tight_layout()
fig.savefig(figures_dir.joinpath('psd_pca_01_scree_comparison.png'), dpi=150)
plt.close(fig)
print("Saved: psd_pca_01_scree_comparison.png")

# %%  ── 6. Side-by-side loadings (top 3 PCs each) ────────────────────────
def shorten(name):
    return (name.replace('psd_', '').replace('_csd_diff1', '_d1')
                .replace('_csd', '_CSD'))

n_pc_show = 3
fig, axes = plt.subplots(n_pc_show, 2, figsize=(12, 3.2 * n_pc_show), sharey='col')

for col, (name, pca, feats, color) in enumerate([
    ('RAW', pca_raw, RAW_FEATURES, 'steelblue'),
    ('CSD', pca_csd, CSD_FEATURES, 'darkorange'),
]):
    evr = pca.explained_variance_ratio_
    labels = [shorten(f) for f in feats]
    for row in range(n_pc_show):
        ax = axes[row, col]
        loadings = pca.components_[row]
        bar_colors = ['crimson' if v >= 0 else 'steelblue' for v in loadings]
        ax.barh(range(len(feats)), loadings, color=bar_colors)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Loading weight')
        ax.set_title(f'{name}  PC{row+1}  ({evr[row]*100:.1f}% var)', fontsize=10)

fig.suptitle('PC loadings — what each PC is made of\nRed = positive, Blue = negative',
             fontsize=12)
fig.tight_layout()
fig.savefig(figures_dir.joinpath('psd_pca_02_loadings_comparison.png'), dpi=150)
plt.close(fig)
print("Saved: psd_pca_02_loadings_comparison.png")

# %%  ── 7. Side-by-side PC1 vs PC2 scatter ───────────────────────────────
def _clip_ax(ax, x, y, pct=(1, 99)):
    lox, hix = np.percentile(x, pct)
    loy, hiy = np.percentile(y, pct)
    px, py = (hix - lox) * 0.1, (hiy - loy) * 0.1
    ax.set_xlim(lox - px, hix + px)
    ax.set_ylim(loy - py, hiy + py)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for col, (name, pca, scores, color) in enumerate([
    ('RAW', pca_raw, scores_raw, 'steelblue'),
    ('CSD', pca_csd, scores_csd, 'darkorange'),
]):
    evr = pca.explained_variance_ratio_
    ax = axes[col]
    pc1_s = scores[idx_s, 0]
    pc2_s = scores[idx_s, 1]
    for rid in unique_ids:
        m = labels_s == rid
        if m.sum() < 5:
            continue
        ax.scatter(pc1_s[m], pc2_s[m], s=1.5, alpha=0.2,
                   color=id2color[rid], rasterized=True)
    _clip_ax(ax, pc1_s, pc2_s)
    ax.set_xlabel(f'PC1  ({evr[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2  ({evr[1]*100:.1f}% var)')
    ax.set_title(f'{name} PCA — PC1 vs PC2\n'
                 f'({n_plot:,} channels, clipped to 1–99th pct)', fontsize=10)
    ax.legend(handles=legend_handles, fontsize=7, loc='upper right', framealpha=0.8)

fig.suptitle(f'PC1 vs PC2 coloured by Cosmos region  [{vintage}]', fontsize=12)
fig.tight_layout()
fig.savefig(figures_dir.joinpath('psd_pca_03_scatter_comparison.png'), dpi=150)
plt.close(fig)
print("Saved: psd_pca_03_scatter_comparison.png")

# %%  ── 8. RAW PC1 vs CSD PC1 comparison scatter ────────────────────────
# Each dot is one channel.  X = its score on the first PC of the RAW PCA,
# Y = its score on the first PC of the CSD PCA.
# If the two PC1s are identical → dots lie on a diagonal line.
# Scatter off the diagonal = the two "total-power" summaries disagree,
# meaning CSD and RAW are picking up different things even in their top axis.
raw_pc1_s = scores_raw[idx_s, 0]
csd_pc1_s = scores_csd[idx_s, 0]

r_pc1 = float(np.corrcoef(scores_raw[:, 0], scores_csd[:, 0])[0, 1])
print(f"\nPearson r(RAW PC1, CSD PC1) = {r_pc1:.4f}")

fig, ax = plt.subplots(figsize=(8, 7))
for rid in unique_ids:
    m = labels_s == rid
    if m.sum() < 5:
        continue
    ax.scatter(raw_pc1_s[m], csd_pc1_s[m], s=2, alpha=0.25,
               color=id2color[rid], rasterized=True)

_clip_ax(ax, raw_pc1_s, csd_pc1_s)

# diagonal reference
xlim, ylim = ax.get_xlim(), ax.get_ylim()
diag = [max(xlim[0], ylim[0]), min(xlim[1], ylim[1])]
ax.plot(diag, diag, 'k--', lw=1, alpha=0.5)

ax.text(0.05, 0.95, f'r = {r_pc1:.3f}', transform=ax.transAxes,
        fontsize=12, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

ax.set_xlabel(f'RAW PC1  ({pca_raw.explained_variance_ratio_[0]*100:.1f}% of RAW variance)')
ax.set_ylabel(f'CSD PC1  ({pca_csd.explained_variance_ratio_[0]*100:.1f}% of CSD variance)')
ax.set_title(
    f'RAW PC1 vs CSD PC1 — {n_plot:,} channels  [{vintage}]\n'
    'Each axis is the "total power" summary of its group',
    fontsize=11,
)
ax.legend(handles=legend_handles, fontsize=8, loc='lower right', framealpha=0.8)

fig.tight_layout()
fig.savefig(figures_dir.joinpath('psd_pca_04_raw_pc1_vs_csd_pc1.png'), dpi=150)
plt.close(fig)
print("Saved: psd_pca_04_raw_pc1_vs_csd_pc1.png")

# %%  ── 10. Print summary ─────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────")
for name, pca, feats in [('RAW', pca_raw, RAW_FEATURES), ('CSD', pca_csd, CSD_FEATURES)]:
    evr = pca.explained_variance_ratio_
    cumevr = np.cumsum(evr)
    n90 = int(np.searchsorted(cumevr, 0.90)) + 1
    n95 = int(np.searchsorted(cumevr, 0.95)) + 1
    print(f"  {name} ({len(feats)} feat): PC1={evr[0]*100:.1f}%  "
          f"→ {n90} PCs for 90%,  {n95} PCs for 95%")
print("─────────────────────────────────────────────────────────────────────")
