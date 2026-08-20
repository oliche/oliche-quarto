# %%
"""
Ephys-atlas region boundary landmark discovery.

Brute-force search for sharp ephys feature transitions across cosmos/beryl
region boundaries that can serve as anatomical landmarks.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from one.api import ONE
import ephysatlas.data
import ephysatlas.anatomy

sns.set_theme(context="talk", palette="muted", style="whitegrid")

PROJECT = 'ea_active'
root_path_features = Path.home() / 'data' / 'ephys-atlas' / 'features'

# %%  Load features
one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
VINTAGE = ephysatlas.data.get_latest_label(one=one, project=PROJECT)
print(f"Using vintage: {VINTAGE}")

path_features = root_path_features / PROJECT / VINTAGE / 'agg_full'
if not path_features.exists():
    ephysatlas.data.download_tables(root_path_features, label=VINTAGE, one=one)

brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
df_features = ephysatlas.data.read_features_from_disk(path_features, brain_atlas=brain_atlas, strict=False)

print(f"Loaded {len(df_features):,} channels, {df_features.shape[1]} columns")

# %% Build Cosmos-level boundary transition count matrix
regions = brain_atlas.regions

# Refine root (997): reclassify by Allen_id into fiber tracts (1009), VS (73), or root (997)
ft_ids = set(regions.subtree(1009)['id']) - {997}
vs_ids = set(regions.subtree(73)['id']) - {997}
allen_int = df_features['Allen_id'].astype(int)
root_mask = df_features['Cosmos_id'] == 997
df_features['Cosmos_refined_id'] = df_features['Cosmos_id'].copy()
df_features.loc[root_mask & allen_int.isin(ft_ids), 'Cosmos_refined_id'] = 1009
df_features.loc[root_mask & allen_int.isin(vs_ids), 'Cosmos_refined_id'] = 73

cosmos_ids = sorted(df_features['Cosmos_refined_id'].unique())
id2acr = {cid: regions.acronym[regions.id2index(cid)[1]][0][0] for cid in cosmos_ids}
print("Refined Cosmos regions:", {cid: id2acr[cid] for cid in cosmos_ids})

# Aggregate channels to one row per (pid, depth): modal refined Cosmos_id at each depth
df_pid = df_features[['axial_um', 'Cosmos_refined_id']].reset_index(level='pid')
df_depth = (
    df_pid.groupby(['pid', 'axial_um'])['Cosmos_refined_id']
    .agg(lambda x: x.mode()[0])
    .reset_index()
    .sort_values(['pid', 'axial_um'])
)

# Nearest-neighbour interpolation of interior root/void channels.
# Channels labelled root (997) or void-type regions that sit between two real
# brain regions cause spurious A→root→B double-hops.  Replace them with the
# nearest valid label along the probe.  Channels at the probe extremities
# (before the first or after the last valid label) are left unchanged so that
# surface transitions (e.g. void_fluid→Isocortex) are preserved.
INTERP_IDS = frozenset(
    cid for cid, acr in id2acr.items()
    if any(kw in acr.lower() for kw in ('root', 'void', 'fiber', 'vs'))
)
print(f"Interpolating interior labels for IDs: {INTERP_IDS}")

# Nearest-neighbour interpolation — fully vectorised.
# df_depth is sorted by (pid, axial_um).
ids = df_depth['Cosmos_refined_id'].values.copy().astype(float)
pids = df_depth['pid'].values
bad = np.isin(ids, list(INTERP_IDS))

# Null out bad positions so we can use pandas ffill/bfill within probes
df_fill = pd.DataFrame({'pid': pids, 'ids': np.where(bad, np.nan, ids)})
ids_ff = df_fill.groupby('pid')['ids'].ffill().values   # forward fill within probe
ids_bf = df_fill.groupby('pid')['ids'].bfill().values   # backward fill within probe

# Distance to the nearest valid position (index distance, same probe)
pos = np.arange(len(ids))
df_pos = pd.DataFrame({'pid': pids, 'pos': np.where(bad, np.nan, pos.astype(float))})
pos_ff = df_pos.groupby('pid')['pos'].ffill().values    # index of last valid before i
pos_bf = df_pos.groupby('pid')['pos'].bfill().values    # index of next valid after i
dist_prev = pos - pos_ff   # distance to previous valid (NaN if none in this probe)
dist_next = pos_bf - pos   # distance to next valid (NaN if none in this probe)

# Interior bad: both sides have a valid neighbour → pick nearer one.
# Extremity bad (only one side filled) → keep original label.
interior = bad & ~np.isnan(ids_ff) & ~np.isnan(ids_bf)
use_ff = interior & (dist_prev <= dist_next)
use_bf = interior & (dist_prev > dist_next)

new_ids = ids.copy()
new_ids[use_ff] = ids_ff[use_ff]
new_ids[use_bf] = ids_bf[use_bf]

orig_ids = df_depth['Cosmos_refined_id'].values.copy()
df_depth = df_depth.copy()
df_depth['Cosmos_refined_id'] = new_ids.astype(int)
n_interp = (df_depth['Cosmos_refined_id'].values != orig_ids).sum()
print(f"Interpolated {n_interp:,} interior depth positions")

# Detect transitions between adjacent depth levels (shallow → deep = upper → lower)
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

print(count_matrix)

output_fig_path = Path.home().joinpath('Documents', 'PYTHON', 'oliche-quarto', 'analyses', '2026-05-EA-boundaries', 'figures')
output_fig_path.mkdir(exist_ok=True)
count_matrix.to_csv(output_fig_path.joinpath(f'cosmos_transition_matrix_{VINTAGE}.csv'))
print(f"Saved transition matrix CSV for {VINTAGE}")

# %% Graph of high-count transitions
from boundaries_utils import plot_transition_graph

fig = plot_transition_graph(
    count_matrix,
    brain_atlas,
    min_count=50,
    vintage=VINTAGE,
    output_fig_path=output_fig_path,
)
if fig is not None:
    plt.show()
    plt.close(fig)

# %% Heatmap of the transition count matrix
# Burn the diagonal: set to NaN so it renders as a distinct color
plot_matrix = count_matrix.astype(float).values.copy()
np.fill_diagonal(plot_matrix, np.nan)
plot_matrix = pd.DataFrame(plot_matrix, index=count_matrix.index, columns=count_matrix.columns)

# Color range from off-diagonal values only
off_diag = plot_matrix.values[~np.isnan(plot_matrix.values)]
vmax = np.percentile(off_diag, 97)

cmap = plt.cm.YlOrRd.copy()
cmap.set_bad(color='#cccccc')  # diagonal shown in grey

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
ax.set_title(f'Cosmos boundary transitions ({VINTAGE})', pad=12)
ax.set_xlabel('upper region')
ax.set_ylabel('lower region')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)
fig.tight_layout()
fig.savefig(output_fig_path.joinpath('cosmos_transition_matrix.png'), dpi=150)
plt.close()

# %% Step 4: Find sharp feature transitions across each Cosmos boundary
from scipy import stats

WINDOW_UM = 200   # µm on each side of each boundary
MIN_TRANSITIONS = 16  # minimum depth-level transitions required for a boundary pair

EXCLUDE_COLS = {
    'axial_um', 'lateral_um', 'x', 'y', 'z', 'x_target', 'y_target', 'z_target',
    'acronym', 'atlas_id', 'Allen_id', 'Cosmos_id', 'Cosmos_refined_id', 'Beryl_id', 'outside',
    'channel_labels', 'spike_count', 'polarity', 'decay_n_peaks',
    'decay_fit_r_squared', 'decay_fit_error',
}
feature_cols = [c for c in df_features.columns if c not in EXCLUDE_COLS]
print(f"{len(feature_cols)} features to test")

# Build transition events: (pid, trans_depth, from_cosmos, to_cosmos)
df_trans_events = (
    df_depth.assign(to_cosmos=df_depth.groupby('pid')['Cosmos_refined_id'].shift(-1))
    .dropna(subset=['to_cosmos'])
    .pipe(lambda d: d[d['to_cosmos'] != d['Cosmos_refined_id']].copy())
)
df_trans_events['to_cosmos'] = df_trans_events['to_cosmos'].astype(int)
df_trans_events = df_trans_events.rename(columns={'Cosmos_refined_id': 'from_cosmos', 'axial_um': 'trans_depth'})
pair_counts = df_trans_events.groupby(['from_cosmos', 'to_cosmos']).size()

df_feat_reset = df_features[['axial_um', 'Cosmos_refined_id'] + feature_cols].reset_index(level='pid')

results = []
for (from_c, to_c), n_trans in pair_counts.items():
    if n_trans < MIN_TRANSITIONS:
        continue

    pair_events = df_trans_events.loc[
        (df_trans_events['from_cosmos'] == from_c) & (df_trans_events['to_cosmos'] == to_c),
        ['pid', 'trans_depth'],
    ]
    pids = pair_events['pid'].unique()
    merged = pair_events.merge(df_feat_reset[df_feat_reset['pid'].isin(pids)], on='pid')

    before = merged[
        (merged['Cosmos_refined_id'] == from_c) &
        (merged['axial_um'] >= merged['trans_depth'] - WINDOW_UM) &
        (merged['axial_um'] <= merged['trans_depth'])
    ]
    after = merged[
        (merged['Cosmos_refined_id'] == to_c) &
        (merged['axial_um'] >= merged['trans_depth']) &
        (merged['axial_um'] <= merged['trans_depth'] + WINDOW_UM)
    ]

    for feat in feature_cols:
        a = before[feat].dropna().values
        b = after[feat].dropna().values
        if len(a) < 10 or len(b) < 10:
            continue
        _, pval = stats.mannwhitneyu(a, b, alternative='two-sided')
        pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        d = abs(np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0
        results.append({
            'from': id2acr[from_c], 'to': id2acr[to_c],
            'feature': feat, 'n_trans': int(n_trans), 'n_probes': len(pids),
            'cohens_d': round(d, 4), 'pval': pval,
        })

df_results = pd.DataFrame(results)
df_results['pval_bonf'] = (df_results['pval'] * len(df_results)).clip(upper=1.0)
df_results = df_results.sort_values('cohens_d', ascending=False).reset_index(drop=True)

landmarks = df_results[(df_results['cohens_d'] > 0.8) & (df_results['pval_bonf'] < 0.01)]
print(f"\n{len(landmarks)} landmark candidates (Cohen's d > 0.8, Bonferroni p < 0.01):")
print(landmarks.head(40).to_string())
df_results.to_csv(output_fig_path.joinpath('boundary_feature_stats.csv'), index=False)

# %% Step 5: Multi-feature depth profiles aligned to each Cosmos boundary
# Layout inspired by figure_01_features_with_histology_columns in ephysatlas/reveal.py:
# depth on y-axis, narrow region-colour strip on the left, one column per feature.
try:
    import addcopyfighandler  # noqa: F401
except Exception:
    pass

BIN_WIDTH = 25   # µm — binning resolution along depth axis
MAX_FEATS = 6    # max feature columns per figure
D_THRESH_PLOT = 0.8  # minimum Cohen's d to include feature in plot

acr2id = {v: k for k, v in id2acr.items()}
fig_dir_home = Path.home().joinpath('Documents', 'figures')
fig_dir_home.mkdir(exist_ok=True)

# Fallback colours for custom region IDs not in the Allen atlas
_CUSTOM_COLORS = {0: (0.55, 0.55, 0.55), 2000: (0.60, 0.82, 0.93)}


def _region_color(cid):
    """Return (r, g, b) floats for a Cosmos region id."""
    if cid in _CUSTOM_COLORS:
        return _CUSTOM_COLORS[cid]
    try:
        return tuple(regions.get(np.array([cid])).rgb[0].astype(float) / 255)
    except Exception:
        return (0.5, 0.5, 0.5)


def _text_color(bg_rgb):
    """Return 'white' or 'black' for maximum contrast on bg_rgb."""
    r, g, b = bg_rgb
    return 'white' if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else 'black'


# All boundary pairs that have at least one feature above the landmark threshold
landmark_pairs = (
    df_results[(df_results['cohens_d'] > D_THRESH_PLOT) & (df_results['pval_bonf'] < 0.01)]
    .groupby(['from', 'to'])['cohens_d'].max()
    .sort_values(ascending=False)
    .index.tolist()
)
print(f"\nPlotting {len(landmark_pairs)} landmark boundaries …")

for from_acr, to_acr in landmark_pairs:
    from_c, to_c = acr2id[from_acr], acr2id[to_acr]
    col_from = _region_color(from_c)
    col_to = _region_color(to_c)

    # Features significant for this boundary, sorted by Cohen's d descending
    pair_feats = (
        df_results[
            (df_results['from'] == from_acr) & (df_results['to'] == to_acr) &
            (df_results['cohens_d'] > D_THRESH_PLOT) & (df_results['pval_bonf'] < 0.01)
        ]
        .sort_values('cohens_d', ascending=False)
        .head(MAX_FEATS)
    )
    feats = pair_feats['feature'].tolist()

    # Gather and align all channels for this boundary
    pair_events = df_trans_events.loc[
        (df_trans_events['from_cosmos'] == from_c) & (df_trans_events['to_cosmos'] == to_c),
        ['pid', 'trans_depth'],
    ]
    pids = pair_events['pid'].unique()
    n_trans, n_probes = len(pair_events), len(pids)

    feat_sub = df_feat_reset[df_feat_reset['pid'].isin(pids)][
        ['pid', 'axial_um', 'Cosmos_refined_id'] + feats
    ].copy()
    merged = pair_events.merge(feat_sub, on='pid')
    merged['rel_depth'] = merged['axial_um'] - merged['trans_depth']
    in_window = merged[merged['rel_depth'].abs() <= WINDOW_UM].copy()
    in_window['bin'] = (in_window['rel_depth'] // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2

    n_feats = len(feats)
    fig, axes = plt.subplots(
        1, n_feats + 1,
        figsize=(1.4 + 2.2 * n_feats, 7),
        sharey=True,
        gridspec_kw={'width_ratios': [0.28] + [1] * n_feats, 'wspace': 0.06},
    )
    fig.suptitle(
        f'{from_acr}  →  {to_acr}    ({n_trans} transitions, {n_probes} probes, {VINTAGE})',
        fontsize=11, y=1.01,
    )

    # --- Leftmost column: region colour strip ---
    ax0 = axes[0]
    ax0.fill_betweenx([-WINDOW_UM, 0], 0, 1, color=col_from, alpha=0.9)
    ax0.fill_betweenx([0, WINDOW_UM], 0, 1, color=col_to, alpha=0.9)
    ax0.axhline(0, color='k', lw=1.2)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(-WINDOW_UM, WINDOW_UM)
    ax0.set_xticks([])
    ax0.set_ylabel(
        '← lower/deeper    ·    boundary    ·    upper/superficial →',
        fontsize=8, labelpad=6,
    )
    ax0.text(0.5, -WINDOW_UM * 0.5, from_acr, ha='center', va='center', fontsize=9,
             color=_text_color(col_from), fontweight='bold', rotation=90)
    ax0.text(0.5, WINDOW_UM * 0.5, to_acr, ha='center', va='center', fontsize=9,
             color=_text_color(col_to), fontweight='bold', rotation=90)
    sns.despine(ax=ax0, bottom=True, left=True, right=True, top=True)

    # --- Feature columns ---
    for ax, feat in zip(axes[1:], feats):
        d_val = pair_feats.loc[pair_feats['feature'] == feat, 'cohens_d'].values[0]
        binned = in_window.groupby('bin')[feat].agg(['mean', 'sem']).reset_index()

        # rel_depth ≤ 0 → "from" (lower) region colour; > 0 → "to" (upper) region colour
        for mask, color in [(binned['bin'] <= 0, col_from), (binned['bin'] > 0, col_to)]:
            seg = binned[mask]
            ax.plot(seg['mean'], seg['bin'], color=color, lw=2)
            ax.fill_betweenx(
                seg['bin'],
                seg['mean'] - seg['sem'],
                seg['mean'] + seg['sem'],
                alpha=0.35, color=color,
            )

        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.65)
        ax.set_xlabel(feat.replace('_', '\n'), fontsize=8)
        ax.set_title(f'd = {d_val:.2f}', fontsize=8, pad=3)
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        sns.despine(ax=ax, left=True)

    fig.tight_layout()
    slug = f'{from_acr}_to_{to_acr}'.replace(' ', '_')
    fig_name = f'2026-05-23_landmark_{slug}.png'
    fig.savefig(fig_dir_home.joinpath(fig_name), dpi=150, bbox_inches='tight')
    fig.savefig(output_fig_path.joinpath(f'landmark_{slug}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_name}")

print(f"Done — {len(landmark_pairs)} boundary figures.")

# %% Step 6: Cross-validation — 50/50 probe split per boundary pair
# For every candidate that passes the full-data threshold, re-test in two independent halves.
# Robust landmarks should show similar Cohen's d in both folds.

RNG_SEED = 42
D_THRESH = 0.5   # relaxed threshold per fold (full-data d > 0.8 already gating entry)
P_THRESH = 0.05


def _cohens_d_mwu(a, b):
    """Return (cohens_d, pval) for two 1-D arrays; (nan, nan) if too few samples."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    _, pval = stats.mannwhitneyu(a, b, alternative='two-sided')
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = abs(np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
    return d, pval


# Work on all candidates from the full scan
candidates = df_results[(df_results['cohens_d'] > 0.8) & (df_results['pval_bonf'] < 0.01)].copy()

rng = np.random.default_rng(RNG_SEED)
cv_rows = []

for _, row in candidates.iterrows():
    from_acr, to_acr, feat = row['from'], row['to'], row['feature']
    from_c, to_c = acr2id[from_acr], acr2id[to_acr]

    pair_events = df_trans_events.loc[
        (df_trans_events['from_cosmos'] == from_c) & (df_trans_events['to_cosmos'] == to_c),
        ['pid', 'trans_depth'],
    ]
    pids = pair_events['pid'].unique()
    if len(pids) < 4:
        continue

    # Random 50/50 split on probe IDs
    shuffled = rng.permutation(pids)
    fold_a_pids = set(shuffled[:len(shuffled) // 2])
    fold_b_pids = set(shuffled[len(shuffled) // 2:])

    fold_ds = []
    fold_ps = []
    for fold_pids in (fold_a_pids, fold_b_pids):
        ev = pair_events[pair_events['pid'].isin(fold_pids)]
        feat_sub = df_feat_reset[df_feat_reset['pid'].isin(fold_pids)][['pid', 'axial_um', 'Cosmos_refined_id', feat]]
        m = ev.merge(feat_sub, on='pid')
        before_vals = m.loc[
            (m['Cosmos_refined_id'] == from_c) &
            (m['axial_um'] >= m['trans_depth'] - WINDOW_UM) &
            (m['axial_um'] <= m['trans_depth']), feat
        ].values
        after_vals = m.loc[
            (m['Cosmos_refined_id'] == to_c) &
            (m['axial_um'] >= m['trans_depth']) &
            (m['axial_um'] <= m['trans_depth'] + WINDOW_UM), feat
        ].values
        d, p = _cohens_d_mwu(before_vals, after_vals)
        fold_ds.append(d)
        fold_ps.append(p)

    cv_rows.append({
        'from': from_acr, 'to': to_acr, 'feature': feat,
        'cohens_d_full': row['cohens_d'],
        'cohens_d_A': fold_ds[0], 'cohens_d_B': fold_ds[1],
        'pval_A': fold_ps[0], 'pval_B': fold_ps[1],
        'n_probes': len(pids),
        'replicated': (
            not np.isnan(fold_ds[0]) and not np.isnan(fold_ds[1]) and
            fold_ds[0] > D_THRESH and fold_ds[1] > D_THRESH and
            fold_ps[0] < P_THRESH and fold_ps[1] < P_THRESH
        ),
    })

df_cv = pd.DataFrame(cv_rows)
print(f"\nCross-validation results ({df_cv['replicated'].sum()}/{len(df_cv)} replicated in both folds):")
print(df_cv.to_string(index=False))
df_cv.to_csv(output_fig_path.joinpath('boundary_cv_results.csv'), index=False)

# Scatter: fold A vs fold B Cohen's d, coloured by boundary pair
fig, ax = plt.subplots(figsize=(7, 6))
boundaries = df_cv[['from', 'to']].drop_duplicates()
palette_cv = sns.color_palette('tab10', n_colors=len(boundaries))
col_map = {(r['from'], r['to']): c for (_, r), c in zip(boundaries.iterrows(), palette_cv)}

seen_boundaries = set()
for _, row in df_cv.iterrows():
    key = (row['from'], row['to'])
    color = col_map[key]
    marker = 'o' if row['replicated'] else 'x'
    label = f"{key[0]}→{key[1]}" if key not in seen_boundaries else ''
    ax.scatter(row['cohens_d_A'], row['cohens_d_B'], color=color, marker=marker,
               s=80, zorder=3, label=label)
    seen_boundaries.add(key)

# Threshold lines and identity line
d_max = df_cv[['cohens_d_A', 'cohens_d_B']].max().max() * 1.05
ax.axhline(D_THRESH, color='k', lw=0.8, ls='--', alpha=0.5)
ax.axvline(D_THRESH, color='k', lw=0.8, ls='--', alpha=0.5)
ax.plot([0, d_max], [0, d_max], color='grey', lw=0.6, ls=':')

# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# rebuild legend from col_map so every boundary appears once
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=f"{k[0]}→{k[1]}")
    for k, c in col_map.items()
]
ax.legend(handles=legend_handles, fontsize=8, title='boundary', loc='lower right')
ax.set_xlabel("Cohen's d — fold A")
ax.set_ylabel("Cohen's d — fold B")
ax.set_title(f'Cross-validation: 50/50 probe split ({VINTAGE})\n'
             f'o replicated (d>{D_THRESH} & p<{P_THRESH} in both folds)   x did not replicate')
fig.tight_layout()
fig_name_cv = '2026-05-23_landmark_crossval.png'
fig.savefig(fig_dir_home.joinpath(fig_name_cv), dpi=150)
fig.savefig(output_fig_path.joinpath('landmark_crossval.png'), dpi=150)
plt.close()
print(f"Saved {fig_name_cv}")

# %% Step 5b: Three probe-sort versions of feature-profile figures (AP, ML, Rastermap)
from boundaries_utils import (
    compute_feature_vlims,
    plot_boundary_feature_profiles,
)

feature_vlims, feature_scaler, scaler_features = compute_feature_vlims(
    df_features, return_scaler=True
)
rastermap_cache_dir = output_fig_path.joinpath('rastermap_cache')
rastermap_cache_dir.mkdir(exist_ok=True)

for from_acr, to_acr in landmark_pairs:
    for sort_mode in ('ap', 'ml', 'rastermap'):
        fig = plot_boundary_feature_profiles(
            df_features,
            brain_atlas,
            from_acr,
            to_acr,
            df_results=df_results,
            window_um=1500,
            max_probes=150,
            feature_vlims=feature_vlims,
            sort=sort_mode,
            feature_scaler=(feature_scaler, scaler_features),
            cache_dir=rastermap_cache_dir,
        )
        if fig is not None:
            slug = f'{from_acr}_to_{to_acr}'
            out_path = output_fig_path.joinpath(f'profiles_{slug}_{sort_mode}.png')
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  Saved profiles_{slug}_{sort_mode}.png")

print(f"Done — {len(landmark_pairs)} boundaries × 3 sort modes.")
