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
# Map Cosmos_id -> acronym for readable labels
regions = brain_atlas.regions
cosmos_ids = sorted(df_features['Cosmos_id'].unique())
id2acr = {cid: regions.acronym[regions.id2index(cid)[1]][0][0] for cid in cosmos_ids}

# Aggregate channels to one row per (pid, depth): modal Cosmos_id at each depth
df_pid = df_features[['axial_um', 'Cosmos_id']].reset_index(level='pid')
df_depth = (
    df_pid.groupby(['pid', 'axial_um'])['Cosmos_id']
    .agg(lambda x: x.mode()[0])
    .reset_index()
    .sort_values(['pid', 'axial_um'])
)

# Detect transitions: adjacent depth levels on the same probe with different Cosmos_id
next_cosmos = df_depth.groupby('pid')['Cosmos_id'].shift(-1)
mask = next_cosmos.notna() & (next_cosmos != df_depth['Cosmos_id'])
transitions = pd.DataFrame({
    'from': df_depth.loc[mask, 'Cosmos_id'].astype(int).values,
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

# %% Heatmap of the transition count matrix
output_fig_path = Path('/home/olivier/Documents/PYTHON/oliche-quarto/analyses/2026-05-EA-boundaries/figures')
output_fig_path.mkdir(exist_ok=True)

# Burn the diagonal: set to NaN so it renders as a distinct color
plot_matrix = count_matrix.astype(float).values.copy()
np.fill_diagonal(plot_matrix, np.nan)
plot_matrix = pd.DataFrame(plot_matrix, index=count_matrix.index, columns=count_matrix.columns)

# Color range from off-diagonal values only
off_diag = plot_matrix.values[~np.isnan(plot_matrix.values)]
vmax = np.percentile(off_diag, 97)

cmap = plt.cm.YlOrRd.copy()
cmap.set_bad(color='#cccccc')  # diagonal shown in grey

fig, ax = plt.subplots(figsize=(9, 8))
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
ax.set_xlabel('to region')
ax.set_ylabel('from region')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)
fig.tight_layout()
fig.savefig(output_fig_path / 'cosmos_transition_matrix.png', dpi=150)
plt.show()

# %% Step 4: Find sharp feature transitions across each Cosmos boundary
from scipy import stats

WINDOW_UM = 200   # µm on each side of each boundary
MIN_TRANSITIONS = 30  # minimum depth-level transitions required for a boundary pair

EXCLUDE_COLS = {
    'axial_um', 'lateral_um', 'x', 'y', 'z', 'x_target', 'y_target', 'z_target',
    'acronym', 'atlas_id', 'Allen_id', 'Cosmos_id', 'Beryl_id', 'outside',
    'channel_labels', 'spike_count', 'polarity', 'decay_n_peaks',
    'decay_fit_r_squared', 'decay_fit_error',
}
feature_cols = [c for c in df_features.columns if c not in EXCLUDE_COLS]
print(f"{len(feature_cols)} features to test")

# Build transition events: (pid, trans_depth, from_cosmos, to_cosmos)
df_trans_events = (
    df_depth.assign(to_cosmos=df_depth.groupby('pid')['Cosmos_id'].shift(-1))
    .dropna(subset=['to_cosmos'])
    .pipe(lambda d: d[d['to_cosmos'] != d['Cosmos_id']].copy())
)
df_trans_events['to_cosmos'] = df_trans_events['to_cosmos'].astype(int)
df_trans_events = df_trans_events.rename(columns={'Cosmos_id': 'from_cosmos', 'axial_um': 'trans_depth'})
pair_counts = df_trans_events.groupby(['from_cosmos', 'to_cosmos']).size()

df_feat_reset = df_features[['axial_um', 'Cosmos_id'] + feature_cols].reset_index(level='pid')

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
        (merged['Cosmos_id'] == from_c) &
        (merged['axial_um'] >= merged['trans_depth'] - WINDOW_UM) &
        (merged['axial_um'] <= merged['trans_depth'])
    ]
    after = merged[
        (merged['Cosmos_id'] == to_c) &
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
df_results.to_csv(output_fig_path / 'boundary_feature_stats.csv', index=False)
