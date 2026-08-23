# %%
"""
PCA-based boundary depth-profile display.

For each Cosmos-level boundary with ≥42 transitions (excluding void, root,
fiber tracts, and VS), plots aligned depth profiles of the first 6 PC scores
alongside the brain-region colour (histology) row.

Each figure: 7 rows (histology + PC1–PC6), one column = one probe.
All probes are aligned so that depth=0 is the detected boundary transition.
"""
import sys
from pathlib import Path

if (_here := '/Users/olivier/Documents/oliche-quarto/analyses/2026-05-EA-boundaries') not in sys.path:
    sys.path.insert(0, _here)

import addcopyfighandler  # noqa: F401
import pandas as pd
import seaborn as sns

from boundaries_utils import (
    compute_pca_features,
    load_features,
    plot_boundary_pca_profiles,
)

sns.set_theme(context='notebook', palette='muted', style='whitegrid')

fig_dir = Path(__file__).parent.joinpath('figures')
fig_dir.mkdir(exist_ok=True)

# %% Load features and compute PCA
df_features, brain_atlas, VINTAGE = load_features()

N_PCS = 6
df_pca, pca, scaler, feature_cols = compute_pca_features(df_features, n_pcs=N_PCS)
print(f"PCA ready: {N_PCS} components, {df_pca.shape[0]:,} channels")

# %% Identify qualifying boundaries
EXCLUDE_REGIONS = {'void_fluid', 'void', 'root', 'fiber tracts', 'VS', 'ft'}
MIN_TRANSITIONS = 42

stats = pd.read_csv(fig_dir.joinpath('boundary_feature_stats.csv'))
boundaries = (
    stats[['from', 'to', 'n_trans', 'n_probes']]
    .drop_duplicates()
    .query('n_trans >= @MIN_TRANSITIONS')
)
boundaries = boundaries[
    ~boundaries['from'].isin(EXCLUDE_REGIONS) & ~boundaries['to'].isin(EXCLUDE_REGIONS)
].sort_values('n_trans', ascending=False)

print(f"\nQualifying boundaries (n_trans ≥ {MIN_TRANSITIONS}, excluding void/root/ft/VS):")
print(boundaries.to_string(index=False))

# %% Plot one figure per qualifying boundary
for _, row in boundaries.iterrows():
    from_acr = row['from']
    to_acr = row['to']
    n_trans = int(row['n_trans'])

    print(f"\nPlotting {from_acr} → {to_acr}  ({n_trans} transitions)…")
    fig = plot_boundary_pca_profiles(
        df_pca=df_pca,
        df_features=df_features,
        brain_atlas=brain_atlas,
        from_acr=from_acr,
        to_acr=to_acr,
        n_pcs=N_PCS,
        window_um=200.0,
        depth_bin_um=20.0,
    )
    if fig is None:
        continue

    fname = f'pca_boundary_{from_acr}_to_{to_acr}.png'
    fig.savefig(fig_dir.joinpath(fname), dpi=150, bbox_inches='tight')
    fig.savefig(
        Path.home().joinpath('Documents', 'figures', f'2026-06-08_{fname}'),
        dpi=150, bbox_inches='tight',
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Saved {fname}")
