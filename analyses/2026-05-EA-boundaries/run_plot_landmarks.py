"""Generate fig-3b-style landmark line-profile figures for all significant Cosmos boundaries."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import ephysatlas.anatomy
import ephysatlas.data

sys.path.insert(0, str(Path(__file__).parent))
from boundaries_utils import plot_landmark_line_profiles

VINTAGE = '2026_W24'
ROOT_PATH = Path.home().joinpath('data', 'ephys-atlas', 'features')
FIGURES_DIR = Path(__file__).parent.joinpath('figures')

path_features = ROOT_PATH.joinpath('ea_active', VINTAGE, 'agg_full')
brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
df_features = ephysatlas.data.read_features_from_disk(
    path_features, brain_atlas=brain_atlas, strict=False
)
print(f"Loaded {len(df_features):,} channels, {df_features.shape[1]} columns")

df_results = pd.read_csv(FIGURES_DIR.joinpath('boundary_feature_stats.csv'))

landmark_pairs = (
    df_results[(df_results['cohens_d'] > 0.8) & (df_results['pval_bonf'] < 0.01)]
    .groupby(['from', 'to'])['cohens_d'].max()
    .sort_values(ascending=False)
    .index.tolist()
)
print(f"Generating {len(landmark_pairs)} landmark figures …\n")

for from_acr, to_acr in landmark_pairs:
    print(f"  {from_acr} → {to_acr}")
    fig = plot_landmark_line_profiles(
        df_features,
        brain_atlas,
        from_acr,
        to_acr,
        df_results=df_results,
        window_um=500.0,
        depth_bin_um=25.0,
    )
    if fig is not None:
        slug = f'{from_acr}_to_{to_acr}'.replace(' ', '_')
        out = FIGURES_DIR.joinpath(f'landmark_{slug}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    → {out.name}")

print("\nDone.")
