"""Generate profile figures for the 5 remaining landmark boundaries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from boundaries_utils import (
    compute_feature_vlims,
    plot_boundary_feature_profiles,
    load_features,
)

VINTAGE = '2026_W24'
OUTPUT = Path(__file__).parent.joinpath('figures')

df_features, brain_atlas, vintage = load_features(vintage=VINTAGE)

df_results = pd.read_csv(OUTPUT.joinpath('boundary_feature_stats.csv'))

remaining_pairs = [
    ('HPF', 'Isocortex'),
    ('HPF', 'fiber tracts'),
    ('fiber tracts', 'root'),
    ('MB', 'HPF'),
    ('CB', 'void_fluid'),
]

print("=== Computing global feature vlims ===")
feature_vlims, feature_scaler, scaler_features = compute_feature_vlims(
    df_features, return_scaler=True
)
rastermap_cache_dir = OUTPUT.joinpath('rastermap_cache')
rastermap_cache_dir.mkdir(exist_ok=True)

for from_acr, to_acr in remaining_pairs:
    slug = f'{from_acr}_to_{to_acr}'.replace(' ', '_')
    print(f"\n--- {from_acr} → {to_acr} ---")
    for sort_mode in ('depth', 'ap', 'ml', 'rastermap'):
        suffix = '' if sort_mode == 'depth' else f'_{sort_mode}'
        out_path = OUTPUT.joinpath(f'profiles_{slug}{suffix}.png')
        fig = plot_boundary_feature_profiles(
            df_features, brain_atlas, from_acr, to_acr,
            df_results=df_results,
            window_um=1500,
            max_probes=150,
            feature_vlims=feature_vlims,
            sort=sort_mode,
            feature_scaler=(feature_scaler, scaler_features),
            cache_dir=rastermap_cache_dir,
        )
        if fig is not None:
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  Saved {out_path.name}")
        else:
            print(f"  No crossings for {from_acr} → {to_acr}, skipping")
            break

print("\nDone.")
