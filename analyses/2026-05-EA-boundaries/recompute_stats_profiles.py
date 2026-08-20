"""Recompute boundary_feature_stats.csv and all profile figures with updated NNI (VS included)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from boundaries_utils import (
    compute_boundary_feature_stats,
    compute_feature_vlims,
    plot_boundary_feature_profiles,
    load_features,
)

VINTAGE = '2026_W24'
OUTPUT = Path(__file__).parent.joinpath('figures')
OUTPUT.mkdir(exist_ok=True)

# --- 1. Load features ---
df_features, brain_atlas, vintage = load_features(vintage=VINTAGE)

# --- 2. Recompute feature stats ---
print("\n=== Computing boundary feature stats ===")
df_results = compute_boundary_feature_stats(df_features, brain_atlas)
df_results.to_csv(OUTPUT.joinpath('boundary_feature_stats.csv'), index=False)
print(f"Saved boundary_feature_stats.csv  ({len(df_results)} rows)")

landmarks = df_results[(df_results['cohens_d'] > 0.8) & (df_results['pval_bonf'] < 0.01)]
landmark_pairs = (
    landmarks.groupby(['from', 'to'])['cohens_d'].max()
    .sort_values(ascending=False)
    .index.tolist()
)
print(f"\n{len(landmarks)} landmark candidates ({len(landmark_pairs)} unique boundaries):")
summary = (
    landmarks.groupby(['from', 'to'])
    .agg(best_d=('cohens_d', 'max'), n_trans=('n_trans', 'first'), n_probes=('n_probes', 'first'))
    .sort_values('best_d', ascending=False)
)
print(summary.to_string())

# --- 3. Remove stale profile figures for boundaries that no longer qualify ---
valid_slugs = {f'{f}_to_{t}' for f, t in landmark_pairs}
for png in OUTPUT.glob('profiles_*.png'):
    slug = png.stem.replace('profiles_', '')
    # Strip sort suffix (_ap, _ml, _rastermap)
    base_slug = slug
    for suffix in ('_ap', '_ml', '_rastermap'):
        if slug.endswith(suffix):
            base_slug = slug[:-len(suffix)]
            break
    if base_slug not in valid_slugs:
        png.unlink()
        print(f"Removed stale figure: {png.name}")

# --- 4. Compute global feature limits and scaler ---
print("\n=== Computing global feature vlims ===")
feature_vlims, feature_scaler, scaler_features = compute_feature_vlims(
    df_features, return_scaler=True
)

rastermap_cache_dir = OUTPUT.joinpath('rastermap_cache')
rastermap_cache_dir.mkdir(exist_ok=True)

# --- 5. Regenerate profile figures ---
print(f"\n=== Generating profiles for {len(landmark_pairs)} boundaries ===")
for from_acr, to_acr in landmark_pairs:
    slug = f'{from_acr}_to_{to_acr}'.replace(' ', '_')
    print(f"\n--- {from_acr} → {to_acr} ---")

    # Default depth sort (full window)
    fig = plot_boundary_feature_profiles(
        df_features, brain_atlas, from_acr, to_acr,
        df_results=df_results,
        window_um=1500,
        max_probes=150,
        feature_vlims=feature_vlims,
        sort='depth',
        feature_scaler=(feature_scaler, scaler_features),
        cache_dir=rastermap_cache_dir,
    )
    if fig is not None:
        fig.savefig(OUTPUT.joinpath(f'profiles_{slug}.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved profiles_{slug}.png")

    # AP, ML, rastermap sorts
    for sort_mode in ('ap', 'ml', 'rastermap'):
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
            fig.savefig(OUTPUT.joinpath(f'profiles_{slug}_{sort_mode}.png'), dpi=150)
            plt.close(fig)
            print(f"  Saved profiles_{slug}_{sort_mode}.png")

print(f"\nDone — {len(landmark_pairs)} boundaries × 4 sort modes.")