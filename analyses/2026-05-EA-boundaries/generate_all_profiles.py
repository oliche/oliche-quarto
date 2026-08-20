"""Generate depth-profile heatmaps for all Cosmos boundary pairs with count >= 50.

Features: all columns after PSD/CSD PCA reduction (plan step 2).
Sort: transition depth (shallow → deep along x-axis).
Output: figures/profiles_<from>_to_<to>.png
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import boundaries_utils as bu
import ephysatlas.anatomy
import ephysatlas.data
from ephysatlas.features import psd_pca_dataframe

VINTAGE = '2026_W24'
FIGURES = Path(__file__).parent.joinpath('figures')
CACHE_DIR = Path.home().joinpath('data', 'ephys-atlas', 'features', 'ea_active', VINTAGE, 'agg_full')
MIN_COUNT = 50


def main():
    # ── 1. Load features ──────────────────────────────────────────────────────
    print("Loading features …")
    brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
    df_features = ephysatlas.data.read_features_from_disk(
        CACHE_DIR, brain_atlas=brain_atlas, strict=False
    )
    print(f"Loaded {len(df_features):,} channels, {df_features.shape[1]} columns")

    # ── 2. PSD/CSD PCA (plan step 2) ─────────────────────────────────────────
    print("Applying PSD/CSD PCA …")
    df_features = psd_pca_dataframe(df_features, n_components_psd=3, n_components_csd=3)

    # ── 3. Collect all feature columns ───────────────────────────────────────
    all_feat_cols = tuple(c for c in df_features.columns if c not in bu.EXCLUDE_COLS)
    print(f"{len(all_feat_cols)} feature columns: {list(all_feat_cols)}")

    # ── 4. Compute global colour limits ──────────────────────────────────────
    print("Computing global vlims …")
    vlims = bu.compute_feature_vlims(df_features, features=list(all_feat_cols))

    # ── 5. Boundary pairs (count >= MIN_COUNT) ────────────────────────────────
    count_matrix = pd.read_csv(
        FIGURES.joinpath(f'cosmos_transition_matrix_{VINTAGE}.csv'), index_col=0
    )
    pairs = [
        (r, c, int(count_matrix.loc[r, c]))
        for r in count_matrix.index
        for c in count_matrix.columns
        if count_matrix.loc[r, c] >= MIN_COUNT
    ]
    pairs.sort(key=lambda x: -x[2])
    print(f"\n{len(pairs)} boundary pairs with count >= {MIN_COUNT}:")
    for r, c, n in pairs:
        print(f"  {r} → {c}: {n}")

    # ── 6. Plot all profile figures ───────────────────────────────────────────
    for from_acr, to_acr, count in pairs:
        fname = FIGURES.joinpath(f'profiles_{from_acr}_to_{to_acr}.png')
        print(f"\n[{count:>4d}] {from_acr} → {to_acr}  →  {fname.name}")
        fig = bu.plot_boundary_feature_profiles(
            df_features,
            brain_atlas,
            from_acr=from_acr,
            to_acr=to_acr,
            df_results=None,
            mandatory_features=all_feat_cols,
            window_um=None,
            depth_bin_um=20.0,
            max_probes=150,
            feature_vlims=vlims,
            sort='depth',
        )
        if fig is None:
            print(f"  → skipped (no crossings found)")
            continue
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  → saved")

    print("\nDone.")


if __name__ == '__main__':
    main()
