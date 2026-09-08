"""Whole-brain dense-grid landmark discovery on the encoding volume.

Reverses the original discovery direction: instead of scanning measured channel
features at real probe insertions (capped at 18 Cosmos pairs with >=32 real
crossings), this samples the brainwide encoding volume on a dense 100 um AP x ML
virtual-probe grid and runs the *same* Cohen's-d boundary scan
(``compute_boundary_feature_stats``) used for the original measured-data
landmark search. The dense grid removes the real-probe sparsity floor, so every
Cosmos pair with enough virtual crossings becomes scannable, not just the
previously qualifying ones.

Reuses ``build_virtual_probe_df`` from ``boundary_classifier_volume.py`` (grid
sampling + PCA projection) and ``compute_cosmos_transitions`` /
``compute_boundary_feature_stats`` from ``boundaries_utils.py`` (transition
detection + effect-size ranking) unchanged.

Run with: .venv/bin/python run_volume_landmark_scan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ephysatlas.anatomy

sys.path.insert(0, str(Path(__file__).parent))
import boundaries_utils as bu
from boundary_classifier_volume import build_virtual_probe_df

PROJECT = 'ea_active'
VOL_LABEL = '2026_W26'          # latest vintage with an encoding volume on S3
RES_UM = 50                      # only resolution available for VOL_LABEL
FEATURES_VINTAGE = '2026_W26'    # matched measured-feature vintage (sanity-check step)
CSD_VARIANT = 'diff1'            # this volume carries *_csd_diff1, not plain *_csd

GRID_SPACING_UM = 100.0          # AP/ML virtual-probe grid spacing
WINDOW_UM = 200.0                # boundary-scan window (matches original measured-data scan)
MIN_TRANSITIONS = 30             # floor for compute_boundary_feature_stats
OLD_REAL_DATA_FLOOR = 32         # previous real-probe qualifying threshold, for QC comparison

# Pass-through / non-tissue Cosmos categories (see boundaries_utils._NNI_IDS) — crossings
# into/out of these are trivial (CSF, unsegmented root, fiber tracts) and were already
# excluded when picking "qualifying boundaries" earlier in this project.
NON_TISSUE_ACRONYMS = frozenset({'void_fluid', 'root', 'fiber tracts', 'VS'})

DATADISK_ROOT = Path('/Users/olivier/Documents/datadisk/ephys-atlas-decoding')
VOL_PATH = DATADISK_ROOT.joinpath(
    'encoding_volumes', PROJECT, VOL_LABEL, f'brainwide_ephys_atlas_{RES_UM}um.npz'
)
RAW_FEATURES_PATH = DATADISK_ROOT.joinpath('features', PROJECT, FEATURES_VINTAGE, 'agg_full')

CACHE_DIR = Path(__file__).parent.joinpath('cache')
FIGURES_DIR = Path(__file__).parent.joinpath('figures')

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
})


def main() -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print('=== Loading atlas & PSD/CSD PCA ===')
    brain_atlas = ephysatlas.anatomy.ClassifierAtlas(res_um=RES_UM)
    psd_pca = bu.load_or_fit_psd_pca(
        RAW_FEATURES_PATH, CACHE_DIR, FEATURES_VINTAGE, brain_atlas, csd_variant=CSD_VARIANT,
    )

    print('\n=== Loading encoding volume ===')
    vol_data = np.load(VOL_PATH, allow_pickle=True)
    print(f'  Shape: {vol_data["ephys_atlas_vol"].shape}, res_um={int(vol_data["res_um"][0])}')

    vp_cache = CACHE_DIR.joinpath(f'vp_df_{VOL_LABEL}_{GRID_SPACING_UM:.0f}um.parquet')
    if vp_cache.exists():
        print(f'\nLoading cached virtual-probe df from {vp_cache.name}')
        df = pd.read_parquet(vp_cache)
    else:
        print(f'\n=== Building virtual-probe dataframe ({GRID_SPACING_UM:.0f} um grid) ===')
        df = build_virtual_probe_df(brain_atlas, vol_data, psd_pca, GRID_SPACING_UM)
        df.to_parquet(vp_cache)
        print(f'  Cached -> {vp_cache.name}')
    print(f'Virtual probe df: {len(df):,} channels, {df.shape[1]} columns')

    print('\n=== Dense Cosmos transition matrix (QC vs. real-probe sparsity) ===')
    count_matrix = bu.compute_cosmos_transitions(df, brain_atlas)
    matrix_csv = FIGURES_DIR.joinpath(f'vp_transition_matrix_{VOL_LABEL}_{GRID_SPACING_UM:.0f}um.csv')
    count_matrix.to_csv(matrix_csv)
    n_newly_qualifying = int((count_matrix.values >= OLD_REAL_DATA_FLOOR).sum())
    print(f'  Saved {matrix_csv.name}')

    # plot_transition_matrix() always writes a fixed 'cosmos_transition_matrix.png' name --
    # render into CACHE_DIR (never FIGURES_DIR, which already holds the real-probe version
    # of that exact filename from Step 1) then move the result under its own distinct name.
    bu.plot_transition_matrix(count_matrix, f'{VOL_LABEL}, {GRID_SPACING_UM:.0f} um virtual grid', CACHE_DIR)
    matrix_png = FIGURES_DIR.joinpath(f'vp_transition_matrix_{VOL_LABEL}_{GRID_SPACING_UM:.0f}um.png')
    CACHE_DIR.joinpath('cosmos_transition_matrix.png').rename(matrix_png)
    print(f'  Saved {matrix_png.name}')
    print(
        f'  {n_newly_qualifying} directed Cosmos pairs have >= {OLD_REAL_DATA_FLOOR} dense '
        f'virtual crossings (the old real-probe qualifying floor)'
    )

    print("\n=== Ranking boundary pairs by effect size (Cohen's d) ===")
    df_stats = bu.compute_boundary_feature_stats(
        df, brain_atlas, window_um=WINDOW_UM, min_transitions=MIN_TRANSITIONS,
    )
    stats_csv = FIGURES_DIR.joinpath(f'volume_boundary_feature_stats_{VOL_LABEL}.csv')
    df_stats.to_csv(stats_csv, index=False)
    print(f'Saved {stats_csv.name}: {len(df_stats)} (boundary, feature) rows')

    print('\n=== Per-boundary landmark ranking ===')
    is_landmark_feat = (df_stats['cohens_d'] > 0.8) & (df_stats['pval_bonf'] < 0.01)
    summary = (
        df_stats.assign(is_landmark_feat=is_landmark_feat)
        .groupby(['from', 'to'], as_index=False)
        .agg(
            n_probes=('n_probes', 'first'),
            n_trans=('n_trans', 'first'),
            max_cohens_d=('cohens_d', 'max'),
            n_significant_features=('is_landmark_feat', 'sum'),
        )
        .sort_values(['n_significant_features', 'max_cohens_d'], ascending=False)
        .reset_index(drop=True)
    )
    ranking_csv = FIGURES_DIR.joinpath(f'volume_landmark_ranking_{VOL_LABEL}.csv')
    summary.to_csv(ranking_csv, index=False)
    print(f'Saved {ranking_csv.name}: {len(summary)} ranked boundary pairs')
    print(summary.head(25).to_string(index=False))

    print('\n=== Tissue-only ranking (excludes CSF/root/fiber-tract crossings) ===')
    tissue_only = summary[
        ~summary['from'].isin(NON_TISSUE_ACRONYMS) & ~summary['to'].isin(NON_TISSUE_ACRONYMS)
    ].reset_index(drop=True)
    tissue_csv = FIGURES_DIR.joinpath(f'volume_landmark_ranking_{VOL_LABEL}_tissue_only.csv')
    tissue_only.to_csv(tissue_csv, index=False)
    print(f'Saved {tissue_csv.name}: {len(tissue_only)} ranked tissue-tissue boundary pairs')
    print(tissue_only.to_string(index=False))


if __name__ == '__main__':
    main()
