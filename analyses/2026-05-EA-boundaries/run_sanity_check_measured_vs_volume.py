"""Sanity-check the top volume-discovered landmark candidates against measured data.

For each top-ranked boundary from ``run_volume_landmark_scan.py`` (tissue-only ranking),
plots the aligned depth-profile figure twice — once from measured channel recordings,
once from the dense encoding-volume virtual-probe grid — using the same
``plot_boundary_feature_profiles`` function and the same shared PSD/CSD PCA axes, so the
two panels are directly comparable. A volume-discovered boundary that measured data does
not corroborate is flagged, not silently promoted.

Run with: .venv/bin/python run_sanity_check_measured_vs_volume.py
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

PROJECT = 'ea_active'
VOL_LABEL = '2026_W26'
RES_UM = 50
FEATURES_VINTAGE = '2026_W26'
CSD_VARIANT = 'diff1'
GRID_SPACING_UM = 100.0
WINDOW_UM = 200.0
MIN_TRANSITIONS_MEASURED = 16   # measured data is far sparser than the virtual grid

# Top tissue-tissue candidates from volume_landmark_ranking_2026_W26_tissue_only.csv,
# ranked by n_significant_features then max_cohens_d.
TOP_CANDIDATES = [
    ('HPF', 'MB'),
    ('TH', 'HPF'),
    ('Isocortex', 'CNU'),
    ('CNU', 'HPF'),
    ('CNU', 'Isocortex'),
]

DATADISK_ROOT = Path('/Users/olivier/Documents/datadisk/ephys-atlas-decoding')
RAW_FEATURES_PATH = DATADISK_ROOT.joinpath('features', PROJECT, FEATURES_VINTAGE, 'agg_full')

CACHE_DIR = Path(__file__).parent.joinpath('cache')
FIGURES_DIR = Path(__file__).parent.joinpath('figures', 'volume_vs_measured')

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
})


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print('=== Loading atlas & PSD/CSD PCA ===')
    brain_atlas = ephysatlas.anatomy.ClassifierAtlas(res_um=RES_UM)
    psd_pca = bu.load_or_fit_psd_pca(
        RAW_FEATURES_PATH, CACHE_DIR, FEATURES_VINTAGE, brain_atlas, csd_variant=CSD_VARIANT,
    )

    print('\n=== Loading measured (post-PCA) channel features ===')
    df_measured = bu.load_or_build_pca_df(
        RAW_FEATURES_PATH, CACHE_DIR, FEATURES_VINTAGE, brain_atlas,
        psd_pca=psd_pca, csd_variant=CSD_VARIANT,
    )
    print(f'  {len(df_measured):,} measured channels, '
          f'{df_measured.index.get_level_values("pid").nunique()} probes')

    print('\n=== Loading cached virtual-probe (volume) dataframe ===')
    vp_cache = CACHE_DIR.joinpath(f'vp_df_{VOL_LABEL}_{GRID_SPACING_UM:.0f}um.parquet')
    df_volume = pd.read_parquet(vp_cache)
    print(f'  {len(df_volume):,} virtual-probe channels')

    # Measured raw data still carries plain *_csd columns (unused by the diff1-variant PCA)
    # and volume-absent features like rms_lf_no_car -- restrict both dataframes to the
    # columns present in both *before* computing stats or plotting, so the per-boundary
    # "most informative extra feature" selection inside plot_boundary_feature_profiles picks
    # from the same feature universe on both sides and the two panels are directly comparable.
    meta_cols = [c for c in df_measured.columns if c in bu.EXCLUDE_COLS]
    feature_cols = [
        c for c in df_measured.columns
        if c not in bu.EXCLUDE_COLS and c in df_volume.columns
    ]
    df_measured = df_measured[feature_cols + meta_cols]
    print(f'  Restricted to {len(feature_cols)} features shared by measured + volume')

    print('\n=== Boundary feature stats (measured) ===')
    stats_measured = bu.compute_boundary_feature_stats(
        df_measured, brain_atlas, window_um=WINDOW_UM, min_transitions=MIN_TRANSITIONS_MEASURED,
    )
    stats_measured.to_csv(
        FIGURES_DIR.joinpath(f'measured_boundary_feature_stats_{FEATURES_VINTAGE}.csv'), index=False
    )

    print('\n=== Boundary feature stats (volume) ===')
    stats_volume = pd.read_csv(
        Path(__file__).parent.joinpath('figures', f'volume_boundary_feature_stats_{VOL_LABEL}.csv')
    )

    # Deliberately no shared/global colour limits: the volume is smooth (near-zero
    # variance outside the true anatomical signal) while measured data carries full
    # recording noise, so a scale calibrated across both washes the volume panel out
    # toward the centre of the colormap. Each panel keeps plot_boundary_feature_profiles'
    # default per-boundary 2nd-98th-percentile autoscale instead -- the point of the
    # comparison is matching sign/timing of the transition, not matching absolute colour.
    n_extra = 6
    mandatory_features = ('rms_ap', 'rms_lf', 'spike_count')

    confirmed, unconfirmed = [], []
    for from_acr, to_acr in TOP_CANDIDATES:
        label = f'{from_acr}_to_{to_acr}'
        has_measured = (
            (stats_measured['from'] == from_acr) & (stats_measured['to'] == to_acr)
        ).any()
        print(f'\n--- {label} --- measured data has this pair: {has_measured}')

        # Pick the top-N extra features from the volume ranking (the discovery driver) and
        # force the *same* feature list on both panels, so a difference between them reflects
        # measured-vs-volume signal, not two independently chosen feature selections.
        mask_pair = (stats_volume['from'] == from_acr) & (stats_volume['to'] == to_acr)
        top_extra = (
            stats_volume[mask_pair].sort_values('cohens_d', ascending=False)['feature'].tolist()
        )
        shared_feature_list = list(mandatory_features)
        for feat in top_extra:
            if feat not in shared_feature_list and feat in feature_cols and len(shared_feature_list) < 3 + n_extra:
                shared_feature_list.append(feat)

        fig_v = bu.plot_boundary_feature_profiles(
            df_volume, brain_atlas, from_acr, to_acr,
            mandatory_features=tuple(shared_feature_list), n_extra=0,
            window_um=1000.0, sort_features=False,
        )
        if fig_v is not None:
            out = FIGURES_DIR.joinpath(f'volume_{label}.png')
            fig_v.savefig(out, dpi=150)
            plt.close(fig_v)
            print(f'  Saved {out.name}')

        fig_m = None
        if has_measured:
            fig_m = bu.plot_boundary_feature_profiles(
                df_measured, brain_atlas, from_acr, to_acr,
                mandatory_features=tuple(shared_feature_list), n_extra=0,
                window_um=1000.0, sort_features=False,
            )
        if fig_m is not None:
            out = FIGURES_DIR.joinpath(f'measured_{label}.png')
            fig_m.savefig(out, dpi=150)
            plt.close(fig_m)
            print(f'  Saved {out.name}')
            confirmed.append(label)
        else:
            print(f'  NOT corroborated in measured {FEATURES_VINTAGE} data (encoding-volume-only)')
            unconfirmed.append(label)

    print(f'\n=== Summary ===')
    print(f'Confirmed in measured data:   {confirmed}')
    print(f'Encoding-volume-only (unconfirmed): {unconfirmed}')


if __name__ == '__main__':
    main()
