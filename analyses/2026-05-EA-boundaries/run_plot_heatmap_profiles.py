"""Generate heatmap profile figures (±1000 µm) for the top-6 classifier transitions.

Three probe-sort versions per boundary pair: AP coordinate, |ML| coordinate, Rastermap.
Post-PCA features (psd_pc0/pc1, csd_pc0/pc1) are produced with the volume-compatible
PCA (no diff1 CSD features) so the axes are identical to the encoding-volume figures.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import ephysatlas.anatomy

sys.path.insert(0, str(Path(__file__).parent))
from boundaries_utils import (
    compute_feature_vlims,
    load_or_build_pca_df,
    load_or_fit_psd_pca,
    plot_boundary_feature_profiles,
)

VINTAGE = '2026_W12'
ROOT_PATH = Path.home().joinpath('data', 'ephys-atlas', 'features')
FIGURES_DIR = Path(__file__).parent.joinpath('figures')
LOCAL_CACHE_DIR = Path(__file__).parent.joinpath('cache')
RASTERMAP_CACHE_DIR = LOCAL_CACHE_DIR.joinpath('rastermap_measured')
WINDOW_UM = 1000

DISPLAY_FEATURES = (
    'psd_pc0', 'psd_pc1', 'csd_pc0', 'csd_pc1',
    'rms_ap', 'spike_count', 'aperiodic_offset', 'aperiodic_exponent',
)

brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
raw_features_path = ROOT_PATH.joinpath('ea_active', VINTAGE, 'agg_full')

psd_pca = load_or_fit_psd_pca(raw_features_path, LOCAL_CACHE_DIR, VINTAGE, brain_atlas)
df = load_or_build_pca_df(raw_features_path, LOCAL_CACHE_DIR, VINTAGE, brain_atlas, psd_pca)
print(f'  {len(df):,} channels, {df.shape[1]} columns')

feature_vlims, feature_scaler, scaler_features = compute_feature_vlims(df, return_scaler=True)
RASTERMAP_CACHE_DIR.mkdir(exist_ok=True)

# Read top-6 transitions from classifier accuracy CSV
df_acc = pd.read_csv(FIGURES_DIR.joinpath('classifier_accuracy.csv'))
df_acc = df_acc[(df_acc['model'] == 'gb') & (df_acc['class'] != 'overall')]
top6 = df_acc.nlargest(6, 'accuracy')[['class', 'accuracy']].values.tolist()
boundary_pairs = [(cls.split('_to_')[0], cls.split('_to_')[1]) for cls, _ in top6]

print('\nTop-6 transitions:')
for f, t in boundary_pairs:
    print(f'  {f} → {t}')

out_dir = FIGURES_DIR.joinpath('heatmap_profiles')
out_dir.mkdir(exist_ok=True)

print(f'\nGenerating {len(boundary_pairs)} × 3 heatmap profile figures …\n')
for from_acr, to_acr in boundary_pairs:
    slug = f'{from_acr}_to_{to_acr}'.replace(' ', '_')
    print(f'  {from_acr} → {to_acr}')
    for sort_mode in ('ap', 'ml', 'rastermap'):
        fig = plot_boundary_feature_profiles(
            df,
            brain_atlas,
            from_acr,
            to_acr,
            mandatory_features=DISPLAY_FEATURES,
            df_results=None,
            window_um=WINDOW_UM,
            max_probes=150,
            feature_vlims=feature_vlims,
            sort=sort_mode,
            feature_scaler=(feature_scaler, scaler_features),
            cache_dir=RASTERMAP_CACHE_DIR,
            sort_features=False,
        )
        if fig is not None:
            out = out_dir.joinpath(f'heatmap_{slug}_{sort_mode}.png')
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f'    → {out.name}')
        else:
            print('    (no crossings, skipped)')

print('\nDone.')