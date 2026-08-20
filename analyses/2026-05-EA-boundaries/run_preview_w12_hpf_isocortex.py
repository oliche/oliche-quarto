"""Preview W12 HPF→Isocortex figures for measured data and encoding volume.

Fits PCA on 2026_W12 measured features (downloads if needed), builds both
measured and volume feature dataframes, and regenerates only the
HPF → Isocortex heatmap profiles (ap, ml, rastermap) for both conditions.
No models are retrained.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import ephysatlas.anatomy
import ephysatlas.data

sys.path.insert(0, str(Path(__file__).parent))
from boundaries_utils import (
    build_volume_feature_df,
    compute_feature_vlims,
    load_or_build_pca_df,
    load_or_fit_psd_pca,
    plot_boundary_feature_profiles,
)

VINTAGE = '2026_W12'
ROOT_PATH = Path.home().joinpath('data', 'ephys-atlas', 'features')
VOL_PATH = Path.home().joinpath('data', 'ephys-atlas', 'encoding_volumes', 'brainwide_ephys_atlas_25um.npz')
FIGURES_DIR = Path(__file__).parent.joinpath('figures', 'heatmap_profiles')
LOCAL_CACHE_DIR = Path(__file__).parent.joinpath('cache')
RASTERMAP_CACHE_DIR = LOCAL_CACHE_DIR.joinpath('rastermap_w12')
WINDOW_UM = 1000
FROM_ACR, TO_ACR = 'HPF', 'Isocortex'

DISPLAY_FEATURES = (
    'psd_pc0', 'psd_pc1', 'csd_pc0', 'csd_pc1',
    'rms_ap', 'spike_count', 'aperiodic_offset', 'aperiodic_exponent',
)

brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
raw_features_path = ROOT_PATH.joinpath('ea_active', VINTAGE, 'agg_full')

# Download 2026_W12 measured features if not on disk
if not raw_features_path.exists():
    print(f'Downloading {VINTAGE} measured features …')
    from one.api import ONE
    one = ONE(base_url='https://alyx.internationalbrainlab.org', mode='remote')
    ephysatlas.data.download_tables(ROOT_PATH, label=VINTAGE, one=one)

# Fit PCA on 2026_W12 (cached as psd_pca_2026_W12.pkl)
psd_pca = load_or_fit_psd_pca(raw_features_path, LOCAL_CACHE_DIR, VINTAGE, brain_atlas)

# Build measured df for 2026_W12 (cached as df_2026_W12.parquet)
df_measured = load_or_build_pca_df(raw_features_path, LOCAL_CACHE_DIR, VINTAGE, brain_atlas, psd_pca)
print(f'Measured df: {len(df_measured):,} channels, {df_measured.shape[1]} columns')

# Build volume df using the same 2026_W12 volume
print(f'Loading encoding volume from {VOL_PATH.name} …')
vol_data = np.load(VOL_PATH, allow_pickle=True)
print(f'  Volume shape: {vol_data["ephys_atlas_vol"].shape}')
df_vol = build_volume_feature_df(df_measured, vol_data, brain_atlas, psd_pca)
print(f'  Volume df: {len(df_vol):,} channels, {df_vol.shape[1]} columns')

# Shared vlims from W12 measured data
feature_vlims, feature_scaler, scaler_features = compute_feature_vlims(df_measured, return_scaler=True)

FIGURES_DIR.mkdir(exist_ok=True)
RASTERMAP_CACHE_DIR.mkdir(exist_ok=True)

slug = f'{FROM_ACR}_to_{TO_ACR}'
print(f'\nGenerating {slug} figures (measured + volume, 3 sort modes each) …')

for label, df in [('heatmap', df_measured), ('vol', df_vol)]:
    print(f'\n  [{label}]')
    for sort_mode in ('ap', 'ml', 'rastermap'):
        fig = plot_boundary_feature_profiles(
            df,
            brain_atlas,
            FROM_ACR,
            TO_ACR,
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
            out = FIGURES_DIR.joinpath(f'{label}_{slug}_{sort_mode}.png')
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f'    → {out.name}')
        else:
            print('    (no crossings, skipped)')

print('\nDone.')
