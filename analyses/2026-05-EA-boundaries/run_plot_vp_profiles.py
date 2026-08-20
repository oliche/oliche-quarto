"""Generate AP and ML heatmap profile figures from virtual-probe volume data.

Loads the virtual-probe feature dataframe and precomputed crossings produced
by ``boundary_classifier_volume.py``, then calls
``plot_boundary_feature_profiles()`` for each qualifying boundary pair with
``sort='ap'`` and ``sort='ml'``.

Precomputed crossings are injected directly into the plot function (via the
``crossings`` parameter added to ``boundaries_utils.plot_boundary_feature_profiles``)
to bypass the slow ``find_boundary_crossings()`` groupby on the large virtual-
probe dataframe.

Output: ``figures/heatmap_profiles/vp_{pair}_{ap|ml}.png``
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ephysatlas.anatomy
sys.path.insert(0, str(Path(__file__).parent))
from boundaries_utils import (
    plot_boundary_feature_profiles,
)
from boundary_classifier_volume import (
    GRID_SPACING_UM,
    LOCAL_CACHE_DIR,
    FIGURES_DIR,
    MIN_INSERTIONS,
    WINDOW_UM,
)

DISPLAY_FEATURES = (
    'psd_pc0', 'psd_pc1', 'csd_pc0', 'csd_pc1',
    'rms_ap', 'spike_count', 'aperiodic_offset', 'aperiodic_exponent',
)

# ---------------------------------------------------------------------------
# Load cached data
# ---------------------------------------------------------------------------

print('Loading atlas ...')
brain_atlas = ephysatlas.anatomy.ClassifierAtlas()

vp_cache = LOCAL_CACHE_DIR.joinpath(f'vp_df_{GRID_SPACING_UM:.0f}um.parquet')
print(f'Loading virtual probe df from {vp_cache.name} ...')
df = pd.read_parquet(vp_cache)
print(f'  {len(df):,} channels, {df.shape[1]} columns')

cross_cache = LOCAL_CACHE_DIR.joinpath(f'vp_crossings_{GRID_SPACING_UM:.0f}um.pkl')
print(f'Loading crossings from {cross_cache.name} ...')
with cross_cache.open('rb') as fh:
    crossings_dict, pair_counts = pickle.load(fh)

# Qualifying pairs sorted by count descending
pairs = sorted(
    [(k, v) for k, v in pair_counts.items() if v >= MIN_INSERTIONS],
    key=lambda x: -x[1],
)
print(f'\n{len(pairs)} qualifying pairs (>= {MIN_INSERTIONS} crossings)')
for (fr, to), cnt in pairs:
    print(f'  {fr} -> {to}: {cnt}')

# ---------------------------------------------------------------------------
# Generate figures
# ---------------------------------------------------------------------------

out_dir = FIGURES_DIR.joinpath('vp_profiles')
out_dir.mkdir(exist_ok=True)

print(f'\nGenerating {len(pairs)} x 2 virtual-probe profile figures ...\n')
for (from_acr, to_acr), n_cross in pairs:
    slug = f'{from_acr}_to_{to_acr}'.replace(' ', '_')
    crossings = crossings_dict.get((from_acr, to_acr), [])
    print(f'{from_acr} -> {to_acr}  ({n_cross} crossings)')

    for sort_mode in ('ap', 'ml'):
        fig = plot_boundary_feature_profiles(
            df,
            brain_atlas,
            from_acr,
            to_acr,
            mandatory_features=DISPLAY_FEATURES,
            df_results=None,
            window_um=WINDOW_UM,
            max_probes=150,
            feature_vlims=None,        # per-figure percentile limits for local contrast
            sort=sort_mode,
            sort_features=False,
            crossings=crossings,       # bypass internal find_boundary_crossings()
        )
        if fig is not None:
            out = out_dir.joinpath(f'vp_{slug}_{sort_mode}.png')
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f'  -> {out.name}')
        else:
            print(f'  (no crossings for {sort_mode}, skipped)')

print('\nDone.')
