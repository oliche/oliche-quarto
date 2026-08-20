"""Visualise one encoding-volume feature in the atlasview brain slicer.

Usage (needs a Qt event loop — run from IPython or as a standalone script):
    %run view_encoding_volume.py
    # or:  python view_encoding_volume.py

Change FEATURE to any name in the volume (printed on startup).
"""
# %%
import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, '/home/olivier/PycharmProjects/EphysAtlas')
from atlasview.atlasview import view as atlasview_view
import ephysatlas.anatomy
from qt_helpers import qt

VOL_PATH = Path.home().joinpath(
    'data', 'ephys-atlas', 'encoding_volumes', 'brainwide_ephys_atlas_25um.npz'
)

# ['rms_lf', 'psd_lfp', 'psd_alpha', 'psd_beta', 'psd_gamma', 'psd_delta', 'psd_theta', 'psd_lfp_csd', 'psd_alpha_csd', 'psd_beta_csd', 'psd_gamma_csd', 'psd_delta_csd', 'psd_theta_csd', 'rms_lf_csd', 'psd_residual_lfp', 'psd_residual_alpha', 'psd_residual_beta', 'psd_residual_gamma', 'psd_residual_delta', 'psd_residual_theta', 'decay_fit_error', 'decay_fit_r_squared', 'decay_n_peaks', 'aperiodic_exponent', 'aperiodic_offset', 'cor_ratio', 'rms_ap', 'alpha_mean', 'alpha_std', 'spike_count', 'tip_time_secs', 'recovery_time_secs', 'peak_time_secs', 'trough_time_secs', 'trough_val', 'tip_val', 'peak_val', 'recovery_slope', 'depolarisation_slope', 'repolarisation_slope', 'polarity']
FEATURE = 'aperiodic_exponent'# set to any feature name listed on startup
CMAP = 'inferno'

# ── load ──────────────────────────────────────────────────────────────────────
brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
vol_data = np.load(VOL_PATH, allow_pickle=True)
feat_names = list(vol_data['feature_names'])
print(f"Features in volume: {feat_names}")
feat_vol = vol_data['ephys_atlas_vol'][..., feat_names.index(FEATURE)].astype(float)

# %%
# permute to atlas axes (ml, ap, dv) so it aligns with brain_atlas.label
feat_vol_atlas = np.permute_dims(feat_vol, [1, 0, 2])
print(f"Volume shape: {feat_vol_atlas.shape}  (displaying '{FEATURE}')")

# ── compute vmin/vmax only over labelled brain voxels ─────────────────────────
brain_mask = (np.logical_and(brain_atlas.label > 0, 2000 > brain_atlas.label) ) & np.isfinite(feat_vol_atlas)
vals = feat_vol_atlas[brain_mask]
vmin, vmax = np.percentile(vals, [10, 90])
# ── mask negative-ML half (left hemisphere) to sentinel so it renders transparent ──
# midline index along ML axis (axis 0); negative ML = left hemisphere
i_midline = brain_atlas.bc.x2i(0)
feat_vol_split = feat_vol_atlas.copy()
step = (vmax - vmin) / 254
sentinel = vmin - step  # one step below vmin → maps to LUT index 0 (transparent)
feat_vol_split[:i_midline, :, :] = sentinel

# ── build RGBA LUT: index 0 = transparent, 1-255 = colormap ──────────────────
cmap = matplotlib.colormaps[CMAP]
lut = np.zeros((256, 4), dtype=np.uint8)
lut[1:] = (cmap(np.linspace(0, 1, 255)) * 255).astype(np.uint8)
# shift levels so vmin → index 1 (not 0), keeping index 0 reserved for sentinel
levels = (vmin - step, vmax)

# ── open slicer ───────────────────────────────────────────────────────────────
# qt.create_app()
av = atlasview_view(atlas=brain_atlas)

av.ctrl.set_volume(feat_vol_split, levels=levels, lut=lut)
av._refresh()
# qt.run_app()
