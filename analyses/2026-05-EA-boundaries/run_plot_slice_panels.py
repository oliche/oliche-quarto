"""Coronal and sagittal feature slice panels with virtual-probe overlays.

For each DISPLAY_FEATURE, produces two figures:
  figures/slice_panels/coronal_{feature}.png   — one panel per AP position (every 3rd)
  figures/slice_panels/sagittal_{feature}.png  — one panel per ML position (every 3rd)

Colorbar: RobustScaler fit on masked in-brain, non-void-fluid, finite voxels
(mask: label > 0 and label != 2000 and isfinite), then 1st–99th percentile
of the scaled distribution as display limits.

Volume axis convention
----------------------
vol[iML, iAP, iDV, feat]  -- shape (456, 528, 320, 41)
ba.label[iAP, iML, iDV]   -- shape (528, 456, 320)
bc.xyz2i(xyz) -> (iML, iAP, iDV)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import pickle
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import ephysatlas.anatomy
sys.path.insert(0, str(Path(__file__).parent))
from boundary_classifier_volume import GRID_SPACING_UM, LOCAL_CACHE_DIR, FIGURES_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISPLAY_FEATURES = (
    'psd_pc0', 'psd_pc1', 'csd_pc0', 'csd_pc1',
    'rms_ap', 'spike_count', 'aperiodic_offset', 'aperiodic_exponent',
)
PCA_FEATURES = {'psd_pc0', 'psd_pc1', 'csd_pc0', 'csd_pc1'}
SLICE_STRIDE = 3        # take every Nth grid position
PANEL_W = 2.8           # inches per panel
PANEL_H = 2.0           # inches per panel
PROBE_DOT_SIZE = 0.5
PROBE_COLOR = '#00FFAA'  # neon mint — pops against inferno
CMAP = 'inferno'
CLIM_PCT = (10, 90)
# Exclude void (0, handled by lbl > 0) and void_fluid (2000) atlas label IDs
VOID_FLUID_ID = 2000
# Reference raw feature used to build the "recorded" mask.
# The encoding volume stores 0.0 (not NaN) for unrecorded brain voxels.
# rms_lf is always negative dB for real tissue; exactly 0 means "no data".
FILL_REF_FEATURE = 'rms_lf'

VOL_PATH = Path.home().joinpath(
    'data', 'ephys-atlas', 'encoding_volumes', 'brainwide_ephys_atlas_25um.npz'
)
OUT_DIR = FIGURES_DIR.joinpath('slice_panels')

# ---------------------------------------------------------------------------
# Colorbar limits  (mirrors view_encoding_volume.py)
# ---------------------------------------------------------------------------

def get_recorded_slice_coronal(vol, feat_names, iAP):
    """Return bool (n_ML, n_DV) — True where voxel has real recordings (rms_lf != 0)."""
    raw = vol[:, iAP, :, feat_names.index(FILL_REF_FEATURE)].astype(np.float32)
    return raw != 0.0


def get_recorded_slice_sagittal(vol, feat_names, iML):
    """Return bool (n_AP, n_DV) — True where voxel has real recordings (rms_lf != 0)."""
    raw = vol[iML, :, :, feat_names.index(FILL_REF_FEATURE)].astype(np.float32)
    return raw != 0.0


def compute_clim(slices, label_slices, recorded_slices, pct=CLIM_PCT):
    """Fit RobustScaler on masked voxels; return (scaler, vmin, vmax).

    Mask: labelled brain tissue (not void/void_fluid), actually recorded
    (rms_lf != 0), and finite::

        mask = (lbl > 0) & (lbl != VOID_FLUID_ID) & recorded & np.isfinite(feat)

    Parameters
    ----------
    slices:
        List of 2-D float arrays (feature values, one per slice position).
    label_slices:
        Corresponding list of 2-D int arrays (atlas label values).
    recorded_slices:
        Corresponding list of 2-D bool arrays from ``get_recorded_slice_*``.
    pct:
        (low, high) percentiles of the scaled distribution for display limits.

    Returns
    -------
    scaler : RobustScaler
        Fitted scaler — apply to each slice before display.
    vmin, vmax : float
        Colorbar limits in scaled units.
    """
    all_vals = []
    for slc, lbl, rec in zip(slices, label_slices, recorded_slices):
        mask = (lbl > 0) & (lbl != VOID_FLUID_ID) & rec & np.isfinite(slc)
        all_vals.append(slc[mask].ravel())
    vals = np.concatenate(all_vals)
    scaler = RobustScaler().fit(vals[:, None])
    scaled = scaler.transform(vals[:, None]).ravel()
    return scaler, float(np.percentile(scaled, pct[0])), float(np.percentile(scaled, pct[1]))


# ---------------------------------------------------------------------------
# Feature extraction from volume
# ---------------------------------------------------------------------------

def _get_raw_slice_coronal(vol, feat_names, feature, iAP):
    """Return (n_ML, n_DV) float32 for a raw volume feature at iAP."""
    return vol[:, iAP, :, feat_names.index(feature)].astype(np.float32)


def _get_raw_slice_sagittal(vol, feat_names, feature, iML):
    """Return (n_AP, n_DV) float32 for a raw volume feature at iML."""
    return vol[iML, :, :, feat_names.index(feature)].astype(np.float32)


def _pca_slice_coronal(vol, feat_names, psd_pca, iAP):
    """Apply EphysPsdPCA to one coronal slice; return dict of PC arrays (ML, DV)."""
    n_ml, n_dv = vol.shape[0], vol.shape[2]
    raw = vol[:, iAP, :, :].astype(np.float32).reshape(-1, len(feat_names))
    df_pca = psd_pca.transform(pd.DataFrame(raw, columns=feat_names))
    return {col: df_pca[col].values.reshape(n_ml, n_dv) for col in PCA_FEATURES}


def _pca_slice_sagittal(vol, feat_names, psd_pca, iML):
    """Apply EphysPsdPCA to one sagittal slice; return dict of PC arrays (AP, DV)."""
    n_ap, n_dv = vol.shape[1], vol.shape[2]
    raw = vol[iML, :, :, :].astype(np.float32).reshape(-1, len(feat_names))
    df_pca = psd_pca.transform(pd.DataFrame(raw, columns=feat_names))
    return {col: df_pca[col].values.reshape(n_ap, n_dv) for col in PCA_FEATURES}


def get_feature_slice_coronal(vol, feat_names, psd_pca, feature, iAP):
    """Return (n_ML, n_DV) float32 for *feature* at coronal index iAP."""
    if feature in PCA_FEATURES:
        return _pca_slice_coronal(vol, feat_names, psd_pca, iAP)[feature]
    return _get_raw_slice_coronal(vol, feat_names, feature, iAP)


def get_feature_slice_sagittal(vol, feat_names, psd_pca, feature, iML):
    """Return (n_AP, n_DV) float32 for *feature* at sagittal index iML."""
    if feature in PCA_FEATURES:
        return _pca_slice_sagittal(vol, feat_names, psd_pca, iML)[feature]
    return _get_raw_slice_sagittal(vol, feat_names, feature, iML)


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

def add_cosmos_contours(ax, x_coords, z_coords, lbl, brain_atlas):
    """Overlay white contour lines at Cosmos region boundaries.

    Parameters
    ----------
    ax : matplotlib Axes
    x_coords, z_coords : 1-D float arrays — axis coordinates in µm
    lbl : 2-D int array (H, W) — Allen region index slice (ba.label values)
    brain_atlas : ClassifierAtlas
    """
    cosmos = brain_atlas.regions.mappings['Cosmos'][lbl]
    vals = np.unique(cosmos[lbl > 0])
    if len(vals) < 2:
        return
    levels = (vals[:-1] + vals[1:]) / 2.0
    ax.contour(x_coords, z_coords, cosmos.T.astype(float),
               levels=levels, colors='white', linewidths=0.8, alpha=0.8, zorder=4)


def make_grid(n, ncols=None):
    """Return (nrows, ncols) for a tight panel grid."""
    if ncols is None:
        ncols = min(9, math.ceil(math.sqrt(n * 1.4)))
    return math.ceil(n / ncols), ncols


# ---------------------------------------------------------------------------
# Main figure function
# ---------------------------------------------------------------------------

def plot_feature_slices(
    slice_type,       # 'coronal' or 'sagittal'
    feature,
    positions_m,      # sorted metres, already strided
    vol, feat_names, psd_pca,
    brain_atlas,
    probe_channels,   # dict {pos_m: (horiz_um, z_um)}
    out_dir,
):
    """Render all slices of one feature into a single multi-panel figure.

    Parameters
    ----------
    slice_type:
        ``'coronal'`` for AP slices (x=ML, y=DV) or ``'sagittal'`` (x=AP, y=DV).
    feature:
        One of DISPLAY_FEATURES.
    positions_m:
        Slice positions in metres (strided to every SLICE_STRIDE-th probe position).
    probe_channels:
        ``{pos_m: (horiz_um_array, z_um_array)}`` — channels to scatter per panel.
    """
    bc = brain_atlas.bc
    n_slices = len(positions_m)
    nrows, ncols = make_grid(n_slices)
    print(f'  {n_slices} {slice_type} slices  ({nrows}×{ncols} grid)')

    # -- collect slices, label slices, and recorded masks -----------------------
    slices, lbl_slices, rec_slices, vox_idx = [], [], [], []
    for pos_m in positions_m:
        if slice_type == 'coronal':
            iAP = int(bc.xyz2i(np.array([[0., pos_m, 0.]]))[0, 1])
            slices.append(get_feature_slice_coronal(vol, feat_names, psd_pca, feature, iAP))
            lbl_slices.append(brain_atlas.label[iAP, :, :])   # (ML, DV)
            rec_slices.append(get_recorded_slice_coronal(vol, feat_names, iAP))
            vox_idx.append(iAP)
        else:
            iML = int(bc.xyz2i(np.array([[pos_m, 0., 0.]]))[0, 0])
            slices.append(get_feature_slice_sagittal(vol, feat_names, psd_pca, feature, iML))
            lbl_slices.append(brain_atlas.label[:, iML, :])   # (AP, DV)
            rec_slices.append(get_recorded_slice_sagittal(vol, feat_names, iML))
            vox_idx.append(iML)

    # -- RobustScaler fit on recorded in-brain voxels; limits from scaled distribution --
    scaler, vmin, vmax = compute_clim(slices, lbl_slices, rec_slices)

    # -- figure ------------------------------------------------------------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * PANEL_W, nrows * PANEL_H),
                             facecolor='#111111')
    axes_flat = np.array(axes).ravel()

    for ax_idx, (pos_m, slc, lbl, rec) in enumerate(
            zip(positions_m, slices, lbl_slices, rec_slices)):
        ax = axes_flat[ax_idx]
        ax.set_facecolor('#111111')

        # Grey base: all labelled, non-void-fluid voxels (includes unrecorded ones).
        # Feature overlay: additionally require actual recordings (rms_lf != 0).
        inside_brain = (lbl > 0) & (lbl != VOID_FLUID_ID)
        brain_mask = inside_brain & rec & np.isfinite(slc)
        display = np.full(slc.shape, np.nan, dtype=np.float32)
        display[brain_mask] = scaler.transform(slc[brain_mask][:, None]).ravel()

        if slice_type == 'coronal':
            extent = [bc.xlim[0]*1e6, bc.xlim[1]*1e6, bc.zlim[1]*1e6, bc.zlim[0]*1e6]
            x_coords = np.linspace(bc.xlim[0]*1e6, bc.xlim[1]*1e6, slc.shape[0])
            z_coords = np.linspace(bc.zlim[0]*1e6, bc.zlim[1]*1e6, slc.shape[1])
            panel_label = f'AP {pos_m*1e3:+.1f} mm'
        else:
            extent = [bc.ylim[0]*1e6, bc.ylim[1]*1e6, bc.zlim[1]*1e6, bc.zlim[0]*1e6]
            x_coords = np.linspace(bc.ylim[0]*1e6, bc.ylim[1]*1e6, slc.shape[0])
            z_coords = np.linspace(bc.zlim[0]*1e6, bc.zlim[1]*1e6, slc.shape[1])
            panel_label = f'ML {pos_m*1e3:+.1f} mm'

        # Grey base layer: fills the entire inside-brain region so that
        # low-value voxels don't blend into the dark outside-brain background.
        grey_base = np.where(inside_brain, 0.5, np.nan)
        ax.imshow(grey_base.T, origin='upper', aspect='equal', extent=extent,
                  cmap='Greys_r', vmin=0, vmax=1, zorder=1)

        # Feature overlay (NaN where no data / outside mask → transparent)
        img = ax.imshow(display.T, origin='upper', aspect='equal',
                        extent=extent, cmap=CMAP, vmin=vmin, vmax=vmax, zorder=2)

        # Brain boundary contour
        ax.contour(x_coords, z_coords, inside_brain.T.astype(float),
                   levels=[0.5], colors='white', linewidths=0.5, alpha=0.6, zorder=3)

        # Cosmos region boundaries
        add_cosmos_contours(ax, x_coords, z_coords, lbl, brain_atlas)

        # Probe channel overlay (disabled for diagnostics)
        # ch = probe_channels.get(pos_m)
        # if ch is not None:
        #     ax.scatter(ch[0], ch[1], s=PROBE_DOT_SIZE, c=PROBE_COLOR,
        #                linewidths=0, rasterized=True, zorder=3)

        ax.set_title(panel_label, fontsize=7, color='white', pad=2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes_flat[n_slices:]:
        ax.set_visible(False)

    # Colorbar
    cbar = fig.colorbar(img, ax=axes_flat[:n_slices], shrink=0.45, pad=0.01,
                        fraction=0.015)
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white', labelsize=8)
    cbar.outline.set_edgecolor('white')

    fig.suptitle(f'{feature}   |   {slice_type} slices   '
                 f'(every {SLICE_STRIDE}rd grid position)',
                 fontsize=13, color='white', y=1.005)
    fig.tight_layout(pad=0.4)

    out = out_dir.joinpath(f'{slice_type}_{feature}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'    -> {out.name}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

print('Loading atlas ...')
brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
bc = brain_atlas.bc

print('Loading encoding volume ...')
vol_data = np.load(VOL_PATH, allow_pickle=True)
vol = vol_data['ephys_atlas_vol']   # (ML=456, AP=528, DV=320, 41) float16
feat_names = list(vol_data['feature_names'])

print('Loading PSD/CSD PCA ...')
pca_path = LOCAL_CACHE_DIR.joinpath('psd_pca_2026_W24.pkl')
with pca_path.open('rb') as fh:
    psd_pca = pickle.load(fh)

print('Loading virtual-probe df ...')
vp_cache = LOCAL_CACHE_DIR.joinpath(f'vp_df_{GRID_SPACING_UM:.0f}um.parquet')
df_vp = pd.read_parquet(vp_cache, columns=['x', 'y', 'z'])
print(f'  {len(df_vp):,} inside-brain channels')

# Unique AP/ML positions, every SLICE_STRIDE-th
probe_first = df_vp.groupby(level='pid')[['x', 'y']].first()
ap_positions = np.sort(probe_first['y'].unique())[::-1][::SLICE_STRIDE]   # anterior first
ml_positions = np.sort(probe_first['x'].unique())[::SLICE_STRIDE]          # left → right
print(f'  {len(ap_positions)} AP positions, {len(ml_positions)} ML positions '
      f'(stride={SLICE_STRIDE})')

# Channel-level lookup per slice position
print('Indexing probe channels per slice ...')
df_vp = df_vp.reset_index(level='pid')
df_vp['probe_ap'] = probe_first.loc[df_vp['pid'], 'y'].values
df_vp['probe_ml'] = probe_first.loc[df_vp['pid'], 'x'].values

ap_set = set(ap_positions)
ml_set = set(ml_positions)

coronal_channels: dict = {
    ap_m: (grp['x'].values * 1e6, grp['z'].values * 1e6)
    for ap_m, grp in df_vp.groupby('probe_ap')
    if ap_m in ap_set
}
sagittal_channels: dict = {
    ml_m: (grp['y'].values * 1e6, grp['z'].values * 1e6)
    for ml_m, grp in df_vp.groupby('probe_ml')
    if ml_m in ml_set
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

for feature in DISPLAY_FEATURES:
    print(f'\n=== {feature} ===')
    plot_feature_slices(
        'coronal', feature, ap_positions,
        vol, feat_names, psd_pca, brain_atlas,
        coronal_channels, OUT_DIR,
    )
    plot_feature_slices(
        'sagittal', feature, ml_positions,
        vol, feat_names, psd_pca, brain_atlas,
        sagittal_channels, OUT_DIR,
    )

print('\nDone.')
