"""Allen-region colour slices and encoding-volume coverage mask slices.

Four figures total (one per slice type × plot type):
  figures/slice_panels/allen_coronal.png    — Allen palette per region
  figures/slice_panels/allen_sagittal.png
  figures/slice_panels/mask_coronal.png     — recording coverage (recorded / unrecorded)
  figures/slice_panels/mask_sagittal.png

Same AP/ML positions and panel layout as run_plot_slice_panels.py.

Volume / atlas axis convention
-------------------------------
vol[iML, iAP, iDV, feat]  -- shape (456, 528, 320, 41)
ba.label[iAP, iML, iDV]   -- shape (528, 456, 320)
bc.xyz2i(xyz) -> (iML, iAP, iDV)

Coronal label slice : ba.label[iAP, :, :]  -> (n_ML, n_DV)
Sagittal label slice: ba.label[:, iML, :]  -> (n_AP, n_DV)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import ephysatlas.anatomy
sys.path.insert(0, str(Path(__file__).parent))
from boundary_classifier_volume import GRID_SPACING_UM, LOCAL_CACHE_DIR, FIGURES_DIR

# ---------------------------------------------------------------------------
# Constants  (must match run_plot_slice_panels.py)
# ---------------------------------------------------------------------------

SLICE_STRIDE  = 3
PANEL_W       = 2.8
PANEL_H       = 2.0
VOID_FLUID_ID = 2000
FILL_REF_FEAT = 'rms_lf'   # 0.0 for unrecorded voxels, always negative-dB for tissue

VOL_PATH = Path.home().joinpath(
    'data', 'ephys-atlas', 'encoding_volumes', 'brainwide_ephys_atlas_25um.npz'
)
OUT_DIR = FIGURES_DIR.joinpath('slice_panels')

# Mask display colours (RGBA uint8)
COL_UNRECORDED = np.array([160, 160, 160, 220], dtype=np.uint8)  # grey
COL_RECORDED   = np.array([ 60, 200, 160, 220], dtype=np.uint8)  # teal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_grid(n, ncols=None):
    """Return (nrows, ncols) for a panel grid."""
    if ncols is None:
        ncols = min(9, math.ceil(math.sqrt(n * 1.4)))
    return math.ceil(n / ncols), ncols


def build_rgb_lut(brain_atlas):
    """Return the Allen RGB lookup table indexed by region index.

    ``ba.label`` stores region indices (0 … n_regions-1) into the ``ba.regions``
    arrays — NOT Allen IDs.  ``ba.regions.rgb`` is already ordered by those same
    indices, so it is the correct LUT to use directly.

    Allen IDs (``ba.regions.id``) are a separate concept and should NOT be used
    as indices into ``ba.regions.rgb``.

    Returns
    -------
    lut : np.ndarray, shape (n_regions, 3), dtype uint8
    """
    return np.array(brain_atlas.regions.rgb, dtype=np.uint8)


def label_to_rgba(lbl, rgb_lut):
    """Convert a 2-D label array to an RGBA image using Allen colours.

    Outside-brain voxels (label == 0) are fully transparent.

    Parameters
    ----------
    lbl : np.ndarray (H, W) int
    rgb_lut : np.ndarray (max_id+1, 3) uint8

    Returns
    -------
    rgba : np.ndarray (H, W, 4) uint8
    """
    clipped = np.clip(lbl, 0, len(rgb_lut) - 1)
    rgb  = rgb_lut[clipped]                       # (H, W, 3)
    alpha = np.where(lbl > 0, 255, 0).astype(np.uint8)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


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


def label_to_mask_rgba(lbl, rec):
    """Convert label + recorded mask to a 3-category RGBA image.

    Categories
    ----------
    outside brain (lbl == 0)     : transparent
    unrecorded (rec == False)    : grey
    recorded   (rec == True)     : teal

    Parameters
    ----------
    lbl : np.ndarray (H, W) int
    rec : np.ndarray (H, W) bool — True where rms_lf != 0

    Returns
    -------
    rgba : np.ndarray (H, W, 4) uint8
    """
    rgba = np.zeros((*lbl.shape, 4), dtype=np.uint8)
    inside = lbl > 0
    rgba[inside & ~rec] = COL_UNRECORDED
    rgba[inside &  rec] = COL_RECORDED
    return rgba


# ---------------------------------------------------------------------------
# Generic figure builder
# ---------------------------------------------------------------------------

def plot_slices(slice_type, positions_m, brain_atlas, rgb_lut, vol, feat_names, out_dir):
    """Render Allen-colour and mask figures for one slice orientation.

    Parameters
    ----------
    slice_type : 'coronal' or 'sagittal'
    positions_m : array of float — slice positions in metres
    """
    bc = brain_atlas.bc
    n = len(positions_m)
    nrows, ncols = make_grid(n)
    print(f'  {n} {slice_type} slices  ({nrows}×{ncols} grid)')

    fill_idx = feat_names.index(FILL_REF_FEAT)

    # collect per-slice data
    lbls, recs, extents, x_coords_list, z_coords_list, labels = [], [], [], [], [], []
    for pos_m in positions_m:
        if slice_type == 'coronal':
            iAP  = int(bc.xyz2i(np.array([[0., pos_m, 0.]]))[0, 1])
            lbl  = brain_atlas.label[iAP, :, :]           # (ML, DV)
            raw  = vol[:, iAP, :, fill_idx].astype(np.float32)
            ext  = [bc.xlim[0]*1e6, bc.xlim[1]*1e6, bc.zlim[1]*1e6, bc.zlim[0]*1e6]
            x_c  = np.linspace(bc.xlim[0]*1e6, bc.xlim[1]*1e6, lbl.shape[0])
            z_c  = np.linspace(bc.zlim[0]*1e6, bc.zlim[1]*1e6, lbl.shape[1])
            panel_label = f'AP {pos_m*1e3:+.1f} mm'
        else:
            iML  = int(bc.xyz2i(np.array([[pos_m, 0., 0.]]))[0, 0])
            lbl  = brain_atlas.label[:, iML, :]           # (AP, DV)
            raw  = vol[iML, :, :, fill_idx].astype(np.float32)
            ext  = [bc.ylim[0]*1e6, bc.ylim[1]*1e6, bc.zlim[1]*1e6, bc.zlim[0]*1e6]
            x_c  = np.linspace(bc.ylim[0]*1e6, bc.ylim[1]*1e6, lbl.shape[0])
            z_c  = np.linspace(bc.zlim[0]*1e6, bc.zlim[1]*1e6, lbl.shape[1])
            panel_label = f'ML {pos_m*1e3:+.1f} mm'

        rec = (raw != 0.0) & (lbl > 0) & (lbl != VOID_FLUID_ID)
        lbls.append(lbl);  recs.append(rec)
        extents.append(ext);  x_coords_list.append(x_c);  z_coords_list.append(z_c)
        labels.append(panel_label)

    # ── figure 1: Allen colours ───────────────────────────────────────────────
    fig_a, axes_a = plt.subplots(nrows, ncols,
                                  figsize=(ncols * PANEL_W, nrows * PANEL_H),
                                  facecolor='#111111')
    axes_a = np.array(axes_a).ravel()

    for i, (lbl, ext, x_c, z_c, panel_label) in enumerate(
            zip(lbls, extents, x_coords_list, z_coords_list, labels)):
        ax = axes_a[i]
        ax.set_facecolor('#111111')
        rgba = label_to_rgba(lbl, rgb_lut)          # (ML|AP, DV, 4)
        ax.imshow(rgba.transpose(1, 0, 2), origin='upper', aspect='equal',
                  extent=ext, interpolation='nearest')
        inside = lbl > 0
        ax.contour(x_c, z_c, inside.T.astype(float),
                   levels=[0.5], colors='white', linewidths=0.4, alpha=0.5)
        add_cosmos_contours(ax, x_c, z_c, lbl, brain_atlas)
        ax.set_title(panel_label, fontsize=7, color='white', pad=2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    for ax in axes_a[n:]:
        ax.set_visible(False)

    fig_a.suptitle(f'Allen regions   |   {slice_type} slices   '
                   f'(every {SLICE_STRIDE}rd grid position)',
                   fontsize=13, color='white', y=1.005)
    fig_a.tight_layout(pad=0.4)
    out_a = out_dir.joinpath(f'allen_{slice_type}.png')
    fig_a.savefig(out_a, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close(fig_a)
    print(f'    -> {out_a.name}')

    # ── figure 2: recording coverage mask ────────────────────────────────────
    fig_m, axes_m = plt.subplots(nrows, ncols,
                                  figsize=(ncols * PANEL_W, nrows * PANEL_H),
                                  facecolor='#111111')
    axes_m = np.array(axes_m).ravel()

    for i, (lbl, rec, ext, x_c, z_c, panel_label) in enumerate(
            zip(lbls, recs, extents, x_coords_list, z_coords_list, labels)):
        ax = axes_m[i]
        ax.set_facecolor('#111111')
        rgba = label_to_mask_rgba(lbl, rec)         # (ML|AP, DV, 4)
        ax.imshow(rgba.transpose(1, 0, 2), origin='upper', aspect='equal',
                  extent=ext, interpolation='nearest')
        inside = lbl > 0
        ax.contour(x_c, z_c, inside.T.astype(float),
                   levels=[0.5], colors='white', linewidths=0.4, alpha=0.5)
        add_cosmos_contours(ax, x_c, z_c, lbl, brain_atlas)
        ax.set_title(panel_label, fontsize=7, color='white', pad=2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    for ax in axes_m[n:]:
        ax.set_visible(False)

    legend_patches = [
        mpatches.Patch(color=COL_UNRECORDED[:3] / 255., label='unrecorded (fill 0)'),
        mpatches.Patch(color=COL_RECORDED[:3]   / 255., label='recorded'),
    ]
    fig_m.legend(handles=legend_patches, loc='lower center', ncol=2,
                 fontsize=9, framealpha=0.3,
                 labelcolor='white', facecolor='#222222', edgecolor='none',
                 bbox_to_anchor=(0.5, -0.01))

    fig_m.suptitle(f'Encoding-volume coverage   |   {slice_type} slices   '
                   f'(every {SLICE_STRIDE}rd grid position)',
                   fontsize=13, color='white', y=1.005)
    fig_m.tight_layout(pad=0.4)
    out_m = out_dir.joinpath(f'mask_{slice_type}.png')
    fig_m.savefig(out_m, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close(fig_m)
    print(f'    -> {out_m.name}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

print('Loading atlas ...')
brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
bc = brain_atlas.bc

print('Building Allen RGB LUT ...')
rgb_lut = build_rgb_lut(brain_atlas)
print(f'  LUT: {len(rgb_lut)} regions (indexed by label value, not Allen ID)')

print('Loading encoding volume ...')
vol_data   = np.load(VOL_PATH, allow_pickle=True)
vol        = vol_data['ephys_atlas_vol']   # (ML=456, AP=528, DV=320, 41)
feat_names = list(vol_data['feature_names'])

print('Loading virtual-probe df (positions only) ...')
vp_cache = LOCAL_CACHE_DIR.joinpath(f'vp_df_{GRID_SPACING_UM:.0f}um.parquet')
df_vp = __import__('pandas').read_parquet(vp_cache, columns=['x', 'y'])
probe_first = df_vp.groupby(level='pid')[['x', 'y']].first()
ap_positions = np.sort(probe_first['y'].unique())[::-1][::SLICE_STRIDE]
ml_positions = np.sort(probe_first['x'].unique())[::SLICE_STRIDE]
print(f'  {len(ap_positions)} AP, {len(ml_positions)} ML positions (stride={SLICE_STRIDE})')

OUT_DIR.mkdir(parents=True, exist_ok=True)

print('\n=== coronal ===')
plot_slices('coronal', ap_positions, brain_atlas, rgb_lut, vol, feat_names, OUT_DIR)

print('\n=== sagittal ===')
plot_slices('sagittal', ml_positions, brain_atlas, rgb_lut, vol, feat_names, OUT_DIR)

print('\nDone.')
