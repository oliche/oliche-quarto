"""Debug: show the brain mask for one coronal slice alongside the raw feature values."""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import ephysatlas.anatomy
sys.path.insert(0, str(Path(__file__).parent))
from boundary_classifier_volume import LOCAL_CACHE_DIR

VOID_FLUID_ID = 2000
VOL_PATH = Path.home().joinpath(
    'data', 'ephys-atlas', 'encoding_volumes', 'brainwide_ephys_atlas_25um.npz'
)
FEATURE = 'rms_ap'   # raw feature so values are interpretable
AP_M = -0.002        # ~bregma, in metres

print('Loading atlas + volume ...')
brain_atlas = ephysatlas.anatomy.ClassifierAtlas()
bc = brain_atlas.bc
vol_data = np.load(VOL_PATH, allow_pickle=True)
vol = vol_data['ephys_atlas_vol']
feat_names = list(vol_data['feature_names'])

iAP = int(bc.xyz2i(np.array([[0., AP_M, 0.]]))[0, 1])
print(f'AP {AP_M*1e3:+.1f} mm  ->  iAP={iAP}')

slc = vol[:, iAP, :, feat_names.index(FEATURE)].astype(np.float32)   # (ML, DV)
lbl = brain_atlas.label[iAP, :, :]                                    # (ML, DV)

print(f'label unique values (first 30): {np.unique(lbl)[:30]}')
print(f'label min={lbl.min()}, max={lbl.max()}')
print(f'  lbl > 0 : {(lbl > 0).sum():,} voxels')
print(f'  lbl == {VOID_FLUID_ID}: {(lbl == VOID_FLUID_ID).sum():,} voxels')
print(f'  lbl > 2000: {(lbl > VOID_FLUID_ID).sum():,} voxels')

inside_brain = (lbl > 0) & (lbl != VOID_FLUID_ID)
brain_mask   = inside_brain & np.isfinite(slc)
vals = slc[brain_mask]
print(f'\nMasked voxels: {brain_mask.sum():,}')
print(f'  raw {FEATURE}: min={vals.min():.3f}, median={np.median(vals):.3f}, '
      f'max={vals.max():.3f}')
print(f'  10th pct={np.percentile(vals, 10):.3f}, 90th pct={np.percentile(vals, 90):.3f}')

# ── figure ────────────────────────────────────────────────────────────────────
extent = [bc.xlim[0]*1e6, bc.xlim[1]*1e6, bc.zlim[1]*1e6, bc.zlim[0]*1e6]

fig, axes = plt.subplots(1, 4, figsize=(20, 4), facecolor='#111111')
titles = ['lbl > 0 (all atlas)', f'lbl == {VOID_FLUID_ID} (void_fluid)',
          'brain_mask (used)', f'{FEATURE} raw (masked)']
images = [
    (lbl > 0).astype(float),
    (lbl == VOID_FLUID_ID).astype(float),
    brain_mask.astype(float),
    np.where(brain_mask, slc, np.nan),
]
cmaps = ['Blues', 'Reds', 'Greens', 'inferno']
for ax, img, title, cmap in zip(axes, images, titles, cmaps):
    ax.set_facecolor('#111111')
    ax.imshow(img.T, origin='upper', aspect='equal', extent=extent, cmap=cmap)
    ax.set_title(title, color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('white')

fig.suptitle(f'Mask diagnostic — coronal AP {AP_M*1e3:+.1f} mm', color='white')
fig.tight_layout()

out = Path(__file__).parent.joinpath('figures', 'debug_mask.png')
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#111111')
print(f'\nSaved -> {out}')