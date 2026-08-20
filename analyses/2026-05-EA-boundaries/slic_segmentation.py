# %%
"""
Unsupervised volumetric brain parcellation via SLIC supervoxels.

Applies 3-D SLIC to the brainwide ephys-atlas encoding volume to discover
spatially compact regions of electrophysiologically similar voxels.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic

CACHE_FILE = Path.home().joinpath('data', 'ephys-atlas', 'encoding_volumes', 'brainwide_ephys_atlas_25um.npz')
DOWNSAMPLE = 2       # spatial downsampling factor (2 → 8× fewer voxels)
N_SEGMENTS = 2_000   # target supervoxels on the downsampled volume
COMPACTNESS = 0.01   # low → feature-driven; high → spatially compact

# %% Load (file cached locally, no download needed)
print(f"Loading {CACHE_FILE}")
data = np.load(CACHE_FILE, allow_pickle=True)

vol = data['ephys_atlas_vol']                             # (456, 528, 320, N) float16
feature_names = data['feature_names']
mu = data['mean_per_feature'].astype(np.float32)
sigma = data['std_per_feature'].astype(np.float32)
print(f"Volume shape: {vol.shape}, {len(feature_names)} features")

# %% Spatial downsampling to reduce memory (~8× fewer voxels)
d = DOWNSAMPLE
vol = vol[::d, ::d, ::d, :]                               # simple stride subsample
print(f"Downsampled volume: {vol.shape}")

# %% Brain mask and z-score normalisation
brain_mask = np.all(np.isfinite(vol), axis=-1)
print(f"Brain voxels: {brain_mask.sum():,} / {brain_mask.size:,}")

vol_z = ((vol.astype(np.float32) - mu) / np.where(sigma > 0, sigma, 1.0)).astype(np.float32)
vol_z[~brain_mask] = 0.0                                  # SLIC cannot handle NaN

# %% Run SLIC
print(f"Running SLIC: n_segments={N_SEGMENTS}, compactness={COMPACTNESS}")
labels = slic(
    vol_z,
    n_segments=N_SEGMENTS,
    compactness=COMPACTNESS,
    channel_axis=-1,
    enforce_connectivity=True,
    start_label=1,
    convert2lab=False,                                     # already normalised
)
labels[~brain_mask] = 0                                    # background = 0
n_found = len(np.unique(labels)) - 1                      # exclude background
print(f"Supervoxels found: {n_found}")

# %% Visualise — three orthogonal mid-slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
cx, cy, cz = [s // 2 for s in vol.shape[:3]]

slices = [
    (labels[cx, :, :], "Coronal (mid-x)"),
    (labels[:, cy, :], "Sagittal (mid-y)"),
    (labels[:, :, cz], "Axial (mid-z)"),
]

for ax, (sl, title) in zip(axes, slices):
    ax.imshow(sl.T, origin='lower', cmap='tab20', interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')

fig.suptitle(f"SLIC supervoxels — {n_found} segments, compactness={COMPACTNESS}")
fig.tight_layout()

out_fig = Path('figures').joinpath('slic_supervoxels_overview.png')
fig.savefig(out_fig, dpi=150)
print(f"Saved {out_fig}")
plt.show()

# %% Save labels for downstream use
out_labels = Path('cache').joinpath(f'slic_labels_{VINTAGE}_n{N_SEGMENTS}_c{COMPACTNESS}.npy')
np.save(out_labels, labels)
print(f"Labels saved to {out_labels}")
