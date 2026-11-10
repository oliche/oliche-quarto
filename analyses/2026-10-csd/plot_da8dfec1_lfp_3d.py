"""
Two 16:9 figures for insertion da8dfec1 (first snippet of `plot_cadzow_csd.py`, t = 2141-2144 s):
  - LFP row only: histology regions, original, Cadzow-denoised (lfpack v05) and difference,
  - the probe insertion in 3D, channels coloured by Allen region, in the CCF atlas brain.
"""
# %%
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import binary_erosion
from brainbox.ephys_plots import plot_brain_regions
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from one.api import ONE

sns.set_theme(context='notebook')

PID = 'da8dfec1-d265-44e8-84ce-6ae9c109b8bd'
CACHE_DIR = Path('/Users/olivier/scratch/lfp/csd_lfpack_v04')
FIGURE_DIR = Path.home().joinpath('Documents', 'figures')
DATE = '2026-10-14'
FS_LF, Q = 2500.0, 10
T_SNIPPET = 3.0  # s
T_START = 4.0  # s, snippet start within the production chunk (first snippet)
PRCTILE = 99
REGIONS = BrainRegions()
FIG_KWARGS = {'dpi': 150}
FIGSIZE = (16, 9)

# %%
one = ONE()
ssl = SpikeSortingLoader(pid=PID, one=one)
channels = ssl.load_channels()
atlas_ids = channels['atlas_id'].astype(int)

# %% Figure 1: LFP, original / denoised / difference with histology
orig, den = (np.load(CACHE_DIR.joinpath(f'{PID}_chunk_{k}.npy')).T for k in ('orig', 'cadzow_v05'))
t0 = int(np.load(CACHE_DIR.joinpath(f'{PID}_chunk_i0.npy'))) / (FS_LF / Q) + T_START
sl = slice(int(T_START * FS_LF / Q), int((T_START + T_SNIPPET) * FS_LF / Q))
orig, den = orig[:, sl] * 1e6, den[:, sl] * 1e6  # µV
vmax = np.nanpercentile(np.abs(den), PRCTILE)

fig, axes = plt.subplots(1, 4, figsize=FIGSIZE, layout='constrained', gridspec_kw={'width_ratios': [0.2, 1, 1, 1]})
plot_brain_regions(atlas_ids, channel_depths=np.arange(atlas_ids.size), brain_regions=REGIONS, ax=axes[0], label='left')
axes[0].set_ylim(-0.5, atlas_ids.size - 0.5)
axes[0].set_title('histology', fontsize=11)
extent = [0, T_SNIPPET, -0.5, atlas_ids.size - 0.5]
for ax, x, title in zip(axes[1:], (orig, den, orig - den), ('original', 'Cadzow-denoised (lfpack v05)', 'difference')):
    im = ax.imshow(x, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower', interpolation='none',
                   extent=extent)
    ax.set_yticks([])
    ax.set_xlabel('time [s]')
    ax.set_title(title, fontsize=11)
    ax.grid(False)
fig.colorbar(im, ax=axes[1:], location='right', shrink=0.9, aspect=30, pad=0.01, label='LFP [µV]')
fig.suptitle(f'{PID}, t = {t0:.0f}-{t0 + T_SNIPPET:.0f} s', fontsize=12)
fig.savefig(FIGURE_DIR.joinpath(f'{DATE}_csd_{PID[:8]}_lfp_histology.png'), **FIG_KWARGS)
plt.close(fig)

# %% Figure 2: probe insertion in 3D
ba = AllenAtlas(25)
xyz = np.c_[channels['x'], channels['y'], channels['z']] * 1e6  # µm, CCF bregma-centred (ml, ap, dv)
# full brain shell: voxels of the subsampled mask that touch the void
step = 3
inside = ba.label[::step, ::step, ::step] != 0  # (ml, ap, dv)
shell = np.argwhere(inside & ~binary_erosion(inside))
surf = ba.bc.i2xyz(shell * step) * 1e6
# probe axis: line through channel sites, extended to 1.5 mm above the topmost one
c = xyz.mean(axis=0)
u = np.linalg.svd(xyz - c)[2][0]
u *= np.sign(u[2])  # pointing to dorsal (z up)
tip = c + u * np.min((xyz - c) @ u)
top = c + u * (np.max((xyz - c) @ u) + 1500)
rgb = REGIONS.rgba[[i[0] for i in REGIONS.id2index(atlas_ids)[1]]][:, :3] / 255

fig = plt.figure(figsize=FIGSIZE, facecolor='white')
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.scatter(*surf.T, clip_on=False, s=4, c=surf[:, 2], cmap='Greys', alpha=0.4, linewidths=0, vmin=-9000, vmax=2000)
ax.plot(*np.c_[tip, top], color='k', lw=1.5)
ax.scatter(*xyz.T, c=rgb, s=70, depthshade=False)
for i, ac in enumerate(np.unique(atlas_ids)):  # label each region at the mean position of its channels, alternating sides
    m = atlas_ids == ac
    sign = 1 if i % 2 else -1
    ax.text(*(xyz[m].mean(axis=0) + [sign * 500, 0, 0]), REGIONS.id2acronym(ac)[0], fontsize=12, fontweight='bold',
            ha='left' if sign > 0 else 'right')
ax.set_axis_off()
ax.patch.set_visible(False)
ax.set_box_aspect(np.ptp(surf, axis=0), zoom=2.0)
ax.set_clip_on(False)
ax.view_init(elev=20, azim=-55)
fig.text(0.01, 0.97, f'{PID}, {ssl.pname}', fontsize=12, va='top')
fig.savefig(FIGURE_DIR.joinpath(f'{DATE}_csd_{PID[:8]}_insertion_3d.png'), **FIG_KWARGS)
plt.close(fig)
