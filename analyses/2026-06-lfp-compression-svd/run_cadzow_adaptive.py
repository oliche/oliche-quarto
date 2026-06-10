"""
Re-run Cadzow denoising with adaptive gap-threshold rank selection on all PIDs.

Loads existing cadzow_orig_{i}.npy snippets, applies cadzow_denoise_probe with
gap_threshold=2.0 (all other params unchanged), saves:
  - cadzow_denoised_adaptive_{i}.npy  (per PID output folder)
  - 2026-06-06_cadzow_adaptive_{pid}.png  (4-column comparison figure)

Cols: Original | Fixed rank (r=5) | Adaptive rank (gap≥2.0, max r=5) | Difference
"""
# %%
import sys
from pathlib import Path
sys.path.insert(0, '/Users/olivier/Documents/oliche-quarto/analyses/2026-06-lfp-compression')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler  # noqa: F401
from compress_fcns import cadzow_denoise_probe

sns.set_theme(context='notebook')

ROOT_OUTPUT = Path('/Users/olivier/scratch/lfp')
FIGURE_DIR = Path('/Users/olivier/scratch/lfp/compression')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FS_RS = 250.0
N_JOBS = 4
T_DISPLAY = 2.0
VMAX = 0.15e-3

GAP_THRESHOLD = 2.0

pids = [
    '1a276285-8b0e-4cc9-9f0a-a3a002978724',
    '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
    '6638cfb3-3831-4fc2-9327-194b76cf22e1',
    '749cb2b7-e57e-4453-a794-f6230e4d0226',
    'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',
    'dab512bd-a02d-4c1f-8dbc-9155a163efc0',
    'dc7e9403-19f7-409f-9240-05ee57cb7aea',
    'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
    'eebcaf65-7fa4-4118-869d-a084e84530e2',
    'fe380793-8035-414e-b000-09bfe5ece92a',
]

# %%
display_ns = int(T_DISPLAY * FS_RS)
col_titles = ['Original', f'Fixed rank (r=5)', f'Adaptive (gap≥{GAP_THRESHOLD}, max r=5)', 'Δ (adaptive − fixed)']

for pid in pids:
    output_path = ROOT_OUTPUT.joinpath(pid)
    originals, fixed_list, adaptive_list = [], [], []

    for i_snip in range(3):
        orig_file = output_path.joinpath(f'cadzow_orig_{i_snip}.npy')
        fixed_file = output_path.joinpath(f'cadzow_denoised_{i_snip}.npy')
        adaptive_file = output_path.joinpath(f'cadzow_denoised_adaptive_{i_snip}.npy')

        if not orig_file.exists():
            print(f'{pid}: orig file missing, skipping')
            break

        wav_orig = np.load(orig_file)

        if not adaptive_file.exists():
            print(f'{pid}: snippet {i_snip + 1}/3 — running adaptive Cadzow...')
            wav_adaptive = cadzow_denoise_probe(
                wav_orig, version=1, r=5, nw=64, fmax=100.0, fs=FS_RS,
                ppca_k=2.0, gap_threshold=GAP_THRESHOLD, n_jobs=N_JOBS,
            )
            np.save(adaptive_file, wav_adaptive)
        else:
            wav_adaptive = np.load(adaptive_file)

        originals.append(wav_orig)
        fixed_list.append(np.load(fixed_file))
        adaptive_list.append(wav_adaptive)
    else:
        # Figure: 3 rows × 4 cols
        fig, axes = plt.subplots(3, 4, figsize=(22, 10))
        fig.suptitle(pid, fontsize=9)

        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=10)

        for row, (orig, fixed, adaptive) in enumerate(zip(originals, fixed_list, adaptive_list)):
            panels = [
                orig[:, :display_ns],
                fixed[:, :display_ns],
                adaptive[:, :display_ns],
                (adaptive - fixed)[:, :display_ns],
            ]
            vmaxes = [VMAX, VMAX, VMAX, VMAX / 5]  # tighter scale for diff panel
            for col, (data, vmax) in enumerate(zip(panels, vmaxes)):
                ax = axes[row, col]
                ax.imshow(
                    data, aspect='auto', cmap='RdBu_r',
                    vmin=-vmax, vmax=vmax, interpolation='none', origin='lower',
                )
                if col == 0:
                    ax.set_ylabel(f'snippet {row + 1}', fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

        fig.tight_layout()
        fig_path = FIGURE_DIR.joinpath(f'2026-06-06_cadzow_adaptive_{pid}.png')
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f'{pid}: figure saved → {fig_path}')