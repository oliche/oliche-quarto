"""
Compress and denoise LFP recordings for all benchmark PIDs.

Pipeline per PID:
  1. Temporal compression: 2500 Hz → 250 Hz via decimation (Q=10) with CAR denoising.
  2. Cadzow PPCA denoising on 3 snippets (at 1/6, 3/6, 5/6 of the recording, 3 s each).
  3. 3×3 figure: rows = snippets, cols = original / denoised / difference.
"""
# %%
import sys
from pathlib import Path
sys.path.insert(0, '/Users/olivier/Documents/oliche-quarto/analyses/2026-06-lfp-compression')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import addcopyfighandler  # noqa: F401
import ibldsp.voltage
from one.api import ONE
from compress_fcns import load_pid, cadzow_denoise_probe

sns.set_theme(context='notebook')

STREAM = False
ROOT_OUTPUT = Path('/Users/olivier/scratch/lfp')
FIGURE_DIR = Path('/Users/olivier/scratch/lfp/compression')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
Q = 10
N_JOBS = 4
T_SNIPPET = 3.0   # seconds per snippet (full window for Cadzow)
T_DISPLAY = 2  # seconds shown in figure
SNIPPET_FRACTIONS = [1 / 6, 3 / 6, 5 / 6]

one = ONE(mode='remote', base_url="https://alyx.internationalbrainlab.org")

pids = [
    '1a276285-8b0e-4cc9-9f0a-a3a002978724',  # 00 - Benchmark PIDS start
    '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
    '6638cfb3-3831-4fc2-9327-194b76cf22e1',
    '749cb2b7-e57e-4453-a794-f6230e4d0226',
    'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',  # 05
    'dab512bd-a02d-4c1f-8dbc-9155a163efc0',  # 06 - amazing CSD
    'dc7e9403-19f7-409f-9240-05ee57cb7aea',  # static noise: positive spikes
    'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
    'eebcaf65-7fa4-4118-869d-a084e84530e2',
    'fe380793-8035-414e-b000-09bfe5ece92a',  # Benchmark PIDS stop
]

# %%
for pid in pids:
    ssl, sr, output_path, fs_rs = load_pid(pid, one, ROOT_OUTPUT, stream=STREAM, q=Q)
    nc = sr.nc - sr.nsync

    # Step 1: temporal compression
    output_file = output_path.joinpath('lf_resampled_car.npy')
    if not output_file.exists():
        print(f'{pid}: detecting bad channels...')
        channel_labels = ibldsp.voltage.detect_bad_channels_cbin(sr.file_bin, display=False)
        print(f'{pid}: resampling and denoising...')
        ibldsp.voltage.resample_denoise_lfp_cbin(
            sr.file_bin, q=Q, output=output_file, n_jobs=N_JOBS, channel_labels=channel_labels,
        )
        print(f'{pid}: temporal compression done')

    # Step 2: Cadzow PPCA on 3 snippets
    resampled_map = np.load(output_file, mmap_mode='r')
    ns_total = resampled_map.shape[0]
    snippet_ns = int(T_SNIPPET * fs_rs)

    originals, denoised_list, snippet_starts = [], [], []
    for i_snip, frac in enumerate(SNIPPET_FRACTIONS):
        s0 = int(frac * ns_total)
        snippet_starts.append(s0)
        s1 = s0 + snippet_ns
        orig_file = output_path.joinpath(f'cadzow_orig_{i_snip}.npy')
        den_file = output_path.joinpath(f'cadzow_denoised_{i_snip}.npy')

        if not den_file.exists():
            print(f'{pid}: cadzow snippet {i_snip + 1}/3...')
            wav_orig = np.array(resampled_map[s0:s1, :nc]).T.astype(np.float32)
            wav_den = cadzow_denoise_probe(wav_orig, version=1, r=5, nw=64, fmax=100.0, fs=fs_rs, ppca_k=2.0, gap_threshold=2.0)
            np.save(orig_file, wav_orig)
            np.save(den_file, wav_den)

        originals.append(np.load(orig_file))
        denoised_list.append(np.load(den_file))

    # Step 3: 3×3 figure — rows: snippets, cols: original / denoised / difference
    display_ns = int(T_DISPLAY * fs_rs)
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    col_titles = ['Original', 'Denoised', 'Difference']
    row_labels = [f't = {s0 / fs_rs:.0f} s' for s0 in snippet_starts]

    VMAX = .15 * 1e-3  # µV — shared color scale for all panels
    for row, (orig, den, row_label) in enumerate(zip(originals, denoised_list, row_labels)):
        diff = orig - den
        orig_d, den_d, diff_d = orig[:, :display_ns], den[:, :display_ns], diff[:, :display_ns]

        for col, (data, title) in enumerate(zip([orig_d, den_d, diff_d], col_titles)):
            ax = axes[row, col]
            ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-VMAX, vmax=VMAX, interpolation='none', origin='lower')
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(row_label)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(pid, fontsize=9)
    fig.tight_layout()
    fig_path = FIGURE_DIR.joinpath(f'2026-06-06_cadzow_{pid}.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'{pid}: figure saved → {fig_path}')
