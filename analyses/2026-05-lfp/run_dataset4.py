"""Run the LFP pipeline on dataset 4 and save figures.

Evaluation metric: peak cross-correlation on CSD (1st and 2nd order) between
spatially matched NP1 / NP2 channels.
  bandpass → preprocessing → CSD(n) → xcorr(NP1, NP2)
"""
import sys
sys.path.insert(0, '/home/olivier/PycharmProjects/EphysAtlas/ibl-neuropixel/src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import spikeglx
from pathlib import Path

import ibldsp.voltage
import ibldsp.cadzow
from lfp_pipeline import (
    cadzow_geometry, adaptive_svd_denoise,
    cadzow_dense_csd, match_dense_channels, evaluate_dense_csd_xcorr,
    evaluate_csd_xcorr, sweep_csd_xcorr, sweep_cadzow_params,
    summary_table, plot_pipeline_comparison, plot_xcorr_traces, plot_singular_values,
)

root_path = Path('/datadisk/Data/2026/np1np2')
out_path = Path('/home/olivier/Insync/olivier.winter@internationalbrainlab.org/Google Drive/Clauderies/lfp/figures')
out_path.mkdir(exist_ok=True)

number = 4


def get_lf_file(root_path, number, probe_type):
    session_path = root_path / f"{number:03d}"
    suffix = 'imec1' if probe_type == 'NP1' else 'imec0c'
    for lf_file in session_path.rglob('*.lf.*bin'):
        if lf_file.parent.name.endswith(suffix):
            return lf_file


sr_np1 = spikeglx.Reader(get_lf_file(root_path, number, 'NP1'))
sr_np2 = spikeglx.Reader(get_lf_file(root_path, number, 'NP2'))
print(f"NP1: {sr_np1.nc} ch  fs={sr_np1.fs} Hz  version={sr_np1.major_version}")
print(f"NP2: {sr_np2.nc} ch  fs={sr_np2.fs} Hz  version={sr_np2.major_version}")

sample_slice = slice(30_000, int(30_000 * 4))  # 3 seconds

# Channel matching
intercept, slope = 2601.3, 1.179
idx_np1 = np.where(
    (sr_np1.geometry['y'] >= intercept + slope * sr_np2.geometry['y'].min()) &
    (sr_np1.geometry['y'] <= intercept + slope * sr_np2.geometry['y'].max())
)[0]
idx_np2 = np.argmin(np.abs(
    (intercept + slope * sr_np2.geometry['y'])[:, None] - sr_np1.geometry['y'][idx_np1][None, :]
), axis=0)
print(f"Matched {len(idx_np1)} NP1 channels to {len(np.unique(idx_np2))} unique NP2 channels")

# Load raw
print("Loading raw data...")
raw_np1 = sr_np1[sample_slice, :-1].T
raw_np2 = sr_np2[sample_slice, :-1].T

h1 = sr_np1.geometry
h2 = sr_np2.geometry

# Bandpass (shared first step)
print("Bandpassing...")
bp_np1 = ibldsp.voltage.destripe_lfp(raw_np1.copy().astype(float), fs=sr_np1.fs, h=h1,
                                       neuropixel_version=1, k_filter=None, channel_labels=None)
bp_np2 = ibldsp.voltage.destripe_lfp(raw_np2.copy().astype(float), fs=sr_np2.fs, h=h2,
                                       neuropixel_version=2, k_filter=None, channel_labels=None)

# Denoising stages applied to bandpassed data (no CAR/PCA — CSD already removes DC/common offset)
print("Running NP1 stages...")
data_np1 = {
    'bandpass':    bp_np1,
    'cadzow_orig': ibldsp.cadzow.cadzow_np1(bp_np1.copy(), fs=sr_np1.fs, rank=4, fmax=200, h=h1),
    'cadzow_geom': cadzow_geometry(bp_np1.copy(), fs=sr_np1.fs, rank=4, fmax=200, h=h1),
    'svd_adapt':   adaptive_svd_denoise(bp_np1.copy(), h=h1),
}

print("Running NP2 stages...")
data_np2 = {
    'bandpass':    bp_np2,
    'cadzow_orig': ibldsp.cadzow.cadzow_np1(bp_np2.copy(), fs=sr_np2.fs, rank=4, fmax=200, h=h2),
    'cadzow_geom': cadzow_geometry(bp_np2.copy(), fs=sr_np2.fs, rank=4, fmax=200, h=h2),
    'svd_adapt':   adaptive_svd_denoise(bp_np2.copy(), h=h2),
}

# ─── Evaluation: bandpass → preprocessing → CSD(n) → xcorr ──────────────────
print("\nEvaluating via CSD cross-correlation (n=1 and n=2)...")
metrics_csd = evaluate_csd_xcorr(
    data_np1, data_np2, h1, h2, idx_np1, idx_np2,
    fs=sr_np1.fs, csd_orders=(1, 2), n_pairs=80
)

print("\n--- CSD n=1 ---")
summary_table({k: v for k, v in metrics_csd.items() if k.endswith('_csd1')})
print("\n--- CSD n=2 ---")
summary_table({k: v for k, v in metrics_csd.items() if k.endswith('_csd2')})

# ─── Figure 1: CSD xcorr box plots side by side ──────────────────────────────
stages = list(data_np1.keys())
fig1, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=False)
for ax, n in zip(axes, (1, 2)):
    m_sub = {s: metrics_csd[f'{s}_csd{n}'] for s in stages}
    plot_pipeline_comparison(m_sub,
                             title=f'Dataset {number:03d} — CSD{n} xcorr (NP1 vs NP2)',
                             fig=fig1, ax=ax)
fig1.tight_layout()
fig1.savefig(out_path / f'fig1_csd_xcorr_ds{number:03d}.png', dpi=150)
print("\nSaved fig1")

# ─── Figure 2: trace + xcorr on CSD for a representative channel pair ────────
np_ch2 = 40
ch_np1_ex = idx_np1[np.argmin(np.abs(
    sr_np1.geometry['y'][idx_np1] - (intercept + slope * sr_np2.geometry['y'][np_ch2])
))]

# Build CSD dicts for trace panels
for n in (1, 2):
    csd_np1 = {s: ibldsp.voltage.current_source_density(d, h=h1, n=n, scale=False)
               for s, d in data_np1.items()}
    csd_np2 = {s: ibldsp.voltage.current_source_density(d, h=h2, n=n, scale=False)
               for s, d in data_np2.items()}
    fig2, _ = plot_xcorr_traces(csd_np1, csd_np2, ch_np1_ex, np_ch2, fs=sr_np1.fs)
    fig2.suptitle(f'Dataset {number:03d} — CSD{n}  NP1 ch{ch_np1_ex} vs NP2 ch{np_ch2}', y=1.01)
    fig2.savefig(out_path / f'fig2_csd{n}_traces_ds{number:03d}.png', dpi=150, bbox_inches='tight')
    print(f"Saved fig2 CSD{n}")

# ─── Figure 3: singular values (NP1 and NP2) ─────────────────────────────────
fig3a, _ = plot_singular_values(bp_np1, h=h1, title=f'NP1 singular values — dataset {number:03d}')
fig3a.savefig(out_path / f'fig3a_svd_np1_ds{number:03d}.png', dpi=150)
fig3b, _ = plot_singular_values(bp_np2, h=h2, title=f'NP2 singular values — dataset {number:03d}')
fig3b.savefig(out_path / f'fig3b_svd_np2_ds{number:03d}.png', dpi=150)
print("Saved fig3a/b")

# ─── Cadzow parameter sweep evaluated via CSD xcorr ─────────────────────────
print("\nRunning Cadzow parameter sweep (NP1, rank × fmax)...")
sweep_np1 = sweep_cadzow_params(
    raw_np1, sr_np1, h1,
    param_ranges=dict(rank=[2, 4, 6, 8], fmax=[100, 200, 300]),
    version=1
)
sweep_results = sweep_csd_xcorr(
    sweep_np1, bp_np2, h1, h2, idx_np1, idx_np2,
    fs=sr_np1.fs, csd_orders=(1, 2), n_pairs=80
)

ranks = [2, 4, 6, 8]
fmaxs = [100, 200, 300]

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
for ax, n in zip(axes4, (1, 2)):
    mat = np.array([[sweep_results[f'rank{r}_fmax{f}'][f'csd{n}'] for f in fmaxs] for r in ranks])
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=mat.min(), vmax=mat.max())
    ax.set_xticks(range(len(fmaxs))); ax.set_xticklabels([str(f) for f in fmaxs])
    ax.set_yticks(range(len(ranks))); ax.set_yticklabels([str(r) for r in ranks])
    ax.set(xlabel='fmax (Hz)', ylabel='rank',
           title=f'Cadzow sweep — CSD{n} xcorr (NP1→NP2) ds{number:03d}')
    plt.colorbar(im, ax=ax, label='mean peak xcorr')
    for i, r in enumerate(ranks):
        for j, f in enumerate(fmaxs):
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                    color='w', fontsize=9, fontweight='bold')

print("\nCadzow sweep results:")
print(f"{'Params':<20} {'CSD1 xcorr':>12} {'CSD2 xcorr':>12}")
print('-' * 46)
for key, res in sweep_results.items():
    print(f"{key:<20} {res['csd1']:12.4f} {res['csd2']:12.4f}")

fig4.tight_layout()
fig4.savefig(out_path / f'fig4_cadzow_sweep_csd_ds{number:03d}.png', dpi=150)
print("\nSaved fig4")

# ─── Dense interleaved CSD + Cadzow ──────────────────────────────────────────
print("\nRunning dense CSD + Cadzow experiment...")
dense_methods = {}
for rank in (2, 4):
    for label, bp in [('np1', bp_np1), ('np2', bp_np2)]:
        h = h1 if label == 'np1' else h2
        dn, h_dn = cadzow_dense_csd(bp, fs=sr_np1.fs, h=h, rank=rank, fmax=200)
        dense_methods[(label, rank)] = (dn, h_dn)

# Match dense channels between probes (same coordinate transform as raw channels)
idx_d1, idx_d2 = match_dense_channels(
    dense_methods[('np1', 2)][1], dense_methods[('np2', 2)][1],
    intercept=intercept, slope=slope, n_pairs=80
)
print(f"Dense channel pairs: {len(idx_d1)}")

# Baseline: plain dense CSD (no Cadzow) on bandpass
from lfp_pipeline import compute_dense_csd
dense_bp_np1, h_d1 = compute_dense_csd(bp_np1, h1)
dense_bp_np2, h_d2 = compute_dense_csd(bp_np2, h2)
idx_d1_ref, idx_d2_ref = match_dense_channels(h_d1, h_d2, intercept, slope, n_pairs=80)

results_dense = {}
results_dense['dense_csd_bandpass'] = evaluate_dense_csd_xcorr(
    dense_bp_np1, dense_bp_np2, idx_d1_ref, idx_d2_ref, fs=sr_np1.fs)
for rank in (2, 4):
    dn1, _ = dense_methods[('np1', rank)]
    dn2, _ = dense_methods[('np2', rank)]
    results_dense[f'dense_csd_cadzow_r{rank}'] = evaluate_dense_csd_xcorr(
        dn1, dn2, idx_d1, idx_d2, fs=sr_np1.fs)

print("\nDense CSD results (xcorr on denoised dense CSD):")
summary_table(results_dense)

# Compare against regular CSD1 (cadzow_geom, rank=4) as reference
print("\nReference — regular CSD1 (cadzow_geom rank=4):")
summary_table({'cadzow_geom_csd1': metrics_csd['cadzow_geom_csd1']})

# Figure 5: box plot dense vs regular
fig5, ax5 = plt.subplots(figsize=(9, 5))
all_metrics = {'cadzow_geom\n(regular CSD1)': metrics_csd['cadzow_geom_csd1']} | results_dense
plot_pipeline_comparison(all_metrics,
                         title=f'Dataset {number:03d} — dense CSD Cadzow vs regular CSD1',
                         fig=fig5, ax=ax5)
fig5.savefig(out_path / f'fig5_dense_csd_ds{number:03d}.png', dpi=150)
print("\nSaved fig5")

# Figure 6: trace panel — dense CSD bandpass vs dense CSD cadzow rank=2
# pick a channel pair in the middle of the matched range
mid = len(idx_d1) // 2
ch_d1, ch_d2 = int(idx_d1[mid]), int(idx_d2[mid])
dense_trace_data_np1 = {
    'dense_csd_bandpass':    dense_bp_np1,
    'dense_csd_cadzow_r2':   dense_methods[('np1', 2)][0],
    'dense_csd_cadzow_r4':   dense_methods[('np1', 4)][0],
}
dense_trace_data_np2 = {
    'dense_csd_bandpass':    dense_bp_np2,
    'dense_csd_cadzow_r2':   dense_methods[('np2', 2)][0],
    'dense_csd_cadzow_r4':   dense_methods[('np2', 4)][0],
}
fig6, _ = plot_xcorr_traces(dense_trace_data_np1, dense_trace_data_np2,
                             ch_d1, ch_d2, fs=sr_np1.fs)
fig6.suptitle(f'Dataset {number:03d} — dense CSD traces  ch{ch_d1} vs ch{ch_d2}', y=1.01)
fig6.savefig(out_path / f'fig6_dense_csd_traces_ds{number:03d}.png', dpi=150, bbox_inches='tight')
print("Saved fig6")

print(f"\nAll figures saved to {out_path}")
