"""
Figures and tables for pareto_sweep.py.

1. Projection vs re-decomposition (m = 48, ε = 100) on the size and snr tracks: paired per-probe differences.
2. Pareto of file size vs wheel R² and vs SNR over the (codec, ε, α) grid, mean ± s.e.m. over the 11 probes.

File size is reported two ways, both pooled over the 11 probes (ratio of summed file sizes):
    1/CR        = Σ file bytes / Σ uncompressed bytes (float32 samples at 250 Hz, ns × 384 × 4)
    size / mild = Σ file bytes / Σ mild-tier file bytes
File bytes = format-2 codec datasets (the HDF5 metadata and saturation tables are not counted).
Wheel R² is shown as the paired change against the mild tier (per-chunk U, ε = 100, α = 14), with the
uncompressed level as a dashed line.
"""

# %%
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context='notebook')
ROOT = Path(__file__).parent if '__file__' in globals() else Path.cwd()
FIGURE_DIRS = [Path.home().joinpath('Documents', 'figures'), ROOT.joinpath('figures')]
TODAY = '2026-09-26'
SCRATCH = Path('/Users/olivier/scratch/lfp')
COLORS = {100.0: '#2a78d6', 70.0: '#eb6834', 50.0: '#8a4fd6'}
LINESTYLES = {'48': '-', '32': '--', '24': ':', '16': '-.'}  # by basis size; int8 C drawn with squares

old = pd.read_csv(ROOT.joinpath('2026-09-25_shared-basis_arms.csv'), dtype={'m': str})
arms = pd.read_csv(ROOT.joinpath(f'{TODAY}_pareto_arms.csv'), dtype={'m': str})
categ = pd.read_csv(ROOT.joinpath(f'{TODAY}_pareto_categorical.csv'), dtype={'m': str})
old_wheel = pd.read_csv(ROOT.joinpath('2026-09-05_lfp-compression-decoding_metrics.csv'))
old_categ = pd.read_csv(ROOT.joinpath('2026-09-05_lfp-compression-decoding_categorical_metrics.csv'))

mild = old[(old.m == 'inf') & (old.epsilon == 100) & (old.track == 'size')].set_index('pid')
raw_bytes = pd.Series({pid: np.load(SCRATCH.joinpath(pid, 'lf_resampled_car_cadzow.npy'), mmap_mode='r').size * 4
                       for pid in mild.index})
unc_r2 = old_wheel[old_wheel.variant == 'uncompressed'].set_index('pid').r2
unc_acc = old_categ[old_categ.variant == 'uncompressed'].set_index(['pid', 'target']).balanced_accuracy

# %% 1. Projection vs re-decomposition, paired over probes
COLS = ['bytes', 'bytes_spatial', 'snr', 'snr_chunk_p10', 'snr_delta', 'snr_gamma', 'r2']
for track in ('size', 'snr'):
    a = arms[(arms.m == 'p48') & (arms.track == track)].set_index('pid')[COLS + ['alpha']]
    b = old[(old.m == '48') & (old.epsilon == 100) & (old.track == track)].set_index('pid')[COLS + ['alpha']]
    d = (a - b).loc[a.index]
    d['bytes'], d['bytes_spatial'] = (a.bytes / b.bytes - 1) * 100, (a.bytes_spatial / b.bytes_spatial - 1) * 100
    print(f'\n{track} track: projection − re-decomposition (m = 48, ε = 100); bytes in %')
    print(pd.DataFrame({'proj mean': a.mean(), 'redec mean': b.mean(), 'diff mean': d.mean(), 'diff sem': d.sem()})
          .round(4).to_string())

# %% 2. Pareto: per-point summaries
grid = arms[arms.track == 'grid'].copy()
grid['d_r2'] = grid.r2 - grid.pid.map(mild.r2)
acc = categ[categ.track == 'grid'].pivot_table(index=['pid', 'm', 'epsilon', 'alpha'], columns='target',
                                               values='balanced_accuracy').reset_index()
grid = grid.merge(acc, on=['pid', 'm', 'epsilon', 'alpha'], how='left')
METRICS = ['bytes', 'snr', 'snr_chunk_p10', 'snr_delta', 'snr_gamma', 'r2', 'd_r2', 'choice', 'feedback']
g = grid.groupby(['m', 'epsilon', 'alpha'])
pts = g[METRICS].mean().join(g[METRICS].sem(), rsuffix='_sem').join(g.size().rename('n')).reset_index()
pts['inv_cr'] = g.bytes.sum().values / g.pid.apply(lambda p: raw_bytes[p].sum()).values
pts['vs_mild'] = g.bytes.sum().values / g.pid.apply(lambda p: mild.bytes[p].sum()).values
d_unc = (unc_r2 - mild.r2).mean()
mild_inv_cr = mild.bytes.sum() / raw_bytes[mild.index].sum()


def frontier(df, x, y):
    """Points of df not dominated in (smaller x, larger y), sorted by x."""
    df = df.sort_values(x)
    keep, best = [], -np.inf
    for i, row in df.iterrows():
        if row[y] > best:
            keep.append(i)
            best = row[y]
    return df.loc[keep].sort_values(x)


def label(m, eps):
    if m == 'inf':
        return 'per-chunk U, ε = 100'
    return f"shared m = {m[1:].removesuffix('q8')}{', C int8' if m.endswith('q8') else ''}, ε = {eps:.0f}"


# %% Figure: 1/CR vs Δ wheel R² (top) and SNR (bottom); left: rank threshold ε at m = 48, right: basis at ε = 100
BASIS_COLORS = {'48': '#2a78d6', '32': '#1baf7a', '24': '#eb6834', '16': '#8a4fd6'}
PANELS = [  # (title, [(m, ε, color, linestyle, marker)])
    ('rank threshold ε (shared m = 48)',
     [('inf', 100.0, '0.15', '-', 'o')] + [('p48', e, COLORS[e], '-', 'o') for e in (100.0, 70.0, 50.0)]),
    ('shared basis size and C precision (ε = 100)',
     [('inf', 100.0, '0.15', '-', 'o')] + [(f'p{k}', 100.0, c, '-', 'o') for k, c in BASIS_COLORS.items()]
     + [(f'p{k}q8', 100.0, BASIS_COLORS[k], '--', 's') for k in ('48', '24')]),
]
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True)
for col, (title, curves) in enumerate(PANELS):
    for m, eps, color, ls, marker in curves:
        d = pts[(pts.m == m) & (pts.epsilon == eps)].sort_values('inv_cr')
        for row, y in enumerate(('d_r2', 'snr')):
            yerr = d[f'{y}_sem'] if (m, eps) in (('inf', 100.0), ('p48', 100.0)) else None  # s.e.m. on references
            axes[row, col].errorbar(d.inv_cr * 100, d[y], yerr=yerr, color=color, ls=ls, lw=1.6, marker=marker, ms=4,
                                    capsize=2, alpha=0.85, label=label(m, eps))
        # α labels: per-chunk U and m = 48 on the left, per-chunk U and the recommended m = 32 on the right
        if (m, eps) in ((('inf', 100.0), ('p48', 100.0)), (('inf', 100.0), ('p32', 100.0)))[col]:
            for row, y in enumerate(('d_r2', 'snr')):
                for _, r in d.iterrows():
                    shared = m != 'inf'  # shared labels above-left of the curve bundle, per-chunk U below-right
                    axes[row, col].annotate(f'α={r.alpha:g}', (r.inv_cr * 100, r[y]), textcoords='offset points',
                                            xytext=(-5, 9) if shared else (5, -11), ha='right' if shared else 'left',
                                            fontsize=7.5, color=color, fontweight='bold' if shared else 'normal',
                                            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
    axes[0, col].set_title(title, fontsize=11, pad=36)
    axes[0, col].axhline(0, color='0.5', lw=1)
    axes[0, col].axhline(d_unc, color='0.2', lw=1.2, ls=':', label='uncompressed')
    axes[0, col].plot(mild_inv_cr * 100, 0, marker='*', ms=14, color='0.15', ls='none', label='mild tier (α = 14)')
    axes[0, col].legend(fontsize=8, loc='upper left')
for ax in axes.flat:
    ax.set(xscale='log')
    ax.set_xticks([0.8, 1, 1.5, 2, 3, 4], labels=['0.8', '1', '1.5', '2', '3', '4'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(True, which='both', alpha=0.3)
for ax in axes[0]:
    sec = ax.secondary_xaxis('top', functions=(lambda x: x / (100 * mild_inv_cr), lambda x: x * 100 * mild_inv_cr))
    sec.set_xlabel('file size / mild tier')
    sec.set_xticks([0.5, 0.75, 1, 1.5, 2], labels=['0.5', '0.75', '1', '1.5', '2'])
    sec.xaxis.set_minor_formatter(plt.NullFormatter())
for ax in axes[1]:
    ax.set_xlabel('file size / uncompressed (1/CR, %)')
for row, ylabel in enumerate(('mean wheel R² change vs mild tier', 'mean reconstruction SNR (dB)')):
    axes[row, 0].set_ylabel(ylabel)
fig.suptitle('File size vs behaviour decoding and SNR (11 probes, mean ± s.e.m.)', fontsize=12)
fig.tight_layout()
fname = f'{TODAY}_pareto_size-vs-decoding.png'
for d in FIGURE_DIRS:
    fig.savefig(d.joinpath(fname), dpi=150)
plt.show()

# %% Figure: the v04 run only (shared m = 32, ε = 100, α = 14 / 7 / 2.5) against the mild baseline
V04_DATE = '2026-09-27'
v04 = pts[(pts.m == 'p32') & (pts.epsilon == 100) & pts.alpha.isin([14, 7, 2.5])].sort_values('inv_cr')
base = pts[(pts.m == 'inf') & (pts.epsilon == 100) & (pts.alpha == 14)]
V04_COLOR = BASIS_COLORS['32']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
for ax, y, ylabel in zip(axes, ('d_r2', 'snr'), ('mean wheel R² change vs mild tier', 'mean reconstruction SNR (dB)')):
    ax.errorbar(v04.inv_cr * 100, v04[y], yerr=v04[f'{y}_sem'], color=V04_COLOR, lw=1.8, marker='o', ms=7,
                capsize=3, label='v04: shared m = 32, ε = 100')
    ax.errorbar(base.inv_cr * 100, base[y], yerr=base[f'{y}_sem'], color='0.15', marker='*', ms=15, ls='none',
                capsize=3, label='mild baseline (per-chunk U, α = 14)')
    # v03 as shipped: format 1 is 1 / 0.71 the size of format 2 at identical decoding
    ax.plot(base.inv_cr * 100 / 0.71, base[y], marker='*', ms=15, mfc='none', mec='0.15', ls='none',
            label='v03 mild as shipped (format 1, estimated size)')
    for _, r in v04.iterrows():  # labels above-left, the lowest α left of its point (clear of the uncompressed line)
        ax.annotate(f'α = {r.alpha:g}', (r.inv_cr * 100, r[y]), textcoords='offset points',
                    xytext=(4, -12) if r.alpha == v04.alpha.min() else (-8, 10),
                    ha='left' if r.alpha == v04.alpha.min() else 'right',
                    va='top' if r.alpha == v04.alpha.min() else 'baseline', fontsize=9, color=V04_COLOR,
                    fontweight='bold')
    ax.set(xscale='log', xlabel='file size / uncompressed (1/CR, %)', ylabel=ylabel)
    ax.set_xticks([0.8, 1, 1.5, 2, 2.5, 3], labels=['0.8', '1', '1.5', '2', '2.5', '3'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(True, which='both', alpha=0.3)
axes[0].axhline(0, color='0.5', lw=1)
axes[0].axhline(d_unc, color='0.2', lw=1.2, ls=':', label='uncompressed')
axes[0].legend(fontsize=8, loc='upper left')
fig.suptitle('v04 tiers vs the mild baseline (11 probes, mean ± s.e.m.)', fontsize=12)
fig.tight_layout()
fname = f'{V04_DATE}_pareto_v04-tiers.png'
for d in FIGURE_DIRS:
    fig.savefig(d.joinpath(fname), dpi=150)
plt.show()

# %% Per-probe robustness: median over probes of the paired change against m = 48 at the same α (ε = 100)
e100 = grid[grid.epsilon == 100].pivot_table(index=['pid', 'alpha'], columns='m', values='r2')
paired = e100.sub(e100['p48'], axis=0).drop(columns='p48')
print('\nwheel R² change vs m = 48 at the same α, ×1000: median over probes (all α pooled), n probes worse')
print(pd.DataFrame({'median': paired.median() * 1000, 'mean': paired.mean() * 1000,
                    'frac_worse': (paired < 0).mean()}).round(2).to_string())
print('\nmedian over probes of Δ wheel R² vs mild, ×1000')
print((grid[grid.epsilon == 100].groupby(['m', 'alpha']).d_r2.median().unstack(0) * 1000).round(2).to_string())
print(f'median uncompressed Δ: {(unc_r2 - mild.r2).median() * 1000:.2f}')

# %% Table: every point
tab = pts.assign(unc_gap_pct=lambda d: 100 * d.d_r2 / d_unc).set_index(['m', 'epsilon', 'alpha'])
print(f'mild 1/CR {100 * mild_inv_cr:.2f} %, uncompressed Δ wheel R² {d_unc:.4f}')
tab['inv_cr_pct'] = 100 * tab.inv_cr
print(tab[['n', 'inv_cr_pct', 'vs_mild', 'snr', 'snr_chunk_p10', 'snr_delta', 'snr_gamma', 'd_r2', 'd_r2_sem', 'unc_gap_pct', 'choice',
           'feedback']].round(4).to_string())
print('uncompressed choice / feedback:', unc_acc.groupby('target').mean().round(4).to_dict())
