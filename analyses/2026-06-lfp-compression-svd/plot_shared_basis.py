"""
Figures for the shared-spatial-basis sweep (shared_basis_sweep.py): SNR and behaviour decoding against the
basis compression ratio, at constant file size.

Basis compression ratio of an arm = median over probes of (spatial bytes of the mild reference) / (spatial bytes
of the arm); the reference (per-chunk float32 U, ε = 100, α = 14) is 1:1.  Behaviour panels show the change
against the mild reference, mean ± s.e.m. over the 11 probes, with the uncompressed level as a dashed line.
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
COLORS = {100.0: '#2a78d6', 150.0: '#eb6834'}  # categorical slots 1-2 of the validated default palette

arms = pd.read_csv(ROOT.joinpath('2026-09-25_shared-basis_arms.csv'), dtype={'m': str})
wheel = pd.read_csv(ROOT.joinpath('2026-09-25_shared-basis_wheel.csv'), dtype={'m': str})
categ = pd.read_csv(ROOT.joinpath('2026-09-25_shared-basis_categorical.csv'), dtype={'m': str})
old_wheel = pd.read_csv(ROOT.joinpath('2026-09-05_lfp-compression-decoding_metrics.csv'))
old_categ = pd.read_csv(ROOT.joinpath('2026-09-05_lfp-compression-decoding_categorical_metrics.csv'))

size = arms[arms.track == 'size']
ref = size[(size.m == 'inf') & (size.epsilon == 100)].set_index('pid')
size = size.assign(
    basis_cr=size.pid.map(ref.bytes_spatial) / size.bytes_spatial,
    d_snr=size.snr - size.pid.map(ref.snr),
    d_p10=size.snr_chunk_p10 - size.pid.map(ref.snr_chunk_p10),
)

# behaviour: one row per (pid, arm, metric) as change against the mild reference
wheel = wheel.assign(metric='wheel R²', value=wheel.r2)
categ = categ.assign(metric=categ.target.map({'choice': 'choice accuracy', 'feedback': 'feedback accuracy'}),
                     value=categ.balanced_accuracy)
beh = pd.concat([wheel[['pid', 'm', 'epsilon', 'metric', 'value']], categ[['pid', 'm', 'epsilon', 'metric', 'value']]])
ref_beh = beh[(beh.m == 'inf') & (beh.epsilon == 100)].set_index(['pid', 'metric']).value
beh['delta'] = beh.value.values - ref_beh.reindex(pd.MultiIndex.from_frame(beh[['pid', 'metric']])).values
beh = beh.merge(size[['pid', 'm', 'epsilon', 'basis_cr']], on=['pid', 'm', 'epsilon'])

# uncompressed level, as the mean change against the mild reference
unc = pd.concat([
    old_wheel[old_wheel.variant == 'uncompressed'].assign(metric='wheel R²', value=lambda d: d.r2),
    old_categ[old_categ.variant == 'uncompressed'].assign(
        metric=lambda d: d.target.map({'choice': 'choice accuracy', 'feedback': 'feedback accuracy'}),
        value=lambda d: d.balanced_accuracy),
])[['pid', 'metric', 'value']].set_index(['pid', 'metric']).value
unc_delta = (unc - ref_beh.reindex(unc.index)).groupby('metric').mean()


def summary(df, col):
    """Median basis CR, mean and s.e.m. of `col` per arm, ordered by basis CR."""
    g = df.groupby(['epsilon', 'm'])
    out = pd.DataFrame({'cr': g.basis_cr.median(), 'mean': g[col].mean(), 'sem': g[col].sem()}).reset_index()
    return out.sort_values('cr')


def label_arm(m):
    return {'inf': 'per-chunk U', 'inf-f16': 'U float16'}.get(m, f'm={m}')


# %% Figure: SNR and decoding against the basis compression ratio
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
panels = [
    (axes[0, 0], size, 'd_snr', 'SNR change vs mild (dB)'),
    (axes[0, 1], beh[beh.metric == 'wheel R²'], 'delta', 'wheel R² change vs mild'),
    (axes[1, 0], beh[beh.metric == 'choice accuracy'], 'delta', 'choice balanced accuracy change'),
    (axes[1, 1], beh[beh.metric == 'feedback accuracy'], 'delta', 'feedback balanced accuracy change'),
]
for ax, df, col, ylabel in panels:
    for eps, color in COLORS.items():
        s = summary(df[df.epsilon == eps], col)
        ax.errorbar(s.cr, s['mean'], yerr=s['sem'], color=color, lw=2, marker='o', ms=6, capsize=3,
                    label=f'ε = {eps:.0f}')
        if eps == 100.0:
            for _, row in s.iterrows():
                ax.annotate(label_arm(row.m), (row.cr, row['mean']), textcoords='offset points', xytext=(0, 8),
                            ha='center', fontsize=8, color='0.35')
    ax.axhline(0, color='0.5', lw=1)
    metric = {'wheel R² change vs mild': 'wheel R²', 'choice balanced accuracy change': 'choice accuracy',
              'feedback balanced accuracy change': 'feedback accuracy'}.get(ylabel)
    if metric is not None:
        ax.axhline(unc_delta[metric], color='0.2', lw=1.2, ls='--', label='uncompressed')
    ax.set(xscale='log', ylabel=ylabel)
    ax.grid(True, which='both', alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel('basis compression ratio (spatial bytes, mild / arm)')
axes[0, 0].legend(fontsize=9, loc='lower right')
axes[0, 1].legend(fontsize=9, loc='lower right')
fig.suptitle('Shared spatial basis at constant file size (11 probes, mean ± s.e.m.)', fontsize=12)
fig.tight_layout()
fname = f'{TODAY}_shared-basis_vs-basis-cr.png'
for d in FIGURE_DIRS:
    fig.savefig(d.joinpath(fname), dpi=150)
plt.show()

# %% Table: one row per arm
tab = size.groupby(['epsilon', 'm']).agg(
    basis_cr=('basis_cr', 'median'), bytes_ratio=('bytes', lambda b: np.median(b.values / ref.bytes.reindex(
        size.loc[b.index, 'pid']).values)), d_snr=('d_snr', 'median'), d_p10=('d_p10', 'median'))
for metric in ('wheel R²', 'choice accuracy', 'feedback accuracy'):
    tab[f'Δ {metric}'] = beh[beh.metric == metric].groupby(['epsilon', 'm']).delta.mean()
print(tab.sort_values('basis_cr').round(3).to_string())
