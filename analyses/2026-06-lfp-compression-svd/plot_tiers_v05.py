"""
The lfpack tiers figure of `plot_pareto.py` (v04 tiers and the mild baseline) with the v05 tiers added, decoding
expressed as the paired change against each version's own uncompressed signal:

    v04 tiers, mild, v03 mild    against the June Cadzow output (`lf_resampled_car_cadzow.npy`)
    v05 tiers                    against the v05 Cadzow output of the production pipeline (`decode_v05.py`)

so that the codec cost is compared regardless of the upstream preprocessing and Cadzow.  Mean ± s.e.m. over the
insertions of each version; file size is the summed file size over the summed float32 uncompressed size at 250 Hz.
"""

# %%
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context='notebook')
ROOT = Path(__file__).parent
FIGURE_DIRS = [Path.home().joinpath('Documents', 'figures'), ROOT.joinpath('figures')]
DATE = '2026-10-11'
SCRATCH = Path('/Users/olivier/scratch/lfp')
TARGETS = {'wheel': 'wheel R²', 'feedback': 'feedback accuracy'}
ALPHAS = [14.0, 7.0, 2.5]
COLORS = {'v04': '#2a78d6', 'v05': '#2ca02c', 'base': '0.15'}


def load_v04():
    """Per-insertion wheel / feedback / bytes of the v04 tiers and the mild baseline, and the June uncompressed."""
    arms = pd.read_csv(ROOT.joinpath('2026-09-26_pareto_arms.csv'), dtype={'m': str})
    categ = pd.read_csv(ROOT.joinpath('2026-09-26_pareto_categorical.csv'), dtype={'m': str})
    old_wheel = pd.read_csv(ROOT.joinpath('2026-09-05_lfp-compression-decoding_metrics.csv'))
    old_categ = pd.read_csv(ROOT.joinpath('2026-09-05_lfp-compression-decoding_categorical_metrics.csv'))
    fb = categ[categ.target == 'feedback'].set_index(['pid', 'm', 'epsilon', 'alpha', 'track']).balanced_accuracy
    g = arms[(arms.track == 'grid') & (arms.epsilon == 100)].set_index(['pid', 'm', 'epsilon', 'alpha', 'track'])
    g = g.assign(feedback=fb).reset_index()
    tiers = {'v04': g[(g.m == 'p32') & g.alpha.isin(ALPHAS)], 'base': g[(g.m == 'inf') & (g.alpha == 14)]}
    tiers = {k: v.rename(columns={'r2': 'wheel'})[['pid', 'alpha', 'wheel', 'feedback', 'bytes']]
             for k, v in tiers.items()}
    unc = pd.DataFrame({
        'wheel': old_wheel[old_wheel.variant == 'uncompressed'].set_index('pid').r2,
        'feedback': old_categ[(old_categ.variant == 'uncompressed') & (old_categ.target == 'feedback')]
        .set_index('pid').balanced_accuracy,
    })
    unc['bytes'] = pd.Series({p: np.load(SCRATCH.joinpath(p, 'lf_resampled_car_cadzow.npy'), mmap_mode='r').size * 4
                              for p in unc.index})
    return tiers, unc


def load_v05():
    """Per-insertion v05 tiers and v05 uncompressed, insertions with all four rows only."""
    d = pd.read_csv(ROOT.joinpath('2026-10-11_decoding_v05.csv'))
    d = d[d.tier.str.startswith('v05')]
    d = d[d.pid.isin(d.groupby('pid').filter(lambda x: len(x) == 4).pid)]
    unc = d[d.tier == 'v05 uncompressed'].set_index('pid')[['wheel', 'feedback', 'bytes']]
    tiers = d[d.tier != 'v05 uncompressed'].assign(
        alpha=lambda x: x.tier.map({'v05 small': 14.0, 'v05 default': 7.0, 'v05 fine': 2.5}))
    return tiers[['pid', 'alpha', 'wheel', 'feedback', 'bytes']], unc


def summarize(tiers, unc):
    """Per α: mean ± s.e.m. of the paired change against uncompressed, and 1/CR pooled over insertions."""
    rows = []
    for alpha, d in tiers.groupby('alpha'):
        d = d.set_index('pid')
        row = dict(alpha=alpha, n=len(d), inv_cr=d.bytes.sum() / unc.bytes[d.index].sum())
        for t in TARGETS:
            delta = d[t] - unc[t][d.index]
            row[t], row[f'{t}_sem'] = delta.mean(), delta.sem()
        rows.append(row)
    return pd.DataFrame(rows).sort_values('inv_cr')


# %%
if __name__ == '__main__':
    v04_tiers, v04_unc = load_v04()
    v05_tiers, v05_unc = load_v05()
    pts = {'v04': summarize(v04_tiers['v04'], v04_unc), 'base': summarize(v04_tiers['base'], v04_unc),
           'v05': summarize(v05_tiers, v05_unc)}
    labels = {'v04': f"v04: shared m = 32, ε = 100 (n = {pts['v04'].n.iloc[0]})",
              'v05': f"v05: Cadzow full grid + shrinkage (n = {pts['v05'].n.iloc[0]})",
              'base': 'mild baseline (per-chunk U, α = 14)'}
    for k, p in pts.items():
        print(k, '\n', p.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, (t, tlabel) in zip(axes, TARGETS.items()):
        for k in ('v04', 'v05'):
            p = pts[k]
            ax.errorbar(p.inv_cr * 100, p[t], yerr=p[f'{t}_sem'], color=COLORS[k], lw=1.8, marker='o', ms=7,
                        capsize=3, label=labels[k])
            for _, r in p.iterrows():  # v04 labels above-left, v05 below-right
                ax.annotate(f'α = {r.alpha:g}', (r.inv_cr * 100, r[t]), textcoords='offset points',
                            xytext=(-8, 8) if k == 'v04' else (8, -14), ha='right' if k == 'v04' else 'left',
                            fontsize=9, color=COLORS[k], fontweight='bold')
        b = pts['base']
        ax.errorbar(b.inv_cr * 100, b[t], yerr=b[f'{t}_sem'], color=COLORS['base'], marker='*', ms=15, ls='none',
                    capsize=3, label=labels['base'])
        # v03 as shipped: format 1 is 1 / 0.71 the size of format 2 at identical decoding
        ax.plot(b.inv_cr * 100 / 0.71, b[t], marker='*', ms=15, mfc='none', mec=COLORS['base'], ls='none',
                label='v03 mild as shipped (format 1, estimated size)')
        ax.axhline(0, color='0.2', lw=1.2, ls=':', label='uncompressed')
        ax.set(xscale='log', xlabel='file size / uncompressed (1/CR, %)',
               ylabel=f'Δ {tlabel} vs uncompressed')
        ax.set_xticks([0.6, 0.8, 1, 1.5, 2, 2.5, 3], labels=['0.6', '0.8', '1', '1.5', '2', '2.5', '3'])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(True, which='both', alpha=0.3)
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('lfpack tiers against their own uncompressed signal (mean ± s.e.m.)', fontsize=12)
    fig.tight_layout()
    for d in FIGURE_DIRS:
        fig.savefig(d.joinpath(f'{DATE}_pareto_v04-v05-tiers.png'), dpi=150)
    plt.show()
