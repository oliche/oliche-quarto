"""
Behaviour decoding (wheel R², choice and feedback balanced accuracy) on the 11 benchmark insertions for the
uncompressed signal, the three v03 tiers (format 1, per-chunk U) and the three v04 tiers (shared basis m = 32).

v03 aggressive / default come from the tuning run, v03 mild (ε = 100, α = 14) from the full grid, v04 tiers from the
Pareto sweep (m = 32, ε = 100, α = 14 / 7 / 2.5).  Writes ``<DATE>_decoding_tiers.csv`` (tidy, one row per
insertion and tier) and ``<DATE>_decoding_tiers.png``.
"""

# %%
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(context='notebook')

OUT_DIR = Path(__file__).parent
FIGURE_DIRS = [Path.home().joinpath('Documents', 'figures'), OUT_DIR.joinpath('figures')]
DATE = '2026-10-02'
TIERS = ['uncompressed', 'v03 aggressive', 'v03 default', 'v03 mild', 'v04 small', 'v04 default', 'v04 fine']
TARGETS = {'wheel': 'wheel speed R²', 'choice': 'choice balanced accuracy', 'feedback': 'feedback balanced accuracy'}


def load():
    """Tidy table: pid, tier, wheel, choice, feedback, bytes."""
    tune = pd.read_csv(OUT_DIR.joinpath('2026-08-31_compression_tuning_metrics.csv'))
    grid = pd.read_csv(OUT_DIR.joinpath('2026-09-01_full_grid_metrics.csv'))
    old_names = {'uncompressed': 'uncompressed', 'alpha28 (default)': 'v03 default',
                 'aggressive (eps450, alpha96, shipped)': 'v03 aggressive'}
    old = tune[tune['tier'].isin(old_names)].assign(tier=lambda d: d['tier'].map(old_names))
    mild = grid[(grid['epsilon'] == 100.0) & (grid['alpha'] == 14.0)].assign(tier='v03 mild')
    v03 = pd.concat([old, mild])[['pid', 'tier', 'r2_wheel', 'balanced_accuracy_choice',
                                  'balanced_accuracy_feedback', 'file_bytes']]
    v03.columns = ['pid', 'tier', 'wheel', 'choice', 'feedback', 'bytes']

    arms = pd.read_csv(OUT_DIR.joinpath('2026-09-26_pareto_arms.csv'))
    wheel = pd.read_csv(OUT_DIR.joinpath('2026-09-26_pareto_wheel.csv'))
    cat = pd.read_csv(OUT_DIR.joinpath('2026-09-26_pareto_categorical.csv'))
    key = ['pid', 'm', 'epsilon', 'alpha', 'track']
    sel = lambda d: d[(d['m'] == 'p32') & (d['epsilon'] == 100.0) & (d['track'] == 'grid')]  # noqa: E731
    v04 = sel(wheel).set_index(key)[['r2']].rename(columns={'r2': 'wheel'})
    for t in ('choice', 'feedback'):
        v04[t] = sel(cat[cat['target'] == t]).set_index(key)['balanced_accuracy']
    v04['bytes'] = sel(arms).set_index(key)['bytes']
    v04 = v04.reset_index()
    v04['tier'] = v04['alpha'].map({14.0: 'v04 small', 7.0: 'v04 default', 2.5: 'v04 fine'})
    v04 = v04.dropna(subset=['tier'])[['pid', 'tier', 'wheel', 'choice', 'feedback', 'bytes']]
    return pd.concat([v03, v04], ignore_index=True)


def plot(df):
    """One panel per target: per-insertion lines across tiers, mean as a heavy marker."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    pal = sns.color_palette('deep', n_colors=2)
    for ax, (col, label) in zip(axes, TARGETS.items()):
        wide = df.pivot(index='pid', columns='tier', values=col)[TIERS]
        for _, row in wide.iterrows():
            ax.plot(range(len(TIERS)), row.values, color='grey', lw=0.7, alpha=0.5)
        mean, sem = wide.mean(), wide.sem()
        for i, t in enumerate(TIERS):
            ax.errorbar(i, mean[t], sem[t], fmt='o', color=pal[1] if t.startswith('v04') else pal[0], ms=8, capsize=3)
        ax.set_xticks(range(len(TIERS)), TIERS, rotation=40, ha='right')
        ax.set_ylabel(label)
        if col != 'wheel':
            ax.axhline(0.5, color='k', ls=':', lw=1)
    axes[0].set_title('mean ± s.e.m. (heavy), 11 insertions (grey)', fontsize=10, loc='left')
    fig.tight_layout()
    for d in FIGURE_DIRS:
        if d.exists():
            fig.savefig(d.joinpath(f'{DATE}_decoding_tiers.png'), dpi=120)
    plt.show()


# %%
if __name__ == '__main__':
    tidy = load()
    tidy.to_csv(OUT_DIR.joinpath(f'{DATE}_decoding_tiers.csv'), index=False)
    print(tidy.groupby('tier')[['wheel', 'choice', 'feedback']].agg(['mean', 'sem']).loc[TIERS].round(4))
    print(tidy.groupby('tier').size())
    plot(tidy)
