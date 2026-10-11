"""
Behaviour decoding per tier (`plot_decoding_tiers.py`) with lfpack v05 added: the shipped v04 archives and the v05
tiers, both through the production pipeline (`decode_v05.py`), next to the earlier uncompressed, v03 and v04 points.

Writes ``<DATE>_decoding_tiers_v05.csv`` (tidy, one row per insertion and tier) and ``<DATE>_decoding_tiers_v05.png``.
"""

# %%
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import plot_decoding_tiers as pdt

sns.set_theme(context='notebook')

OUT_DIR = Path(__file__).parent
DATE = '2026-10-11'
TIERS = pdt.TIERS + ['v04 small (archive)', 'v04 default (archive)', 'v04 fine (archive)',
                     'v05 uncompressed', 'v05 small', 'v05 default', 'v05 fine']


def load():
    """Tidy table of `plot_decoding_tiers.load` plus the rows of `decode_v05.py`."""
    return pd.concat([pdt.load(), pd.read_csv(OUT_DIR.joinpath('2026-10-11_decoding_v05.csv'))], ignore_index=True)


def colour(tier, pal):
    """Blue: uncompressed and v03; orange: v04 (sweep, and archive as shipped); green: v05."""
    return pal[2] if tier.startswith('v05') else pal[1] if tier.startswith('v04') else pal[0]


def plot(df):
    """One panel per target: per-insertion lines across tiers, mean ± s.e.m. as heavy markers."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    pal = sns.color_palette('deep', n_colors=3)
    for ax, (col, label) in zip(axes, pdt.TARGETS.items()):
        wide = df.pivot(index='pid', columns='tier', values=col)[TIERS]
        for _, row in wide.iterrows():
            ax.plot(range(len(TIERS)), row.values, color='grey', lw=0.7, alpha=0.5)
        mean, sem = wide.mean(), wide.sem()
        for i, t in enumerate(TIERS):
            ax.errorbar(i, mean[t], sem[t], fmt='o', ms=8, capsize=3, color=colour(t, pal),
                        mfc='white' if 'archive' in t else colour(t, pal))
        for x in (len(pdt.TIERS) - 0.5, len(pdt.TIERS) + 2.5):  # sweep | production v04 | production v05
            ax.axvline(x, color='k', lw=0.6)
        ax.set_xticks(range(len(TIERS)), TIERS, rotation=45, ha='right')
        ax.set_ylabel(label)
        ax.grid(axis='x', visible=False)
        if col != 'wheel':
            ax.axhline(0.5, color='k', ls=':', lw=1)
    axes[0].set_title('mean ± s.e.m. (heavy), 11 insertions (grey)', fontsize=10, loc='left')
    fig.tight_layout()
    for d in pdt.FIGURE_DIRS:
        if d.exists():
            fig.savefig(d.joinpath(f'{DATE}_decoding_tiers_v05.png'), dpi=120)
    plt.show()


# %%
if __name__ == '__main__':
    tidy = load()
    tidy.to_csv(OUT_DIR.joinpath(f'{DATE}_decoding_tiers_v05.csv'), index=False)
    print(tidy.groupby('tier')[['wheel', 'choice', 'feedback']].agg(['mean', 'sem']).loc[TIERS].round(4))
    print(tidy.groupby('tier').size())
    plot(tidy)
