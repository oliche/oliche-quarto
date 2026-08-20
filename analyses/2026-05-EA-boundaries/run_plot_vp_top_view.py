"""Plot the virtual-probe 200 µm grid on a top-view brain map.

Shows only the probes that are actually used in the analysis — i.e. those with
at least one inside-brain channel in the cached virtual-probe parquet.
Uses ba.plot_top for the dorsal brain background.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import ephysatlas.anatomy
sys.path.insert(0, str(Path(__file__).parent))
from boundary_classifier_volume import GRID_SPACING_UM, LOCAL_CACHE_DIR, FIGURES_DIR


def plot_vp_grid_top_view(brain_atlas, df: pd.DataFrame, grid_spacing_um: float) -> plt.Figure:
    """Plot used virtual-probe entry points on a dorsal brain view.

    Parameters
    ----------
    brain_atlas:
        AllenAtlas or ClassifierAtlas instance (25 µm resolution).
    df:
        Virtual-probe feature dataframe (MultiIndex pid x channel) with columns
        'x' (ML) and 'y' (AP) in metres.  Only probes present in this frame
        (i.e. those with at least one inside-brain channel) are plotted.
    grid_spacing_um:
        AP and ML probe spacing in micrometres (used in title only).

    Returns
    -------
    fig:
        Matplotlib figure.
    """
    # One (ML, AP) position per probe — all channels of a probe share the same x/y
    probe_xy = df.groupby(level='pid')[['x', 'y']].first()
    ml_um = probe_xy['x'].values * 1e6   # metres → µm (iblatlas axis units)
    ap_um = probe_xy['y'].values * 1e6
    n_probes = len(probe_xy)
    print(f'Inside-brain probes: {n_probes:,}')

    fig, ax = plt.subplots(figsize=(8, 7))
    # ba.plot_top: dorsal max-projection; axes are in µm, order (ML, AP)
    brain_atlas.plot_top(volume='boundary', ax=ax)
    ax.scatter(ml_um, ap_um, s=3, c='steelblue', alpha=0.7, linewidths=0, rasterized=True)
    ax.set_title(
        f'Virtual-probe grid — {grid_spacing_um:.0f} µm spacing\n'
        f'{n_probes:,} probes with ≥1 inside-brain channel',
        fontsize=13,
    )
    fig.tight_layout()
    return fig


print('Loading atlas ...')
brain_atlas = ephysatlas.anatomy.ClassifierAtlas()

vp_cache = LOCAL_CACHE_DIR.joinpath(f'vp_df_{GRID_SPACING_UM:.0f}um.parquet')
print(f'Loading virtual-probe df from {vp_cache.name} ...')
df = pd.read_parquet(vp_cache, columns=['x', 'y'])
print(f'  {len(df):,} inside-brain channels across {df.index.get_level_values("pid").nunique():,} probes')

fig = plot_vp_grid_top_view(brain_atlas, df, GRID_SPACING_UM)

FIGURES_DIR.mkdir(exist_ok=True)
out = FIGURES_DIR.joinpath('vp_grid_top_view.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved -> {out}')