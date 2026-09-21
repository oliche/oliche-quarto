"""Pairwise region x region Mahalanobis-distance heatmap.

Run after pairwise_mahalanobis.py (needs cache/pairwise_regions.csv, cache/pairwise_mahalanobis.npz
for the default Allen-partition figure; `region_distance_figure.py` produces the Cosmos-/Beryl-level
variants, sharing `plot_pairwise_heatmap` below).

Regions are ordered by CCF hierarchy traversal order (`order`, identical between a region's raw-id
hemisphere twins — see `region_stats_table.pool_by_acronym`) rather than distance-based clustering,
so that each region's own Allen atlas colour, shown as a thin strip along the top and left instead
of text tick labels (too many qualifying regions — several hundred — for legible per-row text),
stays visually coherent: anatomically related regions share colour families in the Allen scheme, so
the colour strip itself reads as a rough anatomical map of the matrix.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE_DIR = HERE / "cache"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_pairwise_heatmap(df, mahalanobis, title, out_path, figsize=11):
    """Shared rendering for every region x region Mahalanobis heatmap in this project: `df` needs
    `order` (sort key) and `hexcolor` columns row-aligned with `mahalanobis` (n, n).

    The colourbar gets its own dedicated grid cell (`cax=`) rather than being carved out of
    `ax_main` after the fact (`fig.colorbar(im, ax=ax_main, ...)`, the previous approach) — the
    latter shrinks `ax_main` post-hoc without shrinking `ax_top`/`ax_left` to match, which is
    exactly what desynced the colour strips from the matrix columns/rows in an earlier version of
    this figure.
    """
    order = np.argsort(df["order"].to_numpy())
    mat = mahalanobis[np.ix_(order, order)]
    rgb = np.array([mcolors.to_rgb(c) for c in df["hexcolor"].to_numpy()[order]])
    n = len(order)

    fig = plt.figure(figsize=(figsize * 1.07, figsize))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 40, 1.6], height_ratios=[1, 40],
                           wspace=0.03, hspace=0.01)
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top_right = fig.add_subplot(gs[0, 2])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    ax_top.imshow(rgb[None, :, :], aspect="auto", interpolation="nearest")
    ax_left.imshow(rgb[:, None, :], aspect="auto", interpolation="nearest")
    im = ax_main.imshow(mat, cmap="magma", aspect="auto", interpolation="nearest")

    for ax in (ax_corner, ax_top, ax_top_right, ax_left, ax_main):
        ax.set_xticks([])
        ax.set_yticks([])
    ax_corner.axis("off")
    ax_top_right.axis("off")

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Mahalanobis distance (PCA-whitened)")

    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def fig_pairwise_heatmap():
    df = pd.read_csv(CACHE_DIR / "pairwise_regions.csv")
    d = np.load(CACHE_DIR / "pairwise_mahalanobis.npz")
    title = (
        f"Pairwise Mahalanobis distance, {len(df)} brain regions (Allen raw partition, >500 voxels)\n"
        "ordered by CCF hierarchy; colour strips show each region's Allen atlas colour"
    )
    plot_pairwise_heatmap(df, d["mahalanobis"], title, FIG_DIR / "fig2_pairwise_mahalanobis.png")


def main():
    fig_pairwise_heatmap()
    print("pairwise heatmap written to", FIG_DIR)


if __name__ == "__main__":
    main()
