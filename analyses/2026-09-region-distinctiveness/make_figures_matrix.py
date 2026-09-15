"""Pairwise region x region Mahalanobis-distance heatmap.

Run after pairwise_mahalanobis.py (needs cache/pairwise_regions.csv, cache/pairwise_mahalanobis.npz).

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


def fig_pairwise_heatmap():
    df = pd.read_csv(CACHE_DIR / "pairwise_regions.csv")
    d = np.load(CACHE_DIR / "pairwise_mahalanobis.npz")
    mahalanobis = d["mahalanobis"]

    order = np.argsort(df["order"].to_numpy())
    mat = mahalanobis[np.ix_(order, order)]
    rgb = np.array([mcolors.to_rgb(c) for c in df["hexcolor"].to_numpy()[order]])
    n = len(order)

    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 40], height_ratios=[1, 40],
                           wspace=0.01, hspace=0.01)
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])

    ax_top.imshow(rgb[None, :, :], aspect="auto", interpolation="nearest")
    ax_left.imshow(rgb[:, None, :], aspect="auto", interpolation="nearest")
    im = ax_main.imshow(mat, cmap="magma", aspect="auto", interpolation="nearest")

    for ax in (ax_corner, ax_top, ax_left, ax_main):
        ax.set_xticks([])
        ax.set_yticks([])
    ax_corner.axis("off")

    cbar = fig.colorbar(im, ax=ax_main, fraction=0.025, pad=0.015)
    cbar.set_label("Mahalanobis distance (PCA-whitened)")

    fig.suptitle(
        f"Pairwise Mahalanobis distance, {n} brain regions "
        f"(Allen raw partition, >500 voxels)\n"
        "ordered by CCF hierarchy; colour strips show each region's Allen atlas colour",
        fontsize=11,
    )
    fig.savefig(FIG_DIR / "fig2_pairwise_mahalanobis.png", dpi=200)
    plt.close(fig)


def main():
    fig_pairwise_heatmap()
    print("pairwise heatmap written to", FIG_DIR)


if __name__ == "__main__":
    main()
