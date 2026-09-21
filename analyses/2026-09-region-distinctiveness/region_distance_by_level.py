"""Pairwise Mahalanobis-distance heatmaps at every distinct CCF ontology-hierarchy level cut,
alongside the Cosmos/Beryl/Allen figures in `region_distance_figure.py`.

`BrainRegions.compute_hierarchy()` builds `br.hierarchy`, shape `(n_levels, n_positions)`: row `L`
gives, for every raw position, the raw position of its ancestor at ontology level `L+1` (a region
shallower than that level is left as itself — nothing to cut). Unlike the atlas's own Cosmos/Beryl
mappings, this walks each hemisphere `+id`/`-id` twin's parent chain independently rather than
merging them into a shared value, so groups are pooled by the *acronym* of each voxel's level-L
ancestor (`region_stats_table.pool_by_group_label`) rather than by raw position.

Restricted to grey matter: `fiber tracts` and `VS` (ventricular systems) voxels are excluded up
front (same mechanism as `no_coverage_mask` — folded into `extra_exclude` before any per-region
stats are computed), not just cropped out of the plot, since they otherwise dominate the figure
(bright, high-distance bands against every grey-matter region) without being the interesting
comparison. The PCA basis itself is still fit brainwide, same as every other figure in this
project — only the per-region grouping/statistics are grey-matter-only.

Consecutive levels are skipped whenever they produce an identical thresholded grouping to the one
before — this happens once every real region has already bottomed out above the query level (the
deepest level here duplicates the one before it) — so the figure count is however many levels are
actually distinct, not the full 10 the ontology nominally has. Level 1 is additionally skipped
outright: with fiber tracts/VS excluded, it has only one surviving group (`grey` itself), too
degenerate to plot.

Usage: python region_distance_by_level.py
Writes figures/fig_distance_level{L}.png for each distinct level L (standalone, not referenced
from index.qmd).
"""

from pathlib import Path

import numpy as np

import ephysatlas.anatomy as anat
from distinctiveness import load_encoding_volume, no_coverage_mask
from make_figures_matrix import FIG_DIR, plot_pairwise_heatmap
from pca_mahalanobis import fit_global_pca, in_mask_valid, project_onto_pcs
from region_stats_table import direct_paint_stats, mahalanobis_matrix, pool_by_group_label

VOL_PATH = Path(
    "/Users/olivier/Documents/datadisk/ephys-atlas-decoding/encoding_volumes/"
    "ea_active/2026_W26/brainwide_ephys_atlas_50um.npz"
)
VARIANCE_THRESHOLD = 0.95
MIN_VOXELS = 500
GREY_MATTER_ONLY = True


def main():
    vol, feature_names = load_encoding_volume(VOL_PATH)
    ba = anat.ClassifierAtlas(res_um=50)
    br = ba.regions
    br.compute_hierarchy()

    extra_exclude = no_coverage_mask(vol)
    if GREY_MATTER_ONLY:
        # every voxel whose level-1 ancestor isn't `grey` (i.e. fiber tracts / VS / the near-empty
        # grv/retina branches) - same idea as EXCLUDE_ACRONYMS elsewhere, but level-dependent so it
        # has to be folded into the voxel mask itself rather than an acronym blocklist.
        top1_acronym = br.acronym[br.hierarchy[0]]
        non_grey_positions = np.where(top1_acronym != "grey")[0]
        extra_exclude = extra_exclude | np.isin(ba.label, non_grey_positions)
        print(f"grey-matter-only: excluding {np.isin(ba.label, non_grey_positions).sum()} "
              f"fiber tracts/VS/other non-grey voxels in addition to no-coverage ones")

    npz_data = np.load(VOL_PATH, allow_pickle=True)
    mean_per_feature, std_per_feature = npz_data["mean_per_feature"], npz_data["std_per_feature"]

    valid_mask = in_mask_valid(ba, vol)
    pca = fit_global_pca(vol, valid_mask, mean_per_feature, std_per_feature, VARIANCE_THRESHOLD)
    print(f"PCA: k={pca['k']} components, {pca['explained_ratio'].sum():.1%} variance")
    pc_vol = project_onto_pcs(vol, mean_per_feature, std_per_feature, pca["components"])
    pc_names = [f"PC{i + 1}" for i in range(pca["k"])]
    count_pc, mean_pc, var_pc = direct_paint_stats(pc_vol, pc_names, ba, extra_exclude)

    n_levels = br.hierarchy.shape[0]
    print(f"ontology hierarchy: {n_levels} levels")

    prev_acronyms, n_written = None, 0
    for row in range(n_levels):
        level = row + 1
        group_label = br.acronym[br.hierarchy[row]]  # per-position acronym of its level-`level` ancestor
        df, mean_g, var_g = pool_by_group_label(count_pc, mean_pc, var_pc, ba, group_label,
                                                 min_voxels=MIN_VOXELS)
        if len(df) < 2:
            print(f"level {level}: only {len(df)} group(s) left, too degenerate to plot, skipped")
            continue
        acronyms_sorted = np.sort(df["acronym"].to_numpy())
        if prev_acronyms is not None and np.array_equal(acronyms_sorted, prev_acronyms):
            print(f"level {level}: identical grouping to the previous level, skipped")
            continue
        prev_acronyms = acronyms_sorted

        mahalanobis, _ = mahalanobis_matrix(df["n_voxels"].to_numpy(), mean_g, var_g)
        grey_note = " (grey matter only)" if GREY_MATTER_ONLY else ""
        title = (f"Pairwise Mahalanobis distance, {len(df)} regions at CCF hierarchy level "
                 f"{level}{grey_note}\n"
                 "ordered by CCF hierarchy; colour strips show each region's Allen atlas colour")
        out = plot_pairwise_heatmap(df, mahalanobis, title,
                                     FIG_DIR / f"fig_distance_level{level}.png")
        n_written += 1
        print(f"level {level}: {len(df)} regions -> {out}")

    print(f"{n_written} distinct level figures written")


if __name__ == "__main__":
    main()
