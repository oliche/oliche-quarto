"""Pairwise Mahalanobis-distance heatmaps at three parcellations of the same brain-region-distance
question as `pairwise_mahalanobis.py`/`fig2_pairwise_mahalanobis.png`, but using the atlas's own
named mappings instead of the raw partition: Cosmos (~11 regions), Beryl (~280), and Allen (the raw
partition, mixing hierarchy depths voxel-by-voxel — same 534-region figure already in the report,
regenerated here standalone for a consistent side-by-side set).

Reuses the same PCA basis and the same `plot_pairwise_heatmap` rendering as the report's figure;
only the region-grouping step differs (`region_stats_table.direct_paint_stats_for_mapping`, which
pools through `ba.regions.mappings[mapping]` instead of `pool_by_acronym`'s manual hemisphere-twin
merge — the official mappings already merge twins by construction, checked directly: both of
piriform cortex's raw positions map to the same Cosmos-level `OLF` position).

Usage: python region_distance_figure.py
Writes figures/fig_distance_{cosmos,beryl,allen}.png (standalone, not referenced from index.qmd).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import ephysatlas.anatomy as anat
from distinctiveness import EXCLUDE_ACRONYMS, load_encoding_volume, no_coverage_mask
from make_figures_matrix import FIG_DIR, plot_pairwise_heatmap
from pca_mahalanobis import fit_global_pca, in_mask_valid, project_onto_pcs
from region_stats_table import direct_paint_stats_for_mapping, mahalanobis_matrix

VOL_PATH = Path(
    "/Users/olivier/Documents/datadisk/ephys-atlas-decoding/encoding_volumes/"
    "ea_active/2026_W26/brainwide_ephys_atlas_50um.npz"
)
CACHE_DIR = Path(__file__).parent / "cache"
VARIANCE_THRESHOLD = 0.95
MIN_VOXELS = {"Cosmos": 0, "Beryl": 500}


def mapping_region_table(count, mean, var, ba, mapping, min_voxels=0):
    """One row per distinct `mapping`-level group actually present in the volume (any raw position
    with real voxels, remapped through `ba.regions.mappings[mapping]`), with `n_voxels`/`hexcolor`/
    `order` read directly off the group's own (canonical) representative position — see
    `direct_paint_stats_for_mapping`'s docstring for why every member position already carries
    the group's pooled stats and why the mapped-to position is itself that representative.
    """
    br = ba.regions
    present = np.where(count > 0)[0]
    present = present[~np.isin(br.acronym[present], list(EXCLUDE_ACRONYMS))]
    groups = np.unique(ba.regions.mappings[mapping][present])
    groups = groups[count[groups] >= min_voxels]
    df = pd.DataFrame({
        "acronym": br.acronym[groups], "n_voxels": count[groups],
        "hexcolor": br.hexcolor[groups], "order": br.order[groups],
    })
    return df, mean[groups], var[groups]


def region_distance_for_mapping(vol, ba, pca, mean_per_feature, std_per_feature, extra_exclude,
                                 mapping, min_voxels=0):
    pc_vol = project_onto_pcs(vol, mean_per_feature, std_per_feature, pca["components"])
    pc_names = [f"PC{i + 1}" for i in range(pca["k"])]
    count_pc, mean_pc, var_pc = direct_paint_stats_for_mapping(pc_vol, pc_names, ba, mapping,
                                                                extra_exclude)
    df, mean_g, var_g = mapping_region_table(count_pc, mean_pc, var_pc, ba, mapping, min_voxels)
    mahalanobis, _ = mahalanobis_matrix(df["n_voxels"].to_numpy(), mean_g, var_g)
    return df, mahalanobis


def main():
    vol, feature_names = load_encoding_volume(VOL_PATH)
    ba = anat.ClassifierAtlas(res_um=50)
    extra_exclude = no_coverage_mask(vol)
    npz_data = np.load(VOL_PATH, allow_pickle=True)
    mean_per_feature, std_per_feature = npz_data["mean_per_feature"], npz_data["std_per_feature"]

    valid_mask = in_mask_valid(ba, vol)
    pca = fit_global_pca(vol, valid_mask, mean_per_feature, std_per_feature, VARIANCE_THRESHOLD)
    print(f"PCA: k={pca['k']} components, {pca['explained_ratio'].sum():.1%} variance")

    for mapping in ["Cosmos", "Beryl"]:
        df, mahalanobis = region_distance_for_mapping(
            vol, ba, pca, mean_per_feature, std_per_feature, extra_exclude,
            mapping, min_voxels=MIN_VOXELS[mapping],
        )
        title = (f"Pairwise Mahalanobis distance, {len(df)} {mapping}-level regions\n"
                 "ordered by CCF hierarchy; colour strips show each region's Allen atlas colour")
        out = plot_pairwise_heatmap(df, mahalanobis, title,
                                     FIG_DIR / f"fig_distance_{mapping.lower()}.png")
        print(f"[{mapping}] {len(df)} regions -> {out}")

    # Allen (raw partition): reuse the already-computed/report figure's data rather than
    # recomputing the acronym-twin-pooled stats a second time.
    df = pd.read_csv(CACHE_DIR / "pairwise_regions.csv")
    d = np.load(CACHE_DIR / "pairwise_mahalanobis.npz")
    title = (f"Pairwise Mahalanobis distance, {len(df)} Allen raw-partition regions (>500 voxels)\n"
             "ordered by CCF hierarchy; colour strips show each region's Allen atlas colour")
    out = plot_pairwise_heatmap(df, d["mahalanobis"], title, FIG_DIR / "fig_distance_allen.png")
    print(f"[Allen] {len(df)} regions -> {out}")


if __name__ == "__main__":
    main()
