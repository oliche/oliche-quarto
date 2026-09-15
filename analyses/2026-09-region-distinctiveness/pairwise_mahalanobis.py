"""Pairwise Mahalanobis distance between every qualifying Allen-level brain region, in PCA-whitened
ephys-atlas feature space.

Scope (2026-09-10, replacing the neighbour-buffer/hierarchy framing in earlier sections of
`PLAN.md`): select every named region in the atlas's own raw painted partition (`ba.label`, not any
Beryl/Cosmos/Swanson remapping) with more than `MIN_VOXELS` directly-painted voxels, project onto
the global PCA basis (`pca_mahalanobis.py`, kept unchanged from the earlier work), and compute the
full region x region Mahalanobis distance matrix (`sqrt(sum_k d_k^2)` over per-PC Cohen's d,
`region_stats_table.mahalanobis_matrix`). No neighbour definition, distance transform or hierarchy
aggregation is involved anywhere in this — a plain N x N matrix over all qualifying regions, so this
is far cheaper than the superseded pipeline.

Usage: python pairwise_mahalanobis.py
Writes cache/pairwise_regions.csv (one row per qualifying region, with hexcolor/order for the
figure) and cache/pairwise_mahalanobis.npz (the full distance matrix + per-PC d), plus
cache/pca_basis.npz (the fitted basis, for make_figures_pca.py's scree plot).
"""

import time
from pathlib import Path

import numpy as np

import ephysatlas.anatomy as anat
from distinctiveness import load_encoding_volume, no_coverage_mask
from region_stats_table import fit_and_project_pca, mahalanobis_matrix, pool_by_acronym

VOL_PATH = Path(
    "/Users/olivier/Documents/datadisk/ephys-atlas-decoding/encoding_volumes/"
    "ea_active/2026_W26/brainwide_ephys_atlas_50um.npz"
)
CACHE_DIR = Path(__file__).parent / "cache"
MIN_VOXELS = 500
VARIANCE_THRESHOLD = 0.95


def main():
    CACHE_DIR.mkdir(exist_ok=True)
    vol, feature_names = load_encoding_volume(VOL_PATH)
    ba = anat.ClassifierAtlas(res_um=50)
    extra_exclude = no_coverage_mask(vol)
    print(f"excluding {extra_exclude.sum()} no-probe-coverage (all-zero) voxels")

    npz_data = np.load(VOL_PATH, allow_pickle=True)
    mean_per_feature, std_per_feature = npz_data["mean_per_feature"], npz_data["std_per_feature"]

    t0 = time.time()
    pca, pc_vol, pc_names, count_pc, mean_pc, var_pc = fit_and_project_pca(
        vol, ba, mean_per_feature, std_per_feature, extra_exclude, VARIANCE_THRESHOLD
    )
    print(f"PCA fit + projection: k={pca['k']} components, "
          f"{pca['explained_ratio'].sum():.1%} variance ({time.time() - t0:.0f}s)")
    np.savez(
        CACHE_DIR / "pca_basis.npz",
        components=pca["components"], explained_ratio=pca["explained_ratio"],
        all_explained_ratio=pca["all_explained_ratio"],
        feature_names=np.array(feature_names, dtype=object), k=pca["k"],
    )

    df, mean_pooled, var_pooled = pool_by_acronym(count_pc, mean_pc, var_pc, ba,
                                                    min_voxels=MIN_VOXELS)
    print(f"{len(df)} regions clear the {MIN_VOXELS}-voxel bar "
          f"(pooled across raw-id hemisphere twins)")

    t0 = time.time()
    mahalanobis, d_pc = mahalanobis_matrix(df["n_voxels"].to_numpy(), mean_pooled, var_pooled)
    print(f"pairwise Mahalanobis matrix: {mahalanobis.shape} ({time.time() - t0:.1f}s)")

    i, j = np.unravel_index(np.argmax(np.triu(mahalanobis, k=1)), mahalanobis.shape)
    print(f"most distinctive pair: {df['acronym'].iat[i]} vs {df['acronym'].iat[j]}, "
          f"mahalanobis={mahalanobis[i, j]:.2f}")

    df.to_csv(CACHE_DIR / "pairwise_regions.csv", index=False)
    np.savez(
        CACHE_DIR / "pairwise_mahalanobis.npz",
        mahalanobis=mahalanobis, d_pc=d_pc, pc_names=np.array(pc_names, dtype=object),
    )
    print(f"saved to {CACHE_DIR / 'pairwise_regions.csv'} and "
          f"{CACHE_DIR / 'pairwise_mahalanobis.npz'}")


if __name__ == "__main__":
    main()
