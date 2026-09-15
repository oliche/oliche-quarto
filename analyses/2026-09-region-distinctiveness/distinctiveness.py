"""Shared data-loading and voxel-exclusion utilities for the region-distinctiveness analysis.

Scope note (2026-09-10): this file used to also hold the neighbour-buffer/adjacency-graph
distinctiveness pipeline (Cohen's d of a region vs. its spatial-neighbour tissue). That framing has
been dropped in favour of a plain pairwise Mahalanobis distance between all qualifying regions (see
`pairwise_mahalanobis.py`), which needs none of it — only the encoding-volume loader and the
voxel-exclusion definitions below carry forward. See `PLAN.md` for why.
"""

from pathlib import Path

import numpy as np

EXCLUDE_ACRONYMS = {"void", "root", "void_fluid"}


def load_encoding_volume(path):
    """Load the encoding-volume npz and permute it to (AP, ML, DV, feature) atlas axis order."""
    data = np.load(Path(path), allow_pickle=True)
    vol = np.transpose(data["ephys_atlas_vol"].astype(np.float32), (1, 0, 2, 3))
    feature_names = list(data["feature_names"])
    return vol, feature_names


def no_coverage_mask(vol):
    """Voxels where all 41 features are exactly 0.0 - an unfilled "no probe coverage" sentinel in
    the encoding volume, not a real physiological measurement (real features are never exactly 0
    across all 41 simultaneously). ~47k such voxels fall inside otherwise-valid regions
    (concentrated in dorsal/medial cortex - RSPv, RSPd, ACAd, MOs, PL - plus a few other
    structures such as MOB, PAG), and must be excluded the same way root/void are: an early
    version that left them in produced an extreme, non-physiological outlier in every region
    statistic that happened to include one.
    """
    return np.all(vol == 0, axis=-1)
