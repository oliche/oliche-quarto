"""Brain-slice figures (coronal / sagittal / horizontal) for each parcellation "family" — the real
anatomical mapping (Cosmos/Beryl, where one exists at that granularity) shown alongside the
optimal/balanced/nested variants built at the same target size, all rendered with the atlas's own
per-region colours.

A `ba.regions.mappings` entry is just a raw-position LUT (see `region_distance_figure.py`'s and
`ibldevtools/olivier/2026-09-17_parcellation_viewer.py`'s notes on what that means) — this script
monkey-patches the cached `cache/parcellation_*.csv` outputs into that same dict (copied from the
GUI viewer script rather than imported, to avoid pulling in its GUI-launching module-level code) so
`BrainAtlas.plot_cslice`/`plot_sslice`/`plot_hslice(volume="annotation", mapping=...)` — the exact
machinery the built-in Cosmos/Beryl mappings already use — renders them identically, with zero
custom slice-rendering or region-colour code of our own.

Usage: python region_slices_figure.py
Writes figures/fig_slices_{cosmos,mid,beryl}.png (standalone, not referenced from index.qmd).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ephysatlas.anatomy as anat
from parcellation import CACHE_DIR, build_ontology_forest

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

RES_UM = 50  # matches the rest of this project (the GUI viewer script uses 25 for display crispness)
AP_CORONAL = -0.0015     # m, bregma-relative
ML_SAGITTAL = 0.0015     # m
DV_HORIZONTAL = -0.0025  # m

# one figure per family: the real anatomical mapping first (where one exists at that granularity),
# then the three parcellation-search variants built at the same target size.
FAMILIES = {
    "cosmos": ["Cosmos", "optimal_cosmos", "balanced_cosmos", "nested_cosmos"],
    "mid": ["optimal_mid", "balanced_mid", "nested_mid"],
    "beryl": ["Beryl", "optimal_beryl", "balanced_beryl", "nested_beryl"],
}


def collect_subtree_positions(children, own_positions, i):
    idx = list(own_positions[i])
    for c in children[i]:
        idx.extend(collect_subtree_positions(children, own_positions, c))
    return idx


def mapping_from_parcellation_csv(csv_path, children, own_positions, n_positions):
    """One `ba.regions.mappings`-shaped LUT from a cached parcellation CSV — positions the
    parcellation doesn't cover (fiber tracts/ventricles, excluded by the grey-matter-only search)
    are left as the identity map, same convention as the atlas's own mappings."""
    df = pd.read_csv(csv_path)
    lut = np.arange(n_positions)
    for row in df.itertuples():
        positions = (collect_subtree_positions(children, own_positions, row.canon_row)
                     if row.kind == "subtree" else list(own_positions[row.canon_row]))
        representative = own_positions[row.canon_row][0]
        lut[positions] = representative
    return lut


def register_parcellation_mappings(ba):
    children, own_positions, roots, canon = build_ontology_forest(ba)
    n_positions = len(ba.regions.id)
    for csv_path in sorted(CACHE_DIR.glob("parcellation_*.csv")):
        name = csv_path.stem.removeprefix("parcellation_")
        ba.regions.mappings[name] = mapping_from_parcellation_csv(
            csv_path, children, own_positions, n_positions
        )


def fig_family_slices(ba, family, mapping_names):
    n = len(mapping_names)
    fig, axs = plt.subplots(3, n, figsize=(3.2 * n, 9.2))
    axs = np.atleast_2d(axs).reshape(3, n)
    row_specs = [
        ("coronal", lambda ax, m: ba.plot_cslice(AP_CORONAL, volume="annotation", mapping=m, ax=ax)),
        ("sagittal", lambda ax, m: ba.plot_sslice(ML_SAGITTAL, volume="annotation", mapping=m, ax=ax)),
        ("horizontal", lambda ax, m: ba.plot_hslice(DV_HORIZONTAL, volume="annotation", mapping=m, ax=ax)),
    ]
    for col, mapping in enumerate(mapping_names):
        axs[0, col].set_title(mapping, fontsize=11)
        for row, (row_label, func) in enumerate(row_specs):
            ax = axs[row, col]
            func(ax, mapping)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel(row_label if col == 0 else "", fontsize=11)
    fig.suptitle(
        f"{family.capitalize()}-family parcellations — coronal / sagittal / horizontal slices, "
        "coloured by each region's Allen atlas colour",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / f"fig_slices_{family}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    ba = anat.ClassifierAtlas(res_um=RES_UM)
    register_parcellation_mappings(ba)
    for family, names in FAMILIES.items():
        out = fig_family_slices(ba, family, names)
        print(f"[{family}] {names} -> {out}")


if __name__ == "__main__":
    main()
