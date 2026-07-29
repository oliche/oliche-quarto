"""Feedback/movement kernel-weight rasters for the v01_smart (row-exclusion) resweep.

One figure per regressor (`feedback_on`, `move_on`), all bands (delta/theta/beta/
gamma/raw) stacked. Unlike `figures_kernelraster.plot_regressor_all_bands` (one region
strip *per band*, since each band's own significance mask gives it a different channel
set/order), `plot_regressor_shared_axis` here shows every channel (no per-band
significance filtering) so all bands share one column order and just **one** region
strip at the bottom -- trading the per-band significance filter for much more vertical
room for the actual kernel data.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figures_kernelraster as fkr

ROOT = Path(__file__).resolve().parent
TIER_DIR = ROOT.joinpath("results_bwm_v01_smart", "default")

# for the v01-vs-v01_smart before/after comparison: same 699 PIDs, same channel/region
# order (verified directly -- byte-identical model_config, row exclusion is the only diff)
V01_SMART_DIR = TIER_DIR


def version_dirs(lfp_tier: str) -> dict[str, Path]:
    """v01/v01_smart tier directories for one LFP compression tier ("default", "aggressive", "uncompressed")."""
    return {
        "v01": ROOT.joinpath("results_bwm_v01", lfp_tier),
        "v01_smart": ROOT.joinpath("results_bwm_v01_smart", lfp_tier),
    }


def plot_regressor_shared_axis(
    tk: fkr.TierKernels, base_name: str, tier: str, bands: list[str] | None = None,
    vmax_by_band: dict[str, float] | None = None,
):
    """One regressor, all bands stacked, all channels, a single shared region strip.

    Parameters
    ----------
    tk : fkr.TierKernels
    base_name : str
        Regressor to display (e.g. ``"feedback_on"``).
    tier : str
        Compression tier label, for the title/filename.
    bands : list of str, optional
        Defaults to delta/theta/beta/gamma/raw.
    vmax_by_band : dict, optional
        Pre-computed colour-scale max per band (e.g. pooled across two runs being
        compared, see :func:`compare_shared_axis`) instead of this call's own
        99th-percentile -- so the same weight magnitude means the same colour across
        multiple figures.

    Returns
    -------
    Path
        Saved PNG path.
    """
    bands = bands or [*fkr.BANDS, "raw"]
    n_ch = len(tk.meta)

    fig, axes = plt.subplots(
        len(bands) + 1, 1, figsize=(16, 1.6 * len(bands)),
        gridspec_kw={"height_ratios": [1] * len(bands) + [0.25]},
    )
    for ax, band in zip(axes[:-1], bands):
        K = tk.raster(base_name, band)
        vmax = (vmax_by_band or {}).get(band) or fkr._vmax(K)
        im = ax.imshow(K, aspect="auto", cmap=fkr._CMAP, vmin=-vmax, vmax=vmax,
                        extent=[0, n_ch, tk.taus[-1], tk.taus[0]])
        ax.set_ylabel(f"{band}\nlag (s)", fontsize=8)
        ax.tick_params(bottom=False, labelbottom=False)
        fig.colorbar(im, ax=ax, fraction=0.01, pad=0.005)

    axr = axes[-1]
    labels = tk.region_boundaries("cosmos")
    axr.imshow(tk.region_strip(), aspect="auto", extent=[0, n_ch, 0, 1])
    axr.set_yticks([])
    axr.set_xticks([c for c, _ in labels])
    axr.set_xticklabels([lab for _, lab in labels], rotation=90, fontsize=6)
    axr.tick_params(bottom=False)
    for spine in axr.spines.values():
        spine.set_visible(False)

    fig.suptitle(f"{base_name}  ·  kernel weight, all bands (all channels)  ·  "
                 f"{tk.meta['pid'].nunique()} PIDs  ·  {tier}")
    fig.tight_layout()
    return fkr._save(fig, f"kernelraster_regressor_{base_name}_{tier}_sharedaxis")


def compare_shared_axis(
    tier_dirs: dict[str, Path], base_name: str, bands: list[str] | None = None, tag_suffix: str = "",
) -> list[Path]:
    """:func:`plot_regressor_shared_axis` for every version in ``tier_dirs``, sharing
    one colour-scale vmax per band (pooled across all versions) and the same x-axis
    (same PIDs/channels/region order), so figures are directly comparable side by side.

    Parameters
    ----------
    tier_dirs : dict[str, Path]
        Version label -> its tier directory (e.g. ``version_dirs("uncompressed")``).
        Must share the same PID set/channel order (verify first).
    base_name : str
    bands : list of str, optional
    tag_suffix : str, default ""
        Appended to each version's filename tag (e.g. the LFP compression tier), so
        e.g. ``uncompressed`` and ``default`` runs don't overwrite each other's figures.

    Returns
    -------
    list of Path
        One saved PNG per version.
    """
    bands = bands or [*fkr.BANDS, "raw"]
    tks = {v: fkr.TierKernels(d) for v, d in tier_dirs.items()}
    vmax_by_band = {
        band: fkr._vmax(np.concatenate([tk.raster(base_name, band).ravel() for tk in tks.values()]))
        for band in bands
    }
    return [
        plot_regressor_shared_axis(tk, base_name, tier=f"{v}{tag_suffix}", bands=bands, vmax_by_band=vmax_by_band)
        for v, tk in tks.items()
    ]


def main(lfp_tier: str = "default") -> None:
    fkr.DATE = "2026-07-26"
    for base_name in ("feedback_on", "move_on"):
        paths = compare_shared_axis(version_dirs(lfp_tier), base_name, tag_suffix=f"_{lfp_tier}")
        for path in paths:
            print("saved", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "default")
