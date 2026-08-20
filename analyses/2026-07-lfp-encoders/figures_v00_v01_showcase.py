"""Single-regressor kernel-raster showcase: v00 vs v01, all 3 tiers, per band.

One illustrative figure per band for the v01-checkpoint report: `stimOn_on`,
2 rows (before/after) x 3 columns (uncompressed/default/aggressive), each with
its own anatomically-sorted region-colour strip. Within a band's figure, all 6
panels share one colour scale (same weight magnitude -> same colour everywhere)
and one x-axis range (channel count), so a shorter bar directly reads as fewer
significant channels rather than being stretched to fill the panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="notebook", style="ticks", font_scale=1.0)

sys.path.insert(0, "/Users/olivier/PycharmProjects/ephys-atlas/sdsc-slurms/2026-07_lfp-encoders")

import figures_kernelraster as fkr  # noqa: E402
import pid_qc  # noqa: E402

FIG_DIR = Path.home().joinpath("Documents", "figures")
DATE = "2026-07-25"
ROOT = Path(__file__).resolve().parent
TIERS = ["uncompressed", "default", "aggressive"]
VERSIONS = {"v00 (before)": ROOT.joinpath("results_bwm_2026-07-07"), "v01 (after)": ROOT.joinpath("results_bwm_v01")}
BANDS = ["raw", "delta", "theta"]
BASE_NAME = "stimOn_on"
ALPHA = 0.05


def make_showcase(tks: dict, band: str) -> Path:
    masks = {k: tk.significant_mask(band, ALPHA) for k, tk in tks.items()}
    rasters = {k: tk.raster(BASE_NAME, band)[:, masks[k]] for k, tk in tks.items()}
    vmax = float(np.nanpercentile(np.abs(np.concatenate([r.ravel() for r in rasters.values()])), 99)) or 1.0
    max_n_ch = max(int(m.sum()) for m in masks.values())

    fig, axes = plt.subplots(
        4, 3, figsize=(15, 6.5),
        gridspec_kw={"height_ratios": [1, 0.12, 1, 0.12]}, sharex="col",
    )
    for col, tier in enumerate(TIERS):
        for row, ver in enumerate(VERSIONS):
            tk = tks[(ver, tier)]
            mask = masks[(ver, tier)]
            K = rasters[(ver, tier)]
            n_ch = int(mask.sum())
            ax = axes[2 * row, col]
            axr = axes[2 * row + 1, col]
            im = ax.imshow(K, aspect="auto", cmap=fkr._CMAP, vmin=-vmax, vmax=vmax,
                            extent=[0, n_ch, tk.taus[-1], tk.taus[0]])
            ax.set_xlim(0, max_n_ch)
            ax.set_title(f"{tier}\n{ver}  ·  {n_ch} channels", fontsize=9)
            if col == 0:
                ax.set_ylabel("lag (s)")
            meta_sub = tk.meta[mask]
            labels = tk.region_boundaries("cosmos", meta_sub)
            axr.imshow(tk.region_strip(meta_sub), aspect="auto", extent=[0, n_ch, 0, 1])
            axr.set_xlim(0, max_n_ch)
            axr.set_yticks([])
            axr.set_xticks([c for c, _ in labels])
            axr.set_xticklabels([lab for _, lab in labels], rotation=90, fontsize=6)
            axr.tick_params(bottom=False)
            for spine in axr.spines.values():
                spine.set_visible(False)

    fig.colorbar(im, ax=axes, fraction=0.01, pad=0.01, label="kernel weight")
    fig.suptitle(
        f"{BASE_NAME}  ·  {band} band  ·  kernel weight, p<{ALPHA}  ·  653 PIDs (KEEP_PIDS)  ·  "
        "before (v00) vs after (v01) the lfpack pre-processing update  ·  shared x-axis (channel count)",
    )
    path = FIG_DIR.joinpath(f"{DATE}_lfpenc_v00vsv01_showcase_{BASE_NAME}_{band}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def main() -> None:
    tks = {
        (ver, tier): fkr.TierKernels(base.joinpath(tier), pids=pid_qc.KEEP_PIDS)
        for ver, base in VERSIONS.items() for tier in TIERS
    }
    for band in BANDS:
        print("saved", make_showcase(tks, band))


if __name__ == "__main__":
    main()
