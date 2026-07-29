"""Regenerate region-significance/Swanson figures and kernel rasters for the v01 resweep.

Runs against the reprocessed encoding results (``results_bwm_v01/``, extracted
from ``results_bwm_v01.tar.gz`` -- updated pre-processing + compression scheme
vs. the ``results_bwm_2026-07-07`` / v00 resweep), restricted to
``pid_qc.KEEP_PIDS`` (same 653-PID QC filter derived from the v00 resweep --
reused as-is for a direct v00-vs-v01 comparison, not recomputed for v01).
"""

from __future__ import annotations

from pathlib import Path

import figures_kernelraster as fkr
import figures_swanson as fsw
import pid_qc

ROOT = Path(__file__).resolve().parent
DATE = "2026-07-25"
fsw.DATE = DATE
fkr.DATE = DATE

TIERS = ["uncompressed", "default", "aggressive"]
V01_DIR = ROOT.joinpath("results_bwm_v01")
TIER_DIRS = {t: V01_DIR.joinpath(t) for t in TIERS}


def main() -> None:
    print(f"KEEP_PIDS: {len(pid_qc.KEEP_PIDS)} / excluded: {len(pid_qc.EXCLUDED_PIDS)} -- see DATA_ISSUES.md")

    print("swanson: real/null grids...")
    for p in fsw.make_real_null_grids(TIER_DIRS, pids=pid_qc.KEEP_PIDS):
        print(" saved", p)

    print("swanson: frac-sig grids...")
    for p in fsw.make_fracsig_grids(TIER_DIRS, pids=pid_qc.KEEP_PIDS):
        print(" saved", p)

    print("swanson: compression-delta grids...")
    for p in fsw.make_compression_delta_grids(TIER_DIRS, pids=pid_qc.KEEP_PIDS):
        print(" saved", p)

    print("kernel rasters (band-regressor family, all tiers)...")
    for p in fkr.make_all_tier_kernelrasters(TIER_DIRS, pids=pid_qc.KEEP_PIDS):
        print(" saved", p)


if __name__ == "__main__":
    main()
