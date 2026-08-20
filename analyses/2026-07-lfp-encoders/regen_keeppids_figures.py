"""Regenerate region-significance/Swanson figures and kernel rasters, KEEP_PIDS only.

Runs against the 2026-07-07 resweep (``results_bwm_2026-07-07/``), restricted
to ``pid_qc.KEEP_PIDS`` (drops the 46 data-quality-excluded PIDs documented in
``DATA_ISSUES.md``: 4 bad recordings, 4 ``nc=24``-corrupted archives, 38
genuine default-compression collapses). The existing 2026-07-06 figures
(pre-fix, unfiltered, ``results_bwm_cluster``) document the earlier state and
are left untouched -- these write under a fresh date prefix instead of
overwriting them.
"""

from __future__ import annotations

import figures_kernelraster as fkr
import figures_swanson as fsw
import pid_qc

from check_resweep import NEW_DIR, TIERS

DATE = "2026-07-07"
fsw.DATE = DATE
fkr.DATE = DATE

TIER_DIRS = {t: NEW_DIR.joinpath(t) for t in TIERS}


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
