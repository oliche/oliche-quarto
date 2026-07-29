"""Per-insertion orchestration of the LFP-encoding model.

Builds the design once, fits both target families (raw voltage and band-power
envelopes), scores them out-of-core, and renders the figure set. Written to be
called per PID so the same entry point drives the brain-wide cluster run later.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

# The shared science modules (design/targets/solve/results_io/lfpack_io) live in the
# SDSC job dir so laptop and cluster run the same code; make them importable here.
sys.path.insert(0, "/Users/olivier/PycharmProjects/ephys-atlas/sdsc-slurms/2026-07_lfp-encoders")

import design as design_mod
import figures
import lfpack_io as io
import solve as solve_mod
import targets as targets_mod

FLAGSHIP_PID = "dab512bd-a02d-4c1f-8dbc-9155a163efc0"
LAMBDAS = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])


@dataclass
class PidRun:
    """Everything fitted for one insertion, kept for figures / downstream saving."""

    pid: str
    eid: str
    design: design_mod.Design
    trials: object
    targets_band: targets_mod.Targets
    targets_raw: targets_mod.Targets
    res_raw: solve_mod.EncodingResult
    res_band: solve_mod.EncodingResult
    lam_curve: tuple[np.ndarray, np.ndarray]
    null_band: np.ndarray
    null_raw: np.ndarray


def run_pid(pid: str = FLAGSHIP_PID, n_basis: int = 10, n_folds: int = 5, n_perm: int = 30) -> PidRun:
    """Fit and score both target families for one insertion.

    Parameters
    ----------
    pid : str
        Probe insertion UUID.
    n_basis : int, default 10
        Raised-cosine bumps in the lag basis.
    n_folds : int, default 5
        Contiguous cross-validation folds.
    n_perm : int, default 30
        Circular-shift permutations for the band-power null.

    Returns
    -------
    PidRun
        Fitted designs, results and diagnostics.
    """
    dsg = design_mod.make_design(pid, n_basis=n_basis)
    trials = io.load_trials(dsg.eid)

    tb = targets_mod.make_targets(pid, kind="band")
    lam_band, curve = solve_mod.select_lambda(dsg, tb, LAMBDAS, n_folds=n_folds)
    res_band = solve_mod.solve_encoding(dsg, tb, lam=lam_band, n_folds=n_folds)
    null_band = solve_mod.permutation_null_r2(dsg, tb, n_perm=n_perm, lam=lam_band, n_folds=n_folds)

    tr = targets_mod.make_targets(pid, kind="raw")
    lam_raw, _ = solve_mod.select_lambda(dsg, tr, LAMBDAS, n_folds=n_folds)
    res_raw = solve_mod.solve_encoding(dsg, tr, lam=lam_raw, n_folds=n_folds)
    null_raw = solve_mod.permutation_null_r2(dsg, tr, n_perm=n_perm, lam=lam_raw, n_folds=n_folds)

    return PidRun(pid, dsg.eid, dsg, trials, tb, tr, res_raw, res_band,
                  (LAMBDAS, curve), null_band, null_raw)


def make_all_figures(run: PidRun, band: str = "delta") -> list:
    """Render the full figure set for a fitted run; returns the saved paths."""
    # validation channels: where stimOn itself contributes most (largest drop-R²),
    # not the strongest overall fit (which is arousal-driven for band power).
    stim_times = run.trials["stimOn_times"].to_numpy()
    band_col = int(np.argmax(run.res_band.dr2["stimOn"]))
    raw_col = int(np.argmax(run.res_raw.dr2["stimOn"]))

    paths = [
        figures.kernel_depth_lag(run.res_band, band=band),
        figures.dr2_depth(run.res_band, band=band),
        figures.kernel_vs_eta(run.res_raw, run.targets_raw, stim_times, "stimOn_on", raw_col),
        figures.kernel_vs_eta(run.res_band, run.targets_band, stim_times, "stimOn_on", band_col),
        figures.raw_vs_band(run.res_raw, run.res_band),
        figures.diagnostics(run.design, run.lam_curve),
        figures.null_comparison(run.res_band, run.null_band, band=band),
    ]
    return paths


def main() -> None:
    run = run_pid()
    print(f"pid {run.pid[:8]}  eid {run.eid[:8]}")
    print(f"band  median CV R² {np.median(run.res_band.r2_cv):.4f}  (λ={run.res_band.lam:g})")
    print(f"raw   median CV R² {np.median(run.res_raw.r2_cv):.4f}  (λ={run.res_raw.lam:g})")
    for path in make_all_figures(run):
        print("saved", path)


if __name__ == "__main__":
    main()