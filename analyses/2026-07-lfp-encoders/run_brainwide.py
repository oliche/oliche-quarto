"""Brain-wide batch runner for the LFP-encoding model.

Fits every available BWM insertion in parallel across PIDs, one worker per
recording, and writes compact per-``(pid, kind)`` result shards via
``results_io``. Designed to run to completion on this laptop (no cluster): at
~90 s per PID including cross-validation and the permutation null, 699
insertions finish in a handful of hours across the usable cores.

Memory, not CPU, is the constraint: each worker holds a probe's band-power ``Y``
(~1.4 GB) transiently, so the default worker count is conservative. Only small
status dicts cross the process boundary -- the large arrays are written to disk
and freed inside the worker.

Usage::

    python run_brainwide.py --outdir results_bwm --workers 6 --n-perm 30
    python run_brainwide.py --limit 8          # smoke test on 8 PIDs
    python run_brainwide.py                    # resume: existing shards are skipped
"""

from __future__ import annotations

# Keep BLAS/FFT single-threaded per worker to avoid oversubscription across the
# process pool; must be set before numpy is imported (spawned children re-read).
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# The shared science modules live in the SDSC job dir; make them importable here.
sys.path.insert(0, "/Users/olivier/PycharmProjects/ephys-atlas/sdsc-slurms/2026-07_lfp-encoders")

import lfpack_io as io
import results_io as rio
import targets as targets_mod

warnings.filterwarnings("ignore")

DEFAULT_OUTDIR = Path("results_bwm")


def available_pids() -> pd.DataFrame:
    """PIDs present in the LFP archive that also have behaviour available.

    Returns
    -------
    DataFrame
        Columns ``pid``, ``eid`` for insertions whose LFP is compressed and
        whose session behaviour shard exists locally.
    """
    with h5py.File(io.LFP_H5, "r") as f:
        lfp_pids = set(f.keys())
    ins = pd.read_parquet(io.resolve_bwm_dir("bwm_ephys").joinpath("metadata", "insertions.parquet"))
    beh_sessions = io.resolve_bwm_dir("bwm_behavior").joinpath("sessions")
    ins = ins[ins["pid"].isin(lfp_pids)].copy()
    has_beh = ins["eid"].map(lambda e: beh_sessions.joinpath(f"{e}.zip").exists())
    return ins.loc[has_beh, ["pid", "eid"]].reset_index(drop=True)


def _already_done(pid: str, outdir: Path) -> bool:
    """True if both target-family shards already exist for this PID."""
    sd = outdir.joinpath("scores")
    return sd.joinpath(f"{pid}_band.parquet").exists() and sd.joinpath(f"{pid}_raw.parquet").exists()


def _fit_pid(pid: str, outdir: str, n_perm: int, n_folds: int) -> dict:
    """Worker: fit both target families for one PID, save shards, return status.

    Imports the heavy modules lazily so the parent stays light and each spawned
    worker builds its own state. Never returns large arrays.
    """
    import design as design_mod
    import solve as solve_mod
    import run_encoding as re_mod

    out = Path(outdir)
    t0 = time.time()
    try:
        dsg = design_mod.make_design(pid, n_basis=10)
        band_median = np.nan
        for kind in ("band", "raw"):
            tgt = targets_mod.make_targets(pid, kind=kind)
            lam, _ = solve_mod.select_lambda(dsg, tgt, re_mod.LAMBDAS, n_folds=n_folds)
            res = solve_mod.solve_encoding(dsg, tgt, lam=lam, n_folds=n_folds)
            null = solve_mod.permutation_null_r2(dsg, tgt, n_perm=n_perm, lam=lam, n_folds=n_folds)
            rio.save_pid_result(res, out, null=null)
            if kind == "band":
                band_median = float(np.median(res.r2_cv))
            del tgt, res, null
        return {"pid": pid, "ok": True, "seconds": time.time() - t0,
                "band_median_r2": band_median, "error": ""}
    except Exception as exc:  # noqa: BLE001 - record and continue the batch
        return {"pid": pid, "ok": False, "seconds": time.time() - t0,
                "band_median_r2": np.nan, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain-wide LFP-encoding batch run")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=6, help="parallel PIDs (~1.8 GB RAM each)")
    parser.add_argument("--n-perm", type=int, default=30, help="circular-shift null permutations")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="fit only the first N PIDs")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    catalogue = available_pids()
    if args.limit:
        catalogue = catalogue.head(args.limit)
    catalogue.to_parquet(args.outdir.joinpath("catalogue.parquet"), index=False)

    # shared config/basis once, from a reference design (identical across PIDs)
    import design as design_mod
    ref = design_mod.make_design(catalogue["pid"].iloc[0], n_basis=10)
    rio.save_shared(ref, args.outdir, targets_mod.BANDS,
                    extra={"n_perm": args.n_perm, "n_folds": args.n_folds})
    del ref

    todo = [p for p in catalogue["pid"] if not _already_done(p, args.outdir)]
    print(f"{len(catalogue)} PIDs available · {len(catalogue) - len(todo)} already done · "
          f"{len(todo)} to fit · {args.workers} workers")

    statuses, t_start = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fit_pid, p, str(args.outdir), args.n_perm, args.n_folds): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            s = fut.result()
            statuses.append(s)
            flag = "ok" if s["ok"] else "FAIL"
            note = f"band R²={s['band_median_r2']:.3f}" if s["ok"] else s["error"]
            print(f"[{i}/{len(todo)}] {s['pid'][:8]} {flag} {s['seconds']:.0f}s · {note}")

    if statuses:
        manifest = pd.DataFrame(statuses)
        manifest.to_parquet(args.outdir.joinpath("run_manifest.parquet"), index=False)
        ok = int(manifest["ok"].sum())
        print(f"done: {ok}/{len(statuses)} succeeded in {(time.time() - t_start) / 3600:.1f} h; "
              f"manifest -> {args.outdir.joinpath('run_manifest.parquet')}")


if __name__ == "__main__":
    main()
