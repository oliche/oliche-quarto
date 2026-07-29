"""Post-resweep check: did the per-band-lambda + coverage-gate fix work?

``PLAN.md`` documents two collapse failure modes found in the pre-fix,
pooled-lambda cluster run (``results_bwm_cluster/<tier>``): (1) a single
per-PID lambda selected on the *median* CV-R² across ~288 targets can badly
under-regularise a minority band/insertion (the ``select_lambda`` collapse),
and (2) partial-camera-coverage regressors get z-scored to non-physiological
magnitudes and poison the one shared solve. Both fixes (``select_lambda_robust``
+ ``solve_encoding_grouped`` + the coverage gate) landed in the 2026-07-06
resweep, transferred here as ``results_bwm_2026-07-07/<tier>``.

This script checks whether the fixes actually reduced (a) the lambda-selection
collapse rate -- reusing ``region_aggregate.collapse_rate`` -- and (b)
cross-compression-tier kernel-weight blow-ups, via a new per-target
weight-norm ratio.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import addcopyfighandler  # noqa: F401  (enables ctrl-c copy of the active figure)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "/Users/olivier/PycharmProjects/ephys-atlas/sdsc-slurms/2026-07_lfp-encoders")
import results_io as rio  # noqa: E402

import pid_qc  # noqa: E402
import region_aggregate as ra  # noqa: E402

sns.set_theme(context="notebook", style="whitegrid")

FIG_DIR = Path.home().joinpath("Documents", "figures")
DATE = "2026-07-07"

OLD_DIR = ROOT.joinpath("results_bwm_cluster")  # pre-fix: pooled lambda
NEW_DIR = ROOT.joinpath("results_bwm_2026-07-07")  # post-fix: per-band lambda + coverage gate
TIERS = ["uncompressed", "default", "aggressive"]
COLLAPSE_THRESHOLD = -0.5
BLOWUP_LOG2 = np.log2(10.0)  # >=10x weight-norm change flagged as "wildly different"
FLAGSHIP_COLLAPSE_PID = "11a5a93e-58a9-4ed0-995e-52279ec16b98"  # PLAN.md's worked example


def _save(fig: plt.Figure, name: str) -> Path:
    """Write ``fig`` to the figures directory with the fixed date prefix."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR.joinpath(f"{DATE}_lfpenc_resweep_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def collapse_before_after() -> pd.DataFrame:
    """Collapse-rate comparison: pre-fix pooled lambda vs post-fix per-band lambda."""
    old = ra.collapse_rate({t: OLD_DIR.joinpath(t) for t in TIERS}, threshold=COLLAPSE_THRESHOLD)
    new = ra.collapse_rate({t: NEW_DIR.joinpath(t) for t in TIERS}, threshold=COLLAPSE_THRESHOLD)
    old["run"], new["run"] = "pre_fix_pooled", "post_fix_per_band"
    return pd.concat([old, new], ignore_index=True)


def plot_collapse_before_after(df: pd.DataFrame) -> Path:
    """Bar chart of ``frac_pids_collapsed`` by tier, pre- vs post-fix."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=df, x="tier", y="frac_pids_collapsed", hue="run", order=TIERS,
        hue_order=["pre_fix_pooled", "post_fix_per_band"], ax=ax,
    )
    ax.set_ylabel(f"fraction of PIDs collapsed\n(median CV R² < {COLLAPSE_THRESHOLD} in ≥1 band)")
    ax.set_xlabel("compression tier")
    ax.set_title("Lambda-selection collapse rate: before vs after fix")
    fig.tight_layout()
    return _save(fig, "collapse_before_after")


def lambda_ceiling_rate(tier_dirs: dict[str, Path]) -> pd.DataFrame:
    """Fraction of (pid, band) fits that fell back to the largest grid lambda.

    A ceiling hit means ``select_lambda_robust``'s worst-case-quantile gate
    found no candidate safe and used the safety fallback rather than the
    objective-best lambda -- expected to fire on exactly the insertions that
    would otherwise have collapsed.
    """
    rows = []
    for tier, path in tier_dirs.items():
        lam_max = max(json.loads(path.joinpath("model_config.json").read_text())["lambdas"])
        df = rio.load_scores(path)
        df = df[df["band"] != "raw"]
        per_pid_band = df.groupby(["pid", "band"], observed=True)["lam"].first().reset_index()
        for band, g in per_pid_band.groupby("band", observed=True):
            rows.append({
                "tier": tier, "band": band, "n": len(g),
                "frac_at_ceiling": float((g["lam"] >= lam_max).mean()),
            })
    return pd.DataFrame(rows)


def flagship_collapse_pid_check(pid: str = FLAGSHIP_COLLAPSE_PID) -> pd.DataFrame:
    """Direct before/after for the PLAN.md flagship collapse example (band kind)."""
    rows = []
    for run_name, base in [("pre_fix", OLD_DIR), ("post_fix", NEW_DIR)]:
        for tier in TIERS:
            path = base.joinpath(tier, "scores", f"{pid}_band.parquet")
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            for band, g in df.groupby("band", observed=True):
                rows.append({
                    "run": run_name, "tier": tier, "band": band,
                    "median_r2_cv": g["r2_cv"].clip(-1.0, 1.0).median(),
                    "lam": g["lam"].iloc[0],
                })
    return pd.DataFrame(rows)


def kernel_norm_table(tier_dir: Path, kind: str, pids: frozenset[str] | None = None) -> pd.DataFrame:
    """Per-(pid, channel, band) L2 norm of the full weight vector for one tier/kind.

    Parameters
    ----------
    pids : frozenset of str, optional
        If given, only these PIDs are loaded (e.g. ``pid_qc.KEEP_PIDS`` to
        drop the data-quality-excluded insertions in ``DATA_ISSUES.md``).
    """
    rows = []
    for npz_path in sorted(tier_dir.joinpath("kernels").glob(f"*_{kind}.npz")):
        pid = npz_path.name[: -len(f"_{kind}.npz")]
        if pids is not None and pid not in pids:
            continue
        w_norm = np.linalg.norm(np.load(npz_path)["W"], axis=0)
        meta = pd.read_parquet(
            tier_dir.joinpath("scores", f"{pid}_{kind}.parquet"), columns=["channel", "band"],
        )
        rows.append(meta.assign(pid=pid, w_norm=w_norm))
    return pd.concat(rows, ignore_index=True)


def cross_tier_weight_ratio(base_dir: Path, kind: str, pids: frozenset[str] | None = pid_qc.KEEP_PIDS) -> pd.DataFrame:
    """Merge per-tier kernel norms and compute log2 ratio vs the uncompressed reference."""
    tabs = {t: kernel_norm_table(base_dir.joinpath(t), kind, pids=pids) for t in TIERS}
    key = ["pid", "channel", "band"]
    out = tabs["uncompressed"].rename(columns={"w_norm": "w_norm_uncompressed"})
    for t in ("default", "aggressive"):
        out = out.merge(tabs[t][key + ["w_norm"]].rename(columns={"w_norm": f"w_norm_{t}"}), on=key)
    for t in ("default", "aggressive"):
        out[f"log2_ratio_{t}"] = np.log2(out[f"w_norm_{t}"] / out["w_norm_uncompressed"])
    return out


def blowup_summary(ratio_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction / magnitude of >=10x cross-tier weight-norm changes."""
    rows = []
    for t in ("default", "aggressive"):
        col = f"log2_ratio_{t}"
        rows.append({
            "tier": t,
            "n": len(ratio_df),
            "frac_blown_up": float((ratio_df[col].abs() >= BLOWUP_LOG2).mean()),
            "median_abs_log2_ratio": float(ratio_df[col].abs().median()),
            "max_abs_log2_ratio": float(ratio_df[col].abs().max()),
        })
    return pd.DataFrame(rows)


def plot_weight_ratio(ratio_df: pd.DataFrame, kind: str) -> Path:
    """Violin of the cross-tier log2 weight-norm ratio, one panel per band."""
    melted = ratio_df.melt(
        id_vars=["pid", "channel", "band"],
        value_vars=["log2_ratio_default", "log2_ratio_aggressive"],
        var_name="tier", value_name="log2_ratio",
    )
    melted["tier"] = melted["tier"].str.replace("log2_ratio_", "", regex=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(data=melted, x="band", y="log2_ratio", hue="tier", cut=0, ax=ax)
    ax.axhline(BLOWUP_LOG2, color="firebrick", ls="--", lw=1, label="10x")
    ax.axhline(-BLOWUP_LOG2, color="firebrick", ls="--", lw=1)
    ax.set_ylabel("log2(compressed / uncompressed weight-norm)")
    ax.set_title(f"Cross-tier kernel-weight stability ({kind}, post-fix)")
    fig.tight_layout()
    return _save(fig, f"weight_ratio_{kind}")


def main() -> None:
    print("=== Collapse rate: pre-fix (pooled) vs post-fix (per-band) ===")
    collapse = collapse_before_after()
    print(collapse.to_string(index=False))
    plot_collapse_before_after(collapse)

    print("\n=== Lambda ceiling-fallback rate (post-fix run only) ===")
    ceiling = lambda_ceiling_rate({t: NEW_DIR.joinpath(t) for t in TIERS})
    print(ceiling.to_string(index=False))

    print(f"\n=== Flagship collapse PID ({FLAGSHIP_COLLAPSE_PID}), before vs after ===")
    print(flagship_collapse_pid_check().to_string(index=False))

    print(f"\n=== Cross-tier kernel-weight stability (KEEP_PIDS only, n={len(pid_qc.KEEP_PIDS)}) ===")
    print(f"excluded {len(pid_qc.EXCLUDED_PIDS)} data-quality PIDs -- see DATA_ISSUES.md")
    for kind in ("band", "raw"):
        ratio = cross_tier_weight_ratio(NEW_DIR, kind)
        print(f"-- {kind} --")
        print(blowup_summary(ratio).to_string(index=False))
        plot_weight_ratio(ratio, kind)


if __name__ == "__main__":
    main()