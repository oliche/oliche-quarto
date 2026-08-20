"""Reusable N-version comparison for the brain-wide LFP-encoder resweeps.

Generalizes `compare_v00_v01.py` (which hardcodes exactly v00 vs v01) so a new
resweep doesn't need its own hand-duplicated script -- register it in `VERSIONS`
below and pass its name to `compare_pair`/`compare_many`.

    v00        results_bwm_2026-07-07   original 2026-07 resweep
    v01        results_bwm_v01          updated lfpack pre-processing (highpass
                                         0.5Hz, alpha/floor fix, saturation muting)
    v01_smart  results_bwm_v01_smart    v01 + saturated-sample ROW EXCLUSION
                                         (PROMPT_saturation_row_exclusion.md) instead
                                         of trusting the muted-to-zero signal
"""

from __future__ import annotations

import sys
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(context="notebook", style="whitegrid")

sys.path.insert(0, "/Users/olivier/PycharmProjects/ephys-atlas/sdsc-slurms/2026-07_lfp-encoders")

import pid_qc  # noqa: E402
import region_aggregate as ra  # noqa: E402

FIG_DIR = Path.home().joinpath("Documents", "figures")
DATE = "2026-07-26"
TIERS = ["uncompressed", "default", "aggressive"]
ROOT = Path(__file__).resolve().parent

VERSIONS: dict[str, dict[str, Path]] = {
    "v00": {t: ROOT.joinpath("results_bwm_2026-07-07", t) for t in TIERS},
    "v01": {t: ROOT.joinpath("results_bwm_v01", t) for t in TIERS},
    "v01_smart": {t: ROOT.joinpath("results_bwm_v01_smart", t) for t in TIERS},
}


def _save(fig: plt.Figure, name: str, tag: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR.joinpath(f"{DATE}_lfpenc_{tag}_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def collapse_comparison(versions: list[str]) -> pd.DataFrame:
    """Catastrophic-collapse rate (unfiltered, all PIDs) across ``versions``."""
    return pd.concat(
        [ra.collapse_rate(VERSIONS[v]).assign(version=v) for v in versions], ignore_index=True
    )


def plot_collapse_comparison(df: pd.DataFrame, versions: list[str], tag: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="tier", y="frac_pids_collapsed", hue="version", hue_order=versions, order=TIERS, ax=ax)
    ax.set_ylabel("fraction of PIDs collapsed\n(median CV R² < -0.5 in >=1 band)")
    ax.set_xlabel("compression tier")
    ax.set_title(f"Catastrophic-overfit rate: {' vs '.join(versions)}")
    fig.tight_layout()
    return _save(fig, "collapse_rate", tag)


def headline_keeppids(tier_dirs: dict[str, Path], pids: frozenset[str]) -> pd.DataFrame:
    """frac_sig / median R² by (tier, band), restricted to ``pids``."""
    rows = []
    for tier, path in tier_dirs.items():
        df = ra.load_channel_scores(path, pids=pids)
        for band, sub in df.groupby("band", observed=True):
            rows.append({
                "tier": tier, "band": band,
                "median_r2_cv": sub["r2_cv"].median(),
                "frac_sig": float((sub["p_value"] < 0.05).mean()),
                "n_channels": len(sub),
            })
    return pd.DataFrame(rows)


def headline_comparison(versions: list[str], pids: frozenset[str] = pid_qc.KEEP_PIDS) -> pd.DataFrame:
    return pd.concat(
        [headline_keeppids(VERSIONS[v], pids).assign(version=v) for v in versions], ignore_index=True
    )


def plot_raw_r2_jump(df: pd.DataFrame, versions: list[str], tag: str) -> Path:
    sub = df[df["band"] == "raw"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=sub, x="tier", y="median_r2_cv", hue="version", hue_order=versions, order=TIERS, ax=ax)
    ax.set_ylabel("median held-out R² (raw broadband)")
    ax.set_xlabel("compression tier")
    ax.set_title(f"Raw-ERP encoding R²: {' vs '.join(versions)}")
    fig.tight_layout()
    return _save(fig, "raw_r2_jump", tag)


def collapsed_pids(tier_dirs: dict[str, Path], tier: str, threshold: float = -0.5) -> set[str]:
    med = ra.pid_band_r2(tier_dirs)
    bad = med[(med["tier"] == tier) & (med["median_r2_cv"] < threshold)]
    return set(bad["pid"].unique())


def newly_collapsed(from_version: str, to_version: str, tier: str, threshold: float = -0.5) -> set[str]:
    """PIDs that collapse in ``to_version`` but did not in ``from_version`` -- a regression."""
    before = collapsed_pids(VERSIONS[from_version], tier, threshold)
    after = collapsed_pids(VERSIONS[to_version], tier, threshold)
    return after - before


def recovered(from_version: str, to_version: str, tier: str, threshold: float = -0.5) -> set[str]:
    """PIDs that collapsed in ``from_version`` and no longer do in ``to_version``."""
    before = collapsed_pids(VERSIONS[from_version], tier, threshold)
    after = collapsed_pids(VERSIONS[to_version], tier, threshold)
    return before - after


def group_recovery_table(from_version: str, to_version: str) -> pd.DataFrame:
    """Did ``to_version`` recover the specific known-bad PID groups from DATA_ISSUES.md?"""
    groups = {
        "GROUP_A_bad_recording": (pid_qc.GROUP_A_BAD_RECORDING, "uncompressed"),
        "GROUP_B_nc24_corruption": (pid_qc.GROUP_B_NC24_CORRUPTION, "default"),
        "DEFAULT_COMPRESSION_ARTIFACT": (pid_qc.DEFAULT_COMPRESSION_ARTIFACT, "default"),
        "AGGRESSIVE_ONLY_COLLAPSE": (pid_qc.AGGRESSIVE_ONLY_COLLAPSE, "aggressive"),
    }
    rows = []
    for name, (pids, tier) in groups.items():
        before = collapsed_pids(VERSIONS[from_version], tier) & pids
        after = collapsed_pids(VERSIONS[to_version], tier) & pids
        rows.append({
            "group": name, "tier": tier, "n": len(pids),
            f"still_collapsed_{from_version}": len(before), f"still_collapsed_{to_version}": len(after),
            "recovered": len(pids) - len(after),
        })
    return pd.DataFrame(rows)


def compare_pair(from_version: str, to_version: str) -> None:
    """Run and print/save the full standard comparison for one ordered version pair."""
    versions = [from_version, to_version]
    tag = f"{from_version}vs{to_version}"

    print(f"=== Collapse rate, {tag} (unfiltered, all PIDs) ===")
    coll = collapse_comparison(versions)
    print(coll[["version", "tier", "n_pids_total", "n_pids_collapsed", "frac_pids_collapsed"]].to_string(index=False))
    print("saved", plot_collapse_comparison(coll, versions, tag))

    print(f"\n=== Known-bad-group recovery (DATA_ISSUES.md groups), {tag} ===")
    print(group_recovery_table(from_version, to_version).to_string(index=False))

    print(f"\n=== Per-tier regression/recovery vs {from_version}, {tag} ===")
    for tier in TIERS:
        new = newly_collapsed(from_version, to_version, tier)
        fixed = recovered(from_version, to_version, tier)
        print(f"  {tier:13s}: newly collapsed={len(new)} {sorted(new) if new else ''}  "
              f"recovered={len(fixed)}")

    print(f"\n=== Tier headline (KEEP_PIDS), {tag} ===")
    head = headline_comparison(versions)
    piv = head.pivot_table(index=["tier", "band"], columns="version", values=["frac_sig", "median_r2_cv"])
    print(piv.to_string())
    print("saved", plot_raw_r2_jump(head, versions, tag))


if __name__ == "__main__":
    args = sys.argv[1:] or ["v01", "v01_smart"]
    compare_pair(*args)
