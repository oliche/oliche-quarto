"""Compare the v00 (July resweep, `results_bwm_2026-07-07`) vs v01 (updated
lfpack pre-processing/compression, `results_bwm_v01`) encoding fits across all
3 LFP sources (uncompressed/default/aggressive).

v01 changed the upstream `lfpack` pipeline, not the encoding model itself
(`model_config.json` is byte-identical): highpass corner 2.0->0.5 Hz (applied
to ALL 3 tiers, since it's part of the shared Cadzow/CAR pre-processing every
tier is decimated from), default-tier alpha 32->28, a WP dominant-component
survival floor (floor_k=64, fixes exact-zero decompression on low-SNR chunks),
and LFP saturation detection/muting. See `packages/lfpack` git log
(c3fa037, 37951b5, 71b4f16, a24b9c9) for the underlying commits.
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
DATE = "2026-07-25"
TIERS = ["uncompressed", "default", "aggressive"]
ROOT = Path(__file__).resolve().parent
V00 = {t: ROOT.joinpath("results_bwm_2026-07-07", t) for t in TIERS}
V01 = {t: ROOT.joinpath("results_bwm_v01", t) for t in TIERS}


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR.joinpath(f"{DATE}_lfpenc_v00vsv01_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def collapse_comparison() -> pd.DataFrame:
    """Catastrophic-collapse rate (unfiltered, all 699 PIDs), v00 vs v01."""
    c00 = ra.collapse_rate(V00).assign(version="v00")
    c01 = ra.collapse_rate(V01).assign(version="v01")
    return pd.concat([c00, c01], ignore_index=True)


def plot_collapse_comparison(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="tier", y="frac_pids_collapsed", hue="version", order=TIERS, ax=ax)
    ax.set_ylabel("fraction of PIDs collapsed\n(median CV R² < -0.5 in >=1 band)")
    ax.set_xlabel("compression tier")
    ax.set_title("Catastrophic-overfit rate: v00 (2026-07 resweep) vs v01 (updated lfpack)")
    fig.tight_layout()
    return _save(fig, "collapse_rate")


def headline_keeppids(tier_dirs: dict[str, Path], pids: frozenset[str]) -> pd.DataFrame:
    """frac_sig / median R² by (tier, band), restricted to v00's KEEP_PIDS."""
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


def headline_comparison() -> pd.DataFrame:
    h00 = headline_keeppids(V00, pid_qc.KEEP_PIDS).assign(version="v00")
    h01 = headline_keeppids(V01, pid_qc.KEEP_PIDS).assign(version="v01")
    return pd.concat([h00, h01], ignore_index=True)


def plot_raw_r2_jump(df: pd.DataFrame) -> Path:
    """The raw-broadband median CV-R², v00 vs v01 -- the highpass-corner story."""
    sub = df[df["band"] == "raw"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=sub, x="tier", y="median_r2_cv", hue="version", order=TIERS, ax=ax)
    ax.set_ylabel("median held-out R² (raw broadband)")
    ax.set_xlabel("compression tier")
    ax.set_title("Raw-ERP encoding R²: v00 (2 Hz highpass) vs v01 (0.5 Hz highpass)")
    fig.tight_layout()
    return _save(fig, "raw_r2_jump")


def collapsed_pids(tier_dirs: dict[str, Path], tier: str, threshold: float = -0.5) -> set[str]:
    med = ra.pid_band_r2(tier_dirs)
    bad = med[(med["tier"] == tier) & (med["median_r2_cv"] < threshold)]
    return set(bad["pid"].unique())


def group_recovery_table() -> pd.DataFrame:
    """Did v01 recover the specific known-bad PID groups from DATA_ISSUES.md?"""
    groups = {
        "GROUP_A_bad_recording": (pid_qc.GROUP_A_BAD_RECORDING, "uncompressed"),
        "GROUP_B_nc24_corruption": (pid_qc.GROUP_B_NC24_CORRUPTION, "default"),
        "DEFAULT_COMPRESSION_ARTIFACT": (pid_qc.DEFAULT_COMPRESSION_ARTIFACT, "default"),
        "AGGRESSIVE_ONLY_COLLAPSE": (pid_qc.AGGRESSIVE_ONLY_COLLAPSE, "aggressive"),
    }
    rows = []
    for name, (pids, tier) in groups.items():
        c00 = collapsed_pids(V00, tier) & pids
        c01 = collapsed_pids(V01, tier) & pids
        rows.append({
            "group": name, "tier": tier, "n": len(pids),
            "still_collapsed_v00": len(c00), "still_collapsed_v01": len(c01),
            "recovered": len(pids) - len(c01),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=== Collapse rate, v00 vs v01 (unfiltered, all 699 PIDs) ===")
    coll = collapse_comparison()
    print(coll[["version", "tier", "n_pids_total", "n_pids_collapsed", "frac_pids_collapsed"]].to_string(index=False))
    print("saved", plot_collapse_comparison(coll))

    print("\n=== Known-bad-group recovery (DATA_ISSUES.md groups) ===")
    recovery = group_recovery_table()
    print(recovery.to_string(index=False))

    print("\n=== Tier headline (KEEP_PIDS), v00 vs v01 ===")
    head = headline_comparison()
    piv = head.pivot_table(index=["tier", "band"], columns="version", values=["frac_sig", "median_r2_cv"])
    print(piv.to_string())
    print("saved", plot_raw_r2_jump(head))


if __name__ == "__main__":
    main()