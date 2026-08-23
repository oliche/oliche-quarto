"""BWM release QC figures: zero-damage survival-floor fix validation + release-wide stats.

Run once to (re)generate the figures referenced by bwm_release_qc.qmd. Reads the
zero-damage scan CSVs and the per-pid QC parquet already produced for the v03 BWM
subset -- no raw data access needed.
"""
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import addcopyfighandler  # noqa: F401

sns.set_theme(context="notebook")

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-olivier-PycharmProjects-ephys-atlas-packages-lfpack/"
    "5405c415-e8b2-4fa0-9a31-d183550a3e8a/scratchpad"
)
QC_DIR = Path("/Users/olivier/Documents/datadisk/lfp-processing/lfpack/v03_bwm")
FIGURE_DIR = Path(__file__).parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
DATE = "2026-07-24"

# %% Figure 1 -- zero-damage fix, before/after, both passes
zd_default = pd.read_csv(SCRATCH / "zero_damage_v00_vs_v03.csv")
zd_aggressive = pd.read_csv(SCRATCH / "zero_damage_v00_vs_v03_aggressive.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, df, title in zip(
    axes, [zd_default, zd_aggressive], ["Default pass (ε=150, α=28)", "Aggressive pass (ε=450, α=96)"]
):
    top = df.sort_values("zero_fraction_v00", ascending=False).head(15).iloc[::-1]
    y = np.arange(len(top))
    ax.barh(y - 0.2, top["zero_fraction_v00"] * 100, height=0.4, label="v00 (live)", color="#d62728")
    ax.barh(y + 0.2, top["zero_fraction_v03"] * 100, height=0.4, label="v03 (fixed)", color="#2ca02c")
    ax.set_yticks(y)
    ax.set_yticklabels([p[:8] for p in top["pid"]], fontsize=8)
    ax.set_xlabel("Zero-time fraction (%)")
    ax.set_title(title)
    ax.legend()
fig.suptitle("Survival-floor fix: 15 worst recordings by pre-fix zero-time fraction")
fig.tight_layout()
fig.savefig(FIGURE_DIR / f"{DATE}_bwm_zero_damage_before_after.png", dpi=150)
plt.close(fig)

# %% Figure 2 -- release-wide RMSE / CR distributions (v03, both passes)
df_pid = pd.read_parquet(QC_DIR / "lfp_qc_per_pid.pqt")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for pass_name, color in zip(["default", "aggressive"], ["#1f77b4", "#ff7f0e"]):
    d = df_pid[df_pid["pass"] == pass_name]
    sns.histplot(d["rmse_mean"], bins=40, ax=axes[0], label=pass_name, color=color, alpha=0.5, element="step")
    sns.histplot(
        np.log10(d["cr_total_mean"].clip(lower=1)), bins=40, ax=axes[1], label=pass_name,
        color=color, alpha=0.5, element="step",
    )

axes[0].axvline(25, color="k", ls="--", lw=1, label="25 µV spec")
axes[0].set_xlabel("Mean RMSE per recording (µV)")
axes[0].legend()

axes[1].set_xlabel("Mean total CR per recording (log10)")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{10**x:.0f}"))
axes[1].legend()

fig.suptitle(f"BWM release v03 — {df_pid['pid'].nunique()} recordings — RMSE / CR")
fig.tight_layout()
fig.savefig(FIGURE_DIR / f"{DATE}_bwm_release_qc_distributions.png", dpi=150)
plt.close(fig)

# %% Figure 3 -- saturation detection & muting (v03 only -- v00 has no saturation dataset at all)
d = df_pid[df_pid["pass"] == "default"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sat = d.loc[d["saturated_fraction"] > 0, "saturated_fraction"]
sns.histplot(np.log10(sat), bins=30, ax=axes[0], color="#9467bd")
axes[0].set_xlabel("Saturated fraction (log10, recordings with detected saturation only)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{10**x * 100:.4g}%"))
axes[0].set_title(f"{len(sat)}/{len(d)} recordings have ≥1 saturated interval")

top = d.sort_values("saturated_fraction", ascending=False).head(15).iloc[::-1]
y = np.arange(len(top))
bars = axes[1].barh(y, top["saturated_fraction"] * 100, color="#9467bd")
axes[1].set_yticks(y)
axes[1].set_yticklabels([p[:8] for p in top["pid"]], fontsize=8)
axes[1].set_xlabel("Saturated fraction (%)")
axes[1].set_title("15 most-saturated recordings")
for bar, sec in zip(bars, top["total_saturated_sec"]):
    axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{sec:.0f}s",
                 va="center", fontsize=7)

fig.suptitle("Saturation detection (default pass) -- new in v03, absent from v00")
fig.tight_layout()
fig.savefig(FIGURE_DIR / f"{DATE}_bwm_saturation_detection.png", dpi=150)
plt.close(fig)

print("Figures written to", FIGURE_DIR)