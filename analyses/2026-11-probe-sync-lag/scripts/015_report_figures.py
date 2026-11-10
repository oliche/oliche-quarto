"""Executive-report figure: wheel-measured offset against the shift inferred from the metadata, for the recordings
labelled `missing edges at start` (014) that have a wheel estimate (BWM from 008, other projects from 013).

Writes `figures/015_wheel_vs_shift.png` and a dated copy in ~/Documents/figures.
"""
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parents[1]
SRC = Path.home().joinpath("Documents", "ibl-ai-agent", "projects", "spike-clock-offset-investigation", "exploratory-analyses")
SNR_MIN = 2
sns.set_theme(context="notebook", style="whitegrid")

g = pd.read_parquet(ROOT.joinpath("artifacts", "010", "012_good_probe.pqt"))
g = g[g.issue.eq("missing edges at start") & g.wheel_offset_s.notna()]
snr = pd.concat([pd.read_csv(SRC.joinpath("008_damage_report.csv"))[["pid", "peak_snr"]],
                 pd.read_csv(ROOT.joinpath("artifacts", "013_wheel_cohort3.csv"))[["pid", "peak_snr"]]]).dropna().drop_duplicates("pid")
g = g.merge(snr, on="pid", how="left")
g["resid"] = g.wheel_offset_s - g.shift_s
g["weak"] = g.peak_snr < SNR_MIN

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})
lim = [-7.6, 1.1]
axs[0].plot(lim, lim, color="0.6", lw=1, zorder=0)
for (bwm, weak), d in g.groupby(["bwm", "weak"]):
    c = "C3" if bwm else "C0"
    kw = {"facecolors": "none", "edgecolors": c} if weak else {"color": c}
    lab = f"{'BWM' if bwm else 'other projects'}{' (weak peak)' if weak else ''}, n={len(d)}"
    for ax, y in zip(axs, [d.wheel_offset_s, d.resid]):
        ax.scatter(d.shift_s, y, s=36, alpha=0.8, label=lab, **kw)
axs[0].set(xlim=lim, ylim=lim, xlabel="shift from metadata, k × 0.5 s (s)", ylabel="offset from wheel (s)",
           title="Wheel offset matches the metadata shift")
axs[0].legend(loc="upper left", fontsize=9)
axs[1].axhline(0, color="0.6", lw=1, zorder=0)
axs[1].axhspan(-0.11, 0.11, color="0.9", zorder=-1)
axs[1].set(xlim=lim, ylim=[-1.1, 0.4], xlabel="shift from metadata (s)", ylabel="wheel − metadata (s)", title="Residual")
fig.tight_layout()
fig.savefig(ROOT.joinpath("figures", "015_wheel_vs_shift.png"), dpi=150)
fig.savefig(Path.home().joinpath("Documents", "figures", "2026-11-10_probe_sync_wheel_vs_shift.png"), dpi=150)
print(len(g), g.weak.sum(), np.abs(g.resid[~g.weak]).max())
