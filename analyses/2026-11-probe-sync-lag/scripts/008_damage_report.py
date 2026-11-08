"""Consolidated damage report: sync-rule offset for every scanned insertion, cross-checked with the wheel xcorr offset.

Sync rule (no behaviour): SpikeGLX starts probe and nidq files on the same phase of the 1 Hz imec sync, so the true
start offset is a whole number k of 0.5 s half-periods and the fronts alone cannot reveal it. ibllib version3B pairs
fronts from index 0, i.e. assumes k = 0. The edge-count difference dn = n_probe - n_nidq splits into fronts gained at
the start and lost at the end; (start_lead - end_lead) / 0.5 measures the same quantity from firstSample, which has a
shared error that cancels in the difference. When both agree, the deficit sits at the file boundaries and the files
end together, so k = dn and spikes are shifted by dn * 0.5 s. Otherwise fronts were lost mid-recording and k = 0.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).parent
FIG_DIR = Path.home().joinpath("Documents", "figures")
HALF = 0.5
sig = pd.read_csv(HERE.joinpath("004_signature_summary.csv"))
wx = pd.read_csv(HERE.joinpath("007_wheel_xcorr_candidates.csv"))[["pid", "role", "offset_s", "ci_lo", "ci_hi", "peak_snr", "peak_lag_s"]]
base = wx.peak_lag_s[wx.role == "control"].median()

sig["boundary_halfperiods"] = (sig.start_lead_s - sig.end_lead_s) / HALF
sig["boundary_explained"] = (sig.boundary_halfperiods - sig.dn).abs() < 0.25
sig["lag_rule"] = np.where(sig.boundary_explained, sig.dn, 0).astype(int)
sig["offset_rule_s"] = sig.lag_rule * HALF
df = sig.merge(wx.drop(columns="role").rename(columns={"offset_s": "offset_wheel_s"}), on="pid", how="left")
df["wheel_ci_width_s"] = df.ci_hi - df.ci_lo
df["wheel_confident"] = (df.peak_snr >= 2) & (df.wheel_ci_width_s < 0.3)
df["offset_firstsample_s"] = df.lag_float.round() * HALF
df["affected"] = df.offset_rule_s != 0
df["method"] = np.where(df.wheel_confident, "sync rule + wheel xcorr", "sync rule")
df["confidence"] = np.select(
    [df.wheel_confident & ((df.offset_wheel_s - df.offset_rule_s).abs() < 0.15), df.wheel_confident, df.boundary_explained | (df.dn == 0)],
    ["high (sync + wheel agree)", "CONFLICT", "medium (sync only)"], "low")
df.to_csv(HERE.joinpath("008_damage_report.csv"), index=False)

pd.set_option("display.width", 250, "display.max_rows", 200)
w = df[df.wheel_confident]
print("wheel-confident insertions:", len(w), "| rule agrees:", ((w.offset_wheel_s - w.offset_rule_s).abs() < 0.15).sum(),
      "| firstSample-only agrees:", ((w.offset_wheel_s - w.offset_firstsample_s).abs() < 0.15).sum())
print(df.confidence.value_counts().to_string())
cols = ["set", "pid", "eid", "lab", "subject", "date", "probe", "dn", "offset_rule_s", "offset_wheel_s", "confidence", "appVersion", "imDatBsc_fw"]
print(df[df.affected].sort_values(["offset_rule_s", "date"])[cols].round(2).to_string())
print(df[df.confidence.isin(["CONFLICT", "low"])][cols + ["peak_snr", "boundary_halfperiods"]].round(2).to_string())

# figures
sns.set_theme(context="notebook", style="whitegrid")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ax = axs[0]
sns.scatterplot(data=w, x="offset_rule_s", y="offset_wheel_s", hue="dn", palette="coolwarm", s=60, ax=ax, legend="full")
ax.plot([-3.2, 1], [-3.2, 1], "k:", lw=1)
ax.set(xlabel="sync-rule offset (s)", ylabel="wheel xcorr offset (s)", title=f"Wheel vs sync rule (n={len(w)} confident)")
ax = axs[1]
sns.scatterplot(data=df, x="dn", y="boundary_halfperiods", hue="affected", palette={True: "tab:red", False: "tab:gray"}, s=40, ax=ax)
ax.plot([-7, 2], [-7, 2], "k:", lw=1)
ax.set(xlabel="edge-count difference n_probe - n_nidq", ylabel="(start lead - end lead) / 0.5 s from firstSample", title=f"Edge deficit at file boundaries (n={len(df)})")
fig.tight_layout()
for d in [HERE.joinpath("008_damage_report.png"), FIG_DIR.joinpath("2026-11-06_probe_sync_lag_damage.png")]:
    fig.savefig(d, dpi=150)
