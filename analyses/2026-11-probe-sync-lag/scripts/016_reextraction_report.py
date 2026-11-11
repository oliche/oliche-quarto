"""Figures and tables of the re-extraction report: wheel check of the re-extracted spike times (67 insertions) and
BWM / prior decoding of the 17 BWM insertions with the original and the re-extracted spike times.

Inputs, pulled from popeye `/mnt/home/owinter/ceph/sync_lag` (`sdsc-slurms/2026-11_probe-sync-lag`):
- `artifacts/sync_lag/wheel.pqt`: `sync_lag.py --step aggregate`
- `artifacts/decoding/decode.pqt`: `decoding.py --aggregate`, one row per condition x target x session x region

Outputs: `figures/016_wheel_reextracted.png`, `figures/016_decoding.png`, `artifacts/decoding/016_*.csv`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parents[1]
FIG = ROOT.joinpath("figures")
ART = ROOT.joinpath("artifacts")
H = 0.5
ALPHA = 0.05
TARGETS = {"stimside": "stimulus side", "choice": "choice", "feedback": "feedback", "pLeft": "prior (pLeft)"}
sns.set_theme(context="notebook", style="whitegrid")
pids = pd.read_csv(Path.home().joinpath("PycharmProjects", "ibllib", "sdsc-slurms", "2026-11_probe-sync-lag", "pids.csv"))
# report sign convention: shift > 0 means the released spikes are late, shift = k * 0.5 s
pids["shift_s"] = pids.k * H
pids["direction"] = np.where(pids.k > 0, "late (+0.5 s)", "early (−1 to −3 s)")

# %% wheel: offset of the population rate vs wheel speed, released and re-extracted spike times
w = pd.read_parquet(ART.joinpath("sync_lag", "wheel.pqt"))
fig, ax = plt.subplots(figsize=(6, 5))
lim = np.abs(w[["original.offset_s", "expected_offset_s"]].to_numpy()).max() + 0.5
ax.axline((0, 0), slope=1, color="grey", lw=0.8)
ax.scatter(w.expected_offset_s, w["original.offset_s"], s=25, color="C3", label="released spike times")
ax.scatter(w.expected_offset_s, w["shifted.offset_s"], s=25, color="C2", label="re-extracted spike times")
ax.set(xlabel="shift predicted from the metadata (s)", ylabel="wheel offset (s)", xlim=(-lim, lim), ylim=(-lim, lim))
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIG.joinpath("016_wheel_reextracted.png"), dpi=150)
wsum = {
    "n": len(w),
    "confident": int(w.wheel_confident.sum()),
    "aligned_confident": int((w.aligned & w.wheel_confident).sum()),
    "max_abs_offset_confident": float(w.loc[w.wheel_confident, "shifted.offset_s"].abs().max()),
}
print(wsum)
print(w.loc[~(w.aligned & w.wheel_confident), ["pid", "subject", "expected_offset_s", "original.offset_s", "shifted.offset_s", "shifted.peak_snr"]])

# %% decoding: one row per target x session x region with both conditions and the full null distribution
d = pd.read_parquet(ART.joinpath("decoding", "decode.pqt"))
d = d[d.n_pseudo == 200]
v = d.pivot_table(index=["target", "pid", "region"], columns="condition", values=["score_minus_null", "p_value", "n_units"])
v.columns = [f"{a}_{b}" for a, b in v.columns]
v = v.dropna().reset_index().merge(pids[["pid", "subject", "date", "probe", "k", "shift_s", "direction"]], on="pid")
v["sig_original"], v["sig_shifted"] = v.p_value_original < ALPHA, v.p_value_shifted < ALPHA
v["change"] = np.select(
    [v.sig_original & ~v.sig_shifted, ~v.sig_original & v.sig_shifted, v.sig_original & v.sig_shifted],
    ["lost", "gained", "kept"],
    "never",
)
v.to_csv(ART.joinpath("decoding", "016_session_region.csv"), index=False)

summary = (
    v.groupby(["target", "direction"])
    .agg(
        sessions=("pid", "nunique"),
        session_regions=("region", "size"),
        median_original=("score_minus_null_original", "median"),
        median_shifted=("score_minus_null_shifted", "median"),
        sig_original=("sig_original", "sum"),
        sig_shifted=("sig_shifted", "sum"),
        lost=("change", lambda c: (c == "lost").sum()),
        gained=("change", lambda c: (c == "gained").sum()),
    )
    .reindex(list(TARGETS), level=0)
)
summary.round(3).to_csv(ART.joinpath("decoding", "016_summary.csv"))
print(summary.round(3).to_string())
lost = v[v.change == "lost"].sort_values(["target", "score_minus_null_original"], ascending=[True, False])
lost.to_csv(ART.joinpath("decoding", "016_lost.csv"), index=False)
print(lost[["target", "subject", "date", "region", "n_units_original", "shift_s", "score_minus_null_original",
            "score_minus_null_shifted", "p_value_original", "p_value_shifted"]].round(3).to_string())  # fmt: skip

# %% figure: corrected score, released vs re-extracted, one panel per target
fig, axs = plt.subplots(1, 4, figsize=(19, 5))
for ax, (target, label) in zip(axs, TARGETS.items()):
    t = v[v.target == target]
    if t.empty:
        ax.set_axis_off()
        continue
    lo = min(t.score_minus_null_original.min(), t.score_minus_null_shifted.min()) - 0.02
    hi = max(t.score_minus_null_original.max(), t.score_minus_null_shifted.max()) + 0.02
    for direction, color in zip(["late (+0.5 s)", "early (−1 to −3 s)"], ["C0", "C1"]):
        s = t[t.direction == direction]
        ax.scatter(s.score_minus_null_original, s.score_minus_null_shifted, s=22, color=color, label=direction)
    s = t[t.change == "lost"]
    ax.scatter(s.score_minus_null_original, s.score_minus_null_shifted, s=70, facecolors="none", edgecolors="k",
               label="significant only with the released spikes")  # fmt: skip
    ax.axline((0, 0), slope=1, color="grey", lw=0.8)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), title=f"{label}: p<0.05 {t.sig_original.sum()} → {t.sig_shifted.sum()}",
           xlabel="released spike times", ylabel="re-extracted spike times")  # fmt: skip
axs[0].legend(loc="upper left", fontsize=8)
fig.supxlabel("decoding score minus null median (balanced accuracy; R² for the prior), one point per session × region")
fig.tight_layout()
fig.savefig(FIG.joinpath("016_decoding.png"), dpi=150)
