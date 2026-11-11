"""Preview of the BWM decoding of the shifted insertions: original vs re-extracted spike times.

Input: `artifacts/decoding/decode_partial.pqt`, pulled from popeye (`sdsc-slurms/2026-11_probe-sync-lag/decoding.py
--aggregate`), restricted to the session x region x target with the full 200 pseudo-sessions in both conditions.
One point per session x region x target: balanced accuracy minus the median of the null, original (x) vs shifted (y).
"""

from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parents[1]
FIG_DIR = Path.home().joinpath("Documents", "figures")
sns.set_theme(context="notebook", style="whitegrid")

d = pd.read_parquet(ROOT.joinpath("artifacts", "decoding", "decode_partial.pqt"))
d = d[d.n_pseudo == 200]
w = d.pivot_table(index=["pid", "target", "region"], columns="condition", values=["score_minus_null", "p_value"]).dropna()
w.columns = [f"{v}_{c}" for v, c in w.columns]
w = w.reset_index()
pids = pd.read_csv(Path.home().joinpath("PycharmProjects", "ibllib", "sdsc-slurms", "2026-11_probe-sync-lag", "pids.csv"))
w = w.merge(pids[["pid", "subject", "date", "k"]], on="pid")

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for ax, target in zip(axs, ["stimside", "choice", "feedback"]):
    t = w[w.target == target]
    sns.scatterplot(data=t, x="score_minus_null_original", y="score_minus_null_shifted", hue="k", palette="deep", ax=ax)
    ax.axline((0, 0), slope=1, color="k", lw=0.8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    n_sig = ((t.p_value_original < 0.05).sum(), (t.p_value_shifted < 0.05).sum())
    ax.set(title=f"{target}: {len(t)} session x region, p<0.05 {n_sig[0]} -> {n_sig[1]}",
           xlabel="original: accuracy - null median", ylabel="shifted: accuracy - null median")  # fmt: skip
fig.suptitle(f"BWM decoding of {w.pid.nunique()} shifted insertions, original vs re-extracted spike times (preview)")
fig.tight_layout()
fig.savefig(FIG_DIR.joinpath("2026-11-10_sync_lag_decoding_preview.png"), dpi=150)
print(w.groupby("target")[["score_minus_null_original", "score_minus_null_shifted"]].median().round(3))
print(w.groupby("target").apply(lambda t: pd.Series({"n": len(t), "sig_original": (t.p_value_original < 0.05).sum(),
                                                       "sig_shifted": (t.p_value_shifted < 0.05).sum()})))  # fmt: skip
