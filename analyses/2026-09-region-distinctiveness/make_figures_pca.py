"""Scree plot for the global PCA basis (`pca_mahalanobis.py`). Kept unchanged from the earlier
neighbour-buffer-framed work (see `PLAN.md`) since the PCA fit itself didn't depend on that framing.

Run after pairwise_mahalanobis.py (needs cache/pca_basis.npz).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="notebook", style="whitegrid")

HERE = Path(__file__).parent
CACHE_DIR = HERE / "cache"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def fig_scree(pca_basis):
    ratio = pca_basis["all_explained_ratio"]
    k = int(pca_basis["k"])
    n_show = 15
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(1, n_show + 1), ratio[:n_show] * 100, color="#4c72b0", label="per-PC")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, n_show + 1), np.cumsum(ratio[:n_show]) * 100, "o-", color="#c44e52",
              label="cumulative")
    ax2.axhline(95, color="grey", ls="--", lw=1)
    ax2.axvline(k + 0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("principal component")
    ax.set_ylabel("% variance explained (per PC)")
    ax2.set_ylabel("cumulative % variance explained")
    ax.set_title(f"PCA scree plot — {k} components reach 95% of variance (of 41 features)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_pca_scree.png", dpi=150)
    plt.close(fig)


def main():
    pca_basis = np.load(CACHE_DIR / "pca_basis.npz", allow_pickle=True)
    fig_scree(pca_basis)
    print("PCA scree figure written to", FIG_DIR)


if __name__ == "__main__":
    main()
