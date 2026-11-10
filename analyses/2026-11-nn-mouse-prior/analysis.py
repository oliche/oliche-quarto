"""Behaviour, prior-model comparison and single-unit analyses of the trained networks.

Usage: MPLBACKEND=Agg python analysis.py
"""

from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression

import task
from models import Decision, FrontEnd, GRUCore, PlasticCore, run_sessions

sns.set_theme(context="notebook", style="whitegrid")
torch.set_num_threads(4)
HERE = Path(__file__).parent
MODELS = HERE.joinpath("models")
FIG_LOCAL = HERE.joinpath("figures")
FIG_HOME = Path.home().joinpath("Documents", "figures")
DATE = "2026-11-08"
MOUSE_TAU = 5.45  # median action-kernel time constant across mice (prior paper, Fig. 4b)
ALPHAS = np.geomspace(0.01, 0.9, 40)
AGENTS = {"no core": None, "A: GRU": "gru", "B: plastic, feedback": "plastic_feedback",
          "B: plastic, choice": "plastic_choice"}


def savefig(fig, name):
    """Save a figure to the report folder and to ~/Documents/figures with a date prefix."""
    FIG_LOCAL.mkdir(exist_ok=True)
    FIG_HOME.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_LOCAL.joinpath(f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(FIG_HOME.joinpath(f"{DATE}_nn-mouse-prior_{name}.png"), dpi=150, bbox_inches="tight")


def tau(alpha):
    """Integration time constant (trials) of an exponential kernel with learning rate alpha."""
    return -1 / np.log(1 - alpha)


def load(arm, ckpt=None):
    """Load frozen front-end, decision layer and the core of an arm (last checkpoint by default)."""
    pre = torch.load(MODELS.joinpath("pretrained.pt"))
    frontend, decision = FrontEnd(), Decision()
    frontend.load_state_dict(pre["frontend"]), decision.load_state_dict(pre["decision"])
    if arm is None:
        return frontend, decision, None
    core = GRUCore() if arm == "gru" else PlasticCore(decision, arm.split("_")[1])
    ckpt = ckpt or max(MODELS.joinpath(arm).glob("ckpt_*.pt"))
    core.load_state_dict(torch.load(ckpt))
    return frontend, decision, core


def evaluate(frontend, decision, core, n_sessions=200, seed=123, ablate=None):
    """Run fixed evaluation sessions and return numpy arrays of task, behaviour and units."""
    rng = np.random.default_rng(seed)
    s = task.sample_sessions(n_sessions, 500, rng)
    x = torch.from_numpy(task.evidence(s["side"], s["contrast"], rng))
    with torch.no_grad():
        out = run_sessions(frontend, decision, core, x, torch.from_numpy(s["side"]).float(),
                           ablate=ablate, generator=torch.Generator().manual_seed(seed))
    s.update(p=torch.sigmoid(out["logit"]).numpy(), action=out["action"].numpy(), d=out["d"].numpy(),
             units=None if out["units"] is None else out["units"].numpy())
    s["bayes"] = task.bayes_optimal_prior(s["side"])
    s["signed"] = s["side"] * s["contrast"]
    return s


def block_bias(s):
    """p(right | 0% contrast, 0.8 block) - p(right | 0% contrast, 0.2 block), from the policy."""
    zero = s["contrast"] == 0
    return s["p"][zero & (s["p_right"] == 0.8)].mean() - s["p"][zero & (s["p_right"] == 0.2)].mean()


def fit_prior_models(s):
    """Held-out log-likelihood gain of choice models with a prior term, and fitted kernel tau.

    The choice model is a logistic regression on one-hot signed contrast plus logit(prior).
    Kernel learning rates are chosen on the first half of sessions, all models are
    evaluated on the second half; only biased trials (t >= 90) are used.
    """
    levels = np.unique(s["signed"])
    sl = np.s_[:, task.N_UNBIASED:]
    y = (s["action"][sl] > 0).ravel()
    onehot = (s["signed"][sl].ravel()[:, None] == levels).astype(float)
    n_train = s["side"].shape[0] // 2 * (s["side"].shape[1] - task.N_UNBIASED)

    def loglik(prior):
        feats = onehot if prior is None else np.c_[onehot, np.log(prior / (1 - prior))[sl].ravel()]
        feats_train, feats_test = feats[:n_train], feats[n_train:]
        model = LogisticRegression(C=1e4, fit_intercept=False, max_iter=2000).fit(feats_train, y[:n_train])
        train_ll = -np.mean(np.log(np.clip(model.predict_proba(feats_train)[np.arange(n_train), y[:n_train].astype(int)], 1e-9, 1)))
        p_test = np.clip(model.predict_proba(feats_test)[:, 1], 1e-6, 1 - 1e-6)
        return -train_ll, np.mean(y[n_train:] * np.log(p_test) + (1 - y[n_train:]) * np.log(1 - p_test))

    base = loglik(None)[1]
    res = {"Bayes-optimal": (loglik(np.clip(s["bayes"], 1e-3, 1 - 1e-3))[1] - base, np.nan)}
    for name, events in [("stimulus kernel", s["side"]), ("action kernel", s["action"])]:
        fits = [loglik(np.clip(task.kernel_prior(events, a), 1e-3, 1 - 1e-3)) for a in ALPHAS]
        best = int(np.argmax([f[0] for f in fits]))
        res[name] = (fits[best][1] - base, tau(ALPHAS[best]))
    return res


def unit_regression(units, regressors):
    """Standardized regression coefficients of each unit on a set of regressors.

    Parameters
    ----------
    units : np.ndarray
        (n_samples, n_units) activity.
    regressors : np.ndarray
        (n_samples, n_regressors).

    Returns
    -------
    np.ndarray
        (n_units, n_regressors) standardized betas.
    """
    zx = (regressors - regressors.mean(0)) / regressors.std(0)
    zy = (units - units.mean(0)) / (units.std(0) + 1e-9)
    return np.linalg.lstsq(zx, zy, rcond=None)[0].T


def unit_kernel_fit(s):
    """Per unit: best |r| with action- and stimulus-kernel priors and the best action-kernel tau."""
    u = s["units"][:, task.N_UNBIASED:].reshape(-1, s["units"].shape[-1])
    out = {}
    for name, events in [("action", s["action"]), ("stimulus", s["side"])]:
        r = np.array([[np.corrcoef(task.kernel_prior(events, a)[:, task.N_UNBIASED:].ravel(), ui)[0, 1]
                       for a in ALPHAS] for ui in u.T])
        out[f"r_{name}"] = np.nanmax(np.abs(r), 1)
        out[f"tau_{name}"] = tau(ALPHAS[np.nanargmax(np.abs(r), 1)])
    return pd.DataFrame(out)


def prior_corr(s):
    """Correlation of each pre-stimulus core unit with the Bayes-optimal prior (biased trials)."""
    u = s["units"][:, task.N_UNBIASED:].reshape(-1, s["units"].shape[-1])
    b = s["bayes"][:, task.N_UNBIASED:].ravel()
    return np.array([np.corrcoef(b, ui)[0, 1] for ui in u.T])


def fig_behaviour(evals):
    """Psychometric curves per block and performance after block switches."""
    fig, axes = plt.subplots(1, len(evals) + 1, figsize=(4 * (len(evals) + 1), 3.6))
    colors = {0.2: "tab:orange", 0.5: "0.5", 0.8: "tab:blue"}
    for ax, (name, s) in zip(axes, evals.items()):
        for block, col in colors.items():
            m = s["p_right"] == block
            df = pd.DataFrame({"c": s["signed"][m] * 100, "p": s["p"][m]}).groupby("c").p.mean()
            ax.plot(df.index, df.values, "o-", color=col, label=f"p(right)={block}")
        ax.set(title=f"{name}\nbias at 0% = {block_bias(s):.2f}", xlabel="signed contrast (%)",
               xscale="symlog", xticks=[-100, -25, 0, 25, 100], xticklabels=[-100, -25, 0, 25, 100], ylim=(0, 1))
    axes[0].set_ylabel("p(right choice)")
    axes[0].legend(fontsize=8)
    ax = axes[-1]
    for name, s in evals.items():
        pr = s["p_right"]
        sw = np.argwhere((np.diff(pr, axis=1) != 0) & (pr[:, :-1] != 0.5))
        lags = np.arange(-5, 21)
        pc = s["p"] * (s["side"] > 0) + (1 - s["p"]) * (s["side"] < 0)  # p(correct)
        curve = [np.nanmean([pc[i, t + 1 + k] for i, t in sw if 0 <= t + 1 + k < pr.shape[1]
                             and s["contrast"][i, t + 1 + k] == 0]) for k in lags]
        ax.plot(lags, curve, label=name)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="trials since block switch", ylabel="p(correct), 0% contrast", title="Block switch recovery")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "01_behaviour")


def fig_prior_models(fits):
    """Held-out log-likelihood gain of the three prior models and fitted kernel tau."""
    df = pd.DataFrame([{"agent": a, "model": m, "gain": v[0], "tau": v[1]} for a, r in fits.items() for m, v in r.items()])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    sns.barplot(df, x="agent", y="gain", hue="model", ax=axes[0])
    axes[0].set(ylabel="held-out log-lik gain / trial\n(vs. no prior)", xlabel="")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(df.dropna(), x="agent", y="tau", hue="model", ax=axes[1])
    axes[1].axhline(MOUSE_TAU, color="k", ls="--", label=f"mice, action kernel ({MOUSE_TAU})")
    axes[1].set(ylabel="fitted kernel tau (trials)", xlabel="", yscale="log")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "02_prior_models")
    return df


def fig_units(evals, kfit):
    """Single units: regression on prior / last action / last reward, example traces, unit tau."""
    s = evals["A: GRU"]
    sl = np.s_[:, task.N_UNBIASED:]
    n_units = s["units"].shape[-1]
    prev_a = np.roll(s["action"], 1, 1)
    prev_r = np.roll(np.where(s["action"] == s["side"], 1.0, -1.0), 1, 1)
    reg = np.c_[s["bayes"][sl].ravel(), prev_a[sl].ravel(), prev_r[sl].ravel(), (prev_a * prev_r)[sl].ravel()]
    beta = unit_regression(s["units"][sl].reshape(-1, n_units), reg)
    rho = prior_corr(s)
    order = np.argsort(-np.abs(rho))
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 2.2, 1])
    ax = fig.add_subplot(gs[:, 0])
    sns.heatmap(beta[order], cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "standardized beta"},
                xticklabels=["Bayes prior", "a(t-1)", "r(t-1)", "side(t-1)"], yticklabels=order)
    ax.set(title="A: GRU units (pre-stimulus)", ylabel="unit (sorted by |r| with prior)")
    ax = fig.add_subplot(gs[0, 1])
    i = 0
    t = np.arange(500)
    ax.fill_between(t, 0, 1, where=s["p_right"][i] == 0.8, color="tab:blue", alpha=0.1, step="mid", label="0.8 block")
    ax.fill_between(t, 0, 1, where=s["p_right"][i] == 0.2, color="tab:orange", alpha=0.1, step="mid", label="0.2 block")
    ax.plot(t, s["bayes"][i], "k", lw=1.5, label="Bayes-optimal prior")
    for k in order[:3]:
        u = s["units"][i, :, k] * np.sign(rho[k])
        ax.plot(t, (u - u.min()) / np.ptp(u), lw=0.8, label=f"GRU unit {k} (r={rho[k]:.2f})")
    ax.set(xlabel="trial", ylabel="normalized activity", title="Example session, arm A", xlim=(0, 500))
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    ax = fig.add_subplot(gs[1, 1])
    ax.fill_between(t, 0, 1, where=s["p_right"][i] == 0.8, color="tab:blue", alpha=0.1, step="mid")
    ax.fill_between(t, 0, 1, where=s["p_right"][i] == 0.2, color="tab:orange", alpha=0.1, step="mid")
    ax.plot(t, s["bayes"][i], "k", lw=1.5)
    for name in ["B: plastic, feedback", "B: plastic, choice"]:
        sb = evals[name]
        rb = prior_corr(sb)
        k = int(np.nanargmax(np.abs(rb)))
        u = sb["units"][i, :, k] * np.sign(rb[k])
        ax.plot(t, (u - u.min()) / np.ptp(u), lw=0.8, label=f"{name}, unit {k} (r={rb[k]:.2f})")
    ax.set(xlabel="trial", ylabel="normalized plastic drive", title="Same session, arm B", xlim=(0, 500))
    ax.legend(fontsize=7, loc="lower left")
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(kfit.r_stimulus, kfit.r_action, c=np.abs(rho), cmap="viridis", vmin=0, vmax=1)
    lim = [0, 1]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set(xlabel="best |r|, stimulus kernel", ylabel="best |r|, action kernel", title="Unit prior type")
    ax = fig.add_subplot(gs[1, 2])
    sel = kfit.r_action > 0.3
    ax.hist(kfit.tau_action[sel], bins=np.geomspace(0.5, 100, 20), color="tab:green")
    ax.axvline(MOUSE_TAU, color="k", ls="--", label=f"mice ({MOUSE_TAU})")
    ax.set(xscale="log", xlabel="unit integration tau (trials)", ylabel="# units (|r| > 0.3)",
           title="Unit time constants")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "03_units")
    return beta, rho


def fig_decision_units(evals):
    """Mixed selectivity of decision-layer units: evidence, prior and choice betas per arm."""
    fig, axes = plt.subplots(1, len(evals), figsize=(3.4 * len(evals), 4.5), sharey=True)
    for ax, (name, s) in zip(axes, evals.items()):
        sl = np.s_[:, task.N_UNBIASED:]
        reg = np.c_[s["signed"][sl].ravel(), s["bayes"][sl].ravel(), s["action"][sl].ravel()]
        beta = unit_regression(s["d"][sl].reshape(-1, s["d"].shape[-1]), reg)
        sns.heatmap(beta, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, cbar=ax is axes[-1],
                    xticklabels=["contrast", "prior", "choice"])
        ax.set(title=name)
    axes[0].set_ylabel("decision unit")
    fig.tight_layout()
    savefig(fig, "04_decision_units")


def fig_ablation(rho, s_full, n_rand=5):
    """Block bias after clamping the top-k prior units of the GRU vs k random units."""
    frontend, decision, core = load("gru")
    means = torch.from_numpy(s_full["units"].reshape(-1, s_full["units"].shape[-1]).mean(0)).float()
    order = np.argsort(-np.abs(rho))
    ks = [0, 1, 2, 4, 8, 16, 32]
    rng = np.random.default_rng(0)
    rows = []
    for k in ks:
        for kind, sets in [("top prior units", [order[:k]]),
                           ("random units", [rng.choice(len(rho), k, replace=False) for _ in range(n_rand)])]:
            for idx in sets:
                idx = torch.as_tensor(idx, dtype=torch.long)
                s = evaluate(frontend, decision, core, n_sessions=100, ablate=(idx, means[idx]) if k else None)
                rows.append({"k": k, "kind": kind, "bias": block_bias(s)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.lineplot(df, x="k", y="bias", hue="kind", marker="o", ax=ax, errorbar="sd")
    ax.set(xlabel="# GRU units clamped to their mean", ylabel="block bias at 0% contrast",
           xscale="symlog", xticks=ks, xticklabels=ks, title="Ablation, arm A")
    fig.tight_layout()
    savefig(fig, "05_ablation")
    return df


def fig_emergence(rho_final):
    """Emergence of prior units and block bias over GRU training checkpoints."""
    ckpts = sorted(MODELS.joinpath("gru").glob("ckpt_*.pt"))
    top = np.argsort(-np.abs(rho_final))[:6]
    rows, traj = [], []
    for c in ckpts:
        s = evaluate(*load("gru", c), n_sessions=50)
        rho = prior_corr(s)
        rows.append({"it": int(c.stem.split("_")[1]), "bias": block_bias(s), "n_prior": int(np.sum(np.abs(rho) > 0.5))})
        traj.append(rho[top])
    df = pd.DataFrame(rows)
    traj = np.array(traj)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].plot(df.it, df.bias, "o-")
    axes[0].set(xlabel="training iteration", ylabel="block bias at 0%", title="Behaviour")
    axes[1].plot(df.it, df.n_prior, "o-", color="tab:green")
    axes[1].set(xlabel="training iteration", ylabel="# units with |r(prior)| > 0.5", title="Prior units")
    for j, k in enumerate(top):
        axes[2].plot(df.it, traj[:, j], label=f"unit {k}")
    axes[2].set(xlabel="training iteration", ylabel="r with Bayes prior", title="Final top prior units over training")
    axes[2].legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=25)
    fig.tight_layout()
    savefig(fig, "06_emergence")
    return df


def main():
    evals = {name: evaluate(*load(arm)) for name, arm in AGENTS.items()}
    fig_behaviour(evals)
    fits = {name: fit_prior_models(s) for name, s in evals.items()}
    df_fits = fig_prior_models(fits)
    core_evals = {k: v for k, v in evals.items() if v["units"] is not None}
    kfit = unit_kernel_fit(evals["A: GRU"])
    _, rho = fig_units(core_evals, kfit)
    fig_decision_units(evals)
    df_abl = fig_ablation(rho, evals["A: GRU"])
    df_em = fig_emergence(rho)
    # summary tables printed for the report
    print(pd.DataFrame({n: {"accuracy": (s["action"] == s["side"])[:, task.N_UNBIASED:].mean(),
                                "bias": block_bias(s)} for n, s in evals.items()}).T.round(3))
    print(df_fits.round(4).to_string())
    for arm in ["plastic_feedback", "plastic_choice"]:
        core = load(arm)[2]
        lam = torch.sigmoid(core.logit_lam).item()
        print(f"{arm}: eta={core.log_eta.exp().item():.3f} lambda={lam:.3f} decay tau={tau(lam):.2f}")
    print(f"GRU units: n={len(rho)}, |r(prior)|>0.5: {np.sum(np.abs(rho) > 0.5)}, "
          f"median unit tau (r_action>0.3): {kfit.tau_action[kfit.r_action > 0.3].median():.2f}, "
          f"action>stimulus kernel: {np.sum(kfit.r_action > kfit.r_stimulus)}")
    print(kfit.assign(rho=rho).round(3).to_string())
    print(df_abl.groupby(["kind", "k"]).bias.agg(["mean", "std"]).round(3).to_string())
    print(df_em.to_string())


if __name__ == "__main__":
    main()
