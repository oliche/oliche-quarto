"""Within-trial signatures that could separate prior storage (activity vs synapses) and update driver.

For each time-resolved core, run evaluation sessions and compute BWM-style analyses on the
recorded units (GRU: core + decision units; synapses: decision units, the only activity):

1. decodability of the prior at every within-trial step, and cross-temporal generalisation;
2. update timing: at which step the activity switches from the current to the next prior;
3. update direction after errors: toward the chosen side (choice) or the correct side (feedback);
4. offset vs gain: prior x contrast interaction in decision units at the decision step;
5. causal test: silence all activity during one ITI and measure how much prior survives.

Usage: MPLBACKEND=Agg python analysis_time.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import Ridge

import task
from analysis import block_bias, fit_prior_models, savefig, unit_regression
from models_time import DECIDE, PHASES, run_timeline
from train_time import NAMES, OUT, load_pretrained, make_core

N_SESSIONS, N_TRIALS, SEED = 100, 500, 123
STEP_LABELS = [f"ITI{i}" if p == "iti" else p for i, p in enumerate(PHASES)]
PALETTE = dict(zip(NAMES, sns.color_palette("Paired", 6)[::-1], strict=True))


def evaluate(name, silence=None):
    """Run the fixed evaluation sessions for one core; return numpy arrays."""
    frontend, decision = load_pretrained()
    core = make_core(name, decision)
    core.load_state_dict(torch.load(OUT.joinpath(f"{name}.pt")))
    rng = np.random.default_rng(SEED)
    s = task.sample_sessions(N_SESSIONS, N_TRIALS, rng)
    x = torch.from_numpy(task.evidence(s["side"], s["contrast"], rng))
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = run_timeline(frontend, decision, core, x, torch.from_numpy(s["side"]).float(),
                           silence=silence, generator=torch.Generator().manual_seed(SEED), record=True)
    units = out["d"] if out["h"] is None else torch.cat([out["h"], out["d"]], -1)
    s.update(p=torch.sigmoid(out["logit"]).numpy(), action=out["action"].numpy(),
             prior=out["prior"].numpy(), units=units.numpy())
    s["bayes"] = task.bayes_optimal_prior(s["side"])
    s["signed"] = s["side"] * s["contrast"]
    return s


def _xy(s, k, target, trials):
    """Units at step k and a per-trial target, for train (first half) and test sessions."""
    half = N_SESSIONS // 2
    u, y = s["units"][:, trials, k], target[:, trials]
    return [(u[sl].reshape(-1, u.shape[-1]), y[sl].ravel()) for sl in (np.s_[:half], np.s_[half:])]


def decoding(s):
    """Cross-validated R^2 of the current prior, trained at step k (rows), tested at step k' (cols)."""
    trials = np.arange(task.N_UNBIASED, N_TRIALS - 1)
    n_steps = len(PHASES)
    r2 = np.zeros((n_steps, n_steps))
    for k in range(n_steps):
        (xtr, ytr), _ = _xy(s, k, s["prior"], trials)
        model = Ridge(alpha=1.0).fit(xtr, ytr)
        for kk in range(n_steps):
            _, (xte, yte) = _xy(s, kk, s["prior"], trials)
            r2[k, kk] = 1 - np.mean((model.predict(xte) - yte) ** 2) / np.var(yte)
    return r2


def prior_axis(s):
    """Ridge decoder of the prior from the last ITI step, free of stimulus and choice signals."""
    trials = np.arange(task.N_UNBIASED, N_TRIALS - 1)
    k_iti = PHASES.index("stim") - 1
    (xtr, ytr), _ = _xy(s, k_iti, s["prior"], trials)
    return Ridge(alpha=1.0).fit(xtr, ytr), k_iti


def update_timing(s):
    """Fraction of the trial-t update expressed by the activity at each step of trial t.

    The activity at each step is projected on the prior axis; the projection is regressed on
    the current prior pi_t and the update pi_{t+1} - pi_t. The ratio of the two coefficients
    is ~0 before the update and ~1 once the activity carries the next prior.
    """
    model, _ = prior_axis(s)
    trials = np.arange(task.N_UNBIASED, N_TRIALS - 1)
    pi, dpi = s["prior"][:, trials], s["prior"][:, trials + 1] - s["prior"][:, trials]
    frac = []
    for k in range(len(PHASES)):
        proj = model.predict(s["units"][:, trials, k].reshape(-1, s["units"].shape[-1]))
        beta = np.linalg.lstsq(np.c_[pi.ravel(), dpi.ravel(), np.ones(pi.size)], proj, rcond=None)[0]
        frac.append(beta[1] / beta[0])
    return np.array(frac)


def update_step(s):
    """First step from which the activity carries the next prior (update fraction > 0.5 onward).

    Activity at a step reflects inputs up to the previous step (one-step, ~100 ms lag).
    "next ITI" if the update only shows after the trial ends; "n/a" if the prior is not
    decodable during the ITI, so the timing cannot be measured with an ITI-anchored axis.
    """
    if np.diag(s["r2"])[: PHASES.index("stim")].max() < 0.2:
        return "n/a"
    above = s["update"] > 0.5
    for k in range(len(PHASES)):
        if above[k:].all():
            return STEP_LABELS[k]
    return "next ITI"


def error_direction(s):
    """Update of the prior after correct and error trials, signed by the chosen side.

    Positive after errors means the prior moved toward the chosen (wrong) side, as an action
    kernel; negative means toward the correct side, as a stimulus kernel. Computed on the
    network's used prior ("behaviour") and on the prior decoded from the last ITI step
    ("neural", the BWM-applicable version).
    """
    model, _ = prior_axis(s)
    trials = np.arange(task.N_UNBIASED, N_TRIALS - 1)
    k_iti = PHASES.index("stim") - 1
    decoded = model.predict(s["units"][:, :, k_iti].reshape(-1, s["units"].shape[-1])).reshape(N_SESSIONS, N_TRIALS)
    a, correct = s["action"][:, trials], s["action"][:, trials] == s["side"][:, trials]
    res = {}
    for label, prior in [("behaviour", s["prior"]), ("neural", decoded)]:
        dpi = (prior[:, trials + 1] - prior[:, trials]) * a
        scale = np.abs(dpi[correct]).mean()
        res[f"{label} correct"] = dpi[correct].mean() / scale
        res[f"{label} error"] = dpi[~correct].mean() / scale
    return res


def gain_vs_offset(s):
    """Median over decision units of |beta(prior x contrast)| / |beta(prior)| at the decision step."""
    trials = np.s_[:, task.N_UNBIASED:]
    d = s["units"][trials][..., DECIDE, -16:].reshape(-1, 16)
    pi, c = s["prior"][trials].ravel(), s["signed"][trials].ravel()
    beta = unit_regression(d, np.c_[pi, c, pi * c])
    return np.median(np.abs(beta[:, 2]) / np.abs(beta[:, 0]))


def silencing(name, s_ctrl):
    """Prior surviving an ITI silencing: slope of the perturbed on the control prior.

    Every 10th biased trial has its ITI silenced; the slope is measured on that trial's
    choice (lag 0) and on the following trials.
    """
    mask = np.zeros(N_TRIALS, bool)
    mask[task.N_UNBIASED + 5::10] = True
    s_pert = evaluate(name, silence=torch.from_numpy(mask))
    out = {}
    for lag in [0, 1, 3]:
        idx = np.flatnonzero(mask) + lag
        ctrl, pert = s_ctrl["prior"][:, idx].ravel(), s_pert["prior"][:, idx].ravel()
        out[lag] = np.dot(ctrl, pert) / np.dot(ctrl, ctrl)
    return out


def fig_timecourse(results):
    """Prior decodability and update fraction along the trial, for every core."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, r in results.items():
        axes[0].plot(np.diag(r["r2"]), "o-", color=PALETTE[name], label=name)
        axes[1].plot(r["update"], "o-", color=PALETTE[name], label=name)
    axes[0].set(ylabel="R² decoding current prior", ylim=(-0.05, 1.02), title="Is the prior in the activity?")
    axes[1].set(ylabel="update fraction", ylim=(-0.3, 1.5), title="When does the activity update?")
    for ax in axes:
        ax.set_xticks(range(len(PHASES)), STEP_LABELS, rotation=45)
        ax.axvspan(PHASES.index("stim") - 0.5, DECIDE + 0.5, color="0.9", zorder=0)
    axes[1].axhline(0, color="0.5", lw=0.8), axes[1].axhline(1, color="0.5", lw=0.8)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    savefig(fig, "07_time_decoding")


def fig_crosstemporal(results):
    """Cross-temporal decoding matrices of the current prior (train step x test step)."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (name, r) in zip(axes.T.ravel(), results.items(), strict=True):
        sns.heatmap(np.clip(r["r2"], 0, 1), vmin=0, vmax=1, cmap="viridis", ax=ax, cbar=False, square=True,
                    xticklabels=STEP_LABELS, yticklabels=STEP_LABELS)
        ax.set(title=name, xlabel="test step", ylabel="train step")
    fig.tight_layout()
    savefig(fig, "08_crosstemporal")


def fig_causal(table):
    """Error-trial update direction (neural) and prior surviving ITI silencing."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = [PALETTE[n] for n in table.index]
    axes[0].bar(table.index, table["neural error"], color=colors)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set(ylabel="update after error\n(+ toward choice, − toward correct)", title="What drives the update?")
    axes[1].bar(table.index, table["silenced lag0"], color=colors)
    axes[1].set(ylabel="prior surviving ITI silencing", ylim=(0, 1.1), title="Where is the prior stored?")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    savefig(fig, "09_drive_storage")


def main():
    results, rows = {}, {}
    for name in NAMES:
        s = evaluate(name)
        s["r2"] = decoding(s)
        s["update"] = update_timing(s)
        fits = fit_prior_models(s)
        sil = silencing(name, s)
        rows[name] = {"bias": block_bias(s), "best kernel": max(fits, key=lambda m: fits[m][0]),
                      "action tau": fits["action kernel"][1],
                      "R2 ITI": s["r2"][3, 3], "R2 stim": s["r2"][DECIDE, DECIDE],
                      "update step": update_step(s),
                      **error_direction(s), "gain/offset": gain_vs_offset(s),
                      **{f"silenced lag{k}": v for k, v in sil.items()}}
        results[name] = {"r2": s["r2"], "update": s["update"]}
        print(name, rows[name], flush=True)
    table = pd.DataFrame(rows).T
    table.to_csv(OUT.joinpath("signatures.csv"))
    with pd.option_context("display.width", 250, "display.float_format", "{:.2f}".format):
        print(table)
    fig_timecourse(results)
    fig_crosstemporal(results)
    fig_causal(table)


if __name__ == "__main__":
    main()
