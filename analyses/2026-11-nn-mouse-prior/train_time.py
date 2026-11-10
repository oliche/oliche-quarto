"""Train the time-resolved model family on top of the frozen pre-trained front-end.

Six cores: storage {GRU activity, tonic synapses, gated synapses} x driver {choice, feedback}.
GRUs are trained with BPTT over all within-trial steps; synaptic cores meta-learn eta, lambda.

Usage: python train_time.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import task
from models import Decision, FrontEnd
from models_time import GRUCoreT, SynapticCoreT, run_timeline

MODELS = Path(__file__).parent.joinpath("models")
OUT = MODELS.joinpath("time")
N_TRIALS, BATCH = 300, 32
torch.set_num_threads(4)


def load_pretrained():
    """Frozen front-end and decision layer from phase 1 (``train.py``)."""
    pre = torch.load(MODELS.joinpath("pretrained.pt"))
    frontend, decision = FrontEnd(), Decision()
    frontend.load_state_dict(pre["frontend"]), decision.load_state_dict(pre["decision"])
    for p in [*frontend.parameters(), *decision.parameters()]:
        p.requires_grad_(False)
    return frontend, decision


def make_core(name, decision):
    """Build a core from its name, e.g. ``gru_choice`` or ``syn-gated_feedback``."""
    storage, driver = name.split("_")
    if storage == "gru":
        return GRUCoreT(driver)
    return SynapticCoreT(decision, driver, presyn=storage.split("-")[1])


NAMES = [f"{s}_{d}" for s in ["gru", "syn-tonic", "syn-gated"] for d in ["choice", "feedback"]]


def train(name, frontend, decision, n_iter, lr, seed=1):
    """Train one core on biased sessions and save its final state.

    Parameters
    ----------
    name : str
        Core name, see ``NAMES``.
    frontend, decision : nn.Module
    n_iter : int
        Iterations of BATCH sessions of N_TRIALS trials.
    lr : float
        Adam learning rate.
    seed : int
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    core = make_core(name, decision)
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    t0 = time.time()
    for i in range(n_iter):
        s = task.sample_sessions(BATCH, N_TRIALS, rng)
        x = torch.from_numpy(task.evidence(s["side"], s["contrast"], rng))
        side = torch.from_numpy(s["side"]).float()
        out = run_timeline(frontend, decision, core, x, side)
        loss = F.binary_cross_entropy_with_logits(out["logit"], (side > 0).float())
        opt.zero_grad(), loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if i % 25 == 0:
            acc = (out["action"] == side).float().mean().item()
            print(f"{name:20s} {i:5d} loss {loss.item():.4f} acc {acc:.3f} {time.time() - t0:6.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(core.state_dict(), OUT.joinpath(f"{name}.pt"))


def main():
    frontend, decision = load_pretrained()
    for name in NAMES:
        if name.startswith("gru"):
            train(name, frontend, decision, n_iter=1500, lr=3e-3)
        else:
            train(name, frontend, decision, n_iter=150, lr=3e-2)


if __name__ == "__main__":
    main()
