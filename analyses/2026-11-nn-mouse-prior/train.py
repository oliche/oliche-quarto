"""Train the networks: pre-train front-end + decision on unbiased trials, then the loose cores.

Phase 1: front-end and decision layer on single unbiased trials (no core), then frozen.
Phase 2: arm A trains a GRU core with BPTT over biased sessions; arm B meta-learns the two
plasticity hyperparameters (eta, lambda) of a plastic core for each learning rule.
Checkpoints are saved along phase 2 to follow the emergence of prior units.

Usage: python train.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import task
from models import Decision, FrontEnd, GRUCore, PlasticCore, run_sessions

OUT = Path(__file__).parent.joinpath("models")
N_TRIALS, BATCH = 500, 32
torch.set_num_threads(4)


def to_torch(sessions, rng):
    """Retinal samples and sides as tensors for a batch of sessions."""
    x = torch.from_numpy(task.evidence(sessions["side"], sessions["contrast"], rng))
    return x, torch.from_numpy(sessions["side"]).float()


def pretrain(n_iter=3000, seed=0):
    """Phase 1: train front-end and decision layer on unbiased single trials."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    frontend, decision = FrontEnd(), Decision()
    opt = torch.optim.Adam([*frontend.parameters(), *decision.parameters()], lr=3e-3)
    for i in range(n_iter):
        x, side = to_torch(task.sample_sessions(1024, 1, rng, unbiased=True), rng)
        logit, _ = decision(frontend(x[:, 0]))
        loss = F.binary_cross_entropy_with_logits(logit, (side[:, 0] > 0).float())
        opt.zero_grad(), loss.backward(), opt.step()
        if i % 500 == 0:
            print(f"pretrain {i:5d} loss {loss.item():.4f}")
    for p in [*frontend.parameters(), *decision.parameters()]:
        p.requires_grad_(False)
    return frontend, decision


def train_core(name, core, frontend, decision, n_iter, lr, seed=1, ckpt_every=25):
    """Phase 2: train the loose core on biased sessions, saving checkpoints.

    Parameters
    ----------
    name : str
        Arm name, used for the checkpoint folder.
    core : GRUCore | PlasticCore
    frontend, decision : nn.Module
        Frozen pre-trained modules.
    n_iter : int
        Number of training iterations (batches of BATCH sessions).
    lr : float
        Adam learning rate.
    seed : int
    ckpt_every : int
        Checkpoint period in iterations.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    folder = OUT.joinpath(name)
    folder.mkdir(parents=True, exist_ok=True)
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    for i in range(n_iter + 1):
        if i % ckpt_every == 0:
            torch.save(core.state_dict(), folder.joinpath(f"ckpt_{i:05d}.pt"))
        if i == n_iter:
            break
        x, side = to_torch(task.sample_sessions(BATCH, N_TRIALS, rng), rng)
        out = run_sessions(frontend, decision, core, x, side)
        loss = F.binary_cross_entropy_with_logits(out["logit"], (side > 0).float())
        opt.zero_grad(), loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if i % 25 == 0:
            acc = (out["action"] == side).float().mean().item()
            print(f"{name} {i:5d} loss {loss.item():.4f} acc {acc:.3f}")


def main():
    OUT.mkdir(exist_ok=True)
    frontend, decision = pretrain()
    torch.save({"frontend": frontend.state_dict(), "decision": decision.state_dict()},
               OUT.joinpath("pretrained.pt"))
    train_core("gru", GRUCore(), frontend, decision, n_iter=1500, lr=3e-3)
    for rule in ["feedback", "choice"]:
        train_core(f"plastic_{rule}", PlasticCore(decision, rule), frontend, decision,
                   n_iter=200, lr=3e-2, ckpt_every=10)


if __name__ == "__main__":
    main()
