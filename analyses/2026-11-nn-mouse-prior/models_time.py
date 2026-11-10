"""Time-resolved versions of the networks: each trial unrolls into within-trial time steps.

One step is ~100 ms. A trial is ``PHASES``: inter-trial interval (ITI, blank screen), two
stimulus steps (the choice is read out at the end of the second one), a movement step and a
feedback step. The frozen front-end and decision layer from ``models.py`` run at every step:
``d_k = tanh(W z_k + u_k) + noise``, with ``z_k`` the evidence code (blank outside stimulus)
and ``u_k`` the drive from the core.

The model family crosses two factors of the hypothesis:

- storage: in recurrent activity (``GRUCoreT``), or in plastic synapses (``SynapticCoreT``)
  whose presynaptic context population is either always active ("tonic") or active only
  during the stimulus ("gated", an activity-silent prior);
- update driver: the own choice ("choice") or the correct side revealed at feedback
  ("feedback").
"""

import torch
from torch import nn

from models import N_D, N_Z

PHASES = ["iti"] * 4 + ["stim", "stim", "move", "fb"]
DECIDE = 5  # step index at which the choice is read out
SIGMA_H, SIGMA_D = 0.05, 0.05  # activity noise per step (core units, decision units)


class GRUCoreT(nn.Module):
    """Recurrent core stepping every ~100 ms; the prior is held in its activity ``h``.

    Inputs per step: evidence code z, plus the own choice on the movement step
    (driver="choice") or the correct side on the feedback step (driver="feedback").
    """

    def __init__(self, driver="choice", n_hidden=32):
        super().__init__()
        self.driver, self.n_hidden = driver, n_hidden
        self.cell = nn.GRUCell(N_Z + 1, n_hidden)
        self.w_u = nn.Linear(n_hidden, N_D)

    def init(self, n):
        return torch.zeros(n, self.n_hidden)

    def drive(self, h, phase):
        return self.w_u(h)

    def step(self, h, phase, z, action, side, **_):
        signal = {"choice": ("move", action), "feedback": ("fb", side)}[self.driver]
        inp = signal[1] if phase == signal[0] else torch.zeros_like(action)
        h = self.cell(torch.cat([z, inp[:, None]], -1), h)
        return h + SIGMA_H * torch.randn_like(h)

    def silence(self, h):
        """Optogenetic-like silencing of the core activity."""
        return torch.zeros_like(h)

    def units(self, h):
        return h


class SynapticCoreT(nn.Module):
    """Plastic synapses P from a context population g onto decision units; the prior is in P.

    ``u_k = P (g * gate_k)``: gate is 1 at every step ("tonic") or only on stimulus steps
    ("gated"). P decays by (1 - lam) per step and is updated once per trial, through an
    eligibility trace of the decision-step activity, with ``e = w_o * (1 - d**2)``:

    - driver="choice": on the movement step, ``P += eta * a e g^T`` (Hebbian, own choice a);
    - driver="feedback": on the feedback step, ``P += eta * (y - p) e g^T`` (prediction error).

    Only ``eta`` and ``lam`` are meta-learned.
    """

    def __init__(self, decision, driver="choice", presyn="tonic", n_ctx=8):
        super().__init__()
        self.driver, self.presyn, self.decision = driver, presyn, decision
        self.register_buffer("g", torch.rand(n_ctx) / n_ctx ** 0.5)
        self.log_eta = nn.Parameter(torch.tensor(-1.0))
        self.logit_lam = nn.Parameter(torch.tensor(-4.0))

    def init(self, n):
        return torch.zeros(n, N_D, self.g.numel())

    def drive(self, p, phase):
        gate = 1.0 if self.presyn == "tonic" or phase == "stim" else 0.0
        return p @ (self.g * gate)

    def step(self, p, phase, action, side, logit, d_dec, **_):
        p = (1 - torch.sigmoid(self.logit_lam)) * p
        if self.driver == "choice" and phase == "move":
            post = action
        elif self.driver == "feedback" and phase == "fb":
            post = (side > 0).float() - torch.sigmoid(logit)
        else:
            return p
        e = self.decision.out.weight.squeeze(0) * (1 - d_dec ** 2)
        return p + self.log_eta.exp() * (post[:, None] * e)[:, :, None] * self.g

    def silence(self, p):
        """Silencing activity leaves the synaptic weights untouched."""
        return p

    def units(self, p):
        return None


def run_timeline(frontend, decision, core, x, side, silence=None, generator=None, record=False):
    """Run sessions step by step within trials, sampling choices at the decision step.

    Parameters
    ----------
    frontend, decision : nn.Module
        Frozen pre-trained modules.
    core : GRUCoreT | SynapticCoreT
    x : torch.Tensor
        (B, T, N_IN) retinal samples, shown on both stimulus steps.
    side : torch.Tensor
        (B, T) true stimulus sides (+-1).
    silence : torch.Tensor, optional
        (T,) bool: trials whose ITI activity (core and decision units) is silenced.
    generator : torch.Generator, optional
        Random generator for choice sampling.
    record : bool
        Return the activity at every step.

    Returns
    -------
    dict
        ``logit``, ``prior`` (B, T): decision-step logit and the core's contribution to it
        (noise-free, the prior used by the network); ``action`` (B, T) +-1; if ``record``,
        ``d`` (B, T, K, N_D) decision units and ``h`` (B, T, K, n) core units or None.
    """
    n, n_trials = side.shape
    z_stim = frontend(x)
    z_blank = frontend(torch.zeros_like(x[:, 0]))
    state = core.init(n)
    w_o = decision.out.weight.squeeze(0)
    out = {k: [] for k in ["logit", "prior", "action", "d", "h"]}
    for t in range(n_trials):
        action = torch.zeros(n)
        ds, hs = [], []
        for k, phase in enumerate(PHASES):
            z = z_stim[:, t] if phase == "stim" else z_blank
            u = core.drive(state, phase)
            drive = decision.w(z)
            d = torch.tanh(drive + u) + SIGMA_D * torch.randn(n, N_D, generator=generator)
            if silence is not None and silence[t] and phase == "iti":
                state, d = core.silence(state), torch.zeros_like(d)
            if k == DECIDE:
                logit, d_dec = decision.out(d).squeeze(-1), d
                prior = (torch.tanh(drive + u) - torch.tanh(drive)) @ w_o
                p = torch.sigmoid(logit.detach())
                action = torch.where(torch.rand(n, generator=generator) < p, 1.0, -1.0)
            if record:
                ds.append(d.detach())
                hs.append(core.units(state).detach() if core.units(state) is not None else None)
            state = core.step(state, phase, z=z, action=action, side=side[:, t],
                              logit=logit if k >= DECIDE else None, d_dec=d_dec if k >= DECIDE else None)
        out["logit"].append(logit), out["prior"].append(prior), out["action"].append(action)
        if record:
            out["d"].append(torch.stack(ds, 1))
            out["h"].append(torch.stack(hs, 1) if hs[0] is not None else None)
    res = {k: torch.stack(out[k], 1) for k in ["logit", "prior", "action"]}
    if record:
        res["d"] = torch.stack(out["d"], 1)
        res["h"] = torch.stack(out["h"], 1) if out["h"][0] is not None else None
    return res
