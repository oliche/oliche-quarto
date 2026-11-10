"""Small networks solving the IBL task: a frozen pre-trained front-end plus a loose core.

The decision layer computes ``d = tanh(W z + b + u)`` and ``logit = w_o . d``, where ``z``
is the front-end evidence code and ``u`` the drive from the core (zero during pre-training).
Two cores hold the block prior in different places:

- ``GRUCore`` (arm A): in recurrent activity, updated after each trial outcome.
- ``PlasticCore`` (arm B): in plastic synapses onto the decision units, updated online
  after each trial by a delta rule, toward the true side ("feedback") or own choice ("choice").
"""

import torch
from torch import nn

N_IN, N_Z, N_D = 16, 16, 16


class FrontEnd(nn.Module):
    """MLP from noisy retinal samples to an evidence code."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_IN, 32), nn.ReLU(), nn.Linear(32, N_Z), nn.Tanh())

    def forward(self, x):
        return self.net(x)


class Decision(nn.Module):
    """Decision layer and choice readout; the core adds its drive ``u`` before the tanh."""

    def __init__(self):
        super().__init__()
        self.w = nn.Linear(N_Z, N_D)
        self.out = nn.Linear(N_D, 1)

    def forward(self, z, u=0.0):
        d = torch.tanh(self.w(z) + u)
        return self.out(d).squeeze(-1), d


class GRUCore(nn.Module):
    """Recurrent core: the state after trial t-1 sets the drive on trial t."""

    def __init__(self, n_hidden=32):
        super().__init__()
        self.n_hidden = n_hidden
        self.cell = nn.GRUCell(N_Z + 2, n_hidden)
        self.w_u = nn.Linear(n_hidden, N_D)

    def init(self, n):
        return torch.zeros(n, self.n_hidden)

    def drive(self, h):
        return self.w_u(h)

    def update(self, h, z, action, reward, **_):
        return self.cell(torch.cat([z, action[:, None], reward[:, None]], -1), h)

    def units(self, h):
        """Pre-stimulus activity of the core units, (B, n_units)."""
        return h


class PlasticCore(nn.Module):
    """Plastic synapses P (N_D x n_ctx) from a fixed context population g onto decision units.

    After each trial, ``P <- (1 - lam) P + eta * delta g^T`` where, with ``e = w_o * (1 - d**2)``
    the sensitivity of the logit to each decision-unit input:

    - rule="feedback": ``delta = (y - p_right) e`` with y the true side, i.e. online SGD of
      this layer on the task loss after feedback (prediction-error rule);
    - rule="choice": ``delta = a e`` with a = +-1 the network's own choice, a Hebbian rule
      that implements an action kernel. (A prediction-error rule toward the own sampled
      choice has zero expected update and cannot build a prior.)

    Only ``eta`` and ``lam`` are meta-learned.
    """

    def __init__(self, decision, rule="feedback", n_ctx=8):
        super().__init__()
        self.rule, self.decision = rule, decision
        self.register_buffer("g", torch.rand(n_ctx) / n_ctx ** 0.5)
        self.log_eta = nn.Parameter(torch.tensor(-1.0))
        self.logit_lam = nn.Parameter(torch.tensor(-2.0))

    def init(self, n):
        return torch.zeros(n, N_D, self.g.numel())

    def drive(self, p):
        return p @ self.g

    def update(self, p, z, action, reward, logit, d, side, **_):
        e = self.decision.out.weight.squeeze(0) * (1 - d ** 2)
        if self.rule == "feedback":
            delta = ((side > 0).float() - torch.sigmoid(logit))[:, None] * e
        else:
            delta = action[:, None] * e
        eta, lam = self.log_eta.exp(), torch.sigmoid(self.logit_lam)
        return (1 - lam) * p + eta * delta[:, :, None] * self.g

    def units(self, p):
        """Plastic drive onto each decision unit, (B, N_D)."""
        return self.drive(p)


def run_sessions(frontend, decision, core, x, side, ablate=None, generator=None):
    """Run a batch of sessions trial by trial, sampling choices from the policy.

    Parameters
    ----------
    frontend, decision : nn.Module
    core : GRUCore | PlasticCore | None
        None runs the pre-trained network without a core (u = 0).
    x : torch.Tensor
        (B, T, N_IN) retinal samples.
    side : torch.Tensor
        (B, T) true stimulus sides (+-1), used for feedback.
    ablate : tuple of (torch.Tensor, torch.Tensor), optional
        (unit indices, clamp values): clamp these GRU units after every update.
    generator : torch.Generator, optional
        Random generator for choice sampling.

    Returns
    -------
    dict
        ``logit`` (B, T), ``action`` (B, T) +-1, ``units`` (B, T, n) pre-stimulus core units,
        ``d`` (B, T, N_D) decision-layer activity.
    """
    n, n_trials = side.shape
    z_all = frontend(x)
    state = core.init(n) if core is not None else None
    logits, actions, units, ds = [], [], [], []
    for t in range(n_trials):
        u = core.drive(state) if core is not None else 0.0
        if core is not None:
            units.append(core.units(state))
        logit, d = decision(z_all[:, t], u)
        p = torch.sigmoid(logit.detach())
        action = torch.where(torch.rand(n, generator=generator) < p, 1.0, -1.0)
        reward = torch.where(action == side[:, t], 1.0, -1.0)
        if core is not None:
            state = core.update(state, z_all[:, t], action, reward, logit=logit, d=d, side=side[:, t])
            if ablate is not None:
                state = state.clone()
                state[:, ablate[0]] = ablate[1]
        logits.append(logit), actions.append(action), ds.append(d)
    return {"logit": torch.stack(logits, 1), "action": torch.stack(actions, 1),
                "units": torch.stack(units, 1) if units else None, "d": torch.stack(ds, 1)}
