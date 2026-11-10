"""IBL biased choice world: session generator, sensory evidence and reference priors.

The generative process follows Findling et al. (prior paper, Methods): 90 unbiased
trials, then blocks with p(right) alternating between 0.2 and 0.8, with block lengths
drawn from an exponential distribution (tau = 60) truncated to [20, 100] trials.
"""

import numpy as np

CONTRASTS = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])
N_UNBIASED = 90
TAU_BLOCK, LMIN, LMAX = 60, 20, 100
SIGMA, N_SAMPLES = 0.28, 8  # per-sample sensory noise, samples per side and trial


def block_length_pmf():
    """Truncated exponential pmf of block lengths over 1..LMAX.

    Returns
    -------
    np.ndarray
        (LMAX,) probabilities for lengths 1..LMAX.
    """
    lengths = np.arange(1, LMAX + 1)
    p = np.exp(-lengths / TAU_BLOCK) * (lengths >= LMIN)
    return p / p.sum()


def sample_sessions(n_sessions, n_trials, rng, unbiased=False):
    """Sample block structure, stimulus sides and contrasts.

    Parameters
    ----------
    n_sessions, n_trials : int
        Batch size and session length.
    rng : np.random.Generator
    unbiased : bool
        If True, p(right) = 0.5 on every trial (pre-training sessions).

    Returns
    -------
    dict
        ``p_right``, ``side`` (+1 right / -1 left) and ``contrast``, each (n_sessions, n_trials).
    """
    p_right = np.full((n_sessions, n_trials), 0.5)
    if not unbiased:
        pmf, lengths = block_length_pmf(), np.arange(1, LMAX + 1)
        for i in range(n_sessions):
            t, pr = N_UNBIASED, rng.choice([0.2, 0.8])
            while t < n_trials:
                n = rng.choice(lengths, p=pmf)
                p_right[i, t:t + n] = pr
                t, pr = t + n, 1 - pr
    side = np.where(rng.random(p_right.shape) < p_right, 1, -1)
    contrast = rng.choice(CONTRASTS, p_right.shape)
    return {'p_right': p_right, 'side': side, 'contrast': contrast}


def evidence(side, contrast, rng):
    """Noisy left/right retinal samples for each trial.

    Parameters
    ----------
    side, contrast : np.ndarray
        (B, T) stimulus side (+-1) and contrast.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        (B, T, 2 * N_SAMPLES) float32: N_SAMPLES left-channel then N_SAMPLES right-channel samples.
    """
    mean_l = contrast * (side < 0)
    mean_r = contrast * (side > 0)
    mean = np.concatenate([np.repeat(mean_l[..., None], N_SAMPLES, -1),
                           np.repeat(mean_r[..., None], N_SAMPLES, -1)], axis=-1)
    return (mean + SIGMA * rng.standard_normal(mean.shape)).astype(np.float32)


def bayes_optimal_prior(side):
    """Bayes-optimal p(right) before each trial, given past stimulus sides.

    Forward filtering of a hidden semi-Markov model over (block type, run length),
    assuming knowledge of the generative process. The prior is 0.5 during the
    unbiased segment and inference starts at trial N_UNBIASED with a flat block belief.

    Parameters
    ----------
    side : np.ndarray
        (B, T) stimulus sides (+-1).

    Returns
    -------
    np.ndarray
        (B, T) prior probability of a right stimulus.
    """
    n_sessions, n_trials = side.shape
    pmf = block_length_pmf()
    survival = np.cumsum(pmf[::-1])[::-1]
    hazard = np.divide(pmf, survival, out=np.ones_like(pmf), where=survival > 0)
    p_block = np.array([0.2, 0.8])
    prior = np.full(side.shape, 0.5)
    belief = np.zeros((n_sessions, 2, LMAX))  # belief[b, l-1]: block type b, run length l
    belief[:, :, 0] = 0.5
    for t in range(N_UNBIASED, n_trials):
        prior[:, t] = np.einsum('bkl,k->b', belief, p_block)
        like = np.where(side[:, t, None] > 0, p_block, 1 - p_block)  # (B, 2)
        post = belief * like[:, :, None]
        post /= post.sum(axis=(1, 2), keepdims=True)
        switch = (post * hazard).sum(-1)  # (B, 2) mass leaving each block type
        belief = np.zeros_like(post)
        belief[:, :, 1:] = post[:, :, :-1] * (1 - hazard[:-1])
        belief[:, :, 0] = switch[:, ::-1]
    return prior


def kernel_prior(x, alpha):
    """Exponentially weighted average of past events (stimulus or action kernel prior).

    Parameters
    ----------
    x : np.ndarray
        (B, T) past events coded +-1 (stimulus sides or actions).
    alpha : float
        Learning rate; the integration time constant is -1 / log(1 - alpha) trials.

    Returns
    -------
    np.ndarray
        (B, T) prior p(right) before each trial, starting at 0.5.
    """
    prior = np.full(x.shape, 0.5)
    for t in range(1, x.shape[1]):
        prior[:, t] = (1 - alpha) * prior[:, t - 1] + alpha * (x[:, t - 1] > 0)
    return prior
