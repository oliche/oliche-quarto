"""
SVD + adaptive wavelet-packet compression prototype.

Math
----
Stage 1 — SVD truncation at rank r (spatial compression):

    X ≈ Uᵣ · diag(σᵣ) · Vᵣᴴ,    r ≈ nc·ns / (CR_svd · (nc+ns))

Noise floor from the SVD spectrum:

    σ_noise = median{ σᵢ : i ≥ nc//2 }     (median of lower half of SVs)

Stage 2 — Adaptive WP thresholding of temporal components:

Each row Vₖ of Vᵣ is unit-L2-norm; its contribution to X is proportional
to σₖ.  A wavelet coefficient wₖⱼ of Vₖ contributes σₖ · |wₖⱼ| to the
total Frobenius reconstruction error when discarded.  To keep that
contribution below α · σ_noise for every discarded coefficient:

    τₖ = α · σ_noise / σₖ         (per-component hard threshold)

Properties:
  - τₖ ∝ 1/σₖ: dominant components (large σₖ) get a small threshold →
    many wavelet coefficients retained (fine temporal detail preserved).
    Weak components (small σₖ) get a large threshold → only the dominant
    wavelet structure is kept.
  - α is the single global aggressiveness knob.
  - We sweep α to find the value where the WP stage achieves CR_wp ≈ 4.

CR accounting:
    n_wp_slots = total wavelet-packet coefficients per temporal component
    CR_wp      = (r · n_wp_slots) / n_kept_avg
    CR_total   = CR_svd × CR_wp
"""
# %%
from pathlib import Path

import addcopyfighandler  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pywt
import seaborn as sns

sns.set_theme(context='notebook')

ROOT_OUTPUT = Path('/Users/olivier/scratch/lfp')
FIGURE_DIR = Path.home().joinpath('Documents', 'figures')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TODAY = '2026-06-12'

pids = [
    '1a276285-8b0e-4cc9-9f0a-a3a002978724',
    '1e104bf4-7a24-4624-a5b2-c2c8289c0de7',
    '6638cfb3-3831-4fc2-9327-194b76cf22e1',
    '749cb2b7-e57e-4453-a794-f6230e4d0226',
    'd7ec0892-0a6c-4f4f-9d8f-72083692af5c',
    'da8dfec1-d265-44e8-84ce-6ae9c109b8bd',
    'dab512bd-a02d-4c1f-8dbc-9155a163efc0',
    'dc7e9403-19f7-409f-9240-05ee57cb7aea',
    'e8f9fba4-d151-4b00-bee7-447f0f3e752c',
    'eebcaf65-7fa4-4118-869d-a084e84530e2',
    'fe380793-8035-414e-b000-09bfe5ece92a',
]

WP_MAXLEVEL = 5
WAVELETS = ['db4', 'sym8']      # families to compare
# SVD-stage CRs to sweep alongside α
INITIAL_CRS_SVD = [12, 16, 24, 32, 48]
# α sweep: small → keep everything, large → zero everything
ALPHAS = np.logspace(0, 3, 80)
TARGET_WP_CR = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svd_noise_floor(sv):
    """Median of the lower half of non-trivial singular values."""
    sv_nz = sv[sv > sv[0] * 1e-4]
    tail = sv_nz[sv_nz.size // 2:] if sv_nz.size else sv
    return float(np.nanmedian(tail)) if tail.size else float(sv[0])


def _count_wp_slots(ns, wavelet, maxlevel=WP_MAXLEVEL):
    """Total number of leaf WP coefficients for a signal of length ns."""
    wp = pywt.WaveletPacket(data=np.zeros(ns), wavelet=wavelet, maxlevel=maxlevel)
    return sum(len(node.data) for node in wp.get_level(maxlevel, 'natural'))


# ---------------------------------------------------------------------------
# Core sweep function
# ---------------------------------------------------------------------------

def sweep_alpha(snippets, initial_cr, wavelet, alphas=ALPHAS):
    """
    Sweep the global WP threshold multiplier α for one (snippets, initial_cr, wavelet) triple.

    For each α the adaptive threshold τₖ = α · σ_noise / σₖ is applied to every
    temporal component; RMSE, SNR, CR_wp, and CR_total are recorded.

    Parameters
    ----------
    snippets : list of ndarray, each (nc, ns), float32
    initial_cr : float
        Target SVD-stage compression ratio; selects rank r.
    wavelet : str
        PyWavelets wavelet name (e.g. 'db4', 'sym8').
    alphas : array-like

    Returns
    -------
    cr_wp : ndarray (n_alpha,)
        Wavelet-packet stage CR (r · n_wp_slots / n_kept_avg).
    cr_total : ndarray (n_alpha,)
        Combined SVD × WP CR.
    rmse_med : ndarray (n_alpha,)
        Median-over-channels RMSE, averaged across snippets [same units as snippets].
    snr_med : ndarray (n_alpha,)
        20·log10(RMS / RMSE) [dB].
    stage1_cr : float
        Actual SVD-stage CR (may differ slightly from initial_cr due to rounding r).
    r_fixed : int
        SVD rank used.
    """
    nc, ns = snippets[0].shape
    r_fixed = max(1, round(nc * ns / (initial_cr * (nc + ns))))
    stage1_cr = (nc * ns) / (r_fixed * (nc + ns))

    svds = [np.linalg.svd(s.astype(np.float64), full_matrices=False) for s in snippets]
    noise_floors = [_svd_noise_floor(sv) for _, sv, _ in svds]
    n_wp_slots = _count_wp_slots(ns, wavelet)   # total WP coef slots per temporal component
    n_wp_total_per_snip = r_fixed * n_wp_slots   # for CR_wp denominator

    # Pre-build WP trees once per (snippet, component); re-apply threshold for each α.
    wp_cache = []
    for s_idx, ((U, sv, Vh), sigma_noise) in enumerate(zip(svds, noise_floors)):
        snip_cache = []
        for k in range(r_fixed):
            wp = pywt.WaveletPacket(data=Vh[k], wavelet=wavelet, maxlevel=WP_MAXLEVEL)
            nodes = wp.get_level(WP_MAXLEVEL, 'natural')
            orig_data = [node.data.copy() for node in nodes]
            snip_cache.append((wp, nodes, orig_data, sv[k], sigma_noise, U[:, :r_fixed], sv[:r_fixed]))
        wp_cache.append(snip_cache)

    n_alpha = len(alphas)
    cr_wp = np.zeros(n_alpha)
    cr_total = np.zeros(n_alpha)
    rmse_med = np.zeros(n_alpha)
    snr_med = np.zeros(n_alpha)

    # RMS of original snippets (per-channel, averaged across snippets)
    rms_per_channel = np.mean(
        [np.sqrt(np.mean(s.astype(np.float64) ** 2, axis=-1)) for s in snippets], axis=0
    )  # (nc,)

    for a_idx, alpha in enumerate(alphas):
        n_kept_total = 0
        rmse_sum = np.zeros(nc)

        for s_idx, ((U, sv, Vh), snip, sigma_noise) in enumerate(
            zip(svds, snippets, noise_floors)
        ):
            Vh_hat = np.zeros((r_fixed, ns))
            for k in range(r_fixed):
                wp, nodes, orig_data, sigma_k, _, _, _ = wp_cache[s_idx][k]
                tau_k = alpha * sigma_noise / (sigma_k + 1e-40)
                n_k = 0
                for node, orig in zip(nodes, orig_data):
                    mask = np.abs(orig) >= tau_k
                    n_k += int(mask.sum())
                    node.data = orig * mask
                Vh_hat[k] = wp.reconstruct(update=True)[:ns]
                n_kept_total += n_k

            x_hat = ((U[:, :r_fixed] * sv[:r_fixed]) @ Vh_hat).astype(np.float32)
            err = snip.astype(np.float64) - x_hat.astype(np.float64)
            rmse_sum += np.sqrt(np.mean(err ** 2, axis=-1))

        rmse_per_channel = rmse_sum / len(snippets)
        rmse_med[a_idx] = float(np.median(rmse_per_channel))
        snr_med[a_idx] = float(20.0 * np.log10(
            np.median(rms_per_channel) / max(rmse_med[a_idx], 1e-12)
        ))

        n_kept_avg = n_kept_total / len(snippets)
        cr_wp[a_idx] = n_wp_total_per_snip / max(n_kept_avg, 1)
        cr_total[a_idx] = stage1_cr * cr_wp[a_idx]

    return cr_wp, cr_total, rmse_med, snr_med, stage1_cr, r_fixed


# ---------------------------------------------------------------------------
# Main sweep: collect results for all PIDs, initial CRs, and wavelet families
# ---------------------------------------------------------------------------
# %%
palette_pid = sns.color_palette('tab10', n_colors=len(pids))
palette_cr = sns.color_palette('viridis_r', n_colors=len(INITIAL_CRS_SVD))

# results[wavelet][initial_cr][pid] = (cr_wp, cr_total, rmse_med, snr_med, stage1_cr, r)
results = {wv: {cr: {} for cr in INITIAL_CRS_SVD} for wv in WAVELETS}

for pid in pids:
    snippet_files = [ROOT_OUTPUT.joinpath(pid, f'cadzow_denoised_{i}.npy') for i in range(3)]
    if not all(f.exists() for f in snippet_files):
        print(f'{pid}: snippets missing, skipping')
        continue
    snippets = [np.load(f) for f in snippet_files]
    print(f'{pid}: running α sweep...', flush=True)
    for wv in WAVELETS:
        for initial_cr in INITIAL_CRS_SVD:
            res = sweep_alpha(snippets, initial_cr, wv)
            results[wv][initial_cr][pid] = res
        print(f'  {wv} done.')
    print(f'  all wavelets done.')

# ---------------------------------------------------------------------------
# Figure: α vs CR_wp — single axis, thin per-PID lines + heavy median
# ---------------------------------------------------------------------------
# %%
QUARTO_FIGURE_DIR = Path(
    '/Users/olivier/Documents/oliche-quarto/analyses/2026-06-lfp-compression-svd/figures'
)

wv0 = WAVELETS[0]
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle(
    f'Adaptive WP — α vs CR_wp  ({wv0}, level {WP_MAXLEVEL})',
    fontsize=12,
)

for initial_cr, color in zip(INITIAL_CRS_SVD, palette_cr):
    r_vals, crwp_all = [], []
    for pid in pids:
        if pid not in results[wv0][initial_cr]:
            continue
        cr_wp = results[wv0][initial_cr][pid][0]
        r_vals.append(results[wv0][initial_cr][pid][5])
        crwp_all.append(cr_wp)
        ax.loglog(ALPHAS, cr_wp, color=color, lw=0.6, alpha=0.25)

    if crwp_all:
        r_med = int(np.median(r_vals))
        crwp_med = np.median(crwp_all, axis=0)
        ax.loglog(ALPHAS, crwp_med, color=color, lw=2.5,
                  label=f'CR_SVD ≈ {initial_cr}  (r = {r_med})')

ax.set_xlabel('α  (threshold multiplier)', fontsize=12)
ax.set_ylabel('WP stage CR  (r · n_slots / n_kept)', fontsize=12)
ax.set_ylim(0.5, 1e4)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=9, loc='upper left')
fig.tight_layout()
fname = f'{TODAY}_svd_adaptive_wp_alpha_sweep.png'
fig.savefig(FIGURE_DIR.joinpath(fname), dpi=150)
if QUARTO_FIGURE_DIR.exists():
    fig.savefig(QUARTO_FIGURE_DIR.joinpath(fname), dpi=150)
plt.show()
print('Figure saved.')

