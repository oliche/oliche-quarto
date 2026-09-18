"""Ground-truth validation of the sub-sample timing pick used for ibleatools #123
slowness (2026-09-18_multichannel_waveform_features.py): does the cross-correlation
+ parabolic-interpolation pick (ibldsp.utils.parabolic_max) lock to the nearest
sample, and would the frequency-domain phase-slope regression idea in
ibldsp.waveforms (get_apf_from2spikes/get_phase_slope/wave_shift_phase) do better?

Method: take 40 real reference-channel spike snippets (denoised, Hanning-windowed)
as templates, apply 200 known sub-sample shifts each via ibldsp.fourier.fshift
(ground truth), recover the shift with both methods, measure the error. Three noise
conditions: clean (no added noise), the snippet's actual denoised low-amplitude-
channel residual noise (what the production pipeline actually picks against), and
the much larger raw (pre-denoising) noise floor.

Findings (2026-09-18, pid=1a276285-8b0e-4cc9-9f0a-a3a002978724):
- Parabolic interpolation is NOT locking to the nearest sample: only ~6% of picks on
  real data land within 0.01 samples of an integer, spread std ~0.28 samples.
- Under "clean" (noiseless) conditions the phase-slope regression is exact (RMSE
  0.0000 samples) vs xcorr+parabolic's RMSE 0.0035 samples -- phase wins, as
  expected in theory for a pure delay.
- Under "denoised_residual_noise" -- the condition that actually matches what the
  production pipeline picks against (real, network-denoised, low-amplitude
  neighbour-channel noise, std ~4e-6 V here) -- xcorr+parabolic RMSE=0.099 samples
  beats phase-regression's RMSE=0.147 samples.
- Under "raw_noise" (much larger, ~1.7e-5 V std, not what we actually pick against
  since picks come from denoised traces) the two are roughly tied (RMSE 0.80 vs
  0.79 samples), both dominated by noise at that point.
- This matches the caveat already documented in ibldsp.waveforms.wave_shift_phase's
  docstring: "this works perfectly in theory, but does not work well with raw data
  sampled at 30kHz" -- and that function's own fix (get_spike_slopeparams, a
  per-template calibration against 50 known shifts via scipy.optimize.curve_fit) is
  too slow to run per spike per channel here (5404 spikes x ~30 channels).

Decision: keep xcorr+parabolic (ibldsp.utils.parabolic_max) as the pick method in
the production prototype script. It's both faster (0.36s vs 2.87s for 5404 spikes,
though both are negligible next to the ~110s waveform extraction) and, in the noise
regime that actually matters here, more accurate than the phase-domain alternative.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ibldsp import waveforms as wf
from ibldsp.utils import parabolic_max
from ibldsp.fourier import fshift

sns.set_theme(context="notebook")
FIG_DIR = Path.home().joinpath("Documents", "figures")
DATE = "2026-09-18"

wdir = Path(
    "/Users/olivier/scratch/ephysatlas_waveform_proto/1a276285-8b0e-4cc9-9f0a-a3a002978724/"
    "1a276285-8b0e-4cc9-9f0a-a3a002978724/probe_1a276285-8b0e-4cc9-9f0a-a3a002978724_000300.0_02.0_02.0/waveforms"
)
denoised = np.load(wdir / "denoised.npy").astype(np.float64)
raw = np.load(wdir / "raw.npy").astype(np.float64)
df_spikes = pd.read_parquet(wdir / "spikes.pqt")
n_spikes, nsw, ncw = denoised.shape
fs = 30_000.0
half_win = int(round(0.5e-3 * fs))
hann = np.hanning(2 * half_win + 1)
df_peak = wf.find_peak(denoised)
pt_all = df_peak["peak_time_idx"].to_numpy()
tr_all = df_peak["peak_trace_idx"].to_numpy()


def picks_xcorr(seg, ref, fs):
    win = seg.shape[0]
    n_fft = 2 * win
    R = np.fft.rfft(ref, n=n_fft)
    S = np.fft.rfft(seg, n=n_fft, axis=0)
    corr_full = np.fft.fftshift(np.fft.irfft(np.conj(R)[:, None] * S, n=n_fft, axis=0), axes=0)
    lags = np.arange(n_fft) - n_fft // 2
    keep = np.abs(lags) <= (win - 1)
    corr, lags_c = corr_full[keep, :], lags[keep]
    ipeak, cpeak = parabolic_max(np.abs(corr).T)
    return (lags_c[0] + ipeak) / fs


def picks_phase(seg, ref, fs, amp_quantile=0.5):
    win = seg.shape[0]
    fscale = np.fft.rfftfreq(win, 1 / fs)
    C = np.fft.rfft(ref)[:, None] * np.conj(np.fft.rfft(seg, axis=0))
    amp = np.abs(C)
    phase = np.unwrap(np.angle(C), axis=0)
    thresh = np.nanquantile(amp, amp_quantile, axis=0, keepdims=True)
    w = np.where(amp >= thresh, amp, 0.0)
    wsum = w.sum(axis=0)
    f_mean = (w * fscale[:, None]).sum(axis=0) / np.where(wsum > 0, wsum, 1)
    p_mean = (w * phase).sum(axis=0) / np.where(wsum > 0, wsum, 1)
    cov = (w * (fscale[:, None] - f_mean) * (phase - p_mean)).sum(axis=0)
    var = (w * (fscale[:, None] - f_mean) ** 2).sum(axis=0)
    slope = np.where(var > 0, cov / var, np.nan)
    return slope / (2 * np.pi)


# %% Ground truth: take real reference-channel spike snippets as templates, apply KNOWN
# sub-sample shifts via fshift, recover dt with each method, measure error vs truth.
rng = np.random.default_rng(1)
template_idx = rng.choice(n_spikes, size=40, replace=False)
true_shifts = rng.uniform(-1.4, 1.4, size=200)  # samples, beyond +-1 to probe robustness too

# realistic noise: pull actual samples from a far-away (no-spike) channel across many
# spikes/snippets, to match this recording's real noise floor
noise_pool_raw = raw[:200, :, -1].ravel()
noise_pool_raw = noise_pool_raw[np.isfinite(noise_pool_raw)]  # raw.npy has NaN padding
noise_pool_raw = noise_pool_raw - noise_pool_raw.mean()

# what our actual pipeline picks from: denoised, far-from-peak (low-amplitude, still
# real network-denoised) channels -- this is the realistic residual noise floor for
# picks_xcorr/picks_phase as used in the production script, not raw
noise_pool_denoised = denoised[:200, :, -1].ravel()
noise_pool_denoised = noise_pool_denoised[np.isfinite(noise_pool_denoised)]
noise_pool_denoised = noise_pool_denoised - noise_pool_denoised.mean()
print(f"raw noise std={noise_pool_raw.std():.2e} V, denoised (far channel) noise std="
      f"{noise_pool_denoised.std():.2e} V")


def make_template(i):
    trace = tr_all[i]
    pt = pt_all[i]
    t0, t1 = pt - half_win, pt + half_win + 1
    w0, w1 = max(0, -t0), len(hann) - max(0, t1 - nsw)
    t0c, t1c = max(0, t0), min(nsw, t1)
    if t1c - t0c != len(hann):
        return None
    return denoised[i, t0c:t1c, trace] * hann[w0:w1]


results = {"xcorr": [], "phase": []}
snr_label = []
for i in template_idx:
    template = make_template(i)
    if template is None:
        continue
    template_amp = np.abs(template).max()
    conditions = [
        ("clean", None),
        ("denoised_residual_noise", noise_pool_denoised),
        ("raw_noise", noise_pool_raw),
    ]
    for snr_name, pool in conditions:
        shifted = np.stack([fshift(template, s) for s in true_shifts], axis=1)  # (win, n_shifts)
        if pool is not None:
            noise = rng.choice(pool, size=shifted.shape, replace=True)
            shifted_noisy = shifted + noise
        else:
            shifted_noisy = shifted
        dt_x = picks_xcorr(shifted_noisy, template, fs) * fs  # back to samples
        dt_p = picks_phase(shifted_noisy, template, fs) * fs
        results["xcorr"].append(dt_x - true_shifts)
        results["phase"].append(dt_p - true_shifts)
        snr_label.extend([snr_name] * len(true_shifts))

err_xcorr = np.concatenate(results["xcorr"])
err_phase = np.concatenate(results["phase"])
snr_label = np.array(snr_label)

for snr_name in ["clean", "denoised_residual_noise", "raw_noise"]:
    m = snr_label == snr_name
    print(f"--- {snr_name} (n={m.sum()}) ---")
    for name, err in [("xcorr+parabolic", err_xcorr), ("phase-regression", err_phase)]:
        e = err[m]
        e = e[np.isfinite(e)]
        print(f"  {name}: bias={e.mean():+.4f} samples, RMSE={np.sqrt(np.mean(e**2)):.4f} samples, "
              f"std={e.std():.4f}")

# %% Plot: error vs true shift, both methods, all SNR conditions
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, snr_name in zip(axes, ["clean", "denoised_residual_noise", "raw_noise"], strict=True):
    m = snr_label == snr_name
    n_templates_trials = m.sum() // len(true_shifts)
    tiled_true = np.tile(true_shifts, n_templates_trials)
    ax.scatter(tiled_true, err_xcorr[m], s=4, alpha=0.3, label="xcorr+parabolic", color="steelblue")
    ax.scatter(tiled_true, err_phase[m], s=4, alpha=0.3, label="phase-regression", color="darkorange")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="true shift (samples)", ylabel="pick error (samples)", title=snr_name)
    ax.legend(markerscale=3)
fig.suptitle("Sub-sample pick accuracy: recovered dt error vs known fshift ground truth\n"
             "(40 real spike templates x 200 random shifts in [-1.4, 1.4] samples)")
fig.tight_layout()
fig.savefig(FIG_DIR.joinpath(f"{DATE}_waveform_proto_pick_accuracy_ground_truth.png"), dpi=150)
print("saved ground truth accuracy figure")
