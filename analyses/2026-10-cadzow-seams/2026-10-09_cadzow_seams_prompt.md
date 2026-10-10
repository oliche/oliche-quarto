# Task: find and fix the spatial seams of the Cadzow LFP denoiser

The CSD (second spatial derivative) of Cadzow-denoised LFP shows horizontal discontinuities. Part of them follow
Cadzow's spatial window grid; part do not. Find out what causes each kind and fix it at the root, in the denoiser
(`ibldsp.cadzow`), not with a detect-and-mask step. Read `analyses/2026-10-cadzow-seams/index.qmd` first. It holds
everything tried so far, and the figures to compare against.

## Context

- lfpack v04 runs Cadzow inside the ibldsp chunk worker `ibldsp.voltage._resample_lfp_chunk`. It is called by
  `resample_denoise_lfp_cbin` ← `lfpack.compress_bin_to_h5` ← `sdsc-slurms/2026-06-lfpack/compress.py`.
  Production kwargs: `rank=5, niter=1, fmax=None, nswx=64, ovx=32, gap_threshold=2.0, ppca_k=2.0`, `fs=250`. After
  Cadzow, bad channels are re-interpolated (`interpolate_bad_channels(..., groupby_y=True)`).
- Denoiser code: `~/PycharmProjects/ephys-atlas/packages/ibl-neuropixel/src/ibldsp/cadzow.py`, mainly
  `cadzow_denoiser`, `_process_window`, `_apply_rank_threshold` and `trajectory`. It takes one rFFT over the whole
  chunk (no temporal windows). For each 64-channel spatial window and each frequency bin it builds a trajectory
  (Hankel-like) matrix from the NP1 geometry, keeps rank ≤ 5 (adaptive by singular-value gap), replaces outlier
  channels (PPCA, median + 2 MAD) and refits. The windows are blended with partition-of-unity Hann ramps
  (`ovx ≤ nswx/2`) or with Hann weights normalised by their summed weight (`ovx > nswx/2`).
- The v04 archive on S3 was built with `ovx=32`, so its CSD has these seams.

## What is known

- **Window seams (production):** horizontal blocks bounded every 32 channels. They are worst where Cadzow keeps
  little CSD power (749cb2b7: blocks at channels 96, 128 and 320). They move with the window grid (shift by 16
  channels).
- **Tried:**
  - `ovx=48`: ~1.6× cost.
  - Cycle spinning over 4 grid offsets (0, 8, 16, 24 channels, mirrored rows at the tip, equal weights): 4× cost.
  - Results: both weaken the seams, give near-identical images and keep the same CSD power (14 % vs 17 % for
    production). **But the user still sees large seams with `ovx=48`**, so this is not solved.
- **Failed:** two runs staggered by 1/4 and 3/4 of the step, blended with complementary raised-cosine weights. The
  CSD of a depth-varying blend picks up w″·(A − B) + 2w′·(A′ − B′), so any blend whose weights vary with depth imprints
  the disagreement between fits on the CSD.
- **Same mechanism in production:** the Hann ramps that blend adjacent windows are depth-varying weights too. With
  `ovx=32` every channel lies in a ramp, and w″ peaks at the ramp ends, i.e. every 32 channels. Blending only shapes
  the seams; their size is set by the disagreement A − B between overlapping window fits. Reduce A − B, don't re-blend.
- **Not explained:** discontinuities at the same depths in all methods, e.g. e8f9fba4 near channels 120, 180–190, 265
  and 300. e8f9fba4 has 18 bad channels (11 dead, 7 noisy). First check whether they are already in the no-Cadzow
  CSD (anatomy, bad channels) before blaming Cadzow.
- **Metric pitfalls:** my seam metrics were too noisy to rank methods.
  - On NP1, adjacent channel indices are in different columns of the checkerboard: channels 0..7 are in columns
    2, 0, 3, 1, ….
  - Bad channels next to a seam inflate any step metric. Channel 191 of 749cb2b7 sits next to the seam at 192.
  - The per-row profile is noisy.
  - Judge visually as well; the user spots seams by eye.
- **Ruled out:** the phantom channel that `cadzow_denoiser` appends with `npad=0` (row 384) never enters a window when
  (384 − ovx) is a multiple of the step, which holds for every grid below; the last window is 320:384.

## Hypotheses to test (one at a time, cheapest first)

1. **Adaptive rank per window and per bin** (`gap_threshold=2.0`): neighbouring windows keep different numbers of
   components, so their fits differ in amplitude → blocks. `_apply_rank_threshold` takes the largest gap ratio, often
   s1/s2, so the rank can jump between 1 and 5 from bin to bin and window to window. Log the rank map (window × bin),
   then test `gap_threshold=None` (fixed rank 5), and rank 1–5.
2. **PPCA outlier replacement per window** (`ppca_k=2.0`): a channel can be an outlier in one window and not in the
   next, so it is replaced in one fit only. Test `ppca_k=None`.
3. **Edge bias of the trajectory fit:** channels near a window edge enter fewer trajectory entries, so the rank-r fit
   is less constrained, or shrunk, there. Measure the per-channel fit gain against position in the window.
4. **Bad channels:** interpolation before Cadzow, Cadzow's own model of those channels, then re-interpolation. Test on
   e8f9fba4 against the depths of its labels.
5. **Window size:** `nswx` 32, 96, 128 at matching overlap ratios.
6. **No window seam at all:** a sliding-centre Cadzow where each output channel comes from the window centred on it
   (an expensive reference, to bound what is achievable).

## How to test

- **Data:** use the cached production chunks in `~/scratch/lfp/csd_lfpack_v04/` (11 benchmark PIDs):
  - `{pid}_chunk_orig.npy`: no-Cadzow output of the production worker, (8192, 384) float32 volts.
  - `{pid}_chunk_cadzow.npy`: production run.
  - `{pid}_chunk_cadzow_ovx48.npy`: `ovx=48` run.
  - `{pid}_chunk_i0.npy`: chunk start in the recording.
  - Bad-channel labels: `/Users/olivier/Documents/datadisk/lfp-processing/lfpack/v04/lf_compressed_v04_a2p5_fine_all.h5`,
    `f[pid]['00/meta'].attrs['labels']`.
  - Geometry: `neuropixel.trace_header(version=1)`; all 11 PIDs are NP1.
  - Running `cadzow_denoiser` on the trimmed chunk reproduces the seam pattern of the production chunk. A 32.8 s chunk
    takes ~6 s on 8 cores (`n_jobs=8`).
- **A clean seam measure without anatomy:** keep each window's fit before blending and measure the disagreement A − B
  (and its CSD) on the channels shared by adjacent windows. This is the quantity that makes the seams, and it needs no
  second run. Cross-check with a run whose grid is shifted by s channels. Plot per insertion, not only the median.
- **Ground truth:** build synthetic data with known smooth CSD sources. Forward-model point/laminar sources onto the
  NP1 geometry, add spatially white noise at a realistic level, run the denoiser and measure seams and bias against
  the truth.
- **Compare against the existing figures:** `analyses/2026-10-cadzow-seams/figures/2026-10-09_cadzow_cyclespin_*.webp`
  (production / cycle spinning / `ovx=48`, CSD of 4–7 s into the chunk, stars at each method's own seams).
- **Track cost:** any fix must stay affordable for reprocessing the archive (1099 PIDs). Report time per chunk against
  production. The production worker runs Cadzow with `n_jobs=1` (one chunk per process), so report single-thread time.

## Constraints and conventions

- Fix the root cause in the algorithm; don't mask or QC-gate the seams.
- When adding an alternative, keep the existing function; add the new one alongside, don't replace it.
- Python: NumPy docstrings, f-strings, `pathlib` with `joinpath()`, ruff (line length 120), seaborn notebook theme.
  Call `ax.grid(False)` on image axes; the user doesn't want grid lines over the data.
- Figures: WebP, quality 80 (`fig.savefig(path, dpi=..., pil_kwargs={'quality': 80})`).
- Report: update `analyses/2026-10-cadzow-seams/index.qmd`, results first and concise. Render from the repo root with
  `QUARTO_PYTHON=/Users/olivier/PycharmProjects/bwm/.venv/bin/python quarto render`. Don't commit unless asked.
- Don't change `ibldsp` in place without asking. Prototype in this folder first, then propose a change to `ibldsp`
  and to the lfpack Cadzow kwargs.
