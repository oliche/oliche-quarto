# Next session: exclude saturated samples from the encoding fit (row-exclusion, not muting)

## Context

Project: `analyses/2026-07-lfp-encoders/` (LFP←behaviour lagged encoding model). Read
`index.qmd` — especially **Result 5** (compression collapse) and **Result 6** (v01
checkpoint) — and the `project_lfp_encoders` memory file before starting; both carry
history this prompt assumes.

`v01` (`results_bwm_v01`, reprocessed via an updated `lfpack`) added saturation
detection: ADC-clipped stretches are detected on the raw LFP band and **muted to exact
zero** before decimation, in the shared Cadzow/CAR stage every one of the 3 LFP sources
(`uncompressed`/`default`/`aggressive`) is derived from. Result 6 found this improved
`aggressive`/`default` collapse rates a lot, but made `uncompressed` collapse *more*
often (4→11 PIDs) — diagnosed as: a CV fold whose window overlaps a muted (exact-zero)
span has near-zero target variance, so R² (dividing by that variance) swings hugely
negative. Not yet confirmed on a raw trace — that's part of this task.

**The user's insight, and the actual task**: the stored saturation start/stop sample
boundaries are not reliable (approximate detection, possible filter smearing beyond the
detected edge). Instead of trusting the muted-to-zero signal as if it were real (if
low-variance) data, **exclude those time samples from the regression fit entirely** —
both from training (sufficient-statistic accumulation) and from held-out CV scoring.
This is mathematically trivial for a ridge/OLS fit: dropping a row just means not
including it in the sums that build `Sxx`/`Sxy`/`n` — no other change needed, and CV R²
is a closed form of those same sums.

## Where the pieces live

**Detection/labels already exist** — read-only, in `packages/lfpack/src/lfpack/_core.py`,
`LFPackReader`:
- `.saturation` — DataFrame `start_sample`/`stop_sample` (raw LFP rate), one row per
  saturated interval, stored once per recording (**scale-independent** — should be
  identical whether read from the `default` or `aggressive` archive for a given PID;
  verify this).
- `.saturation_mask` — boolean array at the reader's own sampling rate, `shape (ns,)`.
- `.saturation_times()` — same intervals as **session-clock seconds**
  (`start_time`/`stop_time`), i.e. directly comparable to `Design.tvec` /
  `Targets.tvec` without any manual sample-rate arithmetic. **Use this one.**

**Encoding pipeline** — `sdsc-slurms/2026-07_lfp-encoders/` (this directory *is* the
source of truth; the laptop quarto repo `sys.path`-shims to it):
- `design.py`: `Design` — has `.tvec` (session-clock time per sample). `X` is
  behaviour-derived and **unaffected** by LFP saturation — no changes needed here beyond
  reading `tvec` to build the mask.
- `targets.py`: `Targets` — has `.tvec` (asserted equal to `design.tvec`'s sample count
  in `accumulate_folds`). `Y` is what's corrupted by saturation.
- `solve.py`: **the one choke point**. `Accumulator.add_chunk(X, Y)` sums *all* rows
  unconditionally. `accumulate_folds(design, targets, n_folds, chunk_samples)` drives
  every chunked pass and is called, directly, by **every** top-level fit/score function:
  `solve_encoding`, `solve_encoding_grouped`, `permutation_null_r2`,
  `permutation_null_r2_grouped`, `select_lambda`, `select_lambda_robust`. Add an optional
  `valid: np.ndarray | None = None` boolean mask (`shape (design.n_samples,)`) to
  `accumulate_folds`; inside its chunk loop, slice `X`/`Y` to `valid[a:b]` before
  `acc.add_chunk(...)` (if `valid is None`, behave exactly as today). Then add the same
  `valid=None` passthrough parameter to all six callers above, so `encode.fit_pid` can
  compute the mask once and pass it everywhere. **Nothing downstream** (`centred_cross`,
  `_r2`, `_cv_r2`, `ridge_fit`) needs to change — they only ever see `Accumulator`
  objects, which already reflect whichever rows were included.
  - Note `permutation_null_r2` circular-shifts `design.base` (`np.roll`) but not
    `targets.Y` — the `valid` mask lives on the `Y`/saturation timeline and must **not**
    be shifted along with it; reuse the same (unshifted) mask for the null as for the
    real fit.
  - There is an existing but unrelated `_mask_targets(acc, mask)` helper — that masks
    the **target/column** axis (used by `solve_encoding_grouped`'s per-group scoring).
    Don't confuse it with the new **row** mask; name the new parameter something
    unambiguous (`valid`, not `mask`).
- `encode.py`: `fit_pid` is the per-PID orchestrator — compute the row-mask once per PID
  (from the `default`-tier archive's `LFPackReader.saturation_times()`, vectorized
  against `dsg.tvec`, since `read_uncompressed` already borrows its `tvec` from the
  `default` archive so one mask should apply to all 3 sources) and pass it into every
  `solve_mod.*` call in the `for kind in ("band", "raw")` loop.

## Open questions to resolve — don't assume, verify

1. **Safety margin around each stored interval.** The user explicitly said the stored
   sample boundaries "are not reliable." The Cadzow/CAR/highpass chain includes
   zero-phase (`filtfilt`-style) filtering — no net delay, but a finite impulse-response
   *extent* that can smear a transient's influence symmetrically in time beyond the
   detected edge. Decide a padding window (some fixed margin, or derived from the
   highpass filter's impulse response length) and validate empirically — e.g. plot the
   raw trace around a detected interval's edge on one of the 7 newly-collapsed
   `uncompressed` PIDs from Result 6 (listed in `project_lfp_encoders` memory /
   `index.qmd` Result 6) and check how far the muted-zero / ringing actually extends
   versus the stored boundary.
2. **Does row-exclusion fully fix `kind="band"` targets?** Band-power targets are
   `|Hilbert(bandpass(Y))|`, computed **globally over the whole trace** (see
   `targets.py` / project memory: "filters WHOLE trace... global, artefact-free at chunk
   edges"). The Hilbert transform and the bandpass filter both have non-compact support
   in principle — a saturated transient could leak spectral energy into the band-power
   envelope at *other*, nominally-clean time samples, not just the exact muted rows.
   Row-exclusion at the regression stage cannot undo that. Investigate whether this
   matters in practice (e.g. compare band-power values near a saturation event with vs.
   without the muting, or vs. a version computed after excising the saturated span
   before filtering) before assuming the row-exclusion fix is sufficient for `band`
   kind — it may need the exclusion applied earlier (at `targets.py`'s filtering stage)
   rather than only at `solve.py`'s accumulation stage. `kind="raw"` (per-channel linear
   detrend) is much more local and less likely to have this problem, but check it isn't
   pulled by outliers either.
3. **Fold-size imbalance.** CV folds are contiguous, equal-length-by-sample-count today
   (`np.linspace(0, design.n_samples, n_folds+1)`). If a fold's window contains a big
   saturated span, its post-exclusion `n` shrinks a lot relative to other folds. Confirm
   nothing downstream implicitly assumes equal fold sizes (skim `_cv_r2`/`_r2` — they
   look generic over `acc.n`, but verify), and decide whether a fold (or a whole PID)
   with too few valid samples remaining needs a new QC skip/flag, analogous to the
   existing wheel/pupil temporal-coverage gating pattern already in this codebase.
4. **Is the mask really tier-independent?** Confirm by directly reading `.saturation`
   from the `default` vs `aggressive` HDF5 archive for a few PIDs and checking the
   interval tables match exactly, before relying on one shared mask for all 3 sources.

## Validation plan before a full 699-PID resweep

- Smoke-test first on: the 7 PIDs newly-flagged as collapsed under `uncompressed` in
  `v01` (see Result 6 / memory), the flagship collapse PID
  `11a5a93e-58a9-4ed0-995e-52279ec16b98`, and a handful of `DATA_ISSUES.md`'s
  `DEFAULT_COMPRESSION_ARTIFACT`/`AGGRESSIVE_ONLY_COLLAPSE` PIDs (some of those may
  *also* be saturation-driven rather than purely the lambda-selection issue Result 5
  diagnosed — worth checking).
- Compare collapse rate / frac_sig before vs after on that subset using the same
  `region_aggregate.py` machinery already built (`collapse_rate`, `load_channel_scores`)
  before committing to a full brain-wide resweep.
- Run the full resweep via the existing SDSC pipeline
  (`sdsc-slurms/2026-07_lfp-encoders/README.md`, `encode.sbatch`/`encode.py`), archive
  and rsync back the same way `v01` was (see the README's rsync line), call it `v02`.
- Compare `v02` against `v01` and `v00` — generalize `analyses/2026-07-lfp-encoders/
  compare_v00_v01.py` (currently hardcodes two versions) into a reusable
  version-comparison helper rather than hand-duplicating it a third time.
- Update `index.qmd` (Result 6, or a new Result 7) with whatever is found — including if
  the row-exclusion *doesn't* fully fix `band` targets (question 2 above); that would be
  a real, reportable finding, not a failure to hide.