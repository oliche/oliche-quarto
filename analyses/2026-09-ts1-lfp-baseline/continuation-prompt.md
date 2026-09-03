We built a first LFP-modality baseline for ibl-benchmark's TS1 suite. Full writeup with
method, results, and discussion is at
`/Users/olivier/Documents/oliche-quarto/analyses/2026-09-ts1-lfp-baseline/index.qmd` — read
that first, it has everything: architecture, the bugs found and fixed, final scores, and
open questions. This prompt is just enough to get oriented and pick a next step.

## Where the code is

`src/ts1/models/single_session/lfp/` in the `ibl-benchmark` repo
(`/Users/olivier/PycharmProjects/brain-wide-bench/ibl-benchmark`). **Untracked** — `git
status` will show it as new files, nothing has been committed yet.

- `features.py` — LFP band-power envelope extraction from the local lfpack archive
  (`/Users/olivier/Documents/datadisk/lfp-processing/lfpack/v03/lf_compressed_all.h5`),
  plus the `EID_TO_PIDS` mapping for all 29 eval sessions.
- `remote_recording.py` — reads trial windows/targets/splits straight off S3 via HTTP range
  requests (no `all_units` download). `materialize_recording()` matters a lot for
  performance — call it before looping over trials, or expect multi-minute stalls.
- `decode.py` — the actual per-session-per-task decoder (PCA+LogisticRegressionCV for
  choice/reward/stimulus_contrast, PCA+RidgeCV for the four continuous tasks). Has a
  `if __name__ == "__main__"` CLI for one session.
- `run_all.py` — loops `decode.py` over all 29 sessions, resumable (checks a marker file
  per session before re-decoding).

Final predictions are already written to `predictions/lfp-band-power/` in the repo (all 8
tasks x 29 sessions, real submission format, scores directly with `bwb_eval.scoring.ts1`).

## Environment note

`ibl-benchmark`'s own venv needed `lfpack` installed from a sibling repo (editable):
`uv pip install --python .venv/bin/python -e /Users/olivier/PycharmProjects/ephys-atlas/packages/lfpack`.
This also pulls in `one-api`/`ibl-neuropixel` from that monorepo's local packages. If the
venv looks incomplete, that's the first thing to check.

## State as of this handoff

- Full 29-session results are final and match the writeup: `choice` (0.539 bacc) and
  `reward` (0.668 bacc) decode above chance; `stimulus_contrast` is flat; the four
  continuous tasks (`wheel_speed`, `{left,right}_paw_speed`, `whisker_motion_energy`,
  `licking_rate`) are negative on average despite positive correlations.
- `wheel_speed`/`licking_rate` predictions are clipped at 0 post-hoc (validated improvement,
  small). Do not add the same clip to `whisker_motion_energy`/paw speeds — those are
  z-scored by the benchmark, negative values are legitimate there.
- log1p on wheel_speed was tried and made things worse — don't redo that experiment.
- A reference-protocol diagnostic (whole-session, 5-fold non-causal CV — NOT a valid TS1
  score) confirmed the negative R² is mostly explained by TS1's harder causal-split +
  movement-window evaluation, not a broken pipeline. Script for this was never committed
  anywhere permanent (`/tmp`, gone) — rewrite from the description in the writeup if needed
  again.

## Open next steps (pick one, or ask what's wanted)

1. **Formalize as a real baseline**: wrap as a `BaseModel` + `TS1EvalTrainer` config (like
   the `CEBRA` baseline in `src/ts1/models/single_session/cebra/`) instead of a standalone
   script — needed to match how other TS1 baselines are integrated.
2. **Run the 5 official `EVAL_SEEDS`** instead of the single seed (42) used so far. Note the
   current decoder is deterministic given fixed `random_state`, so this needs either a
   different `random_state` per seed or an argument for why single-seed is acceptable here.
3. **Investigate `whisker_motion_energy`'s bias** — it's the one continuous task where
   the problem isn't just weak signal, there's a real systematic offset unexplained.
4. **Actually submit**: `ibl-benchmark/CONTRIBUTING.md` still marks the submission upload
   path as TODO — that's a repo-level gap, not something to solve by writing more code here.
5. Something else — the writeup's Conclusion section has the fullest list of what's known
   vs. unknown.
