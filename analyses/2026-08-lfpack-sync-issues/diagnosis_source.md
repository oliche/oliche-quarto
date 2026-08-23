## Diagnosis

All 20 flagged pids ([lfpack#8](https://github.com/int-brain-lab/lfpack/issues/8)) were re-derived
from their raw `_spikeglx_sync.{channels,polarities,times}` fronts and re-run through the
pyproject-migration `ibllib.ephys.sync_probes` code. The 20 pids collapse to 12 unique sessions
(several contribute both probes). Four genuinely distinct raw-pulse pathologies account for 11 of
the 20 probes; the rest are within ordinary hardware tolerance. Residuals below are the post-sync
single global-affine check (see verification note at the end).

- **`dropped_edges` (5 probes / 3 sessions)** -- one channel's pulse rate sustainedly halves
  partway through the recording (e.g. 2 Hz -> 1 Hz for the last ~10-15% of the session) while the
  other channel keeps ticking at the original rate. Real edges are missing on that channel from
  that point to the end. Seen in `367e94f6` (probe), `4836a465`+`d0046384` (both probes of the
  same session, identical onset index/time), `7967a14e`+`ad597f5f` (both probes of the same
  session). Residuals 9-76 seconds.
- **`duplicate_burst` (3 probes / 2 sessions)** -- one transition double-triggers the detector,
  adding a couple of near-simultaneous spurious edges (a bounce), on either channel. Seen on the
  **nidq** channel for `0b8ea3ec`+`a5f2ec22` (6 extra edges within a few ms, ~1.5s residual) and on
  the **probe** channel for `f2ea7211` (2 extra edges 33 microseconds apart, ~0.8s residual).
- **`single_edge_glitch` (2 probes / 2 sessions)** -- a single edge, isolated to the last ~1% of
  the recording (probe channel only; the reference channel stays clean throughout), arrives early
  by an amount well above ordinary clock jitter (baseline jitter on genuinely clean probes is
  <1ms per interval), then the train immediately reverts to its normal cadence. Not a sustained
  rate change and not a near-zero double-trigger -- a one-off mistimed detection, plausibly from
  signal quality degrading right as the probe's acquisition winds down. Seen in `b2746c16` (one
  edge ~139ms early at 99.8% through the session, 154ms residual) and `81f0087b` (one edge ~12ms
  early at 99.3% through, 12ms residual) -- the two "clean"-looking outliers flagged for a closer
  look. `b2746c16`'s sibling probe in the same session, `c09b3c18`, is unaffected (1.4ms),
  confirming this is probe-local, not a shared reference-trace defect (`81f0087b`'s sibling probe
  wasn't in this 20-pid list, so that check couldn't be repeated for it).
- **`irregular_3A_reference` (1 probe)** -- `0fed7207` is a pre-2020 Neuropixel 3A session with no
  nidq; probes are synced against each other via frame2ttl pulses. These are naturally bursty
  (dense within a trial, sparse between), so the rate-change/burst/glitch heuristics above don't
  apply cleanly here. The ~5s residual (of the final smoothed sync output; the raw pre-correction
  affine misfit is much larger, ~42.5s) is more likely accumulated imprecision from using a bursty
  behavioural signal as the sync reference over a long session than a single discrete pulse event.
- **`clean` (9 probes / 6 sessions)** -- no rate change, no burst, no isolated glitch; residuals
  1-4ms, consistent with ordinary nidq/probe crystal-oscillator ppm mismatch, not a genuine pulse
  defect.

In every defective case, total pulse *counts* end up close enough that
`sync_probes.version3B()`'s only integrity check (`np.isclose(..., rtol=0.1)`) never fires -- it
zips `nidq[i] <-> probe[i]` for all `i` up to `min(n_ref, n_probe)` with no per-channel internal
consistency check (constant rate, isolated bounce/glitch detection). Compounding this, whenever a
probe's own pulse spacing already fails the existing 150ppm burst check, the code falls back to
`type='exact'`, whose own QC is mathematically tautological (it interpolates through the very
points it was built from, so residual is always ~0) -- silently passing sessions with
tens-of-seconds-scale defects. All three raw-pulse pathologies above are genuine hardware/signal
defects, not something the current extraction code can fix -- but the code's QC could be made
substantially harder to fool by (a) checking per-channel rate/alternation consistency instead of
only gross count closeness, and (b) not treating the `'exact'`-fallback QC as informative.

**Verification note (extraction-mismatch check):** the residual was independently computed two
ways for every probe -- once from whatever `sync.timestamps` is currently registered on Alyx (via
`SpikeSortingLoader`), and once from actually re-running `sync_probes.version3A()`/`version3B()`
on the raw fronts (both using the same single global affine polyfit + max |residual|, the exact
check `attach_ibl_metadata.py::compute_sync()` runs). 18 of 20 probes matched exactly; the other 2
(`53ecbf4f`, `b543e81e`, both probes of the same `clean` session) differed by <0.25ms, which isn't
meaningfully different from a re-extraction (neither number is "more correct" than the other) --
so none of this is a stale/failed prior extraction artifact; every defect above is reproducible
straight from the raw pulses today. The table below reports only the resync-derived values, since
the two agree.

## Per-probe results

| pid | eid | probe | version | diagnosis | slope | intercept | max residual (ms) | median residual (ms) |
|---|---|---|---|---|---|---|---|---|
| 4836a465-c691-4852-a0b1-dcd2b1ce38a1 | caa5dddc-9290-4e27-9f5e-575ba3598614 | probe00 | 3B | dropped_edges (probe) | 3.33e-05 | 3.4544 | 76360.9 | 2553.0 |
| d0046384-16ea-4f69-bae9-165e8d0aeacf | caa5dddc-9290-4e27-9f5e-575ba3598614 | probe01 | 3B | dropped_edges (probe) | 3.33e-05 | 3.4364 | 76182.1 | 2539.7 |
| 367e94f6-df51-4120-a297-77fa88dcec31 | bd8b204f-a42e-45c1-a8f0-71c6223a6657 | probe00 | 3B | dropped_edges (probe) | 3.33e-05 | 3.5710 | 69506.6 | 2642.9 |
| 7967a14e-1bf0-4666-acb4-9b08ba8f3385 | f30ab838-98bd-4b64-b721-61209cfd6ae9 | probe01 | 3B | dropped_edges (probe) | 3.33e-05 | 0.0608 | 9006.2 | 45.5 |
| ad597f5f-5201-4e02-9a28-1fb1a75746cc | f30ab838-98bd-4b64-b721-61209cfd6ae9 | probe00 | 3B | dropped_edges (probe) | 3.33e-05 | 0.0606 | 8985.5 | 45.3 |
| 0fed7207-f747-428b-b4c0-854cabb50d9e | 1211f4af-d3e4-4c4e-9d0b-75a0bc2bf1f0 | probe01 | 3A | irregular_3A_reference | 3.33e-05 | -3.3179 | 4994.6 | 952.5 |
| 0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a | a4a74102-2af5-45dc-9e41-ef7f5aed88be | probe00 | 3B | duplicate_burst (ref) | 3.33e-05 | 0.8518 | 1455.4 | 692.5 |
| a5f2ec22-0ff3-4249-bd2f-6247c3990e53 | a4a74102-2af5-45dc-9e41-ef7f5aed88be | probe01 | 3B | duplicate_burst (ref) | 3.33e-05 | 0.8517 | 1455.4 | 692.5 |
| f2ea7211-85f3-4394-b03e-1302a1dfe79c | ff700fcb-f72c-4613-9738-1f82cbabc112 | probe00 | 3B | duplicate_burst (probe) | 3.33e-05 | -0.1302 | 781.1 | 107.4 |
| b2746c16-7152-45a3-a7f0-477985638638 | 614e1937-4b24-4ad3-9055-c8253d089919 | probe01 | 3B | single_edge_glitch (probe) | 3.33e-05 | -0.0004 | 154.1 | 0.3 |
| 81f0087b-2bd1-4e48-8e86-e8206aee3d9d | 8b1f4024-3d96-4ee7-95f9-8a1dfd4ce4ef | probe00 | 3B | single_edge_glitch (probe) | 3.33e-05 | -0.0002 | 12.3 | 0.1 |
| e6402305-5028-42aa-975b-c540c882b131 | bc9ea019-b560-4435-ab53-780d9276f15c | probe01 | 3B | clean | 3.33e-05 | -0.0007 | 3.7 | 0.5 |
| 2adc4f5d-bc7b-42a4-be76-f5df33d713d4 | 4aab0f45-54eb-4ba0-9049-8ad1b7598fbe | probe00 | 3B | clean | 3.33e-05 | -0.0003 | 3.5 | 1.3 |
| 0d59e3a1-86c1-44bd-b291-d1f8bc8327ba | 4aab0f45-54eb-4ba0-9049-8ad1b7598fbe | probe01 | 3B | clean | 3.33e-05 | -0.0003 | 3.1 | 1.2 |
| 5c63d860-1e3c-481b-a290-9f299a5421f5 | bc9ea019-b560-4435-ab53-780d9276f15c | probe00 | 3B | clean | 3.33e-05 | -0.0003 | 1.8 | 0.2 |
| b543e81e-4c8f-415e-82ec-631b177d19d2 | 16693458-0801-4d35-a3f1-9115c7e5acfd | probe01 | 3B | clean | 3.33e-05 | 0.0017 | 1.8 | 0.2 |
| 53ecbf4f-e0d8-4fe6-a852-8b934a37a1c2 | 16693458-0801-4d35-a3f1-9115c7e5acfd | probe00 | 3B | clean | 3.33e-05 | 0.0016 | 1.7 | 0.2 |
| 735fa61d-db9b-4289-990d-659793413c75 | 0d7ea15f-86ea-4571-9bc2-859f1ee9ae6a | probe00 | 3B | clean | 3.33e-05 | 0.0015 | 1.6 | 0.4 |
| c09b3c18-c9e5-4551-9a35-7b2a069f57ff | 614e1937-4b24-4ad3-9055-c8253d089919 | probe00 | 3B | clean | 3.33e-05 | 0.0000 | 1.4 | 0.0 |
| 5999eeca-10fa-4e4b-ae7c-02fab4fe41be | 0d7ea15f-86ea-4571-9bc2-859f1ee9ae6a | probe01 | 3B | clean | 3.33e-05 | 0.0010 | 1.1 | 0.3 |