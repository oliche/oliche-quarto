# LFP Denoising Pipeline — Summary

## What this is
Systematic exploration of multi-channel LFP denoising for Neuropixel probes (NP1 checkerboard, NP2 straight columns), evaluated using cross-probe cross-correlation on CSD-transformed signals.

## Files
| File | Purpose |
|---|---|
| `lfp_pipeline.py` | Library: new denoising functions + modular pipeline + evaluation |
| `run_dataset4.py` | Script: runs everything on dataset 004, saves figures to `figures/` |
| `figures/` | Output figures from the last run |

The original data loading script is at `~/Documents/PYTHON/ibldevtools/olivier/2026-03-24_NP1NP2.py`.  
The core DSP library is at `~/PycharmProjects/EphysAtlas/ibl-neuropixel/src/ibldsp/`.

## Data
- Path: `/datadisk/Data/2026/np1np2/`
- Dataset 004: NP1 in `ephysData_g0_imec1/` (384 ch), NP2.4 in `ephysData_g0_imec0c/` (96 ch, single shank)
- Channel matching: linear y-transform `y_np1 = 2601.3 + 1.179 * y_np2` (valid for datasets ≤ 4)

## New functions in `lfp_pipeline.py`

### Denoising
- **`cadzow_geometry(wav, fs, rank, fmax, h)`** — Cadzow denoiser that processes each probe column separately along y. Better than `cadzow_np1` for both NP1 and NP2 because the 1D Hankel matrix per column is better conditioned than the mixed-column 2D version.
- **`adaptive_svd_denoise(wav, h)`** — SVD with automatic rank selection via the Gavish-Donoho (2014) optimal hard threshold. Operates per column. Best CSD1 metric but higher variance than Cadzow.
- **`column_car(wav, h)`** — Median CAR per column. Not useful as a standalone step (CSD removes common offsets anyway) but available.
- **`global_pca_filter(wav, n_components)`** — Removes top N global PCA components. Found to hurt performance — removes signal along with noise.

### Evaluation
- **`evaluate_csd_xcorr(..., csd_orders=(1,2))`** — Main metric: bandpass → preprocessing → CSD(n) → peak xcorr between matched NP1/NP2 channel pairs. CSD removes common-mode signals so the xcorr reflects local signal preservation only.
- **`sweep_csd_xcorr(sweep_np1, bp_np2, ...)`** — Evaluate a parameter sweep dict using CSD xcorr.
- **`sweep_cadzow_params(raw, sr, h, param_ranges, version)`** — Run Cadzow over a grid of `{rank, fmax, nswx, ovx, niter}`.
- **`summary_table(metrics)`** — Print mean ± std table.

### Visualisation
- **`plot_pipeline_comparison(metrics)`** — Box plot of xcorr per stage.
- **`plot_xcorr_traces(data_np1, data_np2, ch1, ch2, fs, t_lim=(27,32))`** — Time traces + xcorr panel for a channel pair. `t_lim` sets the x-axis window in seconds.
- **`plot_singular_values(wav, h)`** — SVD spectrum per column with Gavish-Donoho threshold line.

## Key findings (dataset 004)

### Stage comparison (CSD1 mean peak xcorr)
| Stage | CSD1 | CSD2 |
|---|---|---|
| bandpass | 0.135 | 0.006 |
| cadzow_orig (rank=4) | 0.277 | 0.021 |
| cadzow_geom (rank=4) | 0.341 | 0.074 |
| svd_adapt | 0.377 | 0.030 |

- CAR (global or per-column) has **zero effect** on CSD metrics — CSD already removes common offsets, so don't bother applying CAR before CSD-based evaluation.
- `global_pca` **hurts** — removes spatially coherent signal.
- `cadzow_geom` beats `cadzow_orig` on both CSD orders. Per-column 1D Cadzow is better than cross-column 2D despite NP1's checkerboard pattern (the mixed-column trajectory matrix is less well-conditioned and inter-column noise contaminates the decomposition).
- `svd_adapt` has the best CSD1 mean but higher variance; `cadzow_geom` is more consistent and wins on CSD2.

### Cadzow rank sweep (`cadzow_np1`, rank × fmax)
- **Rank is the dominant parameter** — lower is better. rank=2 (0.250) > rank=4 (0.214) > rank=8 (0.162).
- fmax (100/200/300 Hz) has negligible effect.
- Sweep was run on `cadzow_np1` only; `cadzow_geom` at rank=2 is the logical next experiment.

## What's next
1. Rank sweep on `cadzow_geom` (expect rank=2 to win again but at higher absolute values).
2. Sequential combination: `cadzow_geom` → `svd_adapt`.
3. Run on dataset 005 to check generalisability.
