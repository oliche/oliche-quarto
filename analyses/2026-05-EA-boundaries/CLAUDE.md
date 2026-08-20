# EA Boundaries — LFP Landmark Detection

## Context

Find sharp transitions in ephys features across brain-region boundaries that scientists can use as anatomical landmarks (e.g. the DG–thalamus LFP landmark from repro-ephys). Reference: [repro-ephys fig 3b](https://elifesciences.org/articles/100840#fig3).

## Completed steps

### 1. Cosmos transition matrix — vintage 2026_W24
`figures/cosmos_transition_matrix_2026_W24.csv` — 15×15 directional count matrix.
Interior root/void/fiber-tracts/VS labels replaced by nearest valid label (NNI); 21,435 positions replaced. Probe extremities preserved.
```python
count_matrix = pd.read_csv('figures/cosmos_transition_matrix_2026_W24.csv', index_col=0)
```

### 2. PSD/CSD dimensionality reduction
`psd_pca_dataframe(df, n_components_psd=2, n_components_csd=2)` from `ephysatlas.features` (implemented in `ibleatools/src/ephysatlas/features.py`).
- **PSD group** (7 features): `rms_lf` + `psd_{lfp,delta,theta,alpha,beta,gamma}` → `psd_pc0, psd_pc1`
- **CSD group** (14 features): `rms_lf_csd`, `psd_{…}_csd`, `rms_lf_csd_diff1`, `psd_{…}_csd_diff1` → `csd_pc0, csd_pc1`

Original PSD/CSD columns are **dropped**; 33 features remain for downstream analysis.

### 3. Depth profile figures
`figures/profiles_<from>_to_<to>.png` — generated for all 15 transitions with count ≥ 50.
Pipeline: load → PCA → `compute_boundary_feature_stats()` → top-6 features by Cohen's d → `plot_boundary_feature_profiles(window_um=1000, sort='depth')`.

## Completed — boundary classifier (`boundary_classifier.py`)

- `HistGradientBoostingClassifier` (GB only — LR/RF dropped), 4-fold stratified CV.
- 26 features, 18 transitions (≥32 insertions), 1714 samples. Overall accuracy **47.1%** vs null ~9.3% (p=0.0099).
- Outputs: confusion matrix, permutation importance heatmaps, SHAP per-class heatmaps, null distribution plot, accuracy CSV.
- Coronal + sagittal brain section plots with auto-best AP/ML per pair, Allen-colour boundary contours (done — see below).

## Completed — Allen colours on boundary contours

`plot_boundary_sections` / `plot_sagittal_boundary_sections` draw `from_acr`/`to_acr` contours in the region's actual Allen hex colour (`regions.hexcolor[...]`), `linewidths=3`, dashed for `to_acr`. Legend `Line2D` colours match.

## Completed — encoding-volume ceiling validation (`boundary_classifier_volume.py`)

Same mean-diff-vector + GB pipeline, but trained on **synthetic virtual probes**: a 200 µm AP/ML grid sampled directly from the brainwide encoding volume (no measured data). 26 transitions (broader set than the 18 real ones — includes `void_fluid`/CSF crossings). **94.5% accuracy, 93.8% balanced accuracy** — confirms the ~47% real-data ceiling is a noise floor, not an information floor.
Companion scripts: `run_plot_vp_profiles.py` (volume counterpart of Step 2 depth profiles), `run_plot_vp_top_view.py` (probe-grid top view + `vp_transition_matrix.png`).

## Completed — depth-resolved interpretability (`boundary_mlp_ig.py`)

GPU MLP on the full depth profile per crossing (50 depth positions × 26 features = 1300-d input), 4-fold CV, Captum Integrated Gradients for per-class (depth × feature) attribution. Outputs in `figures/ig/`: `ig_class_signatures.png`, `ig_depth_profile.png`, `ig_feature_importance.png`, `ig_attributions.pkl`. Complements the SHAP fingerprints (which collapse depth) with a spatially-resolved view.

## Completed — heatmap profiles: measured vs encoding volume

**Scripts:** `run_plot_heatmap_profiles.py` and `run_plot_volume_profiles.py`
**Output:** `figures/heatmap_profiles/`
- `heatmap_{pair}_{ap|ml|rastermap}.png` — measured channel features
- `vol_{pair}_{ap|ml|rastermap}.png` — encoding volume predictions

Both use the **same** `EphysPsdPCA` fit (cached in `cache/psd_pca_2026_W24.pkl`) on 7 PSD + 7 CSD non-diff1 features, so `psd_pc0/pc1` and `csd_pc0/pc1` are on identical axes.
The measured parquet (`cache/df_2026_W24.parquet`) was regenerated with this volume-compatible PCA (drops 7 diff1 CSD features vs previous version).
Features shown in fixed order (`sort_features=False`): `psd_pc0, psd_pc1, csd_pc0, csd_pc1, rms_ap, spike_count, aperiodic_offset, aperiodic_exponent`.
Rastermap sort order is shared between measured/volume (same cache hash).

**Key utilities in `boundaries_utils.py`:**
- `load_or_fit_psd_pca()` — fits and caches the shared PCA
- `load_or_build_pca_df()` — builds/caches the post-PCA measured feature df
- `build_volume_feature_df()` — vectorised volume lookup at channel xyz, applies same PCA
- `plot_boundary_feature_profiles(..., sort_features=False)` — fixed feature order for comparisons

## Next steps

- **SLIC supervoxel segmentation** (`slic_segmentation.py`) — unsupervised 3-D SLIC clustering of the encoding volume, checking whether boundaries emerge without labels. Exploratory: single overview figure (`figures/slic_supervoxels_overview.png`), no quantitative overlap/purity metric against Cosmos boundaries yet — needed before this is report-worthy.
- **Atlas-coverage QC panels** (`run_plot_atlas_coverage_slices.py`, `run_plot_slice_panels.py`) — coronal/sagittal per-feature slices with virtual-probe overlays and recording-coverage masks, in `figures/slice_panels/`. Generated but only lightly referenced in `index.qmd`; consider promoting to a full appendix section.
- `debug_mask.py` / `view_encoding_volume.py` are QC/dev scratch tools (brain-mask diagnostics, interactive volume viewer) — not report material, safe to leave untracked or delete once the mask issue they were diagnosing is confirmed resolved.
- `psd_pca_study.py` was rewritten independently of `index.qmd`'s Step 3 figure and its output filenames no longer matched (`psd_pca_04_scatter_cosmos.png` was missing from `figures/`, orphaned from the current script). Fixed by regenerating that exact figure from the cached `cache/df_2026_W24.parquet` + `cache/psd_pca_2026_W24.pkl` (no re-download needed). If `psd_pca_study.py` is run again, reconcile its new filenames (`psd_pca_0{1..4}_*_comparison.png`) with the qmd reference, or update the qmd to point at its outputs directly.
