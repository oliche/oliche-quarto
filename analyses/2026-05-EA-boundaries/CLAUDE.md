# Multi-Channel Neuropixel LFP Pre-Processing Pipeline

## Context

in addition to our atlas it would also be useful to supply the community with a set of landmarks - specific places in the brain where there is a sharp transition in predicted features that scientists can use to help determine their location(the DG-thalamus LFP landmark from the repro-ephys paper is a nice example of this.)

we could search for more such landmarks by brute force: for each region boundary (could start with cosmos and if it works try beryl next): find all penetrations that go through this region boundary  for each ephys feature: make a plot like fig 3b of the repro-ephys paper (https://elifesciences.org/articles/100840#fig3) run a t-test (or similar) to see if there is a sharp transition across the boundary
if yes (and there are enough penetrations and the effect size is large enough) then add this to the list of landmarks to report (or sort the landmarks by p-value)

and then collect the resulting features into a collection of supplemental figures that we could check and then add to the paper to document our claims about these landmarks.

we could also check the results against some known cases - eg the void-cortex transition, DG-thalamus, white matter to gray matter, ventricle to non-ventricle, etc. we can also do some data splitting / xvalidation to further check the results + make sure the suggested landmarks are real + useful.

## Plan

1. Load the ephys-atlas channel features using this skill: /home/olivier/PycharmProjects/EphysAtlas/paper-ephys-atlas/.claude/skills/load-channel-features-dataframe.md 
2. At the Cosmos level, create a matrix of the counts of the number of transitions across the region boundaries. Channels are first aggregated by depth (axial_um) within each probe — taking the modal Cosmos_id across channels at the same depth — before transitions between adjacent depth levels are detected. The matrix is directional: rows = lower region (deeper, small axial_um), columns = upper region (shallower, large axial_um). `root` (id=997) is split into three subclasses using `Allen_id`: fiber tracts (id=1009), ventricular systems / VS (id=73), and the unlabeled remainder (root). Use `regions.subtree(1009)['id']` and `regions.subtree(73)['id']` to build the membership sets.
3. Display the matrix as a heatmap, burn the diagonal and choose the dynamic color range optimized for the off-diagonal terms
4. Look for sharp transitions for each the boundaries in the matrix, above a certain threshold to determine
5. Document the findings in displays

## Checkpoints

### Cosmos transition count matrix — vintage 2026_W12
`figures/cosmos_transition_matrix_2026_W12.csv` — 15×15 directional count matrix (rows = lower region, columns = upper region). Load with:
```python
import pandas as pd
count_matrix = pd.read_csv('figures/cosmos_transition_matrix_2026_W12.csv', index_col=0)
```
Not trivial to recompute: requires downloading/reading the full features parquet (~380k channels) and re-running the depth-aggregation + transition detection pipeline.

### Feature t-test scan + cross-validation + depth profile plots — vintage 2026_W21

Steps 4–6 completed (2026-05-23) using vintage **2026_W21** (383,232 channels, 43 features, MIN_TRANSITIONS=16).

Outputs:
- `figures/boundary_feature_stats.csv` — one row per (boundary, feature), full dataset
- `figures/boundary_cv_results.csv` — 50/50 probe-split CV results per candidate
- `figures/landmark_crossval.png` — CV scatter plot (fold A vs fold B Cohen's d)
- `figures/landmark_<from>_to_<to>.png` — one multi-feature depth profile per boundary

**47 landmark candidates** (Cohen's d > 0.8, Bonferroni p < 0.01), 7 unique boundaries, **44/47 replicate** in 50/50 probe-split CV:

| Boundary | Best feature | Cohen's d | n trans | n probes | Replicated |
|---|---|---|---|---|---|
| TH → HPF | trough_val | **1.42** | 19 | 19 | Yes (all 29 features) |
| TH → root | trough_val | 1.27 | 94 | 92 | Yes |
| void_fluid → Isocortex | peak_val | 0.93 | 42 | 40 | Yes |
| HB → VS | trough_val | 0.90 | 60 | 55 | Yes |
| TH → VS | peak_val | 0.86 | 28 | 23 | No (n too small: ~11/fold) |
| HPF → fiber tracts | trough_val | 0.85 | 343 | 305 | Yes |
| MB → root | trough_val | 0.82 | 66 | 64 | Yes |

**TH → HPF** is the strongest landmark and almost certainly the DG-thalamus boundary from the repro-ephys paper — it shows up across waveform, LFP power, and CSD features.

Plot design: depth on y-axis centred at 0 (boundary), narrow Allen-atlas-colour region strip on left, one column per significant feature (up to 6), shared y-axis. Inspired by `ephysatlas/reveal.py:figure_01_features_with_histology_columns`.

**Next session — display improvements (step 5 figures):**

The multi-feature depth profile figures (`landmark_<from>_to_<to>.png`) work but need polish:

1. **`tight_layout` warning**: `sharey=True` axes are not compatible with `fig.tight_layout()`. Switch to `fig.subplots_adjust()` or use `constrained_layout=True` at figure creation instead.
2. **Missing data on one side**: for HPF→fiber tracts and MB→root the upper (superficial) region side shows no feature curve — because fiber tracts / root have no spiking activity and features are NaN. Options: shade those bins grey, add a note, or only plot the side that has data.
3. **x-axis label crowding**: feature names with underscores replaced by `\n` sometimes still crowd (e.g. long CSD feature names in TH→HPF). Consider rotating labels 45° or abbreviating.
4. **TH→HPF has 29+ significant features** but we cap at 6 columns. Consider a second figure or a summary strip (e.g. heatmap of Cohen's d per feature) for the full feature set.
5. **Region colour for "root"** (id=997) renders white — add a distinct grey fallback for id=997 in `_CUSTOM_COLORS`.
6. **Quarto page**: write `index.qmd` documenting the method, the count matrix, the landmark table, and embedding the figures.

## General Instructions

Use the following skills: 
- /Users/olivier/Documents/ibl-ai-agent/skills/ibl-anatomy
- /Users/olivier/PycharmProjects/ephys-atlas/paper-ephys-atlas/.claude/skills