# Plan: which brain regions are electrophysiologically distinctive from their neighbours?

## Question

Using the brainwide ephys-atlas encoding volume (41 voxel-wise features: LFP power/CSD/residual
bands, spectral/aperiodic shape, spike-waveform shape, spike amplitude/rate), rank brain regions
by how different their local feature distribution is from the tissue immediately surrounding them
— not from the whole-brain average, and not from taxonomically-similar regions elsewhere in the
brain. "Neighbour" means physically adjacent in the CCF volume.

Motivating examples (given, to validate against): Piriform (PIR) and hippocampal fields are
distinctive in LFP-derived features from their surrounding cortex/subcortex; whole cerebellum
(CB) is distinctive in `alpha_mean` from the brainstem it borders.

## Data

`brainwide_ephys_atlas_50um.npz` — `ephys_atlas_vol` (228, 264, 160, 41) float16, axis order
(ML, AP, DV, feature). No NaNs anywhere, including outside the brain — values there are
extrapolated/smoothed, not real, so every step below restricts to in-brain, in-mask voxels only
via `ephysatlas.anatomy.ClassifierAtlas(res_um=50).mask()`.

**Axis alignment** (confirmed against prior art in `2026-05-EA-boundaries/view_encoding_volume.py`):
atlas volumes (`ba.label`, `ba.image`) are (AP, ML, DV); the encoding volume is (ML, AP, DV, feat).
Align with `np.transpose(vol, (1, 0, 2, 3))` before combining with `ba.label`.

**Atlas**: `ephysatlas.anatomy.ClassifierAtlas(res_um=50)`, not plain `AllenAtlas` — it relabels
void voxels trapped inside the skull to a `void_fluid` region so they don't get miscounted as
"outside brain". `root` (unassigned in-mask voxels, mostly registration-edge noise) and `void`/
`void_fluid` are excluded from every regional statistic and from every neighbour pool — an early
prototype that left `root` in the neighbour pool produced spuriously large effect sizes (root's
mean feature values are far from any real tissue's), so this exclusion is load-bearing, not
cosmetic.

## Parcellation resolution

Run the whole pipeline at two mapping levels rather than picking one, since the two motivating
examples live at different levels themselves (PIR is a Beryl-level leaf; CB is a Cosmos-level
grand division that fragments into ~20 individual lobules/nuclei at Beryl level, none of which is
very different from its lobule neighbours — the "cerebellum is different" signal only exists at
the coarser level):

- **Cosmos** (13 regions): coarse, whole-structure contrasts (CB vs MB/HB, HPF vs Isocortex, etc).
- **Beryl** (~309 regions): the field's standard analysis granularity; captures PIR, CA1/CA3/DG,
  individual cerebellar lobules, etc. as separate units.

## Method

1. **Distance-buffer neighbours (current, revised after discussion — see below).** For each
   region, a Euclidean distance transform (isotropic 50 µm grid, so voxel units = physical units)
   gives every other in-mask voxel's distance to the region's boundary, cropped to a bounding box
   padded by the buffer radius for speed. Full weight out to 5 voxels (250 µm), a raised-cosine
   taper to zero between 5 and 10 voxels (500 µm), zero beyond. This is pure geometry, so distance
   tunnels through thin gaps (e.g. a ventricle) — two regions facing each other across such a gap
   still count as neighbours if within the buffer radius. `root`/`void`/`void_fluid` never
   contribute weight but don't block the geometry. Superseded a first version — a 6-connectivity
   voxel-touching adjacency graph (extending `ephysatlas.anatomy.regions_transition_matrix`, which
   only covered the DV axis) that pooled each entire neighbouring region unweighted — kept in
   `distinctiveness.py` for comparison. The plateau/cutoff radii (5/10 voxels) are a starting point
   to be swept later, not a fitted or principled choice.
2. **Per-region sufficient statistics.** One pass over all in-mask voxels accumulates, per region
   and per feature: n, sum, sum-of-squares (own-region stats stay unweighted; only the neighbour
   side uses the distance weights). Cheap to recompute per mapping.
3. **Distinctiveness = feature-wise Cohen's d** between a region (its own full voxel population)
   and its distance-weighted neighbour buffer (weighted mean/variance). This is the effect-size
   language the motivating examples are phrased in ("distinctive for X feature"), so a region's
   result is directly interpretable per feature, not just as an opaque composite.
4. **Composite score**, to get a single ranking without letting the ~20 mutually-correlated LFP
   power/CSD/residual bands dominate a naive average: group the 41 features into ~8 families
   (lfp_power, lfp_csd, lfp_residual, spectral_shape, ap_amplitude, waveform_timing,
   waveform_amplitude, waveform_slope), take the RMS |d| within each family, then RMS across
   families. Report alongside the single max-|d| feature per region, since that's what a reader
   will actually want to know ("distinctive *in what*").
5. Rank all regions per mapping by the composite score; report top-N with their driving
   feature(s)/family, plus the full per-feature d matrix for figures.

### Neighbour-definition discussion (2026-09-09)

Before starting the planned PCA/Mahalanobis extension, we revisited the neighbour definition:

- **Bulk-region vs. boundary-shell pooling.** Chose a boundary-shell/buffer over pooling each
  entire neighbouring region, since the latter (the original adjacency-graph version) diluted a
  region's true edge contrast against a large neighbour's far side — this was the main reason
  piriform ranked only 70/305 under the graph definition despite matching the motivating example
  qualitatively.
- **Buffer shape.** Settled on plateau-then-taper (full weight 0–5 voxels, cosine taper 5–10,
  zero beyond) rather than a zero-at-both-ends Hann bump — simpler, and matches the more literal
  reading of "distinctive from the tissue immediately around it."
- **No minimum-adjacency threshold, replaced by geometric distance instead of a touching-graph.**
  A minimum shared-voxel-face count was considered to filter spurious single-voxel/corner
  adjacencies, but the distance-buffer approach sidesteps the question entirely: adjacency is no
  longer graph-based, and any two regions facing each other within the buffer radius are
  neighbours regardless of whether they literally touch (this is what lets regions separated by a
  ventricle still count as neighbours).
- **Granularity.** Confirmed the atlas's own painted partition (`ba.label`, ~1337 usable ids) is
  already the finest available granularity — it is not a uniform leaf level, since Allen's
  annotation itself mixes hierarchy depths voxel-by-voxel (many of the 2657 ids in the full
  ontology are non-leaf ancestor nodes that never get their own voxels). Running the same
  procedure at every possible hierarchy roll-up level was explicitly ruled out as too expensive;
  Cosmos/Beryl/Allen-painted are the three levels reported.

Net effect: PIR moved from rank 70/305 (adjacency-graph, Beryl) to rank 8/306 (distance-buffer,
Beryl) — a much closer match to the motivating example. CB remains #1/10 at Cosmos level either
way. See `index.qmd` for full validation figures and numbers.

## PCA / Mahalanobis extension (2026-09-09)

Added `pca_mahalanobis.py` / `run_mahalanobis.py` / `make_figures_pca.py`: fit a global PCA basis
(z-scored, all in-mask non-excluded voxels), keep components to 95% variance, and rank regions by
`sqrt(sum_k d_k^2)` over per-PC Cohen's d (region vs. distance-buffer neighbours, same buffer as
the Cohen's-d pipeline) — a Mahalanobis distance replacing the hand-picked 8-feature-family
composite with a data-driven one. Reprojects each region's PC-space difference back onto the
original 41 features for interpretability.

**Bug caught while building this**: a PC1-vs-PC2 scatter of region vs. neighbour-buffer voxels
showed one extreme outlier; tracing it back found the voxel had all 41 features exactly 0.0 — an
unfilled "no probe coverage" sentinel, not real data. 46,950 in-mask voxels (1.2% of valid voxels)
turned out to be like this, concentrated in specific structures (RSPv, RSPd, ACAd, MOs, PL, MOB,
ENTm, COAp, PAG, PFL, PAR, SCs) rather than spread randomly. Added
`distinctiveness.no_coverage_mask()` and threaded an `extra_exclude` parameter through
`region_index_volume`, `rank_regions`, `rank_regions_buffer`, `mahalanobis_rank_regions` and
`pca_mahalanobis.in_mask_valid` so these voxels are excluded from every region's own statistics,
every neighbour buffer, and the global PCA fit. This fully explained why `PVa`/`PVi` (small
periventricular structures right next to the third ventricle — exactly where coverage gaps
concentrate) had topped the Beryl/Allen rankings in the pre-fix version; after the fix, neither
ranks in the top 20 at either level. PIR/DG/CB (the three validated examples) were only mildly
affected by the fix, which is reassuring. Re-ran and re-validated the entire Cohen's-d pipeline
after adding this exclusion, not just the new PCA path — `index.qmd`'s "Data-quality fix" callout
has the full numbers.

Also notable: the pre-fix PCA had PC1 alone absorbing 77% of variance (a spurious "zero vs. real
data" axis); post-fix, variance is spread more evenly across the first 4 components (37/22/20/11%)
and 6 components are needed for 95% — a much more plausible picture of the real LFP/CSD/waveform
feature-correlation structure, and a good illustration of how an artefact-driven axis can dominate
a naively-run PCA.

## Minimum-region-size floor (2026-09-09)

User feedback after reviewing the top-20 table: several top hits were suspiciously tiny nuclei.
Added `min_voxels` to `rank_regions_buffer`/`mahalanobis_rank_regions`: a region needs >= 500 own
voxels (~0.06 mm^3, a ~400 um cube at 50 um) to appear as a *ranked row*, though it can still serve
as neighbour tissue for others regardless of size (a tiny nucleus is still real tissue, just not a
reliable target of its own effect-size estimate). 500 is a round, defensible starting choice
(excludes 16%/30% of Beryl/Allen regions, keeps the ranking substantial) rather than a fitted one.

Effect: `SFO` (145 voxels), the post-no-coverage-fix #1 Beryl hit, drops out; `IO` becomes #1.
PIR/DG both improve (PIR: Beryl rank 8->6/257; Mahalanobis-Beryl rank 18->12/257. DG: Mahalanobis
rank 19->13/257) simply because noisy tiny regions no longer inflate the denominator of "how many
regions rank above them."

**Second bug caught while re-plotting after this change**: the Allen-level top-20 bar chart came
out visibly broken — bars and text-annotation labels were desynced for roughly a third of the
rows. Root cause: several Allen-painted regions share acronym text across genuinely distinct raw
atlas ids (confirmed by inspection — e.g. two `PIR` ids each with ~46,287 voxels spanning both
hemispheres almost identically; not a lateralisation split, a real ontology property).
`sns.barplot(y="acronym", ...)` silently *aggregates* rows with equal categorical values, which
desynced the bar count from the row count and corrupted the positional text-annotation loop. A
second, silent (non-crashing) version of the same root cause existed in the reprojection heatmap:
`acronyms.index(a)` always returns the *first* match for a duplicated acronym, so two different
regions named e.g. `V` would have shown identical reprojected data. Fixed by: (1) switching both
bar charts to `ax.barh` with explicit integer y-positions instead of a seaborn categorical axis,
and (2) adding an explicit `region_idx` column to every ranking dataframe (the internal, always-
unique region index) and using that instead of acronym-text lookups in the reprojection heatmap.
Re-verified visually after the fix: all 20 bars/labels align correctly, and duplicate-acronym rows
now show visibly different (not identical) reprojected values.

## Validation against the given examples (done during prototyping, holds up)

- CB vs {MB, HB} (Cosmos, neighbours excluding root/void): `alpha_mean` Cohen's d = **1.73**.
- PIR (Beryl) vs its 12 real neighbours (ORBl, AIp/AIv, AON, PAA, TR, ENTl, EPd/EPv, BLA, OT, SI):
  driven less by raw LFP power (d ≈ 0.3–0.6) than by CSD-derivative and residual LFP bands
  (d up to 1.6–1.3), `alpha_mean`/`alpha_std` (d ≈ 1.3), and `cor_ratio` (d ≈ 1.1) — a more precise
  version of "distinctive in LFP features" than the raw-power framing suggests.
- DG (Beryl) top feature is `rms_lf_csd_diff1` (d ≈ 1.28) — CSD-derived, consistent with
  hippocampal theta/sharp-wave-ripple laminar structure being a CSD phenomenon.
- Sanity check on plausibility: many top-ranked regions by the global composite are pallidal/
  nigral output nuclei (GPe, GPi, SNr) and other brainstem nuclei, driven by waveform-timing/slope
  features — consistent with the well-documented fast, narrow-spike phenotype of these cell
  populations. This is an unprompted independent confirmation that the metric tracks real biology,
  not registration artefacts (registration-artefact-driven hits would look like small, low-n,
  boundary-adjacent regions with no consistent feature story — that's a diagnostic to keep
  watching for in the full run).

## Deliverables

- `distinctiveness.py`: adjacency graph + sufficient statistics + Cohen's d + composite score,
  parametrised by mapping (`Cosmos`/`Beryl`).
- Ranked tables (Cosmos and Beryl) with driving feature/family, saved for the report.
- Figures: (a) metric-definition/validation figure reproducing the PIR/CB checks above, (b) top-N
  bar chart of composite scores per mapping, (c) region x feature-family effect-size heatmap for
  top hits, (d) brain-slice or Swanson highlight of the top regions.
- `index.qmd` report in this folder, Diátaxis-light: motivation, method, validation, results,
  caveats (small-n regions, single fixed aggregate volume so no replicate/significance testing —
  this is descriptive effect-size ranking, not a hypothesis test).

## Scope note on statistical framing

This is a **descriptive/exploratory** ranking over one already-aggregated atlas volume (itself
pooled across many recordings upstream), not a hypothesis test over experimental replicates — there
is no natural per-subject/per-session split available inside this file to hold out for confirmatory
testing. No p-values are reported; effect sizes (Cohen's d) and cross-checks against known biology
are the validation currency. If a future version needs inferential guarantees, that requires
going back to session/PID-level features and treating recordings as replicates, not this volume.

## Rescoped to pairwise region x region Mahalanobis distance (2026-09-10)

Restarted with a much narrower scope: dropped the entire neighbour-definition problem (distance
buffer, adjacency graph, plateau/taper weighting) and the hierarchy-aggregation machinery
(`aggregate_hierarchy_stats`, `mirror_aware_descendants`, per-node distance-buffer Cohen's d) —
none of that framing is needed for the new question. Dropped because: the neighbour-buffer method
answered "how different is a region from the tissue immediately around it," which required an
arbitrary geometric definition of "neighbour" (buffer radius, taper shape) that was never fully
settled and needed a hierarchy walk to make sense of coarse vs. fine parcellation levels at once.
The new question — "how different is each region from every *other* region" — needs neither: a
plain pairwise distance between all sufficiently-sampled regions sidesteps the neighbour-definition
problem entirely and is far cheaper to compute (no distance transform, no per-node bounding box).

**Kept unchanged**: the global PCA fit (`pca_mahalanobis.fit_global_pca`/`project_onto_pcs`) and its
scree plot — the whitening basis itself never depended on the neighbour-buffer framing.

**New pipeline** (`pairwise_mahalanobis.py`): select every named region in the atlas's raw painted
partition (`ba.label`) with >500 directly-painted voxels, pooling each region's two raw-id
hemisphere "twins" into one row (`region_stats_table.pool_by_acronym` — confirmed both twins hold
real, independent, roughly-equal voxel counts rather than one being a near-empty duplicate, e.g.
PIR: 46286 vs 46289) — 534 regions clear the bar. Compute the full 534×534 Mahalanobis distance
matrix in PCA-whitened space (`region_stats_table.mahalanobis_matrix`, vectorised over all pairs at
once) and visualise as a heatmap with each region's Allen atlas colour shown as a strip along the
top/left instead of text tick labels (too many regions for legible per-row text at this scale).

**Removed**: `distinctiveness.py`'s neighbour-buffer/adjacency-graph pipeline, `run_ranking.py`,
`make_figures.py`, `make_figures_hierarchy.py`, `pca_mahalanobis.mahalanobis_rank_regions`,
`run_mahalanobis.py`, and most of `region_stats_table.py`'s hierarchy-aggregation code — only the
generic, grouping-agnostic pieces (`direct_paint_stats`, `pool_stats`, `cohens_d_between`,
`mahalanobis_between`, `fit_and_project_pca`) survived, plus two new small additions
(`pool_by_acronym`, `mahalanobis_matrix`). `index.qmd` was rewritten from scratch to cover only the
PCA method + scree plot and the new pairwise matrix figure.
