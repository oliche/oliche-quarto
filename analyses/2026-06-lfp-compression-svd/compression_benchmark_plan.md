# LFP Compression Benchmark — SVD Methods

## Scope

Benchmark SVD-based LFP compression on 11 Neuropixels BWM insertions (Cadzow-denoised snippets).
Evaluate RMSE and SNR across compression ratios up to ~3000×.

## Methods (6)

| Name | Description | CR formula | Sweep |
|---|---|---|---|
| SVD | Truncated SVD, fixed rank | nc·ns / (r·(nc+ns)) | r ∈ {1,2,4,8,16,32,64,96} |
| SVD-adapt | SVD with noise-floor rank selection: keep sv > ε × median(lower half of non-trivial SVs) | nc·ns / (r_adapt·(nc+ns)) | ε ∈ logspace(10⁴, 10⁻¹, 30) |
| SVD-adapt-CR16-WP | SVD fixed at CR≈16, then db4 wavelet-packet thresholding on temporal rows | CR16 / f | f ∈ {0.005…0.50} |
| SVD-adapt-CR24-WP | As above, SVD stage at CR≈24 | CR24 / f | f ∈ {0.005…0.50} |
| SVD-adapt-CR32-WP | As above, SVD stage at CR≈32 | CR32 / f | f ∈ {0.005…0.50} |
| SVD-adapt-CR48-WP | As above, SVD stage at CR≈48 | CR48 / f | f ∈ {0.005…0.50} |

SVD_ADAPT_CRS = [16, 24, 32, 48] controls the WP variants; NAMES derives from it automatically.

## Key implementation notes

**Noise floor for SVD-adapt**: computed from `sv[sv > sv[0] * 1e-4]` (non-trivial SVs only).
Using `median(sv[192:])` is unreliable because the lower half of the SV spectrum decays to near-machine-epsilon on these recordings — even at ε=1000 the threshold would only reach rank≈15. The non-trivial floor is ~1e-5, and ε up to 10⁴ is needed to reach rank=1.

**SVD-adapt vs SVD in RMSE-vs-CR plots**: the two curves will overlay. Both are truncated SVD; at equal rank the reconstruction is identical. SVD-adapt's value is automated rank selection for deployment, not improved quality.

**Gallery cell labels**: WP methods show `CR=X\nSVD×Y · WP×Z` (combined + breakdown). Cells where the target CR is below the SVD stage floor are greyed out as N/A.

## Outputs

- Cache: `metrics_v2_{pid}.npz` per PID (`/Users/olivier/scratch/lfp/{pid}/`)
- Figures: `2026-06-09_compression_metrics_{pid}.png`, `_gallery_{pid}.png`, `_gallery_{pid}_residual.png`
- Aggregate: `2026-06-09_compression_aggregate.png`
- Report: `index.qmd` → built into the Analysis Hub quarto site