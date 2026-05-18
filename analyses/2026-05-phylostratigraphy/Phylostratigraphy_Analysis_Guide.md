# Phylostratigraphy Analysis — Mouse Brain Volumetric Expression

## Overview

This analysis assigns evolutionary ages to mouse brain gene expression data from the Allen Gene Expression Atlas (AGEA), then visualises how expression from each phylogenetic age group is spatially distributed across the brain.

The central question: do anatomically distinct brain regions preferentially express genes that arose at different points in vertebrate evolution?

---

## Phylostratum scheme

Genes are grouped into 12 phylostrata (PS) by the oldest clade in which an ortholog can be detected. PS1 is the most ancient, PS12 the most recent.

| PS | Clade | Age (Mya) |
|----|-------|-----------|
| 1 | Cellular organisms | 3500 |
| 2 | Eukaryota | 1500 |
| 3 | Opisthokonta | 1000 |
| 4 | Metazoa | 650 |
| 5 | Bilateria | 550 |
| 6 | Chordata | 525 |
| 7 | Vertebrata | 500 |
| 8 | Gnathostomata | 450 |
| 9 | Tetrapoda | 350 |
| 10 | Mammalia | 200 |
| 11 | Euarchontoglires | 90 |
| 12 | Rodentia | 25 |

**Important caveat:** Ensembl Compara does not include prokaryotes or most unicellular eukaryotes. As a result, genes shared with all life (true PS1) or all eukaryotes (true PS2) have no detectable ortholog in the database and end up assigned to PS3–PS5. The displayed PS labels are therefore systematically 2–4 strata younger than the true evolutionary age. The relative ordering across phylostrata is preserved, but the absolute labels should be interpreted with caution.

In practice, PS1 and PS2 are empty in our cache; PS3 (Opisthokonta) is the oldest populated stratum.

---

## Step 1 — Fetch gene evolutionary ages (`phylostratigraphy_pipeline.py`)

For each mouse gene symbol the Ensembl REST API is queried for orthologues:

```
GET https://rest.ensembl.org/homology/symbol/mus_musculus/{gene}
    ?type=orthologues&format=condensed
```

Each homology record contains a `taxonomy_level` field — the most recent common ancestor (MRCA) between mouse and the ortholog species. This taxonomy level is mapped to a phylostratum number via the `TAXONOMY_TO_PS` dictionary. The gene is assigned the **minimum** PS across all detected orthologs, i.e. the oldest clade where a homolog exists.

Results are cached in `gene_ages_cache.parquet` (columns: `gene`, `phylostratum`) to avoid repeat API calls. Genes not found in Ensembl or whose taxonomy levels are not in `TAXONOMY_TO_PS` receive `NaN` and are excluded from downstream analysis.

Observed gene counts per stratum in the AGEA dataset (~4 000 genes):

| PS | N genes |
|----|---------|
| 3 | 859 |
| 5 | 1438 |
| 6 | 348 |
| 7 | 631 |
| 8 | 518 |
| 9 | 72 |
| 10 | 93 |
| 11 | 4 |
| 12 | 27 |

PS1, PS2 and PS4 are empty due to the Ensembl coverage limitation described above.

---

## Step 2 — Sum expression volumes per phylostratum (`sum_volumes_by_age.py`)

Loads the pre-processed AGEA expression volumes via `iblatlas.genomics.agea.load(label='processed')`, which returns:
- `df_genes`: DataFrame with gene metadata
- `gene_vols`: array of shape `(n_genes, 58, 41, 67)` — one 3-D volume per gene
- `atlas_agea`: AllenAtlas object at 200 µm resolution

**Atlas axis order:** `(ML, DV, AP) = (58, 41, 67)`

| Axis | Dimension | Size | Slice type when fixed |
|------|-----------|------|-----------------------|
| 0 | ML | 58 | Sagittal |
| 1 | DV | 41 | Horizontal |
| 2 | AP | 67 | Coronal |

For each phylostratum `ps` (1–12), all gene volumes where `phylostratum == ps` are summed:

```python
summed_volumes[ps - 1] = gene_vols[mask].sum(axis=0)
```

Output: `outputs/summed_volumes_by_phylostratum.npy`, shape `(12, 58, 41, 67)`.

---

## Step 3 — Visualise deviation from baseline (`view_volumes_by_age.py`)

### Normalisation

Each of the 12 summed volumes is z-scored over in-brain voxels:

```python
v = vol[brain_mask]
vol[brain_mask] = (v - v.mean()) / v.std()
```

The voxel-wise mean across all 12 z-scored volumes forms the **baseline**. Each panel then shows the deviation `z-scored volume − baseline`, making it easy to see which phylostrata are regionally enriched or depleted relative to the average across all evolutionary ages.

### Grid display (`view_phylostratum_grid`)

- **Rows:** 12 evenly-spaced slices spanning the full brain along the chosen axis
- **Columns:** Allen atlas annotation (left) + one column per selected phylostratum

PS1, PS2 and PS4 are excluded (empty or near-empty strata). The remaining 9 strata shown are PS3, PS5–PS12.

The Allen atlas reference column is drawn with `AllenAtlas.plot_cslice` / `plot_sslice` using physical coordinates derived from `atlas_agea.bc`:
- **Coronal** (`orientation='coronal'`): slices along AP axis (axis 2); coordinate via `atlas_agea.bc.i2y(ap_idx)`
- **Sagittal** (`orientation='sagittal'`): slices along ML axis (axis 0); coordinate via `atlas_agea.bc.i2x(ml_idx)`

Outputs:
- `outputs/phylostratum_grid_coronal.png`
- `outputs/phylostratum_grid_sagittal.png`

---

## Key files

| File | Purpose |
|------|---------|
| `phylostratigraphy_pipeline.py` | Fetch gene ages from Ensembl, cache to parquet |
| `sum_volumes_by_age.py` | Sum AGEA volumes by phylostratum → `.npy` |
| `view_volumes_by_age.py` | Z-score, subtract baseline, display grid |
| `gene_ages_cache.parquet` | Cached gene → phylostratum assignments |
| `outputs/summed_volumes_by_phylostratum.npy` | Summed expression per PS, shape (12, 58, 41, 67) |
| `outputs/phylostratum_grid_coronal.png` | Coronal grid summary figure |
| `outputs/phylostratum_grid_sagittal.png` | Sagittal grid summary figure |
