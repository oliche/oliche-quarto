"""
Phylostratigraphy pipeline for mouse brain volumetric gene expression (AGEA atlas).

Steps:
  1. Load AGEA gene expression volumes + atlas
  2. Fetch gene evolutionary ages (phylostrata) from Ensembl Compara, with caching
  3. Compute per-voxel TAI (Transcriptome Age Index)
  4. Aggregate TAI by brain region
  5. Visualize and export results

Usage:
    python phylostratigraphy_pipeline.py
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from iblatlas.genomics import agea

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / 'outputs'
CACHE_FILE = Path(__file__).parent / 'gene_ages_cache.parquet'

PHYLOSTRATA = {
    1:  {'name': 'Cellular_organisms',  'age_mya': 3500},
    2:  {'name': 'Eukaryota',           'age_mya': 1500},
    3:  {'name': 'Opisthokonta',        'age_mya': 1000},
    4:  {'name': 'Metazoa',             'age_mya':  650},
    5:  {'name': 'Bilateria',           'age_mya':  550},
    6:  {'name': 'Chordata',            'age_mya':  525},
    7:  {'name': 'Vertebrata',          'age_mya':  500},
    8:  {'name': 'Gnathostomata',       'age_mya':  450},
    9:  {'name': 'Tetrapoda',           'age_mya':  350},
    10: {'name': 'Mammalia',            'age_mya':  200},
    11: {'name': 'Euarchontoglires',    'age_mya':   90},
    12: {'name': 'Rodentia',            'age_mya':   25},
}

# Map Ensembl taxonomy_level strings → phylostratum number.
# The minimum PS across all orthologs of a gene gives its evolutionary age.
TAXONOMY_TO_PS = {
    # Pre-vertebrate
    'Cellular_organisms': 1,
    'Eukaryota': 2, 'Fungi_Metazoa_group': 2,
    'Opisthokonta': 3,
    'Metazoa': 4,
    'Bilateria': 5, 'Deuterostomia': 5,
    # Chordate / vertebrate
    'Chordata': 6,
    'Vertebrata': 7, 'Craniata': 7,
    'Gnathostomata': 8, 'Euteleostomi': 8, 'Actinopterygii': 8,
    'Clupeocephala': 8, 'Neopterygii': 8, 'Teleostei': 8,
    # Tetrapod / amniote
    'Tetrapoda': 9, 'Amniota': 9, 'Sarcopterygii': 9,
    'Reptilia': 9, 'Sauropsida': 9, 'Archelosauria': 9,
    # Mammalian
    'Mammalia': 10, 'Theria': 10, 'Eutheria': 10,
    'Atlantogenata': 10, 'Boreoeutheria': 10,
    # Euarchontoglires
    'Euarchontoglires': 11, 'Primates': 11, 'Glires': 11,
    'Lagomorpha': 11, 'Scandentia': 11,
    # Rodentia-specific
    'Rodentia': 12, 'Muroidea': 12, 'Murinae': 12,
    'Mus': 12, 'Mus_musculus_domestic': 12,
}

ENSEMBL_REST = 'https://rest.ensembl.org'
HEADERS = {'Content-Type': 'application/json'}


# ---------------------------------------------------------------------------
# Step 1 – Gene age assignment (Ensembl Compara, parallel, cached)
# ---------------------------------------------------------------------------

def _query_ensembl_gene(gene: str, retries: int = 2) -> int | None:
    """Return phylostratum (1-12) for one mouse gene symbol, or None on failure."""
    url = f'{ENSEMBL_REST}/homology/symbol/mus_musculus/{gene}'
    params = {'type': 'orthologues', 'format': 'condensed'}
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                homologies = resp.json()['data'][0]['homologies']
                ps_values = [
                    TAXONOMY_TO_PS[h['taxonomy_level']]
                    for h in homologies
                    if h['taxonomy_level'] in TAXONOMY_TO_PS
                ]
                return min(ps_values) if ps_values else 12  # default: rodent-specific
            if resp.status_code == 400:
                return None  # gene not found
            time.sleep(1 + attempt)
        except Exception as exc:
            log.debug(f'  {gene} attempt {attempt}: {exc}')
            time.sleep(1 + attempt)
    return None


def fetch_gene_ages(gene_symbols: list[str], n_workers: int = 15) -> pd.DataFrame:
    """
    Return a DataFrame with columns [gene, phylostratum] for all supplied symbols.
    Loads from CACHE_FILE if present; fetches from Ensembl otherwise and caches result.
    """
    if CACHE_FILE.exists():
        cached = pd.read_parquet(CACHE_FILE)
        missing = [g for g in gene_symbols if g not in cached['gene'].values]
        if not missing:
            log.info(f'Gene ages loaded from cache ({len(cached)} genes)')
            return cached[cached['gene'].isin(gene_symbols)].reset_index(drop=True)
        log.info(f'Cache hit for {len(cached)} genes; fetching {len(missing)} new genes')
    else:
        cached = pd.DataFrame(columns=['gene', 'phylostratum'])
        missing = gene_symbols

    log.info(f'Fetching phylostrata for {len(missing)} genes using {n_workers} workers ...')
    results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_query_ensembl_gene, g): g for g in missing}
        done = 0
        for fut in as_completed(futures):
            gene = futures[fut]
            results[gene] = fut.result()
            done += 1
            if done % 100 == 0:
                log.info(f'  {done}/{len(missing)} genes done')

    new_rows = pd.DataFrame({
        'gene': list(results.keys()),
        'phylostratum': [float(v) if v is not None else float('nan')
                         for v in results.values()],
    })
    combined = pd.concat([cached, new_rows], ignore_index=True)
    combined.to_parquet(CACHE_FILE)
    log.info(f'Saved gene age cache ({len(combined)} total genes)')
    return combined[combined['gene'].isin(gene_symbols)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 – TAI computation
# ---------------------------------------------------------------------------

def compute_tai(expression_data: np.ndarray, phylostrata: np.ndarray,
                method: str = 'log') -> np.ndarray:
    """
    Compute TAI = Σ(ps_i × expr_i) / Σ(expr_i) per voxel.

    Parameters
    ----------
    expression_data : (n_genes, x, y, z)
    phylostrata     : (n_genes,)  float, NaN for unknown ages
    method          : 'log' | 'raw' | 'binary'

    Returns
    -------
    tai_volume : (x, y, z)
    """
    valid = ~np.isnan(phylostrata)
    expr = expression_data[valid].astype(np.float32)
    ps = phylostrata[valid].astype(np.float32)

    n_genes, x, y, z = expr.shape
    expr_flat = expr.reshape(n_genes, -1)

    with np.errstate(divide='ignore', invalid='ignore'):
        if method == 'log':
            # Clip to 0 first: negative values are below-background artefacts
            weights = np.log2(np.maximum(expr_flat, 0) + 1)
        elif method == 'binary':
            thr = np.percentile(expr_flat, 50, axis=1, keepdims=True)
            weights = (expr_flat > thr).astype(np.float32)
        else:
            weights = expr_flat

        numerator = (ps[:, None] * weights).sum(axis=0)
        denominator = weights.sum(axis=0)
        denominator[denominator == 0] = np.nan
        tai_flat = numerator / denominator
    return tai_flat.reshape(x, y, z)


# ---------------------------------------------------------------------------
# Step 3 – Regional aggregation
# ---------------------------------------------------------------------------

def compute_regional_tai(tai_volume: np.ndarray, atlas) -> pd.DataFrame:
    """
    Aggregate TAI statistics per anatomical region using the atlas label volume.
    Returns DataFrame sorted by tai_mean (oldest first).
    """
    label_vol = atlas.label
    regions = atlas.regions

    records = []
    unique_ids = np.unique(label_vol)
    unique_ids = unique_ids[unique_ids != 0]  # skip void

    for rid in unique_ids:
        mask = label_vol == rid
        tai_vals = tai_volume[mask]
        tai_vals = tai_vals[~np.isnan(tai_vals)]
        if len(tai_vals) == 0:
            continue
        # Map region id to acronym / name
        idx = np.where(regions.id == rid)[0]
        acronym = regions.acronym[idx[0]] if len(idx) else str(rid)
        name = regions.name[idx[0]] if len(idx) else str(rid)
        records.append({
            'region_id': int(rid),
            'acronym': acronym,
            'name': name,
            'tai_mean': float(np.mean(tai_vals)),
            'tai_std': float(np.std(tai_vals)),
            'tai_median': float(np.median(tai_vals)),
            'n_voxels': int(mask.sum()),
        })

    if not records:
        return pd.DataFrame(columns=['region_id', 'acronym', 'name', 'tai_mean',
                                     'tai_std', 'tai_median', 'n_voxels'])
    df = pd.DataFrame(records).sort_values('tai_mean').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 4 – Validation
# ---------------------------------------------------------------------------

def validate_tai(regional_tai: pd.DataFrame) -> pd.DataFrame:
    """
    Check expected evolutionary ordering of brain regions at 200µm resolution.
    Neocortex should have higher TAI than brainstem per Belgard et al. 2013.
    Uses broad acronyms available in the coarse AGEA atlas.
    """
    tests = {
        'Cortex > Brainstem': {
            'newer': ['VISrl', 'ILA', 'ILA5', 'ILA6b', 'MOs', 'MOp'],
            'older': ['SNr', 'TRN', 'P', 'PAG', 'MRN'],
        },
        'Hippocampus > Thalamus': {
            'newer': ['HPF', 'CA1slm', 'CA1so', 'DGmb-mo'],
            'older': ['PR', 'mfbst', 'RT'],
        },
        'Striatum > White matter': {
            'newer': ['CP', 'OT1', 'HEM'],
            'older': ['tspc', 'fa'],
        },
    }

    acr = regional_tai.set_index('acronym')['tai_mean']
    rows = []
    for name, grps in tests.items():
        newer_vals = acr[acr.index.isin(grps['newer'])].values
        older_vals = acr[acr.index.isin(grps['older'])].values
        newer_tai = newer_vals.mean() if len(newer_vals) else np.nan
        older_tai = older_vals.mean() if len(older_vals) else np.nan
        rows.append({
            'test': name,
            'newer_tai': round(newer_tai, 3),
            'older_tai': round(older_tai, 3),
            'diff': round(newer_tai - older_tai, 3),
            'passed': bool(newer_tai > older_tai),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3b – Variance-weighted TAI
# ---------------------------------------------------------------------------

def calculate_variance_weighted_tai(expression_data: np.ndarray,
                                     phylostrata: np.ndarray) -> np.ndarray:
    """
    TAI weighted by each gene's spatial variance across the brain.
    Down-weights ubiquitous housekeeping genes; amplifies regionally-specific ones.
    """
    valid = ~np.isnan(phylostrata)
    expr = np.maximum(expression_data[valid].astype(np.float32), 0)
    ps = phylostrata[valid].astype(np.float32)

    n_genes, x, y, z = expr.shape
    expr_flat = expr.reshape(n_genes, -1)

    gene_var = expr_flat.var(axis=1)
    var_min, var_max = gene_var.min(), gene_var.max()
    var_weights = (gene_var - var_min) / (var_max - var_min + 1e-10)

    log_expr = np.log2(expr_flat + 1)
    combined_weights = var_weights[:, None] * log_expr

    numerator = (ps[:, None] * combined_weights).sum(axis=0)
    denominator = combined_weights.sum(axis=0)
    denominator[denominator == 0] = np.nan

    return (numerator / denominator).reshape(x, y, z)


# ---------------------------------------------------------------------------
# Step 5 – Visualization
# ---------------------------------------------------------------------------

def _ps_label(ps: int) -> str:
    return f"PS{ps} {PHYLOSTRATA[ps]['name']}" if ps in PHYLOSTRATA else str(ps)


def plot_gene_age_distribution(df_ages: pd.DataFrame, output_dir: Path) -> None:
    valid = df_ages.dropna(subset=['phylostratum'])
    counts = valid['phylostratum'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values, color='steelblue', edgecolor='white')
    ax.set_xticks(list(PHYLOSTRATA.keys()))
    ax.set_xticklabels([_ps_label(p) for p in PHYLOSTRATA], rotation=40, ha='right', fontsize=8)
    ax.set_xlabel('Phylostratum')
    ax.set_ylabel('Number of genes')
    ax.set_title(f'Gene age distribution  (n={len(valid)}/{len(df_ages)} genes assigned)')
    plt.tight_layout()
    plt.savefig(output_dir / 'gene_age_distribution.png', dpi=150)
    plt.close()
    log.info('Saved gene_age_distribution.png')


def plot_regional_tai(regional_tai: pd.DataFrame, output_dir: Path, top_n: int = 40) -> None:
    df = regional_tai.dropna(subset=['tai_mean'])
    n = min(top_n, len(df))
    # show the n/2 oldest + n/2 youngest
    half = n // 2
    subset = pd.concat([df.head(half), df.tail(half)]).drop_duplicates()

    cmap = plt.cm.RdYlBu_r
    norm_vals = (subset['tai_mean'] - subset['tai_mean'].min()) / (
        subset['tai_mean'].max() - subset['tai_mean'].min() + 1e-9
    )
    colors = [cmap(v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(10, max(6, len(subset) * 0.25)))
    y = np.arange(len(subset))
    ax.barh(y, subset['tai_mean'], xerr=subset['tai_std'], color=colors,
            ecolor='grey', capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(subset['acronym'], fontsize=8)
    ax.set_xlabel('Transcriptome Age Index (TAI)')
    ax.set_title('Brain region TAI — oldest (bottom) to youngest (top)')
    ax.axvline(df['tai_mean'].median(), color='k', linestyle='--', lw=1, label='median')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'regional_tai_barplot.png', dpi=150)
    plt.close()
    log.info('Saved regional_tai_barplot.png')


def plot_tai_slices(tai_volume: np.ndarray, output_dir: Path) -> None:
    plot_tai_slices_named(tai_volume, output_dir, 'tai_brain_slices',
                          'Mouse brain Transcriptome Age Index')


def plot_tai_slices_named(tai_volume: np.ndarray, output_dir: Path,
                          name: str, title: str) -> None:
    """Plot coronal, horizontal, and sagittal slices through a TAI volume."""
    vmin, vmax = np.nanpercentile(tai_volume, [5, 95])
    x, y, z = tai_volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [
        (tai_volume[x // 2, :, :], 'Coronal (ML mid)'),
        (tai_volume[:, y // 2, :], 'Horizontal (DV mid)'),
        (tai_volume[:, :, z // 2], 'Sagittal (AP mid)'),
    ]
    for ax, (sl, subtitle) in zip(axes, slices):
        im = ax.imshow(sl.T, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, origin='lower',
                       interpolation='nearest')
        ax.set_title(subtitle)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='TAI', shrink=0.8)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}.png', dpi=150)
    plt.close()
    log.info(f'Saved {name}.png')


def plot_phylostratum_heatmap(expression_data: np.ndarray, phylostrata: np.ndarray,
                               atlas, output_dir: Path, top_regions: int = 30) -> None:
    """Heatmap: proportion of expression from each phylostratum per brain region."""
    label_vol = atlas.label
    regions = atlas.regions

    unique_ids = np.unique(label_vol)
    unique_ids = unique_ids[(unique_ids != 0)]

    n_ps = 12
    records = []
    for rid in unique_ids:
        mask = label_vol == rid
        if mask.sum() < 5:
            continue
        idx = np.where(regions.id == rid)[0]
        acronym = regions.acronym[idx[0]] if len(idx) else str(rid)
        row = {'acronym': acronym}
        expr_region = expression_data[:, mask].mean(axis=1).astype(np.float32)
        total = expr_region.sum() + 1e-9
        for ps in range(1, n_ps + 1):
            ps_mask = phylostrata == ps
            row[f'PS{ps}'] = float(expr_region[ps_mask].sum() / total)
        records.append(row)

    df_heat = pd.DataFrame(records).set_index('acronym')
    ps_cols = [f'PS{p}' for p in range(1, n_ps + 1)]
    df_heat = df_heat[ps_cols]

    # Select top_regions by most variance across PS
    df_heat = df_heat.loc[df_heat.var(axis=1).nlargest(top_regions).index]

    fig, ax = plt.subplots(figsize=(13, max(8, top_regions * 0.3)))
    sns.heatmap(df_heat, cmap='YlOrRd', ax=ax, linewidths=0.3,
                cbar_kws={'label': 'Proportion of expression'},
                xticklabels=[_ps_label(p) for p in range(1, n_ps + 1)])
    ax.set_xlabel('Phylostratum')
    ax.set_ylabel('Brain region')
    ax.set_title('Phylostratigraphic composition by brain region')
    plt.xticks(rotation=40, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / 'phylostratum_heatmap.png', dpi=150)
    plt.close()
    log.info('Saved phylostratum_heatmap.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_workers: int = 15, tai_method: str = 'log'):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Load data ---
    log.info('Loading AGEA atlas (processed volumes) ...')
    df_genes, gene_vols, atlas_agea = agea.load(label='processed')
    gene_symbols = df_genes['gene'].tolist()
    log.info(f'  {len(gene_symbols)} genes, volumes shape {gene_vols.shape}')

    # --- Gene ages ---
    df_ages = fetch_gene_ages(gene_symbols, n_workers=n_workers)
    df_ages = df_ages.set_index('gene').reindex(gene_symbols).reset_index()
    phylostrata_arr = df_ages['phylostratum'].values.astype(float)

    n_assigned = int(np.sum(~np.isnan(phylostrata_arr)))
    log.info(f'Phylostrata assigned: {n_assigned}/{len(gene_symbols)} genes '
             f'({100 * n_assigned / len(gene_symbols):.1f}%)')
    plot_gene_age_distribution(df_ages, OUTPUT_DIR)

    # Phylostrata coverage check
    for ps in range(1, 13):
        n = int(np.sum(phylostrata_arr == ps))
        log.info(f'  PS{ps:2d} {PHYLOSTRATA[ps]["name"]:20s}: {n:4d} genes'
                 + (' ⚠ LOW' if n < 50 else ''))

    # --- TAI (standard log-weighted) ---
    log.info(f'Computing TAI volume (method={tai_method}) ...')
    tai_vol = compute_tai(gene_vols, phylostrata_arr, method=tai_method)
    log.info(f'  TAI range: {np.nanmin(tai_vol):.2f} – {np.nanmax(tai_vol):.2f}')
    np.save(OUTPUT_DIR / 'tai_volume.npy', tai_vol)
    plot_tai_slices(tai_vol, OUTPUT_DIR)

    # --- Variance-weighted TAI (emphasises spatially specific genes) ---
    log.info('Computing variance-weighted TAI ...')
    tai_var = calculate_variance_weighted_tai(gene_vols, phylostrata_arr)
    log.info(f'  Variance-TAI range: {np.nanmin(tai_var):.2f} – {np.nanmax(tai_var):.2f}')
    np.save(OUTPUT_DIR / 'tai_var_volume.npy', tai_var)
    plot_tai_slices_named(tai_var, OUTPUT_DIR, name='tai_var_brain_slices',
                          title='Variance-weighted TAI (spatially specific genes)')

    # --- Regional TAI ---
    log.info('Aggregating TAI by brain region ...')
    regional_tai = compute_regional_tai(tai_vol, atlas_agea)
    regional_tai.to_csv(OUTPUT_DIR / 'regional_tai.csv', index=False)
    plot_regional_tai(regional_tai, OUTPUT_DIR)

    log.info('\nOldest 10 regions:')
    log.info(regional_tai[['acronym', 'name', 'tai_mean', 'n_voxels']].head(10).to_string())
    log.info('\nYoungest 10 regions:')
    log.info(regional_tai[['acronym', 'name', 'tai_mean', 'n_voxels']].tail(10).to_string())

    # --- Validation ---
    val = validate_tai(regional_tai)
    log.info('\nValidation:')
    log.info(val.to_string())
    val.to_csv(OUTPUT_DIR / 'validation.csv', index=False)

    # --- Heatmap ---
    log.info('Generating phylostratum heatmap ...')
    plot_phylostratum_heatmap(gene_vols, phylostrata_arr, atlas_agea, OUTPUT_DIR)

    log.info(f'\nAll outputs saved to {OUTPUT_DIR}')
    return regional_tai, tai_vol, df_ages


if __name__ == '__main__':
    main()
