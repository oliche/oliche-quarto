"""
For each phylostratum (gene age group), sum all gene expression volumes.
Produces 12 brain volumes saved as a (12, x, y, z) numpy array.

Requires gene_ages_cache.parquet to be present (run phylostratigraphy_pipeline.py first).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from iblatlas.genomics import agea

CACHE_FILE = Path(__file__).parent / 'gene_ages_cache.parquet'
OUTPUT_DIR = Path(__file__).parent / 'outputs'

N_PS = 12


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load pre-processed AGEA expression volumes
    df_genes, gene_vols, atlas_agea = agea.load(label='processed')
    gene_symbols = df_genes['gene'].tolist()
    print(f'Loaded {len(gene_symbols)} genes, volume shape: {gene_vols.shape}')

    # Load gene ages from cache
    df_ages = pd.read_parquet(CACHE_FILE)
    df_ages = df_ages.set_index('gene').reindex(gene_symbols)
    phylostrata = df_ages['phylostratum'].values  # (n_genes,), NaN for unknown

    out_path = OUTPUT_DIR / 'summed_volumes_by_phylostratum.npy'

    if out_path.exists():
        summed_volumes = np.load(out_path)
        print(f'Loaded cached summed volumes: shape {summed_volumes.shape} ← {out_path}')
        return summed_volumes

    # Sum expression volumes per phylostratum → shape (N_PS, x, y, z)
    _, x, y, z = gene_vols.shape
    summed_volumes = np.zeros((N_PS, x, y, z), dtype=np.float32)

    for ps in range(1, N_PS + 1):
        mask = phylostrata == ps
        n_genes = mask.sum()
        if n_genes == 0:
            print(f'  PS{ps:2d}: no genes assigned')
            continue
        summed_volumes[ps - 1] = gene_vols[mask].sum(axis=0)
        print(f'  PS{ps:2d}: {n_genes:4d} genes  sum range [{summed_volumes[ps-1].min():.1f}, {summed_volumes[ps-1].max():.1f}]')

    np.save(out_path, summed_volumes)
    print(f'\nSaved summed volumes: shape {summed_volumes.shape} → {out_path}')
    return summed_volumes


if __name__ == '__main__':
    main()
