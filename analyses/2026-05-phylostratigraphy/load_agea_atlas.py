import matplotlib.pyplot as plt
from iblatlas.genomics import agea

df_genes, gene_expression_volumes, atlas_agea = agea.load()
_, _, atlas_agea = agea.load(label='processed')
