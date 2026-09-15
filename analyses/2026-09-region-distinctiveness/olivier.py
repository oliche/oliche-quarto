# %%
from pathlib import Path
import pandas as pd
import numpy as np

MIN_VOXELS = 500

csv_file = Path("/Users/olivier/Documents/oliche-quarto/analyses/2026-09-region-distinctiveness/cache/region_hierarchy_stats.csv")
df_regions = pd.read_csv(csv_file)
