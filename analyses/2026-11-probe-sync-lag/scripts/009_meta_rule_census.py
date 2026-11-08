"""Apply the boundary rule to every 3B probe on Alyx from metadata only (no sync files).

boundary = (start_lead - end_lead) / 0.5 s counts the sync half-periods gained at the start minus those lost at the end;
the shared firstSample error cancels in the difference. When the probe and nidq files end together, the spike offset
is boundary * 0.5 s. Cross-checked against the sync-file scan (008) where both exist.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
cen = pd.read_parquet(HERE.joinpath("005_meta_census.pqt"))
cen["boundary"] = (cen.start_lead_s - cen.end_lead_s) / 0.5
cen["boundary_int"] = cen.boundary.round()
cen["boundary_frac"] = (cen.boundary - cen.boundary_int).abs()
cen["rule_class"] = np.select([cen.boundary.isna(), cen.boundary_frac > 0.25, cen.boundary_int == 0], ["no meta", "ambiguous", "aligned"], "shifted")
cen["offset_meta_rule_s"] = np.where(cen.rule_class == "shifted", cen.boundary_int * 0.5, np.nan)
cen["combo"] = cen.appVersion.fillna("?") + " | " + cen.imDatBsc_fw.fillna("?")
cen.to_parquet(HERE.joinpath("009_meta_rule_census.pqt"))
pd.set_option("display.width", 250, "display.max_rows", 200)
print(cen.rule_class.value_counts().to_string())
print(pd.crosstab(cen.combo, cen.rule_class).to_string())
s = cen[cen.rule_class != "aligned"]
print(pd.crosstab([s.lab, s.rule_class], s.boundary_int).to_string())

# agreement with the sync-file scan where both exist
dmg = pd.read_csv(HERE.joinpath("008_damage_report.csv"))
m = dmg.merge(cen[["eid", "probe", "boundary", "rule_class", "offset_meta_rule_s"]], on=["eid", "probe"], how="inner")
m["meta_offset"] = m.offset_meta_rule_s.fillna(0)
print("sync-scan vs meta-only rule on", len(m), "probes:")
print(pd.crosstab(m.offset_rule_s, m.meta_offset.where(m.rule_class != "ambiguous", np.nan).astype(str)).to_string())
