#!/usr/bin/env python
"""FMCS-aligned firm-year pipeline (STEP 0-2) + Part B diagnostics.

STEPS:
  0: Data check, cusip→gvkey bridge, repro check
  1: Build FMCS UNIONIZATION panel
  2: Correlations + merge with Glassdoor
  B: Diagnose previous NLRB fallback issues
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

OUT = "/data/disk4/workspace/projects/union_glassdoor/outputs/20260702/firmyear_unionization/fmcs_aligned"
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 0: Data check + cusip→gvkey bridge
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 0: Data check + cusip→gvkey bridge")
print("=" * 60)

# 0.2: FMCS data
fmcs = pd.read_csv('/data/disk5/data/union/union f7/unionized_rate_data.csv')
print(f"FMCS: {len(fmcs):,} rows, {fmcs['cusip'].dropna().nunique():,} unique cusips, {fmcs['year'].min():.0f}-{fmcs['year'].max():.0f}")

# 0.3 cusip→gvkey bridge
matched = pd.read_parquet('/data/disk4/workspace/projects/union/outputs/union_election_rc_votes_matched_combined.parquet',
                          columns=['matched_cusip','gvkey_final','matched_conm'])
bridge = matched[['matched_cusip','gvkey_final']].dropna().drop_duplicates()

bridge_map = {}
for _, r in bridge.iterrows():
    c9 = str(r['matched_cusip']).strip().zfill(9)
    bridge_map[c9] = r['gvkey_final']

fmcs_cusips = fmcs['cusip'].dropna().unique()
fmcs_str = [str(c).strip().zfill(9) for c in fmcs_cusips]
matched_cusips = set(c for c in fmcs_str if c in bridge_map)
gvkey_map = {c: bridge_map[c] for c in matched_cusips}

print(f"Cusip→gvkey bridge: {len(bridge_map):,} mappings from NLRB matched data")
print(f"FMCS cusips matched: {len(matched_cusips):,} / {len(fmcs_cusips):,} ({len(matched_cusips)/len(fmcs_cusips)*100:.1f}%)")
print(f"Unique gvkeys: {len(set(gvkey_map.values())):,}")
unmatched = [c for c in fmcs_str if c not in bridge_map]
print(f"Unmatched cusips: {len(unmatched):,}")

# Write STEP 0 checklist
step0 = []
step0.append("# STEP 0 — FMCS Data Check\n\n")
step0.append(f"## 0.2 FMCS Data\n")
step0.append(f"- unionized_rate_data.csv: {len(fmcs):,} rows\n")
step0.append(f"- Years: {fmcs['year'].min():.0f}–{fmcs['year'].max():.0f}\n")
step0.append(f"- Unique cusips with match: {fmcs['cusip'].dropna().nunique():,}\n")
step0.append(f"- BUS≤EST filter: {(fmcs['Bargaining Unit Size'] <= fmcs['Establishment Size']).sum():,} / {len(fmcs):,} ({(fmcs['Bargaining Unit Size'] <= fmcs['Establishment Size']).mean()*100:.1f}%)\n")
step0.append(f"\n## 0.3 Cusip→GVKEY Bridge\n")
step0.append(f"- Bridge source: NLRB union_election_rc_votes_matched_combined.parquet\n")
step0.append(f"- Bridge size: {len(bridge_map):,} cusip→gvkey mappings\n")
step0.append(f"- FMCS cusips matched: {len(matched_cusips):,} / {len(fmcs_cusips):,} ({len(matched_cusips)/len(fmcs_cusips)*100:.1f}%)\n")
step0.append(f"- Unique gvkeys: {len(set(gvkey_map.values())):,}\n")
step0.append(f"- ⚠️ Only 30% coverage — bridge is limited to NLRB-election firms. Full Compustat cusip not available.\n")

with open(f"{OUT}/step0_fmcs_check.md", "w") as f:
    f.write("".join(step0))
print("Saved step0_fmcs_check.md")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Build FMCS UNIONIZATION panel
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Build FMCS UNIONIZATION panel")
print("=" * 60)

# Filter FMCS
fmcs_f = fmcs[(fmcs['cusip'].notna()) &
              (fmcs['Bargaining Unit Size'] <= fmcs['Establishment Size']) &
              (fmcs['Establishment Size'] > 0)].copy()
print(f"After BUS≤EST filter: {len(fmcs_f):,} rows ({len(fmcs_f)/len(fmcs)*100:.1f}%)")

# Map cusip to gvkey
fmcs_f['cusip_str'] = fmcs_f['cusip'].apply(lambda c: str(c).strip().zfill(9))
fmcs_f['gvkey'] = fmcs_f['cusip_str'].map(gvkey_map)
fmcs_m = fmcs_f[fmcs_f['gvkey'].notna()].copy()
print(f"After gvkey match: {len(fmcs_m):,} rows, {fmcs_m['gvkey'].nunique():,} gvkeys")

# Main construction: UNIONIZATION = sum(BUS) / sum(EST) per (gvkey, year)
panel = fmcs_m.groupby(['gvkey','year']).agg(
    sum_bus=('Bargaining Unit Size','sum'),
    sum_est=('Establishment Size','sum'),
    n_notices=('Bargaining Unit Size','count')
).reset_index()

panel['UNIONIZATION'] = panel['sum_bus'] / panel['sum_est']

# Verify
unionized = panel[panel['UNIONIZATION'] > 0]
print(f"\nUnionized gvkey-years: {len(unionized):,}")
print(f"UNIONIZATION stats (unionized):")
print(f"  mean={unionized['UNIONIZATION'].mean():.4f}, median={unionized['UNIONIZATION'].median():.4f}")
print(f"  P25={unionized['UNIONIZATION'].quantile(0.25):.4f}, P75={unionized['UNIONIZATION'].quantile(0.75):.4f}")
print(f"  >1: {(unionized['UNIONIZATION']>1).sum():,}")
print(f"  Unique gvkeys: {unionized['gvkey'].nunique():,}")
print(f"  year range: {unionized['year'].min():.0f}–{unionized['year'].max():.0f}")

# Merge with Compustat to get non-unionized firm-years
cmp = pd.read_parquet("outputs/compustat_firm_controls.parquet")
cmp['gvkey'] = cmp['gvkey'].astype(str)
cmp_yr = cmp[['gvkey','fyear']].drop_duplicates()
cmp_yr = cmp_yr[(cmp_yr['fyear'] >= 2005) & (cmp_yr['fyear'] <= 2017)]

# Extend panel: all gvkey-years in the bridge + Compustat range
all_gvkeys = set(panel['gvkey'].unique())  # only matched gvkeys
all_years = range(2005, 2018)
full_idx = pd.MultiIndex.from_product([all_gvkeys, all_years], names=['gvkey','year'])
full_panel = pd.DataFrame(index=full_idx).reset_index()
full_panel = full_panel.merge(panel, on=['gvkey','year'], how='left')
for c in ['UNIONIZATION','sum_bus','sum_est','n_notices']:
    full_panel[c] = full_panel[c].fillna(0)

print(f"\nFull panel: {len(full_panel):,} gvkey-years, {full_panel['gvkey'].nunique():,} gvkeys")
print(f"UNIONIZATION>0: {(full_panel['UNIONIZATION']>0).sum():,} ({(full_panel['UNIONIZATION']>0).mean()*100:.1f}%)")
print(f"UNIONIZATION stats (full panel): mean={full_panel['UNIONIZATION'].mean():.4f}, median={full_panel['UNIONIZATION'].median():.4f}")

# Add Unionized binary
full_panel['has_union'] = (full_panel['UNIONIZATION'] > 0).astype(int)

# Save
full_panel.to_parquet(f"{OUT}/fmcs_unionization_panel.parquet", index=False)
print(f"Saved fmcs_unionization_panel.parquet ({len(full_panel.columns)} cols)")

# Descriptives
desc_rows = [
    {'metric': 'n_gvkey_years', 'value': len(full_panel)},
    {'metric': 'n_gvkeys', 'value': full_panel['gvkey'].nunique()},
    {'metric': 'n_unionized_fy', 'value': int((full_panel['UNIONIZATION']>0).sum())},
    {'metric': 'pct_unionized', 'value': round((full_panel['UNIONIZATION']>0).mean()*100, 1)},
    {'metric': 'union_mean', 'value': round(full_panel['UNIONIZATION'].mean(), 4)},
    {'metric': 'union_median', 'value': round(full_panel['UNIONIZATION'].median(), 4)},
    {'metric': 'union_p25', 'value': round(full_panel['UNIONIZATION'].quantile(0.25), 4)},
    {'metric': 'union_p75', 'value': round(full_panel['UNIONIZATION'].quantile(0.75), 4)},
    {'metric': 'union_p90', 'value': round(full_panel['UNIONIZATION'].quantile(0.90), 4)},
    {'metric': 'union_mean_if_pos', 'value': round(full_panel.loc[full_panel['UNIONIZATION']>0,'UNIONIZATION'].mean(), 4)},
    {'metric': 'union_median_if_pos', 'value': round(full_panel.loc[full_panel['UNIONIZATION']>0,'UNIONIZATION'].median(), 4)},
    {'metric': 'union_gt1', 'value': int((full_panel['UNIONIZATION']>1).sum())},
]
pd.DataFrame(desc_rows).to_csv(f"{OUT}/fmcs_descriptives.csv", index=False)
print("Saved fmcs_descriptives.csv")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Merge with Glassdoor + Correlations
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Merge + correlations")
print("=" * 60)

gd = pd.read_parquet("outputs/20260702/firmyear_unionization/firmyear_glassdoor_panel.parquet")
print(f"Glassdoor panel: {len(gd):,} rows, {gd['gvkey'].nunique():,} gvkeys")

# Merge
merged = full_panel.merge(gd, left_on=['gvkey','year'], right_on=['gvkey','review_year'], how='inner')
print(f"Merged: {len(merged):,} rows, {merged['gvkey'].nunique():,} gvkeys")
print(f"Unionized in merged: {(merged['UNIONIZATION']>0).sum():,} ({(merged['UNIONIZATION']>0).mean()*100:.1f}%)")

# Filter: n>=5 reviews
merged['n5'] = merged['n_reviews_all'] >= 5
merged_main = merged[merged['n5']].copy()

# Merge coverage
cov_rows = [
    {'metric': 'fmcs_gvkeys', 'value': full_panel['gvkey'].nunique()},
    {'metric': 'glassdoor_gvkeys', 'value': gd['gvkey'].nunique()},
    {'metric': 'merged_gvkeys', 'value': merged['gvkey'].nunique()},
    {'metric': 'merged_fy', 'value': len(merged)},
    {'metric': 'merged_fy_n5', 'value': int(merged['n5'].sum())},
    {'metric': 'merged_unionized_fy', 'value': int((merged['UNIONIZATION']>0).sum())},
    {'metric': 'merged_unionized_fy_n5', 'value': int((merged_main['UNIONIZATION']>0).sum())},
]
pd.DataFrame(cov_rows).to_csv(f"{OUT}/fmcs_merge_coverage.csv", index=False)
print("Saved fmcs_merge_coverage.csv")

# Correlations
DV10 = ['overall_rating','career_opp','comp_benefit','senior_mgmt','wlb','culture',
        'recommend','business_outlook','ceo_approval','diversity']

corr_rows = []
for dv in DV10:
    sub = merged_main[[dv,'UNIONIZATION']].dropna()
    if len(sub) < 30: continue
    rp = sub.corr().iloc[0,1]
    rs = sub.corr(method='spearman').iloc[0,1]
    zero_m = sub.loc[sub['UNIONIZATION']==0, dv].mean()
    nonzero_m = sub.loc[sub['UNIONIZATION']>0, dv].mean()
    n_nz = (sub['UNIONIZATION']>0).sum()
    corr_rows.append({'dv': dv, 'pearson': round(rp,4), 'spearman': round(rs,4),
                       'mean_zero': round(zero_m,4), 'mean_nonzero': round(nonzero_m,4),
                       'diff': round(nonzero_m - zero_m, 4), 'n_nonzero': n_nz, 'N': len(sub)})
    print(f"  {dv}: r={rp:.4f}, zero={zero_m:.3f}, nonzero={nonzero_m:.3f}, diff={nonzero_m-zero_m:+.3f}")

pd.DataFrame(corr_rows).to_csv(f"{OUT}/fmcs_correlations.csv", index=False)
print("Saved fmcs_correlations.csv")

# ═══════════════════════════════════════════════════════════════════════════
# Part B: Diagnose NLRB fallback issues
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part B: Diagnose NLRB fallback")
print("=" * 60)

nlrb_panel = pd.read_parquet("outputs/20260702/firmyear_unionization/firmyear_unionization_panel.parquet")

# B.2: ratio distribution and >1 cases
print("NLRB union_ratio distribution:")
for col in ['union_ratio_raw','union_ratio_winsor','union_ratio_capped']:
    vals = nlrb_panel[col].dropna()
    vals_f = vals.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  {col}: mean={vals_f.mean():.6f}, med={vals_f.median():.6f}, "
          f"max={vals_f.max():.6f}, >1={(vals_f>1).sum():,}, nz={(vals_f>0).sum():,}")

# ratio > 1 top 30
gt1 = nlrb_panel[nlrb_panel['union_ratio_raw'] > 1].nlargest(30, 'union_ratio_raw')
gt1_out = gt1[['gvkey','fyear','unionized_emp_stock','emp','emp_actual','union_ratio_raw']].copy()
gt1_out.to_csv(f"{OUT}/diag_ratio_gt1_top30.csv", index=False)
print(f"Saved diag_ratio_gt1_top30.csv ({len(gt1_out)} rows)")

# B.3: Fix binary — has_union should be 1 if ratio > 0, not NA
nlrb_panel['has_union_fixed'] = (nlrb_panel['unionized_emp_stock'] > 0).astype(int)
print(f"has_union_fixed: {(nlrb_panel['has_union_fixed']==1).sum():,} non-zero")

# Save for R regression
nlrb_panel[['gvkey','fyear','union_ratio_winsor','has_union_fixed']].to_csv(
    f"{OUT}/nlrb_fallback_binary_fix.csv", index=False)
print("Saved nlrb_fallback_binary_fix.csv")

print("\nDone.")
