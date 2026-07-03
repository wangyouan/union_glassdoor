#!/usr/bin/env python3
"""Round 13: Panel correction (Compustat-only + cell n>=10) + EMP extension."""
import pandas as pd, numpy as np, os, re, glob

OUT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260704/firmyear_corrected/'
UNI2 = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unified_v2/'
CTAT_DIR = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/'
MATCHES = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension/employer_gvkey_matches.csv'
EMP_DIR = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_emp_denom/'
os.makedirs(OUT, exist_ok=True)

# ====== STEP 0: Diagnose gvkey contamination ======
print("=== STEP 0: gvkey audit ===")
gd = pd.read_parquet('/data/disk4/workspace/projects/union_glassdoor/outputs/20260702/firmyear_unionization/firmyear_glassdoor_panel.parquet')
ctat = pd.read_parquet(CTAT_DIR + 'ctat_id_table.parquet')

all_gvkeys = set(gd.gvkey.unique())
ctat_gvkeys = set(ctat.gvkey.unique())
in_ctat = all_gvkeys & ctat_gvkeys
not_in_ctat = all_gvkeys - ctat_gvkeys

print(f"Total gvkeys in GD panel: {len(all_gvkeys)}")
print(f"  In Compustat: {len(in_ctat)} ({len(in_ctat)/len(all_gvkeys)*100:.1f}%)")
print(f"  NOT in Compustat: {len(not_in_ctat)} ({len(not_in_ctat)/len(all_gvkeys)*100:.1f}%)")

# Check by year: how many firm-years have Compustat records
ctat_years = ctat.groupby('gvkey')['fyear'].apply(set).to_dict()
gd_years = gd.groupby('gvkey')['review_year'].apply(set).to_dict()
fy_in_ctat = 0; fy_total = 0
for gv, yrs in gd_years.items():
    for y in yrs:
        fy_total += 1
        if gv in ctat_years and y in ctat_years.get(gv, set()):
            fy_in_ctat += 1
print(f"Firm-years with Compustat record: {fy_in_ctat}/{fy_total} ({fy_in_ctat/fy_total*100:.1f}%)")

# Sample of non-Compustat gvkeys
sample_bad = list(not_in_ctat)[:20]
print(f"Sample non-Compustat gvkeys: {sample_bad}")

# Check what these gvkeys look like (numeric range)
bad_nums = [int(g) for g in not_in_ctat if str(g).replace('.','').replace('-','').isdigit()]
if bad_nums:
    print(f"Non-Compustat gvkeys: min={min(bad_nums)}, max={max(bad_nums)}, count={len(bad_nums)}")

# Check if the GD gvkey source is another database (e.g. GVKEY-I or internal ID)
# Look at the top non-Compustat gvkeys in the GD panel by review count
gd['in_ctat'] = gd['gvkey'].isin(ctat_gvkeys)
top_non = gd[~gd['in_ctat']].groupby('gvkey').size().sort_values(ascending=False).head(20)
print(f"\nTop non-Compustat gvkeys by review count:")
for gv, n in top_non.items():
    print(f"  gvkey={gv}: {n} fy")

# Save audit
audit = pd.DataFrame({
    'metric': ['total_gvkeys','in_compustat','not_in_compustat','pct_in_ctat',
               'total_fy','fy_with_ctat_record','pct_fy_with_ctat'],
    'value': [len(all_gvkeys), len(in_ctat), len(not_in_ctat), len(in_ctat)/len(all_gvkeys)*100,
              fy_total, fy_in_ctat, fy_in_ctat/fy_total*100]
})
audit.to_csv(OUT + 'gvkey_audit.csv', index=False)

# ====== STEP 1: Build corrected panel ======
print("\n=== STEP 1: Corrected panel ===")

# 1a: Compustat-only filter (with fyear match)
gd['gvkey'] = gd['gvkey'].astype(float)
ctat['gvkey'] = ctat['gvkey'].astype(float)
ctat['fyear'] = ctat['fyear'].astype(int)

# Merge with Compustat to filter
gd_ctat = gd.merge(ctat[['gvkey','fyear']], left_on=['gvkey','review_year'], right_on=['gvkey','fyear'], how='inner')
print(f"After Compustat match (gvkey + year): {len(gd_ctat)} rows, {gd_ctat.gvkey.nunique()} gvkeys")

# Yearly company counts
yr_firms = gd_ctat.groupby('review_year')['gvkey'].nunique()
print(f"Yearly firms (corrected): {yr_firms.min()}-{yr_firms.max()}, mean={yr_firms.mean():.0f}")

# Keep only the columns we need
dv_cols = ['overall_rating','career_opp','comp_benefit','senior_mgmt','wlb','culture',
           'recommend','business_outlook','ceo_approval','diversity']
cur_cols = [c+'_cur' for c in dv_cols]
n_cols = ['n_'+c+'_all' for c in dv_cols]

keep = ['gvkey','review_year'] + dv_cols + cur_cols + n_cols
gd_ctat = gd_ctat[[c for c in keep if c in gd_ctat.columns]]

# 1b: Merge UNIONIZATION (EST) from unified2 panel
est_uni = pd.read_parquet(UNI2 + 'unified2_panel_base.parquet')
est_uni['gvkey'] = est_uni['gvkey'].astype(float)
est_uni['Year'] = est_uni['Year'].astype(int)
gd_ctat = gd_ctat.merge(est_uni[['gvkey','Year','UNIONIZATION','UNIONIZATION_raw']],
                         left_on=['gvkey','review_year'], right_on=['gvkey','Year'], how='left')
gd_ctat['UNIONIZATION'] = gd_ctat['UNIONIZATION'].fillna(0.0)
gd_ctat['UNIONIZATION_raw'] = gd_ctat['UNIONIZATION_raw'].fillna(0.0)
gd_ctat['UNIONIZATION_cap1'] = gd_ctat['UNIONIZATION']
gd_ctat['UNIONIZATION_binary'] = (gd_ctat['UNIONIZATION'] > 0).astype(int)

# 1c: Merge Compustat controls (needed for regressions)
ctat_ctrls = pd.read_parquet(CTAT_DIR + 'ctat_controls.parquet')
ctat_ctrls['gvkey'] = ctat_ctrls['gvkey'].astype(float)
ctat_ctrls['fyear'] = ctat_ctrls['fyear'].astype(int)
gd_ctat = gd_ctat.merge(ctat_ctrls, left_on=['gvkey','review_year'], right_on=['gvkey','fyear'], how='left', suffixes=('','_ctat'))
gd_ctat = gd_ctat.drop_duplicates(subset=['gvkey','review_year'], keep='first')

# Filter to main window
gd_ctat = gd_ctat[(gd_ctat.review_year >= 2005) & (gd_ctat.review_year <= 2022)]

print(f"Corrected panel (main window): {len(gd_ctat)} rows, {gd_ctat.gvkey.nunique()} gvkeys")
print(f"UNIONIZATION>0: {len(gd_ctat[gd_ctat.UNIONIZATION>0])} fy")

# cell n>=10 filter check — will be applied in R per DV
# Report current n≥10 cell counts for each DV
for dv in dv_cols:
    n_col = f'n_{dv}_all'
    if n_col in gd_ctat.columns:
        n10 = (gd_ctat[n_col] >= 10).sum()
        n5 = (gd_ctat[n_col] >= 5).sum()
        n20 = (gd_ctat[n_col] >= 20).sum()
        print(f"  {dv}: total={len(gd_ctat)}, n>=10={n10}, n>=5={n5}, n>=20={n20}")

gd_ctat.to_parquet(OUT + 'corrected_panel.parquet', index=False)
print(f"Saved corrected_panel.parquet")

# Coverage
cov_df = pd.DataFrame({
    'metric': ['total_fy','total_firms','unionized_fy','unionized_firms','mean_un','pct_ctat_matched'],
    'value': [len(gd_ctat), gd_ctat.gvkey.nunique(),
              (gd_ctat.UNIONIZATION>0).sum(), gd_ctat[gd_ctat.UNIONIZATION>0].gvkey.nunique(),
              round(gd_ctat.UNIONIZATION.mean(),4), 100.0]
})
cov_df.to_csv(OUT + 'corrected_coverage.csv', index=False)

# ====== STEP 2: EMP extension to 2024 ======
print("\n=== STEP 2: EMP extension to 2024 ===")
# Load existing EMP panel
emp_old = pd.read_parquet(EMP_DIR + 'emp_panel.parquet')
print(f"Old EMP panel: {len(emp_old)} rows, years {emp_old.review_year.min():.0f}-{emp_old.review_year.max():.0f}")

# Get BUS_sum for 2023-2024 from extension_fix data
# Reuse the notice data from round 5/6
ext_notices = pd.read_csv('/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension_fix/f7_notices_all_years_fix.parquet'
    if os.path.exists('/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension_fix/f7_notices_all_years_fix.parquet')
    else '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension/f7_notices_all_years.csv')

# Actually, use extension_fix base panel for 2023-2024
ext_agg = pd.read_parquet('/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension_fix/unionization_panel_v1_fix.parquet')
ext_agg = ext_agg[ext_agg.Year.isin([2023,2024])][['gvkey','Year','BUS_sum']].copy()
ext_agg.columns = ['gvkey','year','BUS_sum']
print(f"Extension notices 2023-2024: {len(ext_agg)} rows, {ext_agg.gvkey.nunique()} gvkeys")

# Merge with Compustat EMP for 2023-2024
emp_ctat = pd.read_parquet(CTAT_DIR + 'ctat_controls.parquet')[['gvkey','fyear','emp']].copy()
emp_ctat['gvkey'] = emp_ctat['gvkey'].astype(float)
emp_ctat['fyear'] = emp_ctat['fyear'].astype(int)
ext_emp = ext_agg.merge(emp_ctat, left_on=['gvkey','year'], right_on=['gvkey','fyear'], how='left')
ext_emp = ext_emp[ext_emp.emp.notna() & (ext_emp.emp > 0)].copy()
ext_emp['UNIONIZATION_EMP_raw'] = ext_emp['BUS_sum'] / (ext_emp['emp'] * 1000)
ext_emp['UNIONIZATION_EMP_cap1'] = ext_emp['UNIONIZATION_EMP_raw'].clip(upper=1.0)
print(f"EMP extension: {len(ext_emp)} fy, {ext_emp.gvkey.nunique()} gvkeys")

# Get 2005-2022 EMP clean from the existing panel
emp_clean_0522 = emp_old[(emp_old.review_year >= 2005) & (emp_old.review_year <= 2022) &
                          emp_old.emp.notna() & (emp_old.emp > 0)].copy()

# Build extended EMP: take the key columns from old + new
emp_ext = pd.concat([
    emp_clean_0522[['gvkey','review_year','UNIONIZATION_EMP_cap1','UNIONIZATION_EMP_raw','emp','at','dltt','dlc','capx','ebitda','sale','tlcf','ib','xsga','sic'] + dv_cols + cur_cols + n_cols],
], ignore_index=True) if len(emp_clean_0522) > 0 else pd.DataFrame()

# Actually, just save the BUS+EMP data and merge in R
emp_ext_full = pd.concat([
    emp_clean_0522[['gvkey','review_year','UNIONIZATION_EMP_cap1','UNIONIZATION_EMP_raw','emp'] + dv_cols + cur_cols + [c for c in n_cols if c in emp_clean_0522.columns]],
], ignore_index=True)
# Add 2023-2024
for _, r in ext_emp.iterrows():
    row = {'gvkey': r['gvkey'], 'review_year': int(r['year']),
           'UNIONIZATION_EMP_cap1': r['UNIONIZATION_EMP_cap1'],
           'UNIONIZATION_EMP_raw': r['UNIONIZATION_EMP_raw'],
           'emp': r['emp']}
    emp_ext_full = pd.concat([emp_ext_full, pd.DataFrame([row])], ignore_index=True)

emp_ext_full['gvkey'] = emp_ext_full['gvkey'].astype(float)
emp_ext_full['review_year'] = emp_ext_full['review_year'].astype(int)
emp_ext_full = emp_ext_full.drop_duplicates(subset=['gvkey','review_year'], keep='first')

# Merge with full GD panel for DVs
gd_for_emp = gd[['gvkey','review_year'] + dv_cols + cur_cols + n_cols +
    ['at','dltt','dlc','capx','ebitda','sale','tlcf','ib','xsga','sic']]
emp_ext_full = gd_for_emp.merge(emp_ext_full[['gvkey','review_year','UNIONIZATION_EMP_cap1','UNIONIZATION_EMP_raw']],
                                on=['gvkey','review_year'], how='right')
emp_ext_full['UNIONIZATION_EMP_cap1'] = emp_ext_full['UNIONIZATION_EMP_cap1'].fillna(0.0)
emp_ext_full['UNIONIZATION_EMP_raw'] = emp_ext_full['UNIONIZATION_EMP_raw'].fillna(0.0)

print(f"EMP extended panel: {len(emp_ext_full)} rows, years {sorted(emp_ext_full.review_year.unique())}")
emp_ext_full.to_parquet(OUT + 'emp_extended_panel.parquet', index=False)
print("Saved emp_extended_panel.parquet")

print("\n=== STEPS 0-2 DONE ===")
