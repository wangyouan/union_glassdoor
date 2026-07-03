#!/usr/bin/env python3
"""Round 8: Build notice dataset from yearly FOIA files + construct UNIONIZATION panel."""
import pandas as pd, numpy as np, os, re, glob

DATA = '/data/disk5/data/union/union f7/'
CTAT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/ctat_id_table.parquet'
MATCHES = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension/employer_gvkey_matches.csv'
OUT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unified_v2/'
os.makedirs(OUT, exist_ok=True)

# ====== LOAD DEPENDENCIES ======
print("Loading dependencies...")
ctat = pd.read_parquet(CTAT)
cusip_to_gvkey = ctat.drop_duplicates('cusip').set_index('cusip')['gvkey'].to_dict()

em = pd.read_csv(MATCHES)
emp_to_gvk = {}
for _, r in em[em.gvkey.notna()].iterrows():
    emp_to_gvk[str(r['Employer']).strip()] = int(r['gvkey'])
t3_gvkeys = set(em[em.match_tier==3]['gvkey'].dropna().astype(int).unique())
t1t2_gvkeys = set(em[em.match_tier.isin([1,2])]['gvkey'].dropna().astype(int).unique())

# Tier1 dict: Employer → cusip → gvkey (from unionized_rate_data)
ur_dict = pd.read_csv(os.path.join(DATA, 'unionized_rate_data.csv'), low_memory=False)
ur_dict = ur_dict[ur_dict['cusip'].notna()].copy()
ur_dict['cusip_str'] = ur_dict['cusip'].astype(str).str.strip()
ur_dict['gvkey'] = ur_dict['cusip_str'].map(cusip_to_gvkey)
ur_dict = ur_dict[ur_dict['gvkey'].notna()]
# Build employer→gvkey from cusip
t1_emp_gvk = {}
for _, r in ur_dict.iterrows():
    en = str(r['Employer']).strip()
    if en and en not in t1_emp_gvk:
        t1_emp_gvk[en] = int(r['gvkey'])
print(f"Tier1 (cusip) employers: {len(t1_emp_gvk)}, Tier1/2/3 from matches: {len(emp_to_gvk)}")

# ====== READ ALL FOIA FILES ======
print("\n=== Reading FOIA files ===")
STD_COLS = {
    'Notice Date':'Notice Date','Initiated Date':'Initiated Date',
    'Employer':'Employer','Employer State':'Employer State',
    'Bargaining Unit Size':'BUS','Establishment Size':'EST',
    'Union Name & Local Number':'Union','Category':'Category'
}

all_files = sorted(glob.glob(os.path.join(DATA, '*.xls*')))
all_files = [f for f in all_files if not f.startswith('~') and 'union_bargain' not in f]

file_stats = []
dfs = []
for f in all_files:
    fname = os.path.basename(f)
    # Skip: monthly files after 2022 (no EST), unionized_rate (used only for dict)
    m = re.search(r'([A-Z][a-z]+)-(\d{4})-F7', fname)
    if m and int(m.group(2)) >= 2023: continue

    # Auto-detect header: try skiprows 3-8
    header_row = None
    for skip in range(3, 9):
        try:
            tmp = pd.read_excel(f, skiprows=skip, nrows=0)
            if any('Employ' in str(c) for c in tmp.columns):
                header_row = skip
                break
        except: pass
    if header_row is None:
        file_stats.append({'file':fname,'status':'NO_HEADER','rows':0})
        continue

    try:
        df = pd.read_excel(f, skiprows=header_row)
        n_raw = len(df)
        # Map columns
        col_map = {}
        for c in df.columns:
            cs = str(c).strip()
            if cs in STD_COLS: col_map[c] = STD_COLS[cs]
        df = df.rename(columns=col_map)
        keep = [v for v in STD_COLS.values() if v in df.columns]
        if 'Employer' not in keep:
            file_stats.append({'file':fname,'status':'NO_EMPLOYER_COL','rows':n_raw})
            continue
        df = df[keep]
        df['source_file'] = fname
        dfs.append(df)
        file_stats.append({'file':fname,'status':'OK','rows':n_raw})
    except Exception as e:
        file_stats.append({'file':fname,'status':f'ERROR:{str(e)[:60]}','rows':0})

pd.DataFrame(file_stats).to_csv(OUT + 'ingestion_loss_table.csv', index=False)
combined = pd.concat(dfs, ignore_index=True)
print(f"Read {len(dfs)} files, {len(combined)} total rows")
print(f"Files with issues: {sum(1 for s in file_stats if s['status']!='OK')}")

# ====== CLEAN + DEDUP ======
print("\n=== Clean + dedup ===")
# Parse Year from Initiated Date
combined['Year'] = pd.to_datetime(combined['Initiated Date'], errors='coerce').dt.year
combined.loc[combined['Year'].isna(), 'Year'] = pd.to_datetime(combined['Notice Date'], errors='coerce').dt.year
combined = combined[combined.Year.notna()].copy()
combined['BUS'] = pd.to_numeric(combined['BUS'], errors='coerce')
combined['EST'] = pd.to_numeric(combined['EST'], errors='coerce')

# Filter BUS≤EST, EST>0
n_before = len(combined)
combined = combined[(combined.BUS <= combined.EST) & (combined.EST > 0)].copy()

# Dedup key: Employer + Union + Notice Date + BUS
dedup_keys = ['Employer', 'Union', 'Notice Date', 'BUS']
n_pre_dedup = len(combined)
combined = combined.drop_duplicates(subset=[k for k in dedup_keys if k in combined.columns], keep='first')

# Keep years 2005-2022
combined = combined[combined.Year.between(2005, 2022)].copy()

print(f"After BUS≤EST: {n_before}→{n_pre_dedup}, dedup: {n_pre_dedup}→{len(combined)}")
print(f"Years: {sorted(combined.Year.unique())}")

# Year distribution
print("\nYearly notice counts:")
yr_counts = combined.groupby('Year').size()
for y in range(2005, 2023):
    n = yr_counts.get(y, 0)
    print(f"  {y}: {n:>7d}")

# Check: all year counts differ
if yr_counts.nunique() == len(yr_counts):
    print("✓ All yearly counts differ")
else:
    print("*** WARNING: some years have identical counts ***")

# ====== MATCH EMPLOYERS TO GVKEY ======
print("\n=== Matching employers → gvkey ===")
# Tier 1: cusip-based dictionary
combined['gvkey'] = combined['Employer'].str.strip().map(t1_emp_gvk)
n_t1 = combined.gvkey.notna().sum()
print(f"Tier1 (cusip dict): {n_t1}")

# Tier 2+3: name match (reuse existing matches)
mask = combined.gvkey.isna()
combined.loc[mask, 'gvkey'] = combined.loc[mask, 'Employer'].str.strip().map(emp_to_gvk)
n_t23 = combined.gvkey.notna().sum() - n_t1
print(f"Tier2+3 (name match): {n_t23}")
print(f"Total with gvkey: {combined.gvkey.notna().sum()} ({combined.gvkey.notna().mean()*100:.1f}%)")
print(f"Unique gvkeys: {combined.gvkey.nunique()}")

# ====== BUILD UNIONIZATION PANEL ======
print("\n=== Building UNIONIZATION panel ===")
has_gvk = combined[combined.gvkey.notna()].copy()
has_gvk['gvkey'] = has_gvk['gvkey'].astype(int)

agg = has_gvk.groupby(['gvkey','Year']).agg(
    BUS_sum=('BUS','sum'), EST_sum=('EST','sum'), n_notices=('BUS','count')
).reset_index()
agg['UNIONIZATION_raw'] = agg['BUS_sum'] / agg['EST_sum']
agg['UNIONIZATION_cap1'] = agg['UNIONIZATION_raw'].clip(upper=1.0)
agg['UNIONIZATION'] = agg['UNIONIZATION_cap1']

print(f"Panel: {len(agg)} rows, {agg.gvkey.nunique()} gvkeys")
yr_stats = agg.groupby('Year').agg(n=('gvkey','nunique'), m=('UNIONIZATION','mean')).reset_index()
print("\nYearly panel:")
for _, r in yr_stats.iterrows():
    print(f"  {int(r.Year)}: {int(r.n):>5d} gvkeys, mean={r.m:.4f}")

# CRITICAL CHECK: 2005-2013 mean std > 0.001
mean_0513 = yr_stats[(yr_stats.Year>=2005)&(yr_stats.Year<=2013)]['m']
std_0513 = mean_0513.std()
print(f"\n2005-2013 mean std: {std_0513:.6f} (must be > 0.001) — {'PASS' if std_0513>0.001 else 'FAIL: snapshot copy detected!'}")

mean_0517 = agg[(agg.Year>=2005)&(agg.Year<=2017)]['UNIONIZATION'].mean()
print(f"2005-2017 mean: {mean_0517:.4f} (baseline ≈0.69)")

# ====== COMPARE WITH OLD FINISHED PANEL ======
print("\n=== Comparison with old finished panel ===")
old = pd.read_parquet('/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/unionization_panel_main.parquet')
old = old[old.gvkey.notna()].copy()
old['gvkey'] = old['gvkey'].astype(int)
old_0517 = old[(old.year>=2005)&(old.year<=2017)].rename(columns={'year':'Year'})

v2_0517 = agg[(agg.Year>=2005)&(agg.Year<=2017)][['gvkey','Year','UNIONIZATION']]
cmp = v2_0517.merge(old_0517[['gvkey','Year','UNIONIZATION']], on=['gvkey','Year'], how='outer', suffixes=('_new','_old'))
ok = cmp.UNIONIZATION_new.notna() & cmp.UNIONIZATION_old.notna()
corr = cmp.loc[ok,'UNIONIZATION_new'].corr(cmp.loc[ok,'UNIONIZATION_old'])

print(f"Correlation with old finished: {corr:.4f} (must be > 0.5) — {'PASS' if corr>0.5 else 'FAIL: data source mismatch!'}")

# Save comparison
pd.DataFrame({'metric':['correlation','n_new','n_old','n_overlap','mean_new','mean_old'],
    'value':[round(corr,4),len(v2_0517),len(old_0517),ok.sum(),
             round(v2_0517.UNIONIZATION.mean(),4),round(old_0517.UNIONIZATION.mean(),4)]
}).to_csv(OUT + 'unified2_vs_finished.csv', index=False)

# ====== MERGE WITH GLASSDOOR ======
print("\n=== Merging with Glassdoor ===")
gd = pd.read_parquet('/data/disk4/workspace/projects/union_glassdoor/outputs/20260702/firmyear_unionization/firmyear_glassdoor_panel.parquet')
gd['gvkey'] = gd['gvkey'].astype(float)
merged = gd.merge(agg, left_on=['gvkey','review_year'], right_on=['gvkey','Year'], how='left')
merged['UNIONIZATION'] = merged['UNIONIZATION'].fillna(0.0)
merged['UNIONIZATION_cap1'] = merged['UNIONIZATION']
merged['UNIONIZATION_binary'] = (merged['UNIONIZATION']>0).astype(int)
merged['UNIONIZATION_raw'] = merged['UNIONIZATION_raw'].fillna(0.0)

ctat_c = pd.read_parquet('/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/ctat_controls.parquet')
ctat_c['gvkey'] = ctat_c['gvkey'].astype(float)
ctat_c['fyear'] = ctat_c['fyear'].astype(int)
m2 = merged.merge(ctat_c, left_on=['gvkey','review_year'], right_on=['gvkey','fyear'], how='left', suffixes=('','_ctat'))
m2 = m2.drop_duplicates(subset=['gvkey','review_year'], keep='first')

n_var = m2.groupby('gvkey')['UNIONIZATION'].std().gt(0).sum()
print(f"Merged: {len(m2)} rows, {m2.gvkey.nunique()} gvkeys")
print(f"UNIONIZATION>0: {len(m2[m2.UNIONIZATION>0])} fy ({m2.UNIONIZATION.mean()*100:.2f}%)")
print(f"Within-var firms: {n_var}")

m2.to_parquet(OUT + 'unified2_panel.parquet', index=False)
agg.to_parquet(OUT + 'unified2_panel_base.parquet', index=False)
yr_stats.to_csv(OUT + 'unified2_panel_yearly.csv', index=False)

pd.DataFrame({'metric':['total_fy','total_firms','unionized_fy','unionized_firms','mean_un','within_var'],
    'value':[len(m2),m2.gvkey.nunique(),(m2.UNIONIZATION>0).sum(),
             m2[m2.UNIONIZATION>0].gvkey.nunique(),round(m2.UNIONIZATION.mean(),4),n_var]
}).to_csv(OUT + 'unified2_merge_coverage.csv', index=False)

print("\nSTEP 1 DONE — all checks passed" if (std_0513>0.001 and corr>0.5) else "\n*** CHECKS FAILED — STOP ***")
