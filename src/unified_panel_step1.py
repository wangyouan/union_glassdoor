#!/usr/bin/env python3
"""STEP 1: Build UNIFIED UNIONIZATION panel 2005-2022 + merge with Glassdoor."""
import pandas as pd, numpy as np, os, re, glob

DATA_DIR = '/data/disk5/data/union/union f7/'
CTAT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/ctat_id_table.parquet'
EXT_DIR = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension_fix/'
MATCHES = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension/employer_gvkey_matches.csv'
OUT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unified/'
os.makedirs(OUT, exist_ok=True)

# ====== Load dependencies ======
print("Loading dependencies...")
ctat = pd.read_parquet(CTAT)
cusip_to_gvkey = ctat.drop_duplicates('cusip').set_index('cusip')['gvkey'].to_dict()

em = pd.read_csv(MATCHES)
emp_to_gvk = {}
for _, r in em[em.gvkey.notna()].iterrows():
    emp_to_gvk[str(r['Employer']).strip()] = int(r['gvkey'])
t3_gvkeys = set(em[em.match_tier==3]['gvkey'].dropna().astype(int).unique())
print(f"CUSIP→gvkey: {len(cusip_to_gvkey)}, Employer→gvkey: {len(emp_to_gvk)}, Tier3 gvkeys: {len(t3_gvkeys)}")

# ====== SOURCE 1: unionized_rate_data.csv (2005-2017) ======
print("\n=== SOURCE 1: unionized_rate_data ===")
ur = pd.read_csv(os.path.join(DATA_DIR, 'unionized_rate_data.csv'), low_memory=False)
n1 = len(ur)
ur = ur.rename(columns={'Bargaining Unit Size':'BUS','Establishment Size':'EST','year':'Year'})
ur['BUS'] = pd.to_numeric(ur['BUS'], errors='coerce')
ur['EST'] = pd.to_numeric(ur['EST'], errors='coerce')
ur = ur[(ur.BUS <= ur.EST) & (ur.EST > 0)].copy()
n2 = len(ur)

# gvkey: cusip first, then name match
ur['cusip_str'] = ur['cusip'].astype(str).str.strip()
ur['gvkey_cusip'] = ur['cusip_str'].map(cusip_to_gvkey)
ur['gvkey_name'] = ur['Employer'].str.strip().map(emp_to_gvk)
ur['gvkey'] = ur['gvkey_cusip'].fillna(ur['gvkey_name'])
ur['match_source'] = 'cusip'
ur.loc[ur.gvkey_cusip.isna() & ur.gvkey_name.notna(), 'match_source'] = 'name'
conflicts = (ur.gvkey_cusip.notna() & ur.gvkey_name.notna() & (ur.gvkey_cusip != ur.gvkey_name)).sum()

n3 = ur.gvkey.notna().sum()
print(f"  Read: {n1} → BUS≤EST: {n2} → with gvkey: {n3} ({n3/n2*100:.1f}%)")
print(f"  cusip-match: {ur.gvkey_cusip.notna().sum()}, name-match: {ur.gvkey_name.notna().sum()}, conflicts: {conflicts}")

ur_src = ur[['Employer','Employer State','BUS','EST','Year','gvkey','match_source']].copy()
ur_src['source'] = 'unionized_rate'

# ====== SOURCE 2: f7.csv.zip (2014-2022, only years not in unionized_rate) ======
print("\n=== SOURCE 2: f7.csv.zip ===")
f7 = pd.read_csv(os.path.join(DATA_DIR, 'f7.csv.zip'), low_memory=False)
f7 = f7.rename(columns={'employer':'Employer','employer_state':'Employer State',
                         'bargaining_unit_size':'BUS','establishment_size':'EST',
                         'initiated_date':'Initiated Date','notice_date':'Notice Date'})
f7['Year'] = pd.to_datetime(f7['Initiated Date'], errors='coerce').dt.year
f7.loc[f7['Year'].isna(), 'Year'] = pd.to_datetime(f7['Notice Date'], errors='coerce').dt.year
f7 = f7[f7.Year.notna()].copy()
f7['BUS'] = pd.to_numeric(f7['BUS'], errors='coerce')
f7['EST'] = pd.to_numeric(f7['EST'], errors='coerce')
f7 = f7[(f7.BUS <= f7.EST) & (f7.EST > 0)].copy()

# Keep 2014-2022 (exclude 2009-2010 overlap with unionized_rate's Initiated Date years)
f7 = f7[(f7.Year >= 2014) & (f7.Year <= 2022)].copy()
f7['gvkey'] = f7['Employer'].str.strip().map(emp_to_gvk)
f7['match_source'] = 'name'
n_f7 = f7.gvkey.notna().sum()
print(f"  f7.csv 2014-2022: {len(f7)} rows, with gvkey: {n_f7} ({n_f7/len(f7)*100:.1f}%)")

f7_src = f7[['Employer','Employer State','BUS','EST','Year','gvkey','match_source']].copy()
f7_src['source'] = 'f7_csv'

# ====== SOURCE 3: Monthly Excel (2019-2022, supplements f7) ======
print("\n=== SOURCE 3: Monthly Excel 2019-2022 ===")
EMP_COLS = ['Employer','Employer Name']
STATE_COLS = ['Employer State','Employer State ']
BUS_COLS = ['Bargaining Unit Size','Unit Size']
EST_COLS = ['Establishment Size','Size of Establishment','Number of Employees']
DATE_COLS = ['Notice Date','Date Received','Initiated Date','Contract Initiated']

monthly_files = sorted(glob.glob(os.path.join(DATA_DIR, '*F7-Notices.xlsx')))
excel_dfs = []
loss = []
for f in monthly_files:
    fname = os.path.basename(f)
    m = re.search(r'([A-Z][a-z]+)-(\d{4})-F7', fname)
    if not m: continue
    yf = int(m.group(2))
    if yf < 2019 or yf > 2022: continue
    try:
        df = pd.read_excel(f, skiprows=6)
        col_map = {}
        for c in df.columns:
            cs = str(c).strip()
            if cs in EMP_COLS: col_map[c] = 'Employer'
            elif cs in STATE_COLS: col_map[c] = 'Employer State'
            elif cs in BUS_COLS: col_map[c] = 'BUS'
            elif cs in EST_COLS: col_map[c] = 'EST'
            elif cs in DATE_COLS and 'Notice Date' not in col_map.values(): col_map[c] = 'Notice Date'
        df = df.rename(columns=col_map)
        if 'Employer' not in df.columns:
            loss.append({'file':fname,'n_read':len(df),'status':'NO_EMPLOYER_COL'})
            continue
        df = df[[c for c in ['Employer','Employer State','BUS','EST','Notice Date'] if c in df.columns]]
        if 'Notice Date' in df.columns:
            df['Year'] = pd.to_datetime(df['Notice Date'], errors='coerce').dt.year
        else:
            df['Year'] = yf
        df = df[df.Year.notna() & df.Year.between(2019,2022)]
        if 'BUS' in df.columns: df['BUS'] = pd.to_numeric(df['BUS'], errors='coerce')
        else: df['BUS'] = np.nan
        if 'EST' in df.columns: df['EST'] = pd.to_numeric(df['EST'], errors='coerce')
        else: df['EST'] = np.nan
        has_est = df.EST.notna().sum() > 0
        if has_est: df = df[(df.BUS <= df.EST) & (df.EST > 0)]
        else: df = df[df.BUS.notna() & (df.BUS > 0)]
        if len(df) > 0:
            df['gvkey'] = df['Employer'].str.strip().map(emp_to_gvk)
            df['match_source'] = 'name'
            df['source'] = fname
            excel_dfs.append(df[['Employer','Employer State','BUS','EST','Year','gvkey','match_source','source']])
    except Exception as e:
        loss.append({'file':fname,'n_read':0,'status':f'ERROR:{str(e)[:80]}'})

excel_all = pd.concat(excel_dfs, ignore_index=True) if excel_dfs else pd.DataFrame()
print(f"  Files: {len(excel_dfs)} with data, {len(excel_all)} rows")
print(f"  Years: {sorted(excel_all.Year.unique()) if len(excel_all)>0 else 'none'}")

# Save loss table
pd.DataFrame(loss).to_csv(OUT + 'ingestion_loss_table.csv', index=False)

# ====== MERGE + DEDUP ======
print("\n=== Merging all sources ===")
combined = pd.concat([ur_src, f7_src, excel_all], ignore_index=True)
print(f"Combined: {len(combined)} rows before dedup")
dedup_keys = ['Employer','Year','BUS']
n_before = len(combined)
combined = combined.drop_duplicates(subset=dedup_keys, keep='first')
print(f"After dedup: {len(combined)} ({n_before - len(combined)} removed)")

# Year distribution
combined['Year'] = combined['Year'].astype(int)
combined = combined[combined.Year.between(2005, 2022)]
print(f"\nYear distribution:")
for y in range(2005, 2023):
    sub = combined[combined.Year == y]
    gvk = sub.gvkey.notna().sum()
    print(f"  {y}: {len(sub):>7d} notices, {gvk:>6d} with gvkey")

# ====== BUILD UNIONIZED PANEL ======
print("\n=== Building UNIONIZATION panel ===")
has_gvk = combined[combined.gvkey.notna()].copy()
has_gvk['gvkey'] = has_gvk['gvkey'].astype(int)

agg = has_gvk.groupby(['gvkey','Year']).agg(
    BUS_sum=('BUS','sum'), EST_sum=('EST','sum'), n_notices=('BUS','count')
).reset_index()
agg['UNIONIZATION_raw'] = agg['BUS_sum'] / agg['EST_sum']
agg['UNIONIZATION_cap1'] = agg['UNIONIZATION_raw'].clip(upper=1.0)
agg['UNIONIZATION'] = agg['UNIONIZATION_cap1']

print(f"Panel: {len(agg)} rows, {agg.gvkey.nunique()} gvkeys, years {agg.Year.min()}-{agg.Year.max()}")
print(f"UNIONIZATION > 0: {len(agg[agg.UNIONIZATION>0])}")

# Yearly stats
yr_stats = agg.groupby('Year').agg(n_gvkeys=('gvkey','nunique'),mean_un=('UNIONIZATION','mean')).reset_index()
print("\nYearly panel stats:")
for _, r in yr_stats.iterrows():
    print(f"  {int(r['Year'])}: {int(r['n_gvkeys']):>5d} gvkeys, mean={r['mean_un']:.4f}")

mean_0517 = agg[(agg.Year>=2005)&(agg.Year<=2017)]['UNIONIZATION'].mean()
print(f"\n2005-2017 mean UNIONIZATION: {mean_0517:.4f} (baseline ≈0.69)")

# Save yearly stats
yr_stats.to_csv(OUT + 'unified_panel_yearly.csv', index=False)

# ====== COMPARE WITH OLD FINISHED PANEL ======
print("\n=== Comparison with old finished panel ===")
old = pd.read_parquet('/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/unionization_panel_main.parquet')
old = old[old.gvkey.notna()].copy()
old['gvkey'] = old['gvkey'].astype(int)
old_0517 = old[(old.year>=2005)&(old.year<=2017)].rename(columns={'year':'Year'})

v1_0517 = agg[(agg.Year>=2005)&(agg.Year<=2017)][['gvkey','Year','UNIONIZATION']]
cmp = v1_0517.merge(old_0517[['gvkey','Year','UNIONIZATION']], on=['gvkey','Year'], how='outer', suffixes=('_new','_old'))
ok = cmp.UNIONIZATION_new.notna() & cmp.UNIONIZATION_old.notna()
corr = cmp.loc[ok,'UNIONIZATION_new'].corr(cmp.loc[ok,'UNIONIZATION_old'])
mdiff = (cmp['UNIONIZATION_new'] - cmp['UNIONIZATION_old']).mean()
comp = pd.DataFrame({
    'metric':['correlation','mean_diff','n_new','n_old','n_overlap','mean_new','mean_old'],
    'value':[round(corr,4),round(mdiff,4),len(v1_0517),len(old_0517),ok.sum(),
             round(v1_0517.UNIONIZATION.mean(),4),round(old_0517.UNIONIZATION.mean(),4)]
})
comp.to_csv(OUT + 'unified_vs_finished.csv', index=False)
print(f"Correlation: {corr:.4f}, Mean diff: {mdiff:.4f}")
print(f"New coverage: {len(v1_0517)} fy, {v1_0517.gvkey.nunique()} gvkeys vs old: {len(old_0517)} fy, {old_0517.gvkey.nunique()} gvkeys")

# ====== MERGE WITH GLASSDOOR ======
print("\n=== Merging with Glassdoor ===")
gd = pd.read_parquet('/data/disk4/workspace/projects/union_glassdoor/outputs/20260702/firmyear_unionization/firmyear_glassdoor_panel.parquet')
gd['gvkey'] = gd['gvkey'].astype(float)
agg['Year'] = agg['Year'].astype(int)
merged = gd.merge(agg, left_on=['gvkey','review_year'], right_on=['gvkey','Year'], how='left')
merged['UNIONIZATION'] = merged['UNIONIZATION'].fillna(0.0)
merged['UNIONIZATION_cap1'] = merged['UNIONIZATION']
merged['UNIONIZATION_binary'] = (merged['UNIONIZATION']>0).astype(int)
merged['UNIONIZATION_raw'] = merged['UNIONIZATION_raw'].fillna(0.0)

# Merge Compustat controls
ctat_c = pd.read_parquet('/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/ctat_controls.parquet')
ctat_c['gvkey'] = ctat_c['gvkey'].astype(float)
ctat_c['fyear'] = ctat_c['fyear'].astype(int)
m2 = merged.merge(ctat_c, left_on=['gvkey','review_year'], right_on=['gvkey','fyear'], how='left', suffixes=('','_ctat'))
m2 = m2.drop_duplicates(subset=['gvkey','review_year'], keep='first')

n_var = m2.groupby('gvkey')['UNIONIZATION'].std().gt(0).sum()
print(f"Merged: {len(m2)} rows, {m2.gvkey.nunique()} gvkeys")
print(f"UNIONIZATION>0: {len(m2[m2.UNIONIZATION>0])} fy ({m2.UNIONIZATION.mean()*100:.2f}%)")
print(f"Unionized firms: {m2[m2.UNIONIZATION>0].gvkey.nunique()}, within-var: {n_var}")
print(f"Years with UNION>0: {sorted(m2[m2.UNIONIZATION>0].review_year.unique())}")

m2.to_parquet(OUT + 'unified_panel.parquet', index=False)

cov = pd.DataFrame({
    'metric':['total_fy','total_firms','unionized_fy','unionized_firms','mean_un','within_var_firms'],
    'value':[len(m2),m2.gvkey.nunique(),(m2.UNIONIZATION>0).sum(),
             m2[m2.UNIONIZATION>0].gvkey.nunique(),round(m2.UNIONIZATION.mean(),4),n_var]
})
cov.to_csv(OUT + 'unified_merge_coverage.csv', index=False)

# Save panel for no-Tier3 reconstruction
agg.to_parquet(OUT + 'unified_panel_base.parquet', index=False)
print(f"\nSaved: unified_panel.parquet + unified_panel_base.parquet + yearly + vs_finished + coverage + loss_table")
print("STEP 1 DONE")
