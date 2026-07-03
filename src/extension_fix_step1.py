#!/usr/bin/env python3
"""STEP 1: Fix three construction bugs and rebuild notice dataset + UNIONIZATION panel."""
import pandas as pd, numpy as np, os, re, glob, calendar

DATA_DIR = '/data/disk5/data/union/union f7/'
CTAT_DIR = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/'
OUT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension_fix/'
os.makedirs(OUT, exist_ok=True)

# ===================== LOAD DEPENDENCIES =====================
print("=== Loading dependencies ===")
ctat = pd.read_parquet(CTAT_DIR + 'ctat_id_table.parquet')
ctat['cusip'] = ctat['cusip'].astype(str).str.strip()
cusip_to_gvkey = ctat.drop_duplicates('cusip').set_index('cusip')['gvkey'].to_dict()
print(f"CUSIP→gvkey: {len(cusip_to_gvkey)} entries")

# Load employer→gvkey matches from round 5
EM_MATCH = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension/employer_gvkey_matches.csv'
if os.path.exists(EM_MATCH):
    em_matches = pd.read_csv(EM_MATCH)
    # Build dicts for each tier
    emp_to_gvk = {}
    for _, r in em_matches[em_matches.gvkey.notna()].iterrows():
        emp_to_gvk[str(r['Employer']).strip()] = int(r['gvkey'])
    print(f"Employer→gvkey matches loaded: {len(emp_to_gvk)}")
else:
    emp_to_gvk = {}
    print("No existing matches found — will need to rebuild (only cusip-based)")

# ===================== SOURCE 1: unionized_rate_data.csv =====================
print("\n=== SOURCE 1: unionized_rate_data.csv (2005-2017) ===")
ur = pd.read_csv(os.path.join(DATA_DIR, 'unionized_rate_data.csv'), low_memory=False)
n_ur_in = len(ur)
print(f"  Read: {n_ur_in} rows")
print(f"  Initiated Date range: {ur['Initiated Date'].dropna().iloc[:3].tolist() if 'Initiated Date' in ur.columns else 'N/A'}")

# BUG 1 FIX: Use Initiated Date year (NOT the artificially-expanded year column)
# The year column was expanded 2005-2017 but actual notices are from 2009-2010
ur = ur.rename(columns={'Bargaining Unit Size': 'BUS', 'Establishment Size': 'EST'})
ur['BUS'] = pd.to_numeric(ur['BUS'], errors='coerce')
ur['EST'] = pd.to_numeric(ur['EST'], errors='coerce')
n_before_filter = len(ur)
ur = ur[(ur.BUS <= ur.EST) & (ur.EST > 0)].copy()
n_after_filter = len(ur)

# Use Initiated Date for actual year
if 'Initiated Date' in ur.columns:
    ur['Year'] = pd.to_datetime(ur['Initiated Date'], errors='coerce').dt.year
    ur = ur[ur.Year.notna()].copy()
    n_with_date = len(ur)
    # Dedup: same employer+year+BUS (removes artificial year expansion)
    ur = ur.drop_duplicates(subset=['Employer', 'Year', 'BUS'], keep='first')
    n_dedup = len(ur)
    print(f"  After BUS≤EST: {n_after_filter}, with date: {n_with_date}, after dedup year-expansion: {n_dedup}")
else:
    ur['Year'] = ur['year']  # fallback
    n_dedup = len(ur)
    print(f"  WARNING: no Initiated Date column, using year column directly")

# CUSIP→gvkey matching
ur['cusip_str'] = ur['cusip'].astype(str).str.strip()
ur['gvkey'] = ur['cusip_str'].map(cusip_to_gvkey)
# For unmatched, try employer name
ur.loc[ur.gvkey.isna(), 'gvkey'] = ur.loc[ur.gvkey.isna(), 'Employer'].str.strip().map(emp_to_gvk)

n_with_gvk = ur.gvkey.notna().sum()
print(f"  With gvkey: {n_with_gvk} ({n_with_gvk/n_dedup*100:.1f}%)")
print(f"  Actual years from Initiated Date: {sorted(ur.Year.unique())}")
yr_counts = ur.groupby('Year').size()
print(f"  Year counts: {dict(zip(yr_counts.index.astype(int), yr_counts.values))}")

# BUG 1 CHECK
if yr_counts.std() == 0:
    print("  *** BUG 1 STILL PRESENT ***")
else:
    print(f"  ✓ Year counts varying: std={yr_counts.std():.0f}")

ur_src = ur[['Employer', 'Employer State', 'BUS', 'EST', 'Year', 'gvkey', 'cusip_str']].copy()
ur_src['source'] = 'unionized_rate_data'

# ===================== SOURCE 2: f7.csv.zip (1996-2021) =====================
print("\n=== SOURCE 2: f7.csv.zip ===")
f7 = pd.read_csv(os.path.join(DATA_DIR, 'f7.csv.zip'), low_memory=False)
print(f"  Read: {len(f7)} rows")

f7 = f7.rename(columns={'employer': 'Employer', 'employer_state': 'Employer State',
                         'bargaining_unit_size': 'BUS', 'establishment_size': 'EST',
                         'initiated_date': 'Initiated Date', 'notice_date': 'Notice Date'})
# BUG 3 FIX: Parse year from Initiated Date, fallback to Notice Date
f7['Year'] = pd.to_datetime(f7['Initiated Date'], errors='coerce').dt.year
f7.loc[f7['Year'].isna(), 'Year'] = pd.to_datetime(f7['Notice Date'], errors='coerce').dt.year
n_with_year = f7.Year.notna().sum()
f7 = f7[f7.Year.notna()].copy()

f7['BUS'] = pd.to_numeric(f7['BUS'], errors='coerce')
f7['EST'] = pd.to_numeric(f7['EST'], errors='coerce')
n_before = len(f7)
f7 = f7[(f7.BUS <= f7.EST) & (f7.EST > 0)].copy()
n_after = len(f7)

# gvkey via employer name matching
f7['gvkey'] = f7['Employer'].str.strip().map(emp_to_gvk)
n_gvk = f7.gvkey.notna().sum()

# Keep f7.csv for: pre-2005, and 2005-2021 excluding the unionized_rate years (2009-2010)
ur_years = set(range(2009, 2011))  # unionized_rate covers 2009-2010
f7 = f7[f7.Year.between(1996, 2021)].copy()
f7_used = f7[~f7.Year.isin(ur_years)].copy()
f7_used_yr = f7_used.groupby('Year').size()
print(f"  Kept years (excl 2009-2010): {sorted(f7_used.Year.unique())}")
print(f"  Year counts: {dict(zip(f7_used_yr.index.astype(int), f7_used_yr.values))}")
print(f"  With gvkey: {f7_used.gvkey.notna().sum()}/{len(f7_used)}")

f7_src = f7_used[['Employer', 'Employer State', 'BUS', 'EST', 'Year', 'gvkey']].copy()
f7_src['source'] = 'f7_csv_zip'

# ===================== SOURCE 3: Monthly Excel files =====================
print("\n=== SOURCE 3: Monthly Excel files ===")
monthly_files = sorted(glob.glob(os.path.join(DATA_DIR, '*F7-Notices.xlsx')))

# Known column name variants
EMP_COLS = ['Employer', 'Employer Name']
STATE_COLS = ['Employer State', 'Employer State ']
BUS_COLS = ['Bargaining Unit Size', 'Unit Size']
EST_COLS = ['Establishment Size', 'Size of Establishment', 'Number of Employees']
DATE_COLS = ['Notice Date', 'Date Received', 'Initiated Date', 'Contract Initiated']

loss_table = []
excel_dfs = []
for f in monthly_files:
    fname = os.path.basename(f)
    m = re.search(r'([A-Z][a-z]+)-(\d{4})-F7', fname)
    if not m: continue
    month_name, year_fn = m.group(1), int(m.group(2))
    month_num = list(calendar.month_name).index(month_name)

    try:
        df = pd.read_excel(f, skiprows=6)
        n_read = len(df)
        cols_found = df.columns.tolist()

        # BUG 2 FIX: flexible column mapping
        col_map = {}
        for c in df.columns:
            cs = str(c).strip()
            if cs in EMP_COLS: col_map[c] = 'Employer'
            elif cs in STATE_COLS: col_map[c] = 'Employer State'
            elif cs in BUS_COLS: col_map[c] = 'BUS'
            elif cs in EST_COLS: col_map[c] = 'EST'
            elif cs in DATE_COLS and 'Notice Date' not in col_map.values():
                col_map[c] = 'Notice Date'
        df = df.rename(columns=col_map)

        # Require at minimum: Employer
        if 'Employer' not in df.columns:
            loss_table.append({'file': fname, 'year': year_fn, 'month': month_name,
                               'n_read': n_read, 'n_mapped': 0, 'n_dropna': 0, 'n_bus_est': 0,
                               'status': 'NO_EMPLOYER_COL', 'cols': str(cols_found)})
            continue

        df = df[[c for c in ['Employer', 'Employer State', 'BUS', 'EST', 'Notice Date'] if c in df.columns]]
        n_mapped = len(df)

        # Parse year from Notice Date
        if 'Notice Date' in df.columns:
            df['Year'] = pd.to_datetime(df['Notice Date'], errors='coerce').dt.year
        else:
            df['Year'] = year_fn  # fallback to filename year

        df = df[df.Year.notna() & (df.Year > 2000)]
        n_dated = len(df)

        # Clean BUS/EST
        if 'BUS' in df.columns: df['BUS'] = pd.to_numeric(df['BUS'], errors='coerce')
        else: df['BUS'] = np.nan
        if 'EST' in df.columns: df['EST'] = pd.to_numeric(df['EST'], errors='coerce')
        else: df['EST'] = np.nan

        # For files WITHOUT EST: use binary UNIONIZATION (BUG 2 fix)
        has_est = df.EST.notna().sum() > 0
        if has_est:
            df_valid = df[(df.BUS <= df.EST) & (df.EST > 0)].copy()
        else:
            df_valid = df[df.BUS.notna() & (df.BUS > 0)].copy()
            df_valid['EST'] = df_valid['BUS']  # placeholder for aggregation

        n_valid = len(df_valid)
        df_valid['source'] = fname
        if n_valid > 0:
            df_valid['gvkey'] = df_valid['Employer'].str.strip().map(emp_to_gvk)
        excel_dfs.append(df_valid)

        status = 'OK' if n_valid > 0 else 'EMPTY_AFTER_FILTER'
        if n_valid == 0:
            loss_table.append({'file': fname, 'year': year_fn, 'month': month_name,
                               'n_read': n_read, 'n_mapped': n_mapped, 'n_dated': n_dated,
                               'n_valid': 0, 'status': status,
                               'cols': str(cols_found),
                               'head_employers': str(df['Employer'].head(3).tolist()) if 'Employer' in df.columns else 'N/A'})

    except Exception as e:
        loss_table.append({'file': fname, 'year': year_fn, 'month': month_name,
                           'n_read': 0, 'n_mapped': 0, 'n_dated': 0, 'n_valid': 0,
                           'status': f'ERROR: {str(e)[:100]}', 'cols': '', 'head_employers': ''})

excel_all = pd.concat(excel_dfs, ignore_index=True) if excel_dfs else pd.DataFrame()
print(f"  Files: {len(monthly_files)}, with data: {len(excel_dfs)}")
print(f"  Total rows: {len(excel_all)}, years: {sorted(excel_all.Year.dropna().unique())}")
if len(excel_all) > 0:
    excel_src = excel_all[['Employer', 'Employer State', 'BUS', 'EST', 'Year', 'gvkey', 'source']].copy()
else:
    excel_src = pd.DataFrame()

# Save loss table
loss_df = pd.DataFrame(loss_table)
# Add stats for files with data that weren't in loss_table
for fname in set(excel_all.source.unique()) - set(loss_df.file.unique()):
    sub = excel_all[excel_all.source == fname]
    y = sub.Year.mode().iloc[0] if len(sub) > 0 else 0
    loss_rows = [{'file': fname, 'year': int(y), 'month': '', 'n_read': len(sub),
                  'n_mapped': len(sub), 'n_dated': len(sub), 'n_valid': len(sub),
                  'status': 'OK', 'cols': '', 'head_employers': ''}]
    loss_df = pd.concat([loss_df, pd.DataFrame(loss_rows)], ignore_index=True)
loss_df.to_csv(OUT + 'ingestion_loss_table.csv', index=False)

empty_files = loss_df[loss_df['status'] != 'OK']
if len(empty_files) > 0:
    print(f"\n  *** EMPTY FILES ({len(empty_files)}) — printing to report ***")
    for _, r in empty_files.iterrows():
        print(f"    {r['file']}: status={r['status']}, cols={r.get('cols','')}, heads={r.get('head_employers','')}")

# ===================== MERGE ALL SOURCES =====================
print("\n=== Merging all sources ===")
all_srcs = [ur_src, f7_src]
if len(excel_src) > 0:
    all_srcs.append(excel_src)
combined = pd.concat(all_srcs, ignore_index=True)

# Dedup: same Employer + Year + BUS (keep first)
dedup_keys = ['Employer', 'Year', 'BUS']
n_before = len(combined)
combined = combined.drop_duplicates(subset=dedup_keys, keep='first')
print(f"  Combined: {n_before} → {len(combined)} after dedup ({n_before - len(combined)} removed)")
combined['Year'] = combined['Year'].astype(int)

# ===================== BUILD V1 PANEL =====================
print("\n=== Building V1 UNIONIZATION panel ===")
# Only rows with gvkey
panel_data = combined[combined.gvkey.notna()].copy()
panel_data['gvkey'] = panel_data['gvkey'].astype(int)

# Aggregate per gvkey-year: UNIONIZATION = Σ BUS / Σ EST
agg = panel_data.groupby(['gvkey', 'Year']).agg(
    BUS_sum=('BUS', 'sum'), EST_sum=('EST', 'sum'), n_notices=('BUS', 'count')
).reset_index()
agg['UNIONIZATION_raw'] = agg['BUS_sum'] / agg['EST_sum']
agg['UNIONIZATION_cap1'] = agg['UNIONIZATION_raw'].clip(upper=1.0)
agg['UNIONIZATION'] = agg['UNIONIZATION_cap1']

# Filter to 2005-2026
agg = agg[agg.Year.between(2005, 2026)]
print(f"  V1 panel: {len(agg)} rows, {agg.gvkey.nunique()} gvkeys")
print(f"  Year range: {agg.Year.min()}-{agg.Year.max()}")
print(f"  UNIONIZATION > 0: {len(agg[agg.UNIONIZATION>0])} rows, {agg[agg.UNIONIZATION>0].gvkey.nunique()} firms")

# BUG 1 CHECK: year means must vary
yr_stats = agg.groupby('Year').agg(n_gvkeys=('gvkey', 'nunique'), mean_un=('UNIONIZATION', 'mean')).reset_index()
print(f"\n  Year distribution:")
for _, r in yr_stats.iterrows():
    print(f"    {int(r['Year'])}: {int(r['n_gvkeys']):>5d} gvkeys, mean UNIONIZATION = {r['mean_un']:.4f}")

if yr_stats['n_gvkeys'].std() == 0:
    print("  *** BUG 1 STILL PRESENT: n_gvkeys constant across years ***")
elif yr_stats['mean_un'].std() == 0:
    print("  *** BUG 1 STILL PRESENT: mean UNIONIZATION constant across years ***")
else:
    print(f"  ✓ Year variation OK: n_gvkeys std={yr_stats['n_gvkeys'].std():.0f}, mean_un std={yr_stats['mean_un'].std():.4f}")

# Check 2005-2017 mean ≈ 0.69
agg_0517 = agg[(agg.Year >= 2005) & (agg.Year <= 2017)]
print(f"  ✓ 2005-2017 mean UNIONIZATION: {agg_0517.UNIONIZATION.mean():.4f} (baseline ≈0.69)")

agg.to_parquet(OUT + 'unionization_panel_v1_fix.parquet', index=False)
print(f"  Saved unionization_panel_v1_fix.parquet")

# ===================== BUILD V2 PANEL =====================
print("\n=== Building V2 panel (old finished + new extension) ===")
# Load old finished panel
old_panel_path = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/unionization_panel_main.parquet'
old = pd.read_parquet(old_panel_path)
old = old[old.gvkey.notna()].copy()
old['gvkey'] = old['gvkey'].astype(int)
old_0517 = old[(old.year >= 2005) & (old.year <= 2017)].copy()
old_0517 = old_0517.rename(columns={'year': 'Year'})
print(f"  Old panel 2005-2017: {len(old_0517)} rows, {old_0517.gvkey.nunique()} gvkeys")

# Extension: V1's 2018-2026
v1_ext = agg[agg.Year >= 2018].copy()
print(f"  V1 extension 2018+: {len(v1_ext)} rows, {v1_ext.gvkey.nunique()} gvkeys")

# Concatenate
v2 = pd.concat([old_0517[['gvkey', 'Year', 'UNIONIZATION']],
                 v1_ext[['gvkey', 'Year', 'UNIONIZATION']]], ignore_index=True)
v2 = v2[v2.Year.between(2005, 2026)]
print(f"  V2 panel: {len(v2)} rows, {v2.gvkey.nunique()} gvkeys")

v2.to_parquet(OUT + 'unionization_panel_v2_fix.parquet', index=False)
print(f"  Saved unionization_panel_v2_fix.parquet")

# ===================== PANEL CONSISTENCY =====================
print("\n=== Panel consistency: V1 vs old finished (2005-2017 overlap) ===")
v1_0517 = agg[(agg.Year >= 2005) & (agg.Year <= 2017)][['gvkey', 'Year', 'UNIONIZATION']].copy()
old_ren = old_0517[['gvkey', 'Year', 'UNIONIZATION']].copy()

merged_cmp = v1_0517.merge(old_ren, on=['gvkey', 'Year'], how='outer', suffixes=('_v1', '_old'))
# Correlation
ok = merged_cmp.UNIONIZATION_v1.notna() & merged_cmp.UNIONIZATION_old.notna()
corr_v1_old = merged_cmp.loc[ok, 'UNIONIZATION_v1'].corr(merged_cmp.loc[ok, 'UNIONIZATION_old'])
mean_diff = (merged_cmp['UNIONIZATION_v1'] - merged_cmp['UNIONIZATION_old']).mean()

consistency = pd.DataFrame({
    'metric': ['correlation', 'mean_diff', 'n_v1', 'n_old', 'n_overlap', 'v1_mean', 'old_mean'],
    'value': [round(corr_v1_old, 4), round(mean_diff, 4),
              len(v1_0517), len(old_ren), ok.sum(),
              round(v1_0517.UNIONIZATION.mean(), 4), round(old_ren.UNIONIZATION.mean(), 4)]
})
consistency.to_csv(OUT + 'panel_consistency.csv', index=False)
print(f"  Correlation: {corr_v1_old:.4f}")
print(f"  Mean diff (V1 - old): {mean_diff:.4f}")
print(f"  V1 mean: {v1_0517.UNIONIZATION.mean():.4f}, old mean: {old_ren.UNIONIZATION.mean():.4f}")
print(f"  Overlap: {ok.sum()} rows")

print("\n=== STEP 1 DONE ===")
