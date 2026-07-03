#!/usr/bin/env python3
"""
STEP 0+1: Inventory all F-7 files and build full notice-level dataset (1996-2026).
"""
import pandas as pd, numpy as np, os, re, glob, calendar

DATA_DIR = '/data/disk5/data/union/union f7/'
OUT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension/'
os.makedirs(OUT, exist_ok=True)

inventory_rows = []

# ====== Source 1: f7.csv.zip ======
print("=== Source 1: f7.csv.zip ===")
f7csv = pd.read_csv(os.path.join(DATA_DIR, 'f7.csv.zip'), low_memory=False)
f7csv['notice_date'] = pd.to_datetime(f7csv['notice_date'], errors='coerce')
f7csv['initiated_date'] = pd.to_datetime(f7csv['initiated_date'], errors='coerce')
f7csv['source'] = 'f7_csv_zip'

# Standardize columns
f7csv = f7csv.rename(columns={
    'employer': 'Employer', 'employer_state': 'Employer State',
    'bargaining_unit_size': 'Bargaining Unit Size',
    'establishment_size': 'Establishment Size',
    'category': 'Category',
    'notice_date': 'Notice Date', 'initiated_date': 'Initiated Date'
})
f7csv['year'] = f7csv['Initiated Date'].dt.year
f7csv.loc[f7csv['year'].isna(), 'year'] = f7csv['Notice Date'].dt.year

# Keep years not in unionized_rate_data (use f7.csv for 1996-2004, 2014-2021)
f7csv = f7csv[f7csv['year'].between(1996, 2021)]
f7csv_orig = len(f7csv)
print(f"  Rows: {len(f7csv)}, years: {int(f7csv.year.min())}-{int(f7csv.year.max())}")
inventory_rows.append({'source': 'f7_csv_zip', 'rows': len(f7csv), 'years': '1996-2021',
                       'employer_pct': f7csv.Employer.notna().mean(),
                       'state_pct': f7csv['Employer State'].notna().mean()})

# ====== Source 2: unionized_rate_data.csv (2005-2017, has cusip) ======
print("\n=== Source 2: unionized_rate_data.csv ===")
ur = pd.read_csv(os.path.join(DATA_DIR, 'unionized_rate_data.csv'), low_memory=False)
ur['source'] = 'unionized_rate_data'
ur = ur.rename(columns={'year': 'year_raw'})  # avoid conflict
ur['year'] = ur['year_raw']
print(f"  Rows: {len(ur)}, years: {ur.year.min()}-{ur.year.max()}")
print(f"  Columns: {[c for c in ur.columns if 'Employ' in c or 'cusip' in c or 'Bargain' in c or 'Establish' in c]}")
print(f"  CUSIP non-null: {ur.cusip.notna().sum()}/{len(ur)} ({ur.cusip.notna().mean()*100:.1f}%)")
inventory_rows.append({'source': 'unionized_rate_data', 'rows': len(ur), 'years': '2005-2017',
                       'employer_pct': ur.Employer.notna().mean(),
                       'state_pct': ur['Employer State'].notna().mean()})

# ====== Source 3: Monthly Excel files (2022-2026) ======
print("\n=== Source 3: Monthly Excel files (2022-2026) ===")
monthly_files = sorted(glob.glob(os.path.join(DATA_DIR, '*F7-Notices.xlsx')))
std_cols_map = {
    'Notice Date': 'Notice Date', 'Initiated Date': 'Initiated Date',
    'Employer': 'Employer', 'Employer Street': 'Employer Street',
    'Employer City': 'Employer City', 'Employer State': 'Employer State',
    'Employer ZIP': 'Employer ZIP', 'Bargaining Unit Size': 'Bargaining Unit Size',
    'Establishment Size': 'Establishment Size', 'Category': 'Category',
    'Union Name & Local Number': 'Union Name', 'NAICS': 'NAICS', 'Industry': 'Industry',
}

monthly_dfs = []
file_report = []
for f in monthly_files:
    fname = os.path.basename(f)
    # Parse year-month from filename
    m = re.search(r'([A-Z][a-z]+)-(\d{4})-F7', fname)
    if not m:
        continue
    month_name, year = m.group(1), int(m.group(2))
    if year < 2022:
        continue  # f7.csv.zip already covers through 2021

    try:
        df = pd.read_excel(f, skiprows=6)
        file_report.append({'file': fname, 'year': year, 'month': month_name, 'rows': len(df)})

        # Standardize columns — keep only known ones
        keep_cols = {}
        for c in df.columns:
            if c in std_cols_map:
                keep_cols[c] = std_cols_map[c]
        df = df.rename(columns=keep_cols)
        df = df[[c for c in keep_cols.values() if c in df.columns]]
        df['source'] = fname
        df['year'] = year
        monthly_dfs.append(df)
    except Exception as e:
        file_report.append({'file': fname, 'year': year, 'month': month_name, 'rows': f'ERROR: {e}'})

monthly_df = pd.concat(monthly_dfs, ignore_index=True) if monthly_dfs else pd.DataFrame()
print(f"  Files processed: {len(monthly_dfs)}")
print(f"  Total rows: {len(monthly_df)}")
print(f"  Years: {sorted(monthly_df.year.unique())}")

# Save file inventory
inv_df = pd.DataFrame(inventory_rows)
inv_df.to_csv(OUT + 'step0_file_inventory.csv', index=False)

# Save monthly file report
pd.DataFrame(file_report).to_csv(OUT + 'step0_monthly_file_report.csv', index=False)

# ====== STEP 1: Merge all sources ======
print("\n=== STEP 1: Building full notice dataset ===")

# Standardize all sources to common columns
common_cols = ['Employer', 'Employer State', 'Bargaining Unit Size', 'Establishment Size',
               'Category', 'year', 'Notice Date', 'Initiated Date', 'source']

# From f7.csv.zip
f7csv_sub = f7csv[[c for c in common_cols if c in f7csv.columns]].copy()
if 'cusip' in f7csv.columns:
    f7csv_sub['cusip'] = f7csv['cusip']
else:
    f7csv_sub['cusip'] = np.nan

# From unionized_rate_data
ur_sub = ur[[c for c in common_cols if c in ur.columns]].copy()
ur_sub['cusip'] = ur['cusip'] if 'cusip' in ur.columns else np.nan

# From monthly Excel
if len(monthly_df) > 0:
    monthly_sub = monthly_df[[c for c in common_cols if c in monthly_df.columns]].copy()
    monthly_sub['cusip'] = np.nan
else:
    monthly_sub = pd.DataFrame()

# Combine
all_notices = pd.concat([f7csv_sub, ur_sub, monthly_sub], ignore_index=True)
print(f"Combined: {len(all_notices)} rows before dedup")

# Deduplicate by Employer + Union (if available) + Notice Date + BUS
dedup_cols = ['Employer', 'Notice Date', 'Bargaining Unit Size']
dedup_cols = [c for c in dedup_cols if c in all_notices.columns]
n_before = len(all_notices)
all_notices = all_notices.drop_duplicates(subset=dedup_cols, keep='first')
print(f"After dedup: {len(all_notices)} rows (removed {n_before - len(all_notices)})")

# Clean numeric columns
for col in ['Bargaining Unit Size', 'Establishment Size']:
    if col in all_notices.columns:
        all_notices[col] = pd.to_numeric(all_notices[col], errors='coerce')

# Filter BUS <= EST, EST > 0
bus = all_notices.get('Bargaining Unit Size', pd.Series(dtype=float))
est = all_notices.get('Establishment Size', pd.Series(dtype=float))
valid = (bus <= est) & (est > 0)
all_notices = all_notices[valid.fillna(False)].copy()
print(f"After BUS<=EST & EST>0: {len(all_notices)} rows")

# Year coverage
print(f"\nYear distribution (all notices):")
all_notices['year'] = all_notices['year'].astype(int)
yr_counts = all_notices.groupby('year').size()
for y in range(2005, 2027):
    n = yr_counts.get(y, 0)
    marker = " ← NEW" if y >= 2022 else ""
    print(f"  {y}: {n:>7d}{marker}")

# Save
all_notices.to_parquet(OUT + 'f7_notices_all_years.parquet', index=False)
print(f"\nSaved f7_notices_all_years.parquet: {len(all_notices)} rows, {all_notices.Employer.nunique()} unique employers")
print(f"Year range: {all_notices.year.min()}-{all_notices.year.max()}")
