#!/usr/bin/env python
"""Rebuild UNIONIZATION panel properly: contract active only during [Initiated, Expiration]."""

import pandas as pd
import numpy as np
import zipfile, os, warnings
warnings.filterwarnings('ignore')

OUT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260702/firmyear_unionization/fmcs_aligned'

# ─── Cusip→gvkey bridge ─────────────────────────────────────────────────
print("Building cusip→gvkey bridge...")
zf = zipfile.ZipFile('/data/disk5/data/compustat/1950_2023_ctat_firm_identifier.zip')
firm_id = pd.read_csv(zf.open('igrtx61clgeqj9k3.csv'))
firm_id = firm_id.dropna(subset=['cusip'])
firm_id['cusip_clean'] = firm_id['cusip'].astype(str).str.strip().str.zfill(9)
bridge = firm_id.sort_values('fyear').groupby('cusip_clean').agg(
    gvkey=('gvkey','last')).reset_index()
bridge['gvkey'] = bridge['gvkey'].astype(str).str.zfill(6)
bridge_map = dict(zip(bridge['cusip_clean'], bridge['gvkey']))

# ─── Load FMCS ──────────────────────────────────────────────────────────
print("Loading FMCS...")
fmcs = pd.read_csv('/data/disk5/data/union/union f7/unionized_rate_data.csv')

# Filter: BUS <= EST, EST > 0, cusip not null
fmcs = fmcs[(fmcs['cusip'].notna()) &
            (fmcs['Bargaining Unit Size'] <= fmcs['Establishment Size']) &
            (fmcs['Establishment Size'] > 0)].copy()

# Map to gvkey
fmcs['cusip_str'] = fmcs['cusip'].apply(lambda c: str(c).strip().zfill(9))
fmcs['gvkey'] = fmcs['cusip_str'].map(bridge_map)
fmcs = fmcs[fmcs['gvkey'].notna()].copy()
print(f"FMCS matched: {len(fmcs):,} notices, {fmcs['gvkey'].nunique():,} gvkeys")

# ─── Parse dates ────────────────────────────────────────────────────────
fmcs['initiated_date'] = pd.to_datetime(fmcs['Initiated Date'])
fmcs['expiration_date'] = pd.to_datetime(fmcs['Expiration Date'])

# Initiated year
fmcs['init_year'] = fmcs['initiated_date'].dt.year

# Expiration year: if missing, use initiated_year + 3 (typical contract length)
fmcs['exp_year'] = fmcs['expiration_date'].dt.year
mask_no_exp = fmcs['exp_year'].isna()
fmcs.loc[mask_no_exp, 'exp_year'] = fmcs.loc[mask_no_exp, 'init_year'] + 3
fmcs['exp_year'] = fmcs['exp_year'].fillna(fmcs['init_year'] + 3).astype(int)

# Cap at 2017 (Glassdoor data range for matched sample)
fmcs['exp_year'] = fmcs['exp_year'].clip(upper=2017)

print(f"Initiated year range: {fmcs['init_year'].min()}-{fmcs['init_year'].max()}")
print(f"Expiration year range: {fmcs['exp_year'].min()}-{fmcs['exp_year'].max()}")
print(f"Contract duration (years): mean={(fmcs['exp_year'] - fmcs['init_year']).mean():.1f}, "
      f"median={(fmcs['exp_year'] - fmcs['init_year']).median():.0f}")

# ─── Expand: each notice → one row per active year ─────────────────────
print("Expanding contracts to active years...")
rows = []
for _, r in fmcs.iterrows():
    for yr in range(int(r['init_year']), int(r['exp_year']) + 1):
        rows.append({
            'gvkey': r['gvkey'],
            'year': yr,
            'bus': r['Bargaining Unit Size'],
            'est': r['Establishment Size']
        })

expanded = pd.DataFrame(rows)
print(f"Expanded: {len(expanded):,} notice-years")

# ─── Aggregate per (gvkey, year) ───────────────────────────────────────
panel = expanded.groupby(['gvkey','year']).agg(
    sum_bus=('bus','sum'),
    sum_est=('est','sum'),
    n_notices=('bus','count')
).reset_index()
panel['UNIONIZATION'] = panel['sum_bus'] / panel['sum_est']

# Stats
unionized = panel[panel['UNIONIZATION'] > 0]
print(f"\nUnionized gvkey-years: {len(unionized):,}")
print(f"UNIONIZATION (unionized): mean={unionized['UNIONIZATION'].mean():.4f}, med={unionized['UNIONIZATION'].median():.4f}")
print(f"  P25={unionized['UNIONIZATION'].quantile(0.25):.4f}, P75={unionized['UNIONIZATION'].quantile(0.75):.4f}")
print(f"  >1: {(unionized['UNIONIZATION']>1).sum():,}")
print(f"  Unique gvkeys: {unionized['gvkey'].nunique():,}")

# Check within-firm variation
n_uq = panel.groupby('gvkey')['UNIONIZATION'].nunique()
print(f"Firms with >1 unique UNIONIZATION: {(n_uq>1).sum():,} / {len(n_uq):,} ({(n_uq>1).mean()*100:.1f}%)")

# Show examples of variation
var_firms = n_uq[n_uq > 1].index[:3]
for gv in var_firms:
    sub = panel[panel['gvkey']==gv][['year','UNIONIZATION','n_notices']].sort_values('year')
    print(f"\n  gvkey={gv}:")
    print(sub.head(15).to_string(index=False))

# ─── Merge with Compustat + Glassdoor ──────────────────────────────────
print("\nMerging with Compustat + Glassdoor...")
cmp = pd.read_parquet('outputs/compustat_firm_controls.parquet')
cmp['gvkey'] = cmp['gvkey'].astype(str).str.zfill(6)
cmp_0517 = cmp[(cmp['fyear']>=2005)&(cmp['fyear']<=2017)][['gvkey','fyear','emp','at','L_leverage','L_roa','L_log_emp','L_size']].drop_duplicates()

gd = pd.read_parquet('outputs/20260702/firmyear_unionization/firmyear_glassdoor_panel.parquet')
gd['gvkey'] = gd['gvkey'].astype(str).str.zfill(6)

cmp_gd = cmp_0517.merge(gd, left_on=['gvkey','fyear'], right_on=['gvkey','review_year'], how='inner')
print(f"Compustat∩Glassdoor: {len(cmp_gd):,} rows, {cmp_gd['gvkey'].nunique():,} firms")

# Merge UNIONIZATION
panel['gvkey'] = panel['gvkey'].astype(str)
full = cmp_gd.merge(panel[['gvkey','year','UNIONIZATION','n_notices']],
                    left_on=['gvkey','fyear'], right_on=['gvkey','year'], how='left')
full.drop(columns=['year'], inplace=True)
full['UNIONIZATION'] = full['UNIONIZATION'].fillna(0)
full['has_union'] = (full['UNIONIZATION'] > 0).astype(int)
full['SIZE'] = np.log(full['at'])
full['LEVERAGE'] = full['L_leverage']

print(f"Full panel: {len(full):,} rows, {full['gvkey'].nunique():,} firms")
print(f"UNIONIZATION>0: {(full['UNIONIZATION']>0).sum():,} ({(full['UNIONIZATION']>0).mean()*100:.1f}%)")
print(f"UNIONIZATION mean (all): {full['UNIONIZATION'].mean():.4f}")
print(f"UNIONIZATION mean (if>0): {full.loc[full['UNIONIZATION']>0,'UNIONIZATION'].mean():.4f}")

# Within-firm variation in final panel
full_3yr = full.groupby('gvkey').filter(lambda x: len(x) >= 3)
n_f = full_3yr['gvkey'].nunique()
n_var = full_3yr.groupby('gvkey')['UNIONIZATION'].nunique()
print(f"Firms >=3yr: {n_f:,}, with UNIONIZATION var: {(n_var>1).sum():,} ({(n_var>1).mean()*100:.1f}%)")

# Save
full.to_parquet(f'{OUT}/fmcs_unionization_panel.parquet', index=False)
print(f"\nSaved: {len(full.columns)} columns")

# Save descriptives
desc_rows = [
    {'metric': 'n_gvkey_years', 'value': len(full)},
    {'metric': 'n_gvkeys', 'value': full['gvkey'].nunique()},
    {'metric': 'n_unionized_fy', 'value': int((full['UNIONIZATION']>0).sum())},
    {'metric': 'pct_unionized', 'value': round((full['UNIONIZATION']>0).mean()*100, 1)},
    {'metric': 'union_mean_all', 'value': round(full['UNIONIZATION'].mean(), 4)},
    {'metric': 'union_median_all', 'value': round(full['UNIONIZATION'].median(), 4)},
    {'metric': 'union_mean_if_pos', 'value': round(full.loc[full['UNIONIZATION']>0,'UNIONIZATION'].mean(), 4)},
    {'metric': 'union_median_if_pos', 'value': round(full.loc[full['UNIONIZATION']>0,'UNIONIZATION'].median(), 4)},
    {'metric': 'firms_with_var', 'value': int((n_var>1).sum())},
]
pd.DataFrame(desc_rows).to_csv(f'{OUT}/fmcs_descriptives.csv', index=False)
print("Saved descriptives.\nDone.")
