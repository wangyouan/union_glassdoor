import pandas as pd

OUT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260702/firmyear_unionization/fmcs_aligned'

# Check merged panel gvkeys
panel = pd.read_parquet(f'{OUT}/fmcs_unionization_panel.parquet')
print('Merged panel gvkeys (first 10):', sorted(panel['gvkey'].unique())[:10])

# Load raw FMCS
fmcs = pd.read_csv('/data/disk5/data/union/union f7/unionized_rate_data.csv')
fmcs_f = fmcs[(fmcs['cusip'].notna()) & (fmcs['Bargaining Unit Size'] <= fmcs['Establishment Size']) & (fmcs['Establishment Size'] > 0)]
fmcs_f['cusip_str'] = fmcs_f['cusip'].apply(lambda c: str(c).strip().zfill(9))

# Check a specific cusip: does year vary? Does BUS/EST vary?
c = fmcs_f['cusip_str'].iloc[0]
sub = fmcs_f[fmcs_f['cusip_str']==c][['Notice Date','Initiated Date','year','Bargaining Unit Size','Establishment Size']]
print(f'\nCusip {c}:')
print(sub.head(20).to_string())

# Key diagnostics
print(f'\nUnique (cusip, year) combos: {fmcs_f.groupby(["cusip_str","year"]).ngroups:,}')
print(f'Unique (cusip, Initiated Date) combos: {fmcs_f.groupby(["cusip_str","Initiated Date"]).ngroups:,}')

# Count firms where BUS varies across years
bus_var = fmcs_f.groupby('cusip_str')['Bargaining Unit Size'].nunique()
print(f'Cusips with >1 unique BUS: {(bus_var>1).sum():,} / {len(bus_var):,}')

# Show a firm where BUS varies
if (bus_var > 1).any():
    c_var = bus_var[bus_var > 1].index[0]
    sub2 = fmcs_f[fmcs_f['cusip_str']==c_var][['Notice Date','Initiated Date','year','Bargaining Unit Size','Establishment Size']]
    print(f'\nFirm with varying BUS (cusip={c_var}):')
    print(sub2.head(20).to_string())
