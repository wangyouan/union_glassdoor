#!/usr/bin/env python3
"""
STEP 2: Employer Name Fuzzy Match
- 2a: Match 82 unmatched CUSIPs by Employer name → Compustat conm
- 2b: Expand sample using notice-level Employer names for notices without CUSIP
"""
import pandas as pd, numpy as np, re, os
from rapidfuzz import process, fuzz

OUT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_unionization/finished_panel/'
FUZZY_OUT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/firmyear_fill_missing/'
os.makedirs(FUZZY_OUT, exist_ok=True)

# ---- 1. Standardization ----
def normalize_name(name):
    """Standardize company name: uppercase, remove punctuation/suffixes/spaces"""
    if pd.isna(name):
        return ""
    s = str(name).upper().strip()
    s = re.sub(r'[^\w\s]', ' ', s)  # punctuation → space
    # Remove common suffixes
    suffixes = [
        r'\bINC\b', r'\bCORP\b', r'\bCORPORATION\b', r'\bCO\b', r'\bCOMPANY\b',
        r'\bLLC\b', r'\bLTD\b', r'\bLIMITED\b', r'\bLP\b', r'\bPLC\b',
        r'\bHOLDINGS\b', r'\bGROUP\b', r'\bINTERNATIONAL\b', r'\bTHE\b',
        r'\bINCORPORATED\b', r'\bL\s?L\s?C\b', r'\bL\s?P\b',
    ]
    for pat in suffixes:
        s = re.sub(pat, '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def token_sort_ratio(a, b):
    """Token sort ratio using rapidfuzz"""
    return fuzz.token_sort_ratio(a, b, processor=utils.default_process) / 100.0

# ---- 2. Load data ----
print("Loading data...")
# Unmatched CUSIPs from STEP 1
bridge = pd.read_csv(OUT + 'bridge_report.csv')
unmatched = bridge[bridge['match_status'] == 'unmatched_cusip'][['cusip', 'year', 'bargaining_unit_rate']].drop_duplicates()
print(f"Unmatched CUSIPs: {unmatched.cusip.nunique()} ({len(unmatched)} rows)")

# Notice-level data
notices = pd.read_csv('/data/disk5/data/union/union f7/unionized_rate_data.csv',
                       usecols=['Employer', 'Employer State', 'cusip', 'year',
                                'Bargaining Unit Size', 'Establishment Size'])
notices['cusip'] = notices['cusip'].astype(str).str.strip()
notices.loc[notices['cusip'] == 'nan', 'cusip'] = np.nan
print(f"Notices: {len(notices)} rows, cusip non-null: {notices.cusip.notna().sum()} ({notices.cusip.notna().mean()*100:.1f}%)")

# Compustat ID table (unique gvkey × conm, latest year)
ctat = pd.read_parquet(OUT + 'ctat_id_table.parquet')
ctat_latest = ctat.sort_values('fyear').groupby('gvkey').last().reset_index()
print(f"Compustat firms: {len(ctat_latest)}")

# ---- 3. Prepare for matching ----
# Normalize Compustat names
ctat_latest['conm_norm'] = ctat_latest['conm'].apply(normalize_name)

# Build dictionaries for fast lookup
conm_to_gvkey = {}     # exact match: normalized name → gvkey
conm_choices = []      # all normalized names for fuzzy matching
conm_gvkey_map = {}    # normalized name → gvkey (for fuzzy results)
for _, cr in ctat_latest.iterrows():
    cn = cr['conm_norm']
    if cn and len(cn) >= 3:
        if cn in conm_to_gvkey:
            conm_to_gvkey[cn] = None  # mark ambiguous
        else:
            conm_to_gvkey[cn] = cr['gvkey']
        if cn not in conm_gvkey_map:
            conm_gvkey_map[cn] = cr['gvkey']
        if cn not in conm_choices:
            conm_choices.append(cn)

print(f"\nCompustat names: {len(conm_choices)} unique normalized")
print(f"Exact-matchable names: {sum(1 for v in conm_to_gvkey.values() if v is not None)}")

def fuzzy_match_one(emp_name, emp_state, threshold=85):
    """Use rapidfuzz to find best match above threshold. Returns (gvkey, conm, score, match_type)."""
    if not emp_name or len(emp_name) < 3:
        return (None, None, 0, 'too_short')

    # Try exact match first
    gvkey = conm_to_gvkey.get(emp_name)
    if gvkey is not None:
        return (gvkey, emp_name, 100, 'exact_norm')
    elif emp_name in conm_to_gvkey:
        return (None, None, 100, 'ambiguous')  # multiple gvkeys for this name

    # Fuzzy match using rapidfuzz extractOne
    result = process.extractOne(
        emp_name, conm_choices,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold
    )

    if result is None:
        return (None, None, 0, 'no_match')

    matched_name, score, _ = result
    gvkey = conm_gvkey_map.get(matched_name)
    return (gvkey, matched_name, score, 'fuzzy_high')

# ---- Task 2a: Match unmatched CUSIPs via Employer name ----
print("\n=== 2a: Match unmatched CUSIPs ===")
unmatched_notices = notices[notices['cusip'].isin(unmatched['cusip'].unique())]
unmatched_employers = unmatched_notices[['Employer', 'Employer State']].drop_duplicates()
unmatched_employers['name_norm'] = unmatched_employers['Employer'].apply(normalize_name)
print(f"Unique Employer names to match: {len(unmatched_employers)}")

matches_2a = []
for _, row in unmatched_employers.iterrows():
    gvkey, conm_matched, score, mtype = fuzzy_match_one(row['name_norm'], row['Employer State'])
    matches_2a.append({**row.to_dict(), 'gvkey': gvkey, 'conm': conm_matched,
                       'match_type': mtype, 'score': round(score/100.0, 3) if score > 0 else 0})

df_2a = pd.DataFrame(matches_2a)
matched_2a = df_2a[df_2a['match_type'].isin(['exact_norm', 'fuzzy_high'])]
print(f"2a matched: {len(matched_2a)} Employers → {matched_2a.gvkey.nunique()} gvkeys")
print(f"Match types: {df_2a.match_type.value_counts().to_dict()}")

# ---- Task 2b: Expand sample from notices without CUSIP ----
print("\n=== 2b: Expand from notices without CUSIP ===")
no_cusip = notices[notices['cusip'].isna()].copy()
print(f"Notices without CUSIP: {len(no_cusip)}")

no_cusip_emps = no_cusip[['Employer', 'Employer State']].drop_duplicates()
no_cusip_emps['name_norm'] = no_cusip_emps['Employer'].apply(normalize_name)
no_cusip_emps = no_cusip_emps[no_cusip_emps['name_norm'].str.len() >= 5]
print(f"Unique notice Employers without CUSIP: {len(no_cusip_emps)}")

matches_2b = []
for _, row in no_cusip_emps.iterrows():
    gvkey, conm_matched, score, mtype = fuzzy_match_one(row['name_norm'], row['Employer State'])
    if mtype in ['exact_norm', 'fuzzy_high']:
        matches_2b.append({**row.to_dict(), 'gvkey': gvkey, 'conm': conm_matched,
                           'match_type': mtype, 'score': round(score/100.0, 3)})

df_2b = pd.DataFrame(matches_2b)
if len(df_2b) > 0:
    print(f"2b matched: {len(df_2b)} Employers → {df_2b.gvkey.nunique()} unique gvkeys")
    print(f"Match types: {df_2b.match_type.value_counts().to_dict()}")
else:
    print("2b: no matches found")

df_2b = pd.DataFrame(matches_2b) if matches_2b else pd.DataFrame()
if len(df_2b) > 0:
    print(f"2b matched: {len(df_2b)} Employers → {df_2b.gvkey.nunique()} unique gvkeys")
    print(f"Match types: {df_2b.match_type.value_counts().to_dict()}")
else:
    print("2b: no matches found")

# ---- 5. Construct extended UNIONIZATION panel (2b matches only) ----
if len(df_2b) > 0:
    print("\n=== Constructing extended UNIONIZATION panel ===")

    # Map notice Employers to gvkeys
    emp_to_gvkey = dict(zip(df_2b['Employer'], df_2b['gvkey']))

    # For matched notices, compute firm-year UNIONIZATION
    ext_notices = no_cusip[no_cusip['Employer'].isin(df_2b['Employer'])].copy()
    ext_notices['gvkey'] = ext_notices['Employer'].map(emp_to_gvkey)

    # Filter: BUS <= EST, EST > 0
    ext_notices['BUS'] = pd.to_numeric(ext_notices['Bargaining Unit Size'], errors='coerce')
    ext_notices['EST'] = pd.to_numeric(ext_notices['Establishment Size'], errors='coerce')
    ext_notices = ext_notices[(ext_notices['BUS'] <= ext_notices['EST']) & (ext_notices['EST'] > 0)]

    # Compute UNIONIZATION = Σ BUS / Σ EST per gvkey-year, cap at 1
    ext_panel = ext_notices.groupby(['gvkey', 'year']).agg(
        BUS_sum=('BUS', 'sum'),
        EST_sum=('EST', 'sum')
    ).reset_index()
    ext_panel['UNIONIZATION'] = (ext_panel['BUS_sum'] / ext_panel['EST_sum']).clip(upper=1.0)
    ext_panel = ext_panel[ext_panel['UNIONIZATION'] > 0]  # keep only unionized

    print(f"Extended panel: {len(ext_panel)} firm-years, {ext_panel.gvkey.nunique()} gvkeys")
    print(f"Year range: {ext_panel.year.min()}-{ext_panel.year.max()}")
    print(f"UNIONIZATION mean: {ext_panel.UNIONIZATION.mean():.4f}")

    # Check overlap with main panel
    main_gvkeys = set(pd.read_parquet(OUT + 'unionization_panel_main.parquet')['gvkey'].dropna())
    new_gvkeys = set(ext_panel.gvkey.unique()) - main_gvkeys
    print(f"New gvkeys not in main panel: {len(new_gvkeys)}")

    ext_panel.to_parquet(FUZZY_OUT + 'extended_unionization_panel.parquet', index=False)
    print(f"Saved extended_unionization_panel.parquet")
else:
    print("\nNo 2b matches to construct extended panel")

# ---- 6. Audit sample ----
print("\n=== Audit: random samples ===")
# Save all match results
if len(df_2a) > 0:
    df_2a.to_csv(FUZZY_OUT + 'fuzzy_match_2a_results.csv', index=False)
if len(df_2b) > 0:
    df_2b.to_csv(FUZZY_OUT + 'fuzzy_match_2b_results.csv', index=False)

# Combined audit table
all_matches = []
if len(df_2a) > 0:
    df_2a['task'] = '2a_unmatched_cusip'
    all_matches.append(df_2a[['task', 'Employer', 'Employer State', 'conm', 'match_type', 'score', 'gvkey']])
if len(df_2b) > 0:
    df_2b['task'] = '2b_expansion'
    all_matches.append(df_2b[['task', 'Employer', 'Employer State', 'conm', 'match_type', 'score', 'gvkey']])

if all_matches:
    audit = pd.concat(all_matches, ignore_index=True)

    # Random audit samples
    matched = audit[audit['match_type'].isin(['exact_norm', 'fuzzy_high'])]
    ambiguous = audit[audit['match_type'] == 'ambiguous']
    state_mismatch = audit[audit['match_type'] == 'state_mismatch']

    n_match_sample = min(100, len(matched))
    n_ambig_sample = min(50, len(ambiguous))

    audit_sample = pd.concat([
        matched.sample(n=n_match_sample, random_state=42) if n_match_sample > 0 else matched,
        ambiguous.sample(n=n_ambig_sample, random_state=42) if n_ambig_sample > 0 else ambiguous,
        state_mismatch.head(20)
    ], ignore_index=True)
    audit_sample.to_csv(FUZZY_OUT + 'fuzzy_match_audit.csv', index=False)

    print(f"Audit sample: {len(audit_sample)} rows")
    print(f"  Matched (exact+fuzzy_high): {len(matched)}")
    print(f"  Ambiguous: {len(ambiguous)}")
    print(f"  State mismatch: {len(state_mismatch)}")

    # Print 15 random matched pairs for manual check
    print("\n--- Random 15 matched pairs (manual review) ---")
    for _, r in matched.sample(n=min(15, len(matched)), random_state=1).iterrows():
        print(f"  {r['Employer'][:50]:50s} → {str(r['conm'])[:50]:50s} [{r['match_type']}] score={r['score']}")
else:
    print("No matches to audit")

print(f"\nAll results saved to: {FUZZY_OUT}")
