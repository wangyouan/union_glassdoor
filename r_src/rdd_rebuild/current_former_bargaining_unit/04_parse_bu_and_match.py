#!/usr/bin/env python
"""STEP 5-8: BU description parsing, title matching, quality report.

Tier 1: Exact occupation keyword matching
Tier 2: Role category matching (using role_k1500, union_classification, role_* flags)
Tier 3: (reserved for LLM if needed)

Outputs:
  - bargaining_unit_field_inventory.csv
  - bargaining_unit_parsed_terms.parquet
  - review_title_unit_matches.parquet
  - title_match_coverage.csv
  - title_match_manual_audit.csv
"""

import pandas as pd
import numpy as np
import re, os

OUT = "/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/current_former_bargaining_unit/20260624"
os.makedirs(OUT, exist_ok=True)

# ─── Load data ──────────────────────────────────────────────────────────────
print("Loading NLRB data...")
nlrb_cols = ['election_id','case_number','description','filing__number_of_eligible_voters',
             'filing__name','participant__employer_names','participant__union_names',
             'filing__case_type','filing__state','filing__city','unit_size']
nlrb = pd.read_parquet('/data/disk4/workspace/projects/union/outputs/preliminary_election_level.parquet',
                       columns=nlrb_cols)
nlrb = nlrb.drop_duplicates(subset='election_id', keep='first')
print(f"  NLRB rows: {len(nlrb):,}")

print("Loading enriched sample...")
enriched = pd.read_parquet(f"{OUT}/enriched_sample.parquet")
our_eids = set(enriched['election_id'].unique())
print(f"  Enriched rows: {len(enriched):,}, elections: {len(our_eids):,}")

# ─── STEP 5: Field inventory ────────────────────────────────────────────────
print("\n=== STEP 5: BU Field Inventory ===")
nlrb_our = nlrb[nlrb['election_id'].isin(our_eids)].copy()
print(f"  Elections in our sample: {len(nlrb_our):,}")

field_inv = []
for col in nlrb_cols:
    n_nonnull = nlrb_our[col].notna().sum()
    field_inv.append({
        'field': col,
        'non_null': n_nonnull,
        'missing_rate_pct': round((1 - n_nonnull/len(nlrb_our))*100, 2),
        'dtype': str(nlrb_our[col].dtype)
    })

field_inv_df = pd.DataFrame(field_inv)
field_inv_df.to_csv(f"{OUT}/bargaining_unit_field_inventory.csv", index=False)
print(field_inv_df.to_string(index=False))

# Save 20 example descriptions
print("\n--- 20 example BU descriptions ---")
sample_desc = nlrb_our[nlrb_our['description'].notna()]['description'].sample(min(20, nlrb_our['description'].notna().sum()), random_state=123)
desc_examples = []
for i, (idx, desc) in enumerate(sample_desc.items()):
    eid = nlrb_our.loc[idx, 'election_id']
    name = nlrb_our.loc[idx, 'filing__name']
    desc_examples.append({'election_id': eid, 'employer': str(name)[:80], 'description': str(desc)[:500]})
    print(f"  [{i}] eid={eid}, employer={str(name)[:50]}")
    print(f"      {str(desc)[:300]}")
    print()

pd.DataFrame(desc_examples).to_csv(f"{OUT}/bu_description_examples.csv", index=False)

# ─── STEP 6: Parse BU descriptions ──────────────────────────────────────────
print("\n=== STEP 6: Parsing BU Descriptions ===")

# Occupation keyword dictionaries
PRODUCTION_TERMS = [
    'production', 'assembl', 'fabricat', 'machin', 'weld', 'tool', 'die',
    'press', 'mold', 'packag', 'shipping', 'receiving', 'warehous',
    'material handler', 'forklift', 'line worker', 'operator', 'machine operator'
]

MAINTENANCE_TERMS = [
    'maintenance', 'janitor', 'custodian', 'housekeep', 'clean',
    'repair', 'mechanic', 'grounds', 'facilities'
]

CLERICAL_TERMS = [
    'clerical', 'clerk', 'secretar', 'receptionist', 'administrative assistant',
    'office', 'data entry', 'customer service', 'call center', 'accounting',
    'billing', 'payroll', 'human resource', 'hr ', 'bookkeep',
    'dispatcher', 'scheduler'
]

PROFESSIONAL_TERMS = [
    'engineer', 'scientist', 'analyst', 'accountant', 'attorney', 'lawyer',
    'pharmacist', 'architect', 'designer', 'programmer', 'developer',
    'software', 'it ', 'information technology', 'systems', 'database',
    'network', 'consultant', 'specialist'
]

TECHNICAL_TERMS = [
    'technician', 'tech ', 'technologist', 'lab ', 'laboratory',
    'quality', 'inspector', 'tester', 'surveyor', 'drafter', 'cad',
    'emt', 'paramedic', 'lpn', 'cna', 'medical assistant'
]

SERVICE_TERMS = [
    'server', 'waiter', 'waitress', 'bartender', 'cook', 'chef', 'kitchen',
    'food', 'dietary', 'restaurant', 'cafeteria', 'catering',
    'cashier', 'sales', 'retail', 'merchandis', 'stylist', 'cosmetologist',
    'barber', 'attendant', 'valet', 'bell', 'concierge',
    'security', 'guard', 'agent'
]

DRIVER_TERMS = [
    'driver', 'truck', 'delivery', 'shuttle', 'bus', 'chauffeur',
    'transport', 'courier', 'dispatcher', 'yard hostl', 'spotter'
]

HEALTHCARE_TERMS = [
    'nurse', 'rn', 'registered nurse', 'physician', 'doctor', 'surgeon',
    'therapist', 'counselor', 'social worker', 'psychologist', 'psychiatrist',
    'dental', 'hygienist', 'pharmacy', 'pharmacist', 'dietitian', 'dietician',
    'respiratory', 'radiolog', 'sonographer', 'ultrasound',
    'patient care', 'home health', 'caregiver', 'aide', 'cna',
    'medical', 'clinical', 'hospital', 'clinic'
]

EDUCATION_TERMS = [
    'teacher', 'instructor', 'professor', 'educator', 'faculty',
    'paraeducator', 'paraprofessional', 'tutor', 'librar',
    'school', 'academic', 'student'
]

CONSTRUCTION_TERMS = [
    'construction', 'carpenter', 'electrician', 'plumber', 'pipefitter',
    'ironworker', 'painter', 'drywall', 'roofer', 'mason', 'bricklayer',
    'laborer', 'heavy equipment', 'crane', 'bulldozer', 'excavator'
]

# Exclusion terms (professions typically excluded from NLRB units)
EXCLUSION_KEYWORDS = [
    'manager', 'supervisor', 'guard', 'confidential', 'professional',
    'office clerical', 'executive', 'director', 'vice president',
    'human resource', 'hr ', 'attorney', 'lawyer', 'legal',
    'owner', 'president', 'chief ', 'controller', 'auditor'
]

# Term sets organized by scope_type
SCOPE_MAP = {
    'production_and_maintenance': PRODUCTION_TERMS + MAINTENANCE_TERMS,
    'clerical': CLERICAL_TERMS,
    'professional': PROFESSIONAL_TERMS,
    'technical': TECHNICAL_TERMS,
    'service': SERVICE_TERMS,
    'drivers_or_warehouse': DRIVER_TERMS,
    'healthcare': HEALTHCARE_TERMS,
    'education': EDUCATION_TERMS,
    'construction': CONSTRUCTION_TERMS,
}

def parse_bu_description(desc):
    """Parse a BU description into included/excluded terms and scope type."""
    if pd.isna(desc) or not isinstance(desc, str):
        return {
            'included_terms': [], 'excluded_terms': [],
            'scope_type': 'unclear', 'is_all_employees': False,
            'has_included_section': False, 'has_excluded_section': False
        }

    text = desc.lower().strip()

    # Split into included/excluded sections
    # Common patterns: "Included:", "All full-time...", "Excluding:", "Excluded:"
    included_text = ""
    excluded_text = ""

    # Try to find explicit sections
    inc_match = re.search(r'(?:included|all|voting group)[:\-]?\s*(.+?)(?:excluded|excluding|but excluding|except|$)', text, re.DOTALL | re.IGNORECASE)
    if inc_match:
        included_text = inc_match.group(1)

    exc_match = re.search(r'(?:excluded|excluding|but excluding)[:\-]?\s*(.+?)$', text, re.DOTALL | re.IGNORECASE)
    if exc_match:
        excluded_text = exc_match.group(1)

    # If no explicit sections, the whole text might describe the unit
    if not included_text and not excluded_text:
        included_text = text

    # Detect "all employees" pattern
    is_all = bool(re.search(r'all (full.?time|regular|part.?time)?\s*emplo', text))

    # Extract occupation terms from included text
    included_terms = []
    excluded_terms = []

    # Check for specific occupation keywords in included text
    for scope, terms in SCOPE_MAP.items():
        for term in terms:
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            if re.search(pattern, included_text):
                included_terms.append(term)

    for term in EXCLUSION_KEYWORDS:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, excluded_text):
            excluded_terms.append(term)

    # Determine scope type
    scope_scores = {}
    for scope, terms in SCOPE_MAP.items():
        score = sum(1 for t in terms if t in included_terms)
        if score > 0:
            scope_scores[scope] = score

    if is_all and not scope_scores:
        scope_type = 'all_employees'
    elif not scope_scores:
        scope_type = 'unclear'
    elif len(scope_scores) >= 3:
        scope_type = 'mixed_specific_occupations'
    else:
        scope_type = max(scope_scores, key=scope_scores.get)

    return {
        'included_terms': list(set(included_terms)),
        'excluded_terms': list(set(excluded_terms)),
        'scope_type': scope_type,
        'is_all_employees': is_all,
        'has_included_section': bool(included_text),
        'has_excluded_section': bool(excluded_text)
    }

# Parse all BU descriptions
print("Parsing BU descriptions...")
parsed = nlrb_our[['election_id','description']].copy()
parsed_data = parsed['description'].apply(parse_bu_description)
parsed['included_terms'] = parsed_data.apply(lambda x: '|'.join(x['included_terms']) if x['included_terms'] else '')
parsed['excluded_terms'] = parsed_data.apply(lambda x: '|'.join(x['excluded_terms']) if x['excluded_terms'] else '')
parsed['scope_type'] = parsed_data.apply(lambda x: x['scope_type'])
parsed['is_all_employees'] = parsed_data.apply(lambda x: x['is_all_employees'])

parsed.to_parquet(f"{OUT}/bargaining_unit_parsed_terms.parquet", index=False)

# Scope type distribution
print("\nScope type distribution:")
print(parsed['scope_type'].value_counts())

# ─── STEP 7-8: Title matching ──────────────────────────────────────────────
print("\n=== STEP 7-8: Title Matching ===")

# Get review-level data with job titles
# Use job_title_clean and role_k1500 from enriched sample
reviews = enriched[['review_id','election_id','gvkey','job_title_clean','job_title_raw',
                     'role_k1500','is_current_employee','is_former_employee',
                     'sample_type',
                     'role_likely_unionizable','role_likely_excluded_from_union',
                     'role_management_supervisory','role_union_classification',
                     'role_high_level_professional','role_hr_labor_relations',
                     'role_legal','role_owner_nonemployee','role_sales_commission',
                     'role_strategy_corporate','role_ambiguous_union_status',
                     'role_exclusion_reason']].copy()

print(f"  Reviews: {len(reviews):,}")

# Check if role_* columns are in enriched
role_cols = [c for c in reviews.columns if c.startswith('role_')]
print(f"  Role columns available: {role_cols}")

# Join BU parsed data
reviews = reviews.merge(parsed[['election_id','included_terms','excluded_terms','scope_type','is_all_employees']],
                        on='election_id', how='left')

def normalize_title(title):
    """Normalize a job title for matching."""
    if pd.isna(title) or not isinstance(title, str):
        return ""
    t = title.lower().strip()
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def match_title_to_unit(title, included_terms, excluded_terms, scope_type, role_data):
    """
    Tier 1: Exact keyword match in included_terms
    Tier 2: Role category mapping
    Returns: (match, confidence, reason)
    """
    if pd.isna(title) or not isinstance(title, str) or not title.strip():
        return ('ambiguous', 'low', 'missing_title')

    title_norm = normalize_title(title)
    if not title_norm:
        return ('ambiguous', 'low', 'empty_title')

    # ─── Tier 2 first: Check role_* flags ───────────────────────────────────
    # Strong exclusion signals
    if role_data.get('role_owner_nonemployee') == 1:
        return ('not_member', 'high', 'owner/non-employee')
    if role_data.get('role_likely_excluded_from_union') == 1:
        reason = role_data.get('role_exclusion_reason', 'likely_excluded')
        if pd.notna(reason) and reason:
            return ('not_member', 'high', f'excluded: {reason}')
        return ('not_member', 'high', 'likely_excluded_from_union')

    # Strong inclusion signals
    if role_data.get('role_likely_unionizable') == 1:
        return ('member', 'high', 'likely_unionizable')
    if role_data.get('role_rank_and_file_likely') == 1:
        return ('member', 'medium', 'rank_and_file_likely')

    # Ambiguous signals
    if role_data.get('role_management_supervisory') == 1:
        return ('ambiguous', 'medium', 'management/supervisory')
    if role_data.get('role_ambiguous_union_status') == 1:
        return ('ambiguous', 'medium', 'ambiguous_union_status')
    if role_data.get('role_high_level_professional') == 1:
        return ('not_member', 'medium', 'high_level_professional')
    if role_data.get('role_hr_labor_relations') == 1:
        return ('not_member', 'high', 'HR/labor_relations')
    if role_data.get('role_legal') == 1:
        return ('not_member', 'high', 'legal')
    if role_data.get('role_strategy_corporate') == 1:
        return ('not_member', 'medium', 'strategy/corporate')
    if role_data.get('role_sales_commission') == 1:
        return ('ambiguous', 'medium', 'sales_commission')

    # ─── Tier 1: Keyword matching ───────────────────────────────────────────
    if not isinstance(included_terms, str) or not included_terms:
        return ('ambiguous', 'low', 'no_bu_terms')

    inc_terms_list = included_terms.lower().split('|')
    exc_terms_list = excluded_terms.lower().split('|') if isinstance(excluded_terms, str) else []

    # Check excluded first
    for term in exc_terms_list:
        if term and len(term) > 3 and term in title_norm:
            return ('not_member', 'medium', f'title_matches_excluded: {term}')

    # Check included terms
    matched_terms = []
    for term in inc_terms_list:
        if term and len(term) > 2 and term in title_norm:
            matched_terms.append(term)

    if matched_terms:
        return ('member', 'medium' if len(matched_terms) >= 2 else 'low',
                f'matches: {", ".join(matched_terms[:3])}')

    # ─── Additional heuristics ──────────────────────────────────────────────
    # Director/VP/Chief/Head/President → not member
    for kw in ['director', 'vp ', 'vice president', 'chief ', 'head of', 'president', 'partner']:
        if kw in title_norm:
            return ('not_member', 'medium', f'title_suggests_management: {kw}')

    # Manager/supervisor → ambiguous (could be in or out depending on unit)
    for kw in ['manager', 'supervisor', 'lead ', 'team lead']:
        if kw in title_norm:
            return ('ambiguous', 'medium', f'title_suggests_supervisory: {kw}')

    # If scope_type is all_employees, all non-excluded are members
    if scope_type == 'all_employees':
        return ('member', 'low', 'all_employees_unit')

    return ('ambiguous', 'low', 'insufficient_info')

# Apply matching
print("Matching titles to units...")
match_results = []
for idx, row in reviews.iterrows():
    role_data = {c: row.get(c) for c in role_cols if c in row.index}
    match, conf, reason = match_title_to_unit(
        row.get('job_title_clean'), row.get('included_terms'),
        row.get('excluded_terms'), row.get('scope_type'), role_data
    )
    match_results.append({
        'review_id': row['review_id'],
        'election_id': row['election_id'],
        'unit_match': match,
        'unit_match_confidence': conf,
        'unit_match_reason': reason
    })

matches_df = pd.DataFrame(match_results)

# Save
matches_df.to_parquet(f"{OUT}/review_title_unit_matches.parquet", index=False)
print(f"  Saved {len(matches_df):,} title matches")

# ─── STEP 8: Match quality report ──────────────────────────────────────────
print("\n=== STEP 8: Match Quality Report ===")

# Merge back with review data for coverage stats
coverage = reviews[['review_id','election_id','sample_type','job_title_clean']].merge(
    matches_df, on='review_id', how='left')

# Overall match distribution
print("\nMatch distribution:")
print(coverage['unit_match'].value_counts())
print(f"\n  member: {(coverage['unit_match']=='member').sum():,} ({(coverage['unit_match']=='member').mean()*100:.1f}%)")
print(f"  not_member: {(coverage['unit_match']=='not_member').sum():,} ({(coverage['unit_match']=='not_member').mean()*100:.1f}%)")
print(f"  ambiguous: {(coverage['unit_match']=='ambiguous').sum():,} ({(coverage['unit_match']=='ambiguous').mean()*100:.1f}%)")

# By confidence
print("\nMatch confidence distribution:")
print(coverage['unit_match_confidence'].value_counts())

# By sample type
print("\nMatch by sample type:")
for s in ['current','former']:
    sub = coverage[coverage['sample_type'] == s]
    print(f"  {s}: member={sub['unit_match'].eq('member').mean()*100:.1f}%, "
          f"not_member={sub['unit_match'].eq('not_member').mean()*100:.1f}%, "
          f"ambiguous={sub['unit_match'].eq('ambiguous').mean()*100:.1f}%")

# Election-level coverage
election_coverage = coverage.groupby('election_id').agg(
    n_reviews=('review_id','count'),
    n_member=('unit_match', lambda x: (x=='member').sum()),
    n_not_member=('unit_match', lambda x: (x=='not_member').sum()),
    n_ambiguous=('unit_match', lambda x: (x=='ambiguous').sum()),
    pct_member=('unit_match', lambda x: (x=='member').mean()*100),
    pct_not_member=('unit_match', lambda x: (x=='not_member').mean()*100),
    pct_ambiguous=('unit_match', lambda x: (x=='ambiguous').mean()*100)
).reset_index()

print(f"\nElection-level coverage (n={len(election_coverage):,}):")
print(f"  Elections with >=1 member review: {(election_coverage['n_member']>0).sum():,}")
print(f"  Elections with >=1 not_member review: {(election_coverage['n_not_member']>0).sum():,}")
print(f"  Elections with any match (member or not): {((election_coverage['n_member']>0) | (election_coverage['n_not_member']>0)).sum():,}")
print(f"  Median pct_member per election: {election_coverage['pct_member'].median():.1f}%")

# Build coverage report
coverage_rows = [{
    'metric': 'total_reviews', 'value': len(coverage)
}, {
    'metric': 'reviews_matched_member', 'value': (coverage['unit_match']=='member').sum()
}, {
    'metric': 'reviews_matched_not_member', 'value': (coverage['unit_match']=='not_member').sum()
}, {
    'metric': 'reviews_ambiguous', 'value': (coverage['unit_match']=='ambiguous').sum()
}, {
    'metric': 'pct_member', 'value': round((coverage['unit_match']=='member').mean()*100, 1)
}, {
    'metric': 'pct_not_member', 'value': round((coverage['unit_match']=='not_member').mean()*100, 1)
}, {
    'metric': 'pct_ambiguous', 'value': round((coverage['unit_match']=='ambiguous').mean()*100, 1)
}, {
    'metric': 'pct_high_confidence', 'value': round((coverage['unit_match_confidence']=='high').mean()*100, 1)
}, {
    'metric': 'pct_medium_confidence', 'value': round((coverage['unit_match_confidence']=='medium').mean()*100, 1)
}, {
    'metric': 'pct_low_confidence', 'value': round((coverage['unit_match_confidence']=='low').mean()*100, 1)
}, {
    'metric': 'elections_total', 'value': len(election_coverage)
}, {
    'metric': 'elections_with_any_match', 'value': ((election_coverage['n_member']>0) | (election_coverage['n_not_member']>0)).sum()
}, {
    'metric': 'elections_with_members', 'value': (election_coverage['n_member']>0).sum()
}, {
    'metric': 'elections_with_not_members', 'value': (election_coverage['n_not_member']>0).sum()
}]
pd.DataFrame(coverage_rows).to_csv(f"{OUT}/title_match_coverage.csv", index=False)

# Top common matches and ambiguous
print("\n=== Top match reasons ===")
print("\nMember match reasons:")
print(coverage[coverage['unit_match']=='member']['unit_match_reason'].value_counts().head(20))

print("\nNot-member match reasons:")
print(coverage[coverage['unit_match']=='not_member']['unit_match_reason'].value_counts().head(20))

print("\nAmbiguous match reasons:")
print(coverage[coverage['unit_match']=='ambiguous']['unit_match_reason'].value_counts().head(20))

# ─── Manual audit sample ────────────────────────────────────────────────────
print("\n=== Manual audit sample ===")
audit_n = min(200, len(coverage))
audit_sample = coverage.sample(n=audit_n, random_state=42)
# Get employer name and description
audit_sample = audit_sample.merge(nlrb_our[['election_id','filing__name','description']],
                                   on='election_id', how='left')

audit_out = audit_sample[['election_id','filing__name','description','job_title_clean',
                           'sample_type','unit_match','unit_match_confidence','unit_match_reason']].copy()
audit_out.columns = ['election_id','employer','bu_desc','job_title','sample_type',
                      'predicted_match','confidence','reason']
audit_out['manual_review'] = ''
audit_out.to_csv(f"{OUT}/title_match_manual_audit.csv", index=False)
print(f"  Saved audit sample: {len(audit_out)} rows")

print("\nDone.")
