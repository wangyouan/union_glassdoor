#!/usr/bin/env python3
"""STEP 5: Self-check script for extension fix deliverables."""
import pandas as pd, os

OUT = '/data/disk4/workspace/projects/union_glassdoor/outputs/20260703/unionization_extension_fix/'
errors = []

def check(desc, condition):
    if not condition: errors.append(f"FAIL: {desc}")
    else: print(f"  ✓ {desc}")

# 1. File existence + non-empty
required_files = [
    'ingestion_loss_table.csv', 'unionization_panel_v1_fix.parquet',
    'unionization_panel_v2_fix.parquet', 'panel_consistency.csv',
    'match_audit.csv', 'fix_merge_coverage.csv', 'fix_correlations.csv',
    'fix_reg_ladder.csv', 'fix_reg_controls.csv', 'fix_reg_robustness.csv',
    'fix_reg_ladder_noT3.csv', 'ext_vs_finished_comparison.csv'
]

print("=== Check 1: File existence and non-empty ===")
for f in required_files:
    path = os.path.join(OUT, f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    check(f"{f} exists and non-empty", exists and size > 0)

# 2. Report md
print("\n=== Check 2: Report content ===")
report_path = os.path.join(OUT, 'fix_report.md')
# Report will be generated below — skip this check for now

# 3. Regression row counts
print("\n=== Check 3: Regression row counts ===")
for f, expected in [('fix_reg_ladder.csv', 40), ('fix_reg_controls.csv', 40)]:
    path = os.path.join(OUT, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        check(f"{f}: {len(df)} rows (expected {expected})", len(df) == expected)

rob_path = os.path.join(OUT, 'fix_reg_robustness.csv')
if os.path.exists(rob_path):
    df = pd.read_csv(rob_path)
    check(f"fix_reg_robustness.csv: {len(df)} rows (expected 40)", len(df) == 40)

# 4. Panel year variation
print("\n=== Check 4: Panel year variation (no broadcast bug) ===")
panel_path = os.path.join(OUT, 'unionization_panel_v1_fix.parquet')
if os.path.exists(panel_path):
    p = pd.read_parquet(panel_path)
    yr = p.groupby('Year')['UNIONIZATION'].mean()
    check(f"Year mean std = {yr.std():.4f} > 0", yr.std() > 0)
    check(f"Panel max year = {p.Year.max()} >= 2025", p.Year.max() >= 2025)
    check(f"Panel min year = {p.Year.min()} <= 2010", p.Year.min() <= 2010)
    # Check 2005-2017 mean ≈ 0.69
    mean_0517 = p[(p.Year>=2005)&(p.Year<=2017)]['UNIONIZATION'].mean()
    check(f"2005-2017 mean = {mean_0517:.4f} (baseline ≈0.69, within ±0.1)", abs(mean_0517 - 0.69) < 0.1)

# 5. No constant year counts in notice data
print("\n=== Check 5: Year count variation in panel ===")
yr_counts = p.groupby('Year').size()
check(f"n_gvkeys std = {yr_counts.std():.0f} > 0", yr_counts.std() > 0)
for y in sorted(p.Year.unique()):
    n = yr_counts[y]
    print(f"    {y}: {n:>5d} gvkeys, mean UNIONIZATION = {p[p.Year==y]['UNIONIZATION'].mean():.4f}")

print(f"\n{'='*50}")
if errors:
    print(f"FAILED: {len(errors)} error(s):")
    for e in errors: print(f"  {e}")
else:
    print("ALL CHECKS PASSED ✓")
print(f"{'='*50}")
