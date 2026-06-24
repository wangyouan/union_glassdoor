#!/usr/bin/env python
"""Fill missing outputs: coverage, audit, combined Excel, comprehensive report."""

import pandas as pd
import numpy as np
import os

OUT = "/data/disk4/workspace/projects/union_glassdoor/outputs/20260624/current_former_bargaining_unit"
os.makedirs(OUT, exist_ok=True)

# ─── 1. title_match_coverage.csv (STEP 8) ─────────────────────────────────
print("=== Generating title_match_coverage.csv ===")

enriched = pd.read_parquet(f"{OUT}/enriched_sample.parquet", columns=['review_id','election_id','sample_type','job_title_clean'])
matches = pd.read_parquet(f"{OUT}/review_title_unit_matches.parquet")
matches2 = matches.drop(columns=['election_id'])  # avoid merge collision
coverage = enriched.merge(matches2, on='review_id', how='left')

# Election-level coverage
ec = coverage.groupby('election_id').agg(
    n_reviews=('review_id','count'),
    n_member=('unit_match', lambda x: (x=='member').sum()),
    n_not_member=('unit_match', lambda x: (x=='not_member').sum()),
    n_ambiguous=('unit_match', lambda x: (x=='ambiguous').sum()),
).reset_index()
ec['pct_member'] = ec['n_member'] / ec['n_reviews'] * 100
ec['pct_not_member'] = ec['n_not_member'] / ec['n_reviews'] * 100
ec['pct_ambiguous'] = ec['n_ambiguous'] / ec['n_reviews'] * 100
ec['has_any_match'] = (ec['n_member'] > 0) | (ec['n_not_member'] > 0)
has5 = (ec['n_member'] >= 5) & (ec['n_not_member'] >= 5)
has10 = (ec['n_member'] >= 10) & (ec['n_not_member'] >= 10)

# Coverage report
conf = coverage['unit_match_confidence']
cov_rows = [
    {'metric': 'total_reviews', 'value': len(coverage)},
    {'metric': 'reviews_matched_member', 'value': int((coverage['unit_match']=='member').sum())},
    {'metric': 'reviews_matched_not_member', 'value': int((coverage['unit_match']=='not_member').sum())},
    {'metric': 'reviews_ambiguous', 'value': int((coverage['unit_match']=='ambiguous').sum())},
    {'metric': 'pct_member', 'value': round((coverage['unit_match']=='member').mean()*100, 1)},
    {'metric': 'pct_not_member', 'value': round((coverage['unit_match']=='not_member').mean()*100, 1)},
    {'metric': 'pct_ambiguous', 'value': round((coverage['unit_match']=='ambiguous').mean()*100, 1)},
    {'metric': 'pct_high_confidence', 'value': round((conf=='high').mean()*100, 1)},
    {'metric': 'pct_medium_confidence', 'value': round((conf=='medium').mean()*100, 1)},
    {'metric': 'pct_low_confidence', 'value': round((conf=='low').mean()*100, 1)},
    {'metric': 'elections_total', 'value': int(len(ec))},
    {'metric': 'elections_with_any_match', 'value': int(ec['has_any_match'].sum())},
    {'metric': 'elections_with_members', 'value': int((ec['n_member']>0).sum())},
    {'metric': 'elections_with_not_members', 'value': int((ec['n_not_member']>0).sum())},
    {'metric': 'elections_with_5plus_both', 'value': int(has5.sum())},
    {'metric': 'elections_with_10plus_both', 'value': int(has10.sum())},
]
pd.DataFrame(cov_rows).to_csv(f"{OUT}/title_match_coverage.csv", index=False)
print(f"  Saved title_match_coverage.csv ({len(cov_rows)} rows)")

# By sample type
for s in ['current','former']:
    sub = coverage[coverage['sample_type']==s]
    print(f"  {s}: member={sub['unit_match'].eq('member').mean()*100:.1f}%, "
          f"not_member={sub['unit_match'].eq('not_member').mean()*100:.1f}%, "
          f"ambiguous={sub['unit_match'].eq('ambiguous').mean()*100:.1f}%")

# Top reasons
for match_type in ['member','not_member','ambiguous']:
    reasons = coverage[coverage['unit_match']==match_type]['unit_match_reason'].value_counts().head(5)
    print(f"  Top {match_type} reasons: {dict(reasons)}")

# ─── 2. title_match_manual_audit.csv (STEP 8) ────────────────────────────
print("\n=== Generating title_match_manual_audit.csv ===")

nlrb = pd.read_parquet("/data/disk4/workspace/projects/union/outputs/preliminary_election_level.parquet",
                       columns=['election_id','filing__name','description'])
nlrb = nlrb.drop_duplicates(subset='election_id')

audit_n = min(200, len(coverage))
audit = coverage.sample(n=audit_n, random_state=42)
audit = audit.merge(nlrb, on='election_id', how='left')
audit_out = audit[['election_id','filing__name','description','job_title_clean',
                    'sample_type','unit_match','unit_match_confidence','unit_match_reason']].copy()
audit_out.columns = ['election_id','employer','bu_desc','job_title','sample_type',
                      'predicted_match','confidence','reason']
audit_out['manual_review'] = ''
audit_out.to_csv(f"{OUT}/title_match_manual_audit.csv", index=False)
print(f"  Saved title_match_manual_audit.csv ({len(audit_out)} rows)")

# ─── 3. Save unit_share as parquet ───────────────────────────────────────
print("\n=== Saving unit_share as parquet ===")
ushare = pd.read_csv(f"{OUT}/unit_share_election_data.csv")
ushare.to_parquet(f"{OUT}/unit_share_election_data.parquet", index=False)
print(f"  Saved unit_share_election_data.parquet ({len(ushare)} rows)")

# ─── 4. Comprehensive report ──────────────────────────────────────────────
print("\n=== Writing comprehensive report ===")

# Read all result files
try:
    all_outcomes = pd.read_csv(f"{OUT}/current_former_all_outcomes.csv")
    diff_tests = pd.read_csv(f"{OUT}/current_former_difference_tests.csv")
    member_reg = pd.read_csv(f"{OUT}/unit_member_regression_results.csv")
    ushare_cont = pd.read_csv(f"{OUT}/unit_share_regression_continuous.csv")
    ushare_above = pd.read_csv(f"{OUT}/unit_share_regression_above_median.csv")
    ushare_log = pd.read_csv(f"{OUT}/unit_share_regression_log_size.csv")
    sample_sum = pd.read_csv(f"{OUT}/sample_summary.csv")
    dv_inv = pd.read_csv(f"{OUT}/dv_field_inventory.csv")
    ushare_data = pd.read_csv(f"{OUT}/unit_share_election_data.csv")
except Exception as e:
    print(f"  Warning: some files missing: {e}")

# Build report
report = []
report.append("# Current vs Former + Bargaining-Unit Analysis — Comprehensive Report\n")
report.append(f"**Date**: 2026-06-24  \n")
report.append(f"**Spec**: v7c DiD-RD (4-FE), review-level, cluster gvkey×review_year  \n")
report.append(f"**Main filter**: total≥10  \n\n")

# A. Sample sizes
report.append("## A. Current/Former Sample Summary\n\n")
report.append("| Sample | Reviews | Elections | Firms |\n")
report.append("|--------|---------|-----------|-------|\n")
for _, r in sample_sum.iterrows():
    report.append(f"| {r['sample']} | {r['n_reviews']:,} | {r['n_elections']:,} | {r['n_firms']:,} |\n")
report.append("\n")

# B. 10-DV results
report.append("## B. 10-DV Results: Current vs Former vs All (total≥10)\n\n")
report.append("| DV | Current Coef | Current p | Former Coef | Former p | All Coef | All p |\n")
report.append("|----|-------------|-----------|-------------|----------|----------|-------|\n")
for dv in all_outcomes['dv'].unique():
    cur = all_outcomes[(all_outcomes['dv']==dv) & (all_outcomes['sample']=='current')]
    frm = all_outcomes[(all_outcomes['dv']==dv) & (all_outcomes['sample']=='former')]
    all_ = all_outcomes[(all_outcomes['dv']==dv) & (all_outcomes['sample']=='all')]
    cur_c, cur_p = (cur['coef'].values[0], cur['p'].values[0]) if len(cur) else (np.nan, np.nan)
    frm_c, frm_p = (frm['coef'].values[0], frm['p'].values[0]) if len(frm) else (np.nan, np.nan)
    all_c, all_p = (all_['coef'].values[0], all_['p'].values[0]) if len(all_) else (np.nan, np.nan)

    sig = lambda p: "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.1 else ""))
    report.append(f"| {dv} | {cur_c:+.3f}{sig(cur_p)} | {cur_p:.3f} | "
                  f"{frm_c:+.3f}{sig(frm_p)} | {frm_p:.3f} | "
                  f"{all_c:+.3f}{sig(all_p)} | {all_p:.3f} |\n")
report.append("\n")

# C. Difference tests
report.append("## C. Current–Former Difference Tests (Pooled Interaction)\n\n")
report.append("| DV | Current Effect | Former Effect | Diff | Diff p |\n")
report.append("|----|---------------|---------------|------|--------|\n")
for _, r in diff_tests.iterrows():
    if r.get('note','') == '':
        report.append(f"| {r['dv']} | {r['current_effect']:+.4f} | {r['former_effect']:+.4f} | "
                      f"{r['diff']:+.4f} | {r['diff_p']:.4f} |\n")
    else:
        report.append(f"| {r['dv']} | — | — | — | {r['note']} |\n")
report.append("\n")

# D. Unit-member
report.append("## D. Unit-Member vs Non-Unit Regression\n\n")
report.append("| DV | Non-Member | Member | Diff | Diff p | N Elections |\n")
report.append("|----|-----------|--------|------|--------|-------------|\n")
for _, r in member_reg.iterrows():
    if r.get('note','') == '':
        report.append(f"| {r['dv']} | {r['non_member_effect']:+.4f} | {r['member_effect']:+.4f} | "
                      f"{r['diff']:+.4f} | {r['diff_p']:.4f} | {r.get('n_elections','')} |\n")
    else:
        report.append(f"| {r['dv']} | — | — | — | — | {r['note']} |\n")
report.append("\n")

# E. Unit-share
report.append("## E. Unit-Share Distribution\n\n")
ushare_valid = ushare_data['unit_share_raw'].replace([np.inf, -np.inf], np.nan).dropna()
report.append(f"- Median unit_share: {ushare_valid.median()*100:.3f}%\n")
report.append(f"- P25: {ushare_valid.quantile(0.25)*100:.3f}%, P75: {ushare_valid.quantile(0.75)*100:.3f}%\n")
report.append(f"- P90: {ushare_valid.quantile(0.90)*100:.3f}%\n")
report.append(f"- unit_share > 100%: {(ushare_valid>1).sum()} elections\n\n")

report.append("### Unit-Share Interaction (WLB, Current Only)\n\n")
report.append("| Specification | Main Effect | Interaction | Interaction p |\n")
report.append("|---------------|------------|-------------|---------------|\n")
for name, df_ in [("Continuous", ushare_cont), ("Above/Below Median", ushare_above), ("Log(unit_size)", ushare_log)]:
    wlb = df_[df_['dv']=='wlb']
    if len(wlb):
        if name == "Continuous":
            report.append(f"| {name} | {wlb['win_post_coef'].values[0]:+.4f} (p={wlb['win_post_p'].values[0]:.3f}) | "
                          f"{wlb['win_post_x_ushare_coef'].values[0]:+.4f} | {wlb['win_post_x_ushare_p'].values[0]:.3f} |\n")
        elif name == "Above/Below Median":
            report.append(f"| {name} | {wlb['win_post_below_med'].values[0]:+.4f} | "
                          f"{wlb['win_post_diff_above'].values[0]:+.4f} | {wlb['diff_p'].values[0]:.3f} |\n")
        elif name == "Log(unit_size)":
            report.append(f"| {name} | {wlb['win_post_coef'].values[0]:+.4f} (p={wlb['win_post_p'].values[0]:.3f}) | "
                          f"{wlb['win_post_x_log_us_coef'].values[0]:+.4f} | {wlb['win_post_x_log_us_p'].values[0]:.3f} |\n")
report.append("\n")

# F. DV inventory
report.append("## F. DV Field Inventory\n\n")
report.append("| DV | Type | Missing Rate |\n")
report.append("|----|------|-------------|\n")
for _, r in dv_inv.iterrows():
    report.append(f"| {r['dv']} | {r['type']} | {r['missing_rate_pct']}% |\n")
report.append("\n")

# G. Recommendations
report.append("## G. Recommendations\n\n")
report.append("1. **Current-only is the right main sample**: WLB significant (p=0.025), Former not (p=0.282). Formal interaction doesn't reject equality, but current provides cleaner identification.\n")
report.append("2. **Comp is robustly zero**: strongest null result — |coef|<0.02, all p>0.78 across all samples.\n")
report.append("3. **Diversity is notably significant**: +0.094 (p=0.050) in current, +0.104 (p=0.033) in all — worth reporting.\n")
report.append("4. **WLB robust to filter**: significant at 5% for total≥10/20 global, stable in narrow bandwidth.\n")
report.append("5. **Title matching feasible but member-vs-nonmember differences all NS**: 53% ambiguous. Unit-member interaction tests all p>0.14. Use unit-share instead of title matching for main paper.\n")
report.append("6. **Unit-share shows flat effect**: WLB effect invariant to unit_share (median 0.03% of firm). Suggests firm-level spillover, not direct member-only mechanism.\n")
report.append("7. **Former shows consistently weaker effects**: all 10 DVs have smaller (but not significantly different) coefficients in former sample.\n")
report.append("8. **Recommend/business_outlook/ceo_approval show no significant effects**: all three categorical DVs have p>0.14 across all samples.\n")

# Write report
with open(f"{OUT}/current_former_bargaining_unit_report.md", "w") as f:
    f.write("".join(report))
print(f"  Saved current_former_bargaining_unit_report.md")

# ─── 5. Combined Excel workbook ──────────────────────────────────────────
print("\n=== Building combined Excel workbook ===")

with pd.ExcelWriter(f"{OUT}/current_former_bargaining_unit_results.xlsx", engine='openpyxl') as writer:
    # Sheet 1: README
    pd.DataFrame({'description': [
        'Current vs Former + Bargaining-Unit Analysis Results',
        'Date: 2026-06-24',
        'Spec: v7c DiD-RD (4-FE), review-level, cluster gvkey×review_year',
        'Main filter: total>=10',
        '10 DVs: overall_rating, career_opp, comp_benefit, senior_mgmt, wlb, culture, recommend, business_outlook, ceo_approval, diversity'
    ]}).to_excel(writer, sheet_name='README', index=False)

    # Sheet 2: Sample_Construction
    sample_sum.to_excel(writer, sheet_name='Sample_Construction', index=False)

    # Sheet 3: Current_vs_Former_All_DVs
    all_outcomes.to_excel(writer, sheet_name='Current_vs_Former_All_DVs', index=False)

    # Sheet 4: Current_Former_Difference
    diff_tests.to_excel(writer, sheet_name='Current_Former_Difference', index=False)

    # Sheet 5: Filter_Stability (if full sweep exists)
    fbr_path = f"{OUT}/filter_bandwidth_robustness_full.csv"
    if os.path.exists(fbr_path):
        pd.read_csv(fbr_path).to_excel(writer, sheet_name='Filter_Stability', index=False)
    else:
        # Fall back to focused version
        fbr_focused = f"{OUT}/filter_bandwidth_robustness.csv"
        if os.path.exists(fbr_focused):
            pd.read_csv(fbr_focused).to_excel(writer, sheet_name='Filter_Stability', index=False)

    # Sheet 6: BU_Field_Inventory
    pd.read_csv(f"{OUT}/bargaining_unit_field_inventory.csv").to_excel(writer, sheet_name='BU_Field_Inventory', index=False)

    # Sheet 7: BU_Title_Match_Coverage
    pd.read_csv(f"{OUT}/title_match_coverage.csv").to_excel(writer, sheet_name='BU_Title_Match_Coverage', index=False)

    # Sheet 8: BU_Title_Match_Examples
    pd.read_csv(f"{OUT}/title_match_manual_audit.csv").to_excel(writer, sheet_name='BU_Title_Match_Examples', index=False)

    # Sheet 9: BU_Member_vs_Nonmember
    member_reg.to_excel(writer, sheet_name='BU_Member_vs_Nonmember', index=False)

    # Sheet 10: BU_Share_Descriptives
    ushare_data.describe().to_excel(writer, sheet_name='BU_Share_Descriptives')

    # Sheet 11: BU_Share_All_DVs
    ushare_cont.to_excel(writer, sheet_name='BU_Share_All_DVs', index=False)

    # Sheet 12: BU_Share_Marginal_Effects
    pd.read_csv(f"{OUT}/unit_share_marginal_effects.csv").to_excel(writer, sheet_name='BU_Share_Marginal_Effects', index=False)

    # Additional sheets
    ushare_above.to_excel(writer, sheet_name='BU_Share_Above_Median', index=False)
    ushare_log.to_excel(writer, sheet_name='BU_Share_Log_Size', index=False)
    dv_inv.to_excel(writer, sheet_name='DV_Field_Inventory', index=False)

print(f"  Saved current_former_bargaining_unit_results.xlsx")

print("\n=== All gaps filled ===")
