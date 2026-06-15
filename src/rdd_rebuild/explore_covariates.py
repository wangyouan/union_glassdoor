"""
Explore covariates in Glassdoor review data for regression control variable selection.

Output: /data/disk4/workspace/projects/glassdoor/outputs/glassdoor_covariate_report.md
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from io import StringIO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FULL_DATA = Path(
    "/data/disk4/workspace/projects/glassdoor/outputs/"
    "sentiment_individual_reviews_with_gvkey.parquet"
)
RDD_SAMPLE = Path(
    "/data/disk4/workspace/projects/union_glassdoor/outputs/rdd_rebuild/"
    "rdd_review_event_sample_from_raw.parquet"
)
OUTPUT_MD = Path("/data/disk4/workspace/projects/glassdoor/outputs/glassdoor_covariate_report.md")

# Global buffer for markdown output
_md: StringIO = StringIO()


def p(*args, **kwargs) -> None:
    print(*args, file=_md, **kwargs)


# ---------------------------------------------------------------------------
# Schema info
# ---------------------------------------------------------------------------
def get_schema_info(path: Path) -> tuple[list[tuple[str, str]], int]:
    pf = pq.ParquetFile(str(path))
    fields = [(field.name, str(field.type)) for field in pf.schema_arrow]
    nrows = pf.metadata.num_rows
    return fields, nrows


def compute_missing(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    records = []
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        records.append((col, str(df[col].dtype), pct))
    result = pd.DataFrame(records, columns=["column", "dtype", "pct_missing"])
    return result.sort_values("pct_missing", ascending=True)


# ---------------------------------------------------------------------------
# Rating cols
# ---------------------------------------------------------------------------
_RATING_COLS = {
    "rating_overall", "overall_rating",
    "rating_career_opportunities", "career_opp",
    "rating_compensation_and_benefits", "comp_benefit",
    "rating_senior_leadership", "senior_mgmt",
    "rating_work_life_balance", "wlb",
    "rating_culture_and_values", "culture",
    "rating_diversity_and_inclusion", "diversity",
}


# ---------------------------------------------------------------------------
# Categorical value counts → markdown table
# ---------------------------------------------------------------------------
def categorical_report_md(df: pd.DataFrame, label: str) -> None:
    p(f"### {label}")
    p()

    candidates = {}
    for col in df.columns:
        if col in _RATING_COLS:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        n_unique = s.nunique()
        dtype = df[col].dtype
        if dtype == "object" or dtype == "string" or dtype == "bool":
            candidates[col] = s
        elif n_unique <= 50:
            candidates[col] = s

    if not candidates:
        p("*No categorical variables found.*")
        p()
        return

    for col, s in sorted(candidates.items()):
        n_unique = s.nunique()
        n_total = len(df)
        pct_present = (len(s) / n_total * 100) if n_total else 0

        # Skip high-cardinality free-text columns beyond top-N display
        p(f"#### `{col}`")
        p()
        p(f"- Unique values: **{n_unique:,}**")
        p(f"- Non-missing: {len(s):,} / {n_total:,} ({pct_present:.1f}%)")
        p()

        vc = s.value_counts().head(30)
        total_shown = vc.sum()
        pct_shown = (total_shown / len(s) * 100) if len(s) > 0 else 0
        p(f"*Top {len(vc)} values (covering {pct_shown:.1f}% of non-missing):*")
        p()
        p("| Rank | Value | Count |")
        p("|------|-------|-------|")
        for rank, (val, cnt) in enumerate(vc.items(), 1):
            val_str = str(val).replace("|", "/").replace("\n", " ")
            # Truncate very long values
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
            p(f"| {rank} | `{val_str}` | {cnt:,} |")
        p()


# ---------------------------------------------------------------------------
# Numeric statistics → markdown
# ---------------------------------------------------------------------------
def numeric_report_md(df: pd.DataFrame, label: str) -> None:
    p(f"### {label}")
    p()

    numeric_cols = []
    for col in df.columns:
        if df[col].dtype in ("int64", "int32", "float64", "float32", "Int64", "Float64"):
            numeric_cols.append(col)

    if not numeric_cols:
        p("*No numeric variables found.*")
        p()
        return

    # Build a summary table
    p("| Column | N | % Missing | Mean | Median | Std | Min | Max |")
    p("|--------|---|----------|------|--------|-----|-----|-----|")
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            p(f"| `{col}` | 0 | 100.0% | — | — | — | — | — |")
            continue
        tag = " ⭐" if col in _RATING_COLS else ""
        p(
            f"| `{col}`{tag} | {len(s):,} | {df[col].isna().mean()*100:.1f}% "
            f"| {s.mean():.2f} | {s.median():.2f} | {s.std():.2f} "
            f"| {s.min():.2f} | {s.max():.2f} |"
        )

    p()
    # Detailed percentiles for ratings only
    rating_in_data = [c for c in numeric_cols if c in _RATING_COLS]
    if rating_in_data:
        p("#### Rating variable percentiles")
        p()
        p("| Column | P1 | P5 | P25 | P50 | P75 | P95 | P99 |")
        p("|--------|----|----|-----|-----|-----|-----|-----|")
        for col in rating_in_data:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            p(
                f"| `{col}` | {np.percentile(s, 1):.0f} | {np.percentile(s, 5):.0f} "
                f"| {np.percentile(s, 25):.0f} | {np.percentile(s, 50):.0f} "
                f"| {np.percentile(s, 75):.0f} | {np.percentile(s, 95):.0f} "
                f"| {np.percentile(s, 99):.0f} |"
            )
        p()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    full_schema, full_nrows = get_schema_info(FULL_DATA)
    rdd_schema, rdd_nrows = get_schema_info(RDD_SAMPLE)

    # Load data
    full_df = pd.read_parquet(FULL_DATA)
    rdd_df = pd.read_parquet(RDD_SAMPLE)

    full_missing = compute_missing(full_df)
    rdd_missing = compute_missing(rdd_df)

    full_col_names = set(full_missing["column"])
    rdd_col_names = set(rdd_missing["column"])
    only_in_full = full_col_names - rdd_col_names
    only_in_rdd = rdd_col_names - full_col_names
    in_both = full_col_names & rdd_col_names

    # ====================================================================
    # MARKDOWN REPORT
    # ====================================================================
    p("# Glassdoor Covariate Exploration Report")
    p()
    p(f"**Full data:** `sentiment_individual_reviews_with_gvkey.parquet` "
      f"— {full_nrows:,} rows × {len(full_schema)} columns")
    p()
    p(f"**RDD sample:** `rdd_review_event_sample_from_raw.parquet` "
      f"— {rdd_nrows:,} rows × {len(rdd_schema)} columns")
    p()
    p("---")
    p()

    # ====================================================================
    # SECTION 1: Full data column list
    # ====================================================================
    p("## 1. Full Data — Column List")
    p()
    p("| Column | Dtype | % Missing |")
    p("|--------|-------|----------|")
    for _, row in full_missing.iterrows():
        p(f"| `{row['column']}` | {row['dtype']} | {row['pct_missing']:.2f}% |")
    p()

    # ====================================================================
    # SECTION 2: RDD sample column list + comparison
    # ====================================================================
    p("## 2. RDD Sample — Column List + Comparison")
    p()
    p("| Column | Dtype | % Missing | Note |")
    p("|--------|-------|----------|------|")
    for _, row in rdd_missing.iterrows():
        note = "← RDD-only" if row["column"] in only_in_rdd else ""
        p(f"| `{row['column']}` | {row['dtype']} | {row['pct_missing']:.2f}% | {note} |")
    p()

    if only_in_full:
        p(f"### Columns in FULL data but NOT in RDD sample ({len(only_in_full)})")
        p()
        for c in sorted(only_in_full):
            p(f"- `{c}`")
        p()
    if only_in_rdd:
        p(f"### Columns in RDD sample but NOT in FULL data ({len(only_in_rdd)})")
        p()
        for c in sorted(only_in_rdd):
            p(f"- `{c}`")
        p()
    p(f"**Columns in both:** {len(in_both)} — {', '.join(f'`{c}`' for c in sorted(in_both))}")
    p()

    # ====================================================================
    # SECTION 3: Categorical distributions
    # ====================================================================
    p("---")
    p()
    p("## 3. Categorical Variable Distributions (RDD Sample)")
    p()
    categorical_report_md(rdd_df, "RDD Sample")

    p("---")
    p()
    p("## 3B. Categorical Variable Distributions (Full Data)")
    p()
    categorical_report_md(full_df, "Full Data")

    # ====================================================================
    # SECTION 4: Numeric statistics
    # ====================================================================
    p("---")
    p()
    p("## 4. Numeric Variable Statistics (RDD Sample)")
    p()
    numeric_report_md(rdd_df, "RDD Sample")

    p("---")
    p()
    p("## 4B. Numeric Variable Statistics (Full Data)")
    p()
    numeric_report_md(full_df, "Full Data")

    # ====================================================================
    # SECTION 5: Overlap check
    # ====================================================================
    p("---")
    p()
    p("## 5. Overlap Check — Candidate Control Variables")
    p()

    skip_for_controls = _RATING_COLS | {
        "review_id", "rcid", "company_id", "company", "gvkey",
        "ultimate_parent_rcid", "ultimate_parent_company_name",
        "review_date", "review_time", "gvkey_match_source",
        "review_summary", "review_advice", "review_pros", "review_cons",
        "job_title_raw", "role_k1500",
    }
    candidate_cols = [c for c in sorted(full_col_names) if c not in skip_for_controls]

    p("| Variable | % Non-missing (Full) | % Non-missing (RDD) | N Non-missing (RDD) | Status |")
    p("|----------|---------------------|--------------------|--------------------|--------|")
    for col in candidate_cols:
        full_pct = (1 - full_df[col].isna().mean()) * 100
        if col in rdd_df.columns:
            rdd_pct = (1 - rdd_df[col].isna().mean()) * 100
            n_rdd = rdd_df[col].notna().sum()
            status = "✅ present"
        else:
            rdd_pct = 0.0
            n_rdd = 0
            status = "❌ NOT IN RDD"
        p(
            f"| `{col}` | {full_pct:.1f}% | {rdd_pct:.1f}% "
            f"| {n_rdd:,} | {status} |"
        )
    p()

    rdd_only_candidates = only_in_rdd - skip_for_controls
    if rdd_only_candidates:
        p("### RDD-only Variables (not in full data)")
        p()
        p("| Variable | % Non-missing (RDD) | N Non-missing (RDD) |")
        p("|----------|--------------------|--------------------|")
        for col in sorted(rdd_only_candidates):
            rdd_pct = (1 - rdd_df[col].isna().mean()) * 100
            n_rdd = rdd_df[col].notna().sum()
            p(f"| `{col}` | {rdd_pct:.1f}% | {n_rdd:,} |")
        p()

    # ====================================================================
    # SECTION 6: Sample rows
    # ====================================================================
    p("---")
    p()
    p("## 6. Sample Rows from RDD Sample")
    p()

    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(len(rdd_df), size=min(3, len(rdd_df)), replace=False)
    for i, idx in enumerate(sample_idxs):
        p(f"### Row {i+1} (index {idx})")
        p()
        row = rdd_df.iloc[idx]
        p("| Column | Value |")
        p("|--------|-------|")
        for col in rdd_df.columns:
            val = row[col]
            if pd.isna(val):
                val_str = "*missing*"
            else:
                val_str = str(val).replace("|", "/")
                if len(val_str) > 100:
                    val_str = val_str[:97] + "..."
            p(f"| `{col}` | `{val_str}` |")
        p()

    # ====================================================================
    # SECTION 7: Summary & Recommendations
    # ====================================================================
    p("---")
    p()
    p("## 7. Summary & Recommendations")
    p()
    p("### Key Finding")
    p()
    p("The RDD sample retains **only 1 individual-level covariate from the full data:** "
      f"`state`. All other candidate controls "
      f"(`reviewer_employment_status`, `reviewer_current_job`, `reviewer_length_of_employment`, "
      f"`seniority`, `review_language_id`, `country`, `metro_area`, `reviewer_job_ending_year`, "
      f"`review_count_helpful`, `review_iscovid19`) are missing from the RDD sample.")
    p()
    p("### Available control variables (full data)")
    p()
    p("| Variable | Coverage | Type | Notes |")
    p("|----------|----------|------|-------|")
    p("| `state` | 63.8% | categorical | **Already in RDD**; use as location FE |")
    p("| `reviewer_employment_status` | 90.8% | categorical | REGULAR, PART_TIME, INTERN, CONTRACT, FREELANCE, TEMPORARY… |")
    p("| `reviewer_current_job` | 100% | bool | Current vs former employee |")
    p("| `reviewer_length_of_employment` | 60.5% | numeric (years) | Categories: 1, 2, 4, 6, 9, 20; median = 2 |")
    p("| `seniority` | 100% | ordinal 1–7 | 1=entry, 5–7=executive |")
    p("| `review_language_id` | 100% | categorical | eng (87%), por (6%), spa (3%), fra (2%) |")
    p("| `country` | 64.7% | categorical | 238 unique; US 45%, India 15%, UK 7% |")
    p("| `metro_area` | 64.7% | categorical | 825 metro areas |")
    p("| `rating_business_outlook` | 63.2% | categorical | POSITIVE / NEUTRAL / NEGATIVE |")
    p("| `rating_ceo` | 54.4% | categorical | APPROVE / NO_OPINION / DISAPPROVE |")
    p("| `rating_recommend_to_friend` | 70.5% | binary | POSITIVE / NEGATIVE |")
    p("| `review_count_helpful` | 19.0% | numeric | Sparse; mean=3.0, median=2 |")
    p("| `role_k1500` | 80.9% | categorical | 1,500 standardized job categories |")
    p()
    p("### Recommended actions")
    p()
    p("1. **Rebuild the RDD sample** to include `reviewer_employment_status`, "
      "`reviewer_length_of_employment`, `seniority`, `review_language_id`, and `role_k1500` "
      "by modifying `build_rdd_review_event_sample_from_raw.py` to carry these columns through the merge.")
    p()
    p("2. **Or post-merge** by joining the full Glassdoor data on `(gvkey, review_id)` "
      "— verify `review_id` consistency first; ~25% of RDD reviews lack `job_title_raw` and "
      "~38% lack `state`, so check whether IDs match.")
    p()
    p("3. **For immediate use:** the only covariate available in the RDD sample as-is is "
      "`state` (37.6% missing). `employee_filter` (current/former, 0% missing) is already "
      "encoded as `is_current_employee` / `is_former_employee` and can be used for subsample analysis.")
    p()

    # ====================================================================
    # Write output
    # ====================================================================
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    content = _md.getvalue()
    OUTPUT_MD.write_text(content, encoding="utf-8")
    print(f"Report written to: {OUTPUT_MD}")
    print(f"  {len(content):,} chars / {content.count(chr(10)) + 1} lines")

    # Also print to stdout for convenience
    print(content)


if __name__ == "__main__":
    main()
