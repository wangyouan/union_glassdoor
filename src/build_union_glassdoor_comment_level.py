from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


COMMENT_WINDOW_DAYS = 365

UNION_PATH = Path(
    "/data/disk4/workspace/projects/union/outputs/union_election_rc_votes_gvkey_only.parquet"
)
REVIEWS_PATH = Path(
    "/data/disk5/projects/shared_glassdoor_data/2008_2023_glassdoor_reviews_with_gvkey_v2.pkl"
)
CONTROLS_PATH = Path(
    "/data/disk4/workspace/projects/union_glassdoor/outputs/compustat_firm_controls.parquet"
)

OUT_DIR = Path("/data/disk4/workspace/projects/union_glassdoor/outputs")
OUT_PARQUET = OUT_DIR / f"union_glassdoor_comment_level_window{COMMENT_WINDOW_DAYS}.parquet"
OUT_DTA = OUT_DIR / f"union_glassdoor_comment_level_window{COMMENT_WINDOW_DAYS}.dta"
OUT_WIN_DTA = OUT_DIR / (
    f"union_glassdoor_comment_level_window{COMMENT_WINDOW_DAYS}_winsor_1_99.dta"
)
OUT_WIN_PARQUET = OUT_DIR / (
    f"union_glassdoor_comment_level_window{COMMENT_WINDOW_DAYS}_winsor_1_99.parquet"
)
OUT_WINSOR_LOG = OUT_DIR / (
    f"union_glassdoor_comment_level_window{COMMENT_WINDOW_DAYS}_winsorized_vars.json"
)

MAIN_COMMENT_OUTCOMES = [
    "GD_rating",
    "recommend",
    "ceo_approve",
    "business_outlook",
    "GD_career_opp",
    "GD_comp_benefit",
    "GD_senior_mgmt",
    "GD_wlb",
    "GD_culture",
    "GD_diversity",
]

LAG_CONTROL_VARS = [
    "L_size",
    "L_log_me",
    "L_leverage",
    "L_cash_ratio",
    "L_roa",
    "L_profitability",
    "L_tangibility",
    "L_capx_at",
    "L_rd_at",
    "L_book_to_market",
    "L_sales_growth",
    "L_log_emp",
]

RAW_CONTROL_VARS = [
    "size",
    "log_me",
    "leverage",
    "cash_ratio",
    "roa",
    "profitability",
    "tangibility",
    "capx_at",
    "rd_at",
    "book_to_market",
    "sales_growth",
    "log_emp",
]


def print_banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def first_available(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    col = first_available(df, candidates)
    if required and col is None:
        raise KeyError(f"Could not find any of columns: {candidates}")
    return col


def detect_review_date_column(df: pd.DataFrame) -> str:
    # Start with explicit known variants, then fall back to a heuristic guess.
    explicit = [
        "review_date",
        "date",
        "review_datetime",
        "time",
        "created_at",
        "date_reviewed",
        "GD_ReviewDate",
        "gd_reviewdate",
    ]
    col = first_available(df, explicit)
    if col is not None:
        return col

    lowered = {c.lower(): c for c in df.columns}
    tokens = ("date", "time", "created", "posted")
    guesses = [orig for low, orig in lowered.items() if any(t in low for t in tokens)]
    if guesses:
        # Prefer names that also include "review" if available.
        guesses = sorted(guesses, key=lambda c: ("review" not in c.lower(), c.lower()))
        chosen = guesses[0]
        print(f"Auto-detected review date column: {chosen}")
        return chosen

    raise KeyError(
        "Could not identify a review date column. "
        f"Available columns: {list(df.columns)}"
    )


def standardize_gvkey(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    is_digits = s.str.fullmatch(r"\d+")
    s.loc[is_digits.fillna(False)] = s.loc[is_digits.fillna(False)].str.zfill(6)
    return s


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    union = pd.read_parquet(UNION_PATH)
    reviews = pd.read_pickle(REVIEWS_PATH)
    controls = pd.read_parquet(CONTROLS_PATH)
    print(f"Union rows: {len(union):,}")
    print(f"Review rows: {len(reviews):,}")
    print(f"Controls rows: {len(controls):,}")
    return union, reviews, controls


def clean_union(union: pd.DataFrame) -> pd.DataFrame:
    print_banner("Clean Union Election Data")
    df = union.copy()

    gv_col = detect_column(df, ["gvkey_final", "gvkey"])
    date_col = detect_column(df, ["election_date", "date"])
    vf_col = detect_column(df, ["votes_for_union"])
    va_col = detect_column(df, ["votes_against_union"])
    support_col = detect_column(df, ["union_support_rate"])
    tv_col = detect_column(df, ["total_valid_votes"])

    df[gv_col] = standardize_gvkey(df[gv_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    before = len(df)
    df = df[df[gv_col].notna() & df[date_col].notna()].copy()
    print(f"Drop missing gvkey/date: {before - len(df):,}")

    df[tv_col] = pd.to_numeric(df[tv_col], errors="coerce")
    before_votes = len(df)
    df = df[df[tv_col] > 10].copy()
    print(f"Drop total_valid_votes <= 10: {before_votes - len(df):,}")

    df[vf_col] = pd.to_numeric(df[vf_col], errors="coerce")
    df[va_col] = pd.to_numeric(df[va_col], errors="coerce")
    df[support_col] = pd.to_numeric(df[support_col], errors="coerce")

    df["win_union"] = (df[vf_col] > df[va_col]).astype("Int8")
    df["lose_union"] = (df[vf_col] < df[va_col]).astype("Int8")
    df["union_tie"] = (df[vf_col] == df[va_col]).astype("Int8")
    df["union_margin"] = df[support_col] - 0.5
    df["close_election_abs_margin"] = df["union_margin"].abs()

    tie_n = int(df["union_tie"].sum())
    print(f"Tie elections flagged: {tie_n:,}")

    sort_cols = [gv_col, date_col]
    df = df.sort_values(sort_cols).copy()

    dup_before = int(df.duplicated([gv_col]).sum())
    print(f"Multiple elections per firm before baseline rule: {dup_before:,}")

    first_per_firm = df.drop_duplicates([gv_col], keep="first").copy()
    dropped = len(df) - len(first_per_firm)
    print(f"Dropped by first-election-per-firm baseline rule: {dropped:,}")

    keep_cols = [
        c
        for c in ["election_id", "case_number", gv_col, date_col, vf_col, va_col, tv_col, support_col]
        if c in first_per_firm.columns
    ] + ["win_union", "lose_union", "union_tie", "union_margin", "close_election_abs_margin"]

    out = first_per_firm[keep_cols].rename(
        columns={
            gv_col: "gvkey",
            date_col: "election_date",
            vf_col: "votes_for_union",
            va_col: "votes_against_union",
            tv_col: "total_valid_votes",
            support_col: "union_support_rate",
        }
    )
    return out


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    print_banner("Clean Glassdoor Review-Level Data")
    df = reviews.copy()

    gv_col = detect_column(df, ["gvkey", "gvkey_final", "gvkey6"])
    review_date_col = detect_review_date_column(df)

    review_id_col = detect_column(df, ["review_id", "id", "reviewid"], required=False)
    if review_id_col is None:
        df["review_id"] = np.arange(1, len(df) + 1)
        review_id_col = "review_id"

    df[gv_col] = standardize_gvkey(df[gv_col])
    df[review_date_col] = pd.to_datetime(df[review_date_col], errors="coerce")

    before = len(df)
    df = df[df[gv_col].notna() & df[review_date_col].notna()].copy()
    print(f"Drop missing gvkey/review_date: {before - len(df):,}")

    rename_map = {gv_col: "gvkey", review_date_col: "review_date", review_id_col: "review_id"}
    df = df.rename(columns=rename_map)

    # Harmonize outcome columns where close alternatives exist.
    candidates = {
        "GD_rating": ["gd_rating", "overall_rating", "rating"],
        "recommend": ["recommend", "recommendation", "is_recommend"],
        "ceo_approve": ["ceo_approve", "ceo_approval", "approve_ceo"],
        "business_outlook": ["business_outlook", "outlook", "positive_outlook"],
        "GD_career_opp": ["gd_career_opp", "career_opp", "career_opportunities"],
        "GD_comp_benefit": ["gd_comp_benefit", "comp_benefits", "compensation_benefits"],
        "GD_senior_mgmt": ["gd_senior_mgmt", "senior_management"],
        "GD_wlb": ["gd_wlb", "work_life_balance", "wlb"],
        "GD_culture": ["gd_culture", "culture_values", "culture"],
        "GD_diversity": ["gd_diversity", "diversity_inclusion", "diversity"],
        "is_current_emp": ["is_current_emp", "current_employee", "current_emp"],
        "is_former_emp": ["is_former_emp", "former_employee", "former_emp"],
        "job_title": ["job_title", "title"],
        "job_group": ["job_group", "job_family"],
        "pros": ["pros"],
        "cons": ["cons"],
    }
    lower_to_orig = {c.lower(): c for c in df.columns}
    remap = {}
    for target, alts in candidates.items():
        if target in df.columns:
            continue
        for c in alts:
            if c.lower() in lower_to_orig:
                remap[lower_to_orig[c.lower()]] = target
                break
    if remap:
        df = df.rename(columns=remap)

    # Text length controls, if text columns exist.
    if "pros" in df.columns:
        df["pros_len"] = df["pros"].astype("string").str.len().astype("Float64")
    if "cons" in df.columns:
        df["cons_len"] = df["cons"].astype("string").str.len().astype("Float64")

    print("Available key rating columns:")
    print([c for c in MAIN_COMMENT_OUTCOMES if c in df.columns])

    return df


def prepare_controls(controls: pd.DataFrame) -> pd.DataFrame:
    print_banner("Prepare Compustat Controls")
    df = controls.copy()
    gv_col = detect_column(df, ["gvkey"])
    year_col = detect_column(df, ["fyear", "year"])

    df[gv_col] = standardize_gvkey(df[gv_col])
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df = df[df[gv_col].notna() & df[year_col].notna()].copy()

    keep_cols = [
        c
        for c in [
            gv_col,
            year_col,
            "conm",
            "tic",
            "cik",
            *RAW_CONTROL_VARS,
            *LAG_CONTROL_VARS,
        ]
        if c in df.columns
    ]

    df = df[keep_cols].drop_duplicates([gv_col, year_col], keep="first").copy()
    df = df.rename(columns={gv_col: "gvkey", year_col: "review_year"})
    return df


def merge_reviews_with_events(reviews: pd.DataFrame, elections: pd.DataFrame) -> pd.DataFrame:
    print_banner("Merge Reviews with Elections by Firm")
    merged = reviews.merge(elections, on="gvkey", how="left", indicator="merge_reviews_elections")

    potential_multi = (
        merged.groupby("review_id", dropna=False)
        .size()
        .rename("n_event_matches")
        .reset_index()
    )
    multi_n = int((potential_multi["n_event_matches"] > 1).sum())
    print(f"Reviews potentially matching multiple elections: {multi_n:,}")

    merged = merged.merge(potential_multi, on="review_id", how="left")

    merged["days_from_election"] = (
        merged["review_date"] - merged["election_date"]
    ).dt.days
    merged["post_election_comment"] = (merged["days_from_election"] >= 0).astype("Int8")
    merged["abs_days_from_election"] = merged["days_from_election"].abs()

    before_window = len(merged)
    in_window = merged["abs_days_from_election"].le(COMMENT_WINDOW_DAYS)
    merged = merged[in_window.fillna(False)].copy()
    print(f"Rows before window filter: {before_window:,}")
    print(f"Rows after window filter (+/-{COMMENT_WINDOW_DAYS} days): {len(merged):,}")

    merged["review_year"] = merged["review_date"].dt.year.astype("Int64")
    return merged


def summarize_final(df: pd.DataFrame) -> None:
    print_banner("Final Validation")
    print(f"Final shape: {df.shape}")
    print(f"Unique firms: {df['gvkey'].nunique():,}")
    if "election_id" in df.columns:
        print(f"Unique elections: {df['election_id'].nunique():,}")

    if "days_from_election" in df.columns:
        print("\ndays_from_election distribution:")
        print(df["days_from_election"].describe(percentiles=[0.01, 0.5, 0.99]))

    before_after_cols = [c for c in MAIN_COMMENT_OUTCOMES if c in df.columns]
    if before_after_cols:
        means = (
            df.groupby("post_election_comment", dropna=False)[before_after_cols]
            .mean(numeric_only=True)
            .T
            .round(4)
        )
        print("\nMean outcomes before vs after election:")
        print(means)

    dup_review = int(df.duplicated(["review_id", "election_id"], keep=False).sum()) if "election_id" in df.columns else 0
    print(f"Duplicate review-election evidence rows: {dup_review:,}")

    major = [
        c
        for c in [
            *MAIN_COMMENT_OUTCOMES,
            "days_from_election",
            "post_election_comment",
            "union_margin",
            "win_union",
            "is_current_emp",
            "is_former_emp",
            "job_title",
            "job_group",
            "pros_len",
            "cons_len",
            *LAG_CONTROL_VARS,
        ]
        if c in df.columns
    ]
    if major:
        miss = (
            df[major]
            .isna()
            .mean()
            .rename("missing_share")
            .sort_values(ascending=False)
            .to_frame()
        )
        print("\nMissingness report for major variables:")
        print(miss.round(4))


def make_stata_compatible(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype("Int8")
        elif pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")

    drop_cols = []
    for col in out.columns:
        if out[col].dtype == "object":
            sample = out[col].dropna().head(20)
            if sample.map(lambda x: isinstance(x, (list, dict, tuple, set))).any():
                drop_cols.append(col)
    if drop_cols:
        print(f"Dropping unsupported object columns for Stata: {drop_cols}")
        out = out.drop(columns=drop_cols)

    # Stata supports string/object columns only when values are strings or None.
    # Coerce mixed object columns safely and drop all-null text columns.
    text_drop_cols = []
    for col in out.columns:
        if out[col].dtype == "object" or pd.api.types.is_string_dtype(out[col]):
            s = out[col]
            non_null = s.dropna()
            if non_null.empty:
                text_drop_cols.append(col)
                continue

            out[col] = s.map(lambda v: None if pd.isna(v) else str(v))

    if text_drop_cols:
        print(f"Dropping all-null text columns for Stata: {text_drop_cols}")
        out = out.drop(columns=text_drop_cols)

    rename = {}
    used = set()
    for c in out.columns:
        new = c.lower()
        new = re.sub(r"[^a-z0-9_]", "_", new)
        if not re.match(r"^[a-z_]", new):
            new = f"v_{new}"
        new = new[:32]
        base = new
        i = 1
        while new in used:
            suffix = f"_{i}"
            new = (base[: 32 - len(suffix)] + suffix)[:32]
            i += 1
        used.add(new)
        if new != c:
            rename[c] = new
    if rename:
        out = out.rename(columns=rename)

    return out


def winsorize_for_regression(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()

    # Excludes identifiers, dummies, event-time vars including days_from_election,
    # running variable union_margin, and integer count-like columns.
    candidate_vars = [
        *MAIN_COMMENT_OUTCOMES,
        *RAW_CONTROL_VARS,
        *LAG_CONTROL_VARS,
        "close_election_abs_margin",
        "pros_len",
        "cons_len",
    ]

    winsor_vars = [
        c for c in candidate_vars if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
    ]

    for c in winsor_vars:
        low = out[c].quantile(0.01)
        high = out[c].quantile(0.99)
        out[c] = out[c].clip(lower=low, upper=high)

    return out, winsor_vars


def export_outputs(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved parquet: {OUT_PARQUET}")

    stata_df = make_stata_compatible(df)
    stata_df.to_stata(OUT_DTA, write_index=False, version=118)
    print(f"Saved Stata: {OUT_DTA}")

    win_df, winsor_vars = winsorize_for_regression(df)
    win_df.to_parquet(OUT_WIN_PARQUET, index=False)
    print(f"Saved winsorized parquet: {OUT_WIN_PARQUET}")

    win_stata = make_stata_compatible(win_df)
    win_stata.to_stata(OUT_WIN_DTA, write_index=False, version=118)
    print(f"Saved winsorized Stata: {OUT_WIN_DTA}")

    winsor_meta = {
        "comment_window_days": COMMENT_WINDOW_DAYS,
        "winsorization": "1st and 99th percentiles",
        "winsorized_variables": winsor_vars,
        "excluded_by_design_examples": [
            "identifiers",
            "fixed-effect indexing variables",
            "binary treatment variables",
            "union_margin",
            "days_from_election",
            "event-time variables",
        ],
    }
    OUT_WINSOR_LOG.write_text(json.dumps(winsor_meta, indent=2), encoding="utf-8")
    print(f"Saved winsorization log: {OUT_WINSOR_LOG}")


def main() -> None:
    print_banner("Build Union x Glassdoor Comment-Level Regression Dataset")
    print(f"Configured COMMENT_WINDOW_DAYS: {COMMENT_WINDOW_DAYS}")

    union_raw, reviews_raw, controls_raw = load_inputs()

    elections = clean_union(union_raw)
    reviews = clean_reviews(reviews_raw)
    controls = prepare_controls(controls_raw)

    merged = merge_reviews_with_events(reviews, elections)
    merged = merged.merge(
        controls,
        on=["gvkey", "review_year"],
        how="left",
        indicator="merge_controls",
        suffixes=("", "_ctrl"),
    )

    # Keep one row per review-election pair.
    key = [c for c in ["review_id", "election_id"] if c in merged.columns]
    if key:
        merged = merged.drop_duplicates(key, keep="first").copy()

    summarize_final(merged)
    export_outputs(merged)


if __name__ == "__main__":
    main()
