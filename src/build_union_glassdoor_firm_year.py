from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


UNION_PATH = Path(
    "/data/disk4/workspace/projects/union/outputs/union_election_rc_votes_gvkey_only.parquet"
)
GLASSDOOR_FIRM_YEAR_ZIP = Path(
    "/data/disk5/projects/shared_glassdoor_data/firm_year_glassdoor.zip"
)
CONTROLS_PATH = Path(
    "/data/disk4/workspace/projects/union_glassdoor/outputs/compustat_firm_controls.parquet"
)

OUT_DIR = Path("/data/disk4/workspace/projects/union_glassdoor/outputs")
OUT_PARQUET = OUT_DIR / "union_glassdoor_firm_year_regression.parquet"
OUT_DTA = OUT_DIR / "union_glassdoor_firm_year_regression.dta"
OUT_WIN_DTA = OUT_DIR / "union_glassdoor_firm_year_regression_winsor_1_99.dta"
OUT_WIN_PARQUET = OUT_DIR / "union_glassdoor_firm_year_regression_winsor_1_99.parquet"
OUT_WINSOR_LOG = OUT_DIR / "union_glassdoor_firm_year_winsorized_vars.json"

MAIN_OUTCOMES = [
    "GD_rating",
    "GD_career_opp",
    "GD_comp_benefit",
    "GD_senior_mgmt",
    "GD_wlb",
    "GD_culture",
    "GD_diversity",
    "pct_recommend",
    "pct_ceo_approve",
    "pct_positive_outlook",
]

REVIEW_VOLUME_CONTROLS = [
    "n_reviews",
    "n_current_emp",
    "n_former_emp",
    "pct_current",
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


def standardize_gvkey(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.where(s.str.fullmatch(r"\d+"), s)
    # For numeric gvkeys, keep 6-digit zero-padded format.
    is_digits = s.str.fullmatch(r"\d+")
    s.loc[is_digits.fillna(False)] = s.loc[is_digits.fillna(False)].str.zfill(6)
    return s


def detect_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    col = first_available(df, candidates)
    if required and col is None:
        raise KeyError(f"Could not find any of columns: {candidates}")
    return col


def load_glassdoor_firm_year_zip(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with zipfile.ZipFile(path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            raise ValueError("Zip file has no data files")

        preferred_order = sorted(
            names,
            key=lambda n: (
                0 if n.lower().endswith(".parquet") else 1,
                0 if n.lower().endswith(".csv") else 1,
                0 if n.lower().endswith(".pkl") else 1,
                n,
            ),
        )
        target = preferred_order[0]
        print(f"Reading from zip member: {target}")

        with zf.open(target) as f:
            lower = target.lower()
            if lower.endswith(".csv"):
                return pd.read_csv(f)
            if lower.endswith(".parquet"):
                return pd.read_parquet(f)
            if lower.endswith(".pkl") or lower.endswith(".pickle"):
                return pd.read_pickle(f)
            raise ValueError(f"Unsupported file inside zip: {target}")


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    union = pd.read_parquet(UNION_PATH)
    gd = load_glassdoor_firm_year_zip(GLASSDOOR_FIRM_YEAR_ZIP)
    controls = pd.read_parquet(CONTROLS_PATH)
    print(f"Union rows: {len(union):,}")
    print(f"Glassdoor firm-year rows: {len(gd):,}")
    print(f"Compustat controls rows: {len(controls):,}")
    return union, gd, controls


def clean_union(union: pd.DataFrame) -> pd.DataFrame:
    print_banner("Clean Union Election Data")
    df = union.copy()

    gv_col = detect_column(df, ["gvkey_final", "gvkey"])
    date_col = detect_column(df, ["election_date", "date"])

    votes_for_col = detect_column(df, ["votes_for_union"])
    votes_against_col = detect_column(df, ["votes_against_union"])
    support_col = detect_column(df, ["union_support_rate"])
    total_votes_col = detect_column(df, ["total_valid_votes"])

    df[gv_col] = standardize_gvkey(df[gv_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    before = len(df)
    df = df[df[gv_col].notna() & df[date_col].notna()].copy()
    print(f"Drop missing gvkey/date: {before - len(df):,}")

    df["election_year"] = df[date_col].dt.year.astype("Int64")
    df[total_votes_col] = pd.to_numeric(df[total_votes_col], errors="coerce")
    before_votes = len(df)
    df = df[df[total_votes_col] > 10].copy()
    print(f"Drop total_valid_votes <= 10: {before_votes - len(df):,}")

    df[votes_for_col] = pd.to_numeric(df[votes_for_col], errors="coerce")
    df[votes_against_col] = pd.to_numeric(df[votes_against_col], errors="coerce")
    df[support_col] = pd.to_numeric(df[support_col], errors="coerce")

    df["win_union"] = (df[votes_for_col] > df[votes_against_col]).astype("Int8")
    df["lose_union"] = (df[votes_for_col] < df[votes_against_col]).astype("Int8")
    df["union_tie"] = (df[votes_for_col] == df[votes_against_col]).astype("Int8")
    df["union_margin"] = df[support_col] - 0.5
    df["close_election_abs_margin"] = df["union_margin"].abs()

    tie_n = int(df["union_tie"].sum())
    print(f"Tie elections flagged: {tie_n:,}")

    key_cols = [c for c in ["election_id", "case_number", gv_col, date_col, "election_year"] if c in df.columns]
    keep_cols = key_cols + [
        votes_for_col,
        votes_against_col,
        total_votes_col,
        support_col,
        "win_union",
        "lose_union",
        "union_tie",
        "union_margin",
        "close_election_abs_margin",
    ]

    out = df[keep_cols].copy()
    out = out.rename(
        columns={
            gv_col: "gvkey",
            date_col: "election_date",
            votes_for_col: "votes_for_union",
            votes_against_col: "votes_against_union",
            total_votes_col: "total_valid_votes",
            support_col: "union_support_rate",
        }
    )
    return out


def resolve_firm_year_elections(df: pd.DataFrame) -> pd.DataFrame:
    print_banner("Resolve Multiple Elections within Firm-Year")
    base_n = len(df)
    sorted_df = df.sort_values(["gvkey", "election_year", "election_date"]).copy()

    dup_before = int(sorted_df.duplicated(["gvkey", "election_year"]).sum())
    print(f"Firm-year duplicate elections before: {dup_before:,}")

    out = sorted_df.drop_duplicates(["gvkey", "election_year"], keep="first").copy()
    dropped = base_n - len(out)
    print(f"Dropped by first-election-within-firm-year rule: {dropped:,}")

    dup_after = int(out.duplicated(["gvkey", "election_year"]).sum())
    print(f"Firm-year duplicate elections after: {dup_after:,}")
    return out


def clean_glassdoor_firm_year(gd: pd.DataFrame) -> pd.DataFrame:
    print_banner("Clean Glassdoor Firm-Year Data")
    df = gd.copy()

    gv_col = detect_column(df, ["gvkey", "gvkey_final", "gvkey6"])
    year_col = detect_column(df, ["year", "fyear", "calendar_year"])

    df[gv_col] = standardize_gvkey(df[gv_col])
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")

    before = len(df)
    df = df[df[gv_col].notna() & df[year_col].notna()].copy()
    print(f"Drop missing gvkey/year: {before - len(df):,}")

    dup_n = int(df.duplicated([gv_col, year_col]).sum())
    print(f"Duplicate gvkey-year rows before resolution: {dup_n:,}")
    if dup_n > 0:
        print("Resolving duplicates by taking mean for numeric columns and first for non-numeric columns.")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num_cols = [c for c in df.columns if c not in num_cols]
        agg = {c: "mean" for c in num_cols}
        agg.update({c: "first" for c in non_num_cols if c not in [gv_col, year_col]})
        df = df.groupby([gv_col, year_col], as_index=False).agg(agg)

    df = df.rename(columns={gv_col: "gvkey", year_col: "year"})

    # Harmonize outcome and review volume names when close alternatives are present.
    rename_map = {}
    candidates = {
        "GD_rating": ["gd_rating", "overall_rating", "rating"],
        "GD_career_opp": ["gd_career_opp", "career_opp", "career_opportunities"],
        "GD_comp_benefit": ["gd_comp_benefit", "comp_benefits", "compensation_benefits"],
        "GD_senior_mgmt": ["gd_senior_mgmt", "senior_management"],
        "GD_wlb": ["gd_wlb", "work_life_balance", "wlb"],
        "GD_culture": ["gd_culture", "culture_values", "culture"],
        "GD_diversity": ["gd_diversity", "diversity_inclusion", "diversity"],
        "pct_recommend": ["recommend_pct", "pct_recommend", "recommendation_pct"],
        "pct_ceo_approve": ["ceo_approve_pct", "pct_ceo_approve"],
        "pct_positive_outlook": ["positive_outlook_pct", "pct_positive_outlook"],
        "n_reviews": ["n_reviews", "review_count"],
        "n_current_emp": ["n_current_emp", "current_emp_reviews"],
        "n_former_emp": ["n_former_emp", "former_emp_reviews"],
        "pct_current": ["pct_current", "share_current"],
    }
    lower_to_orig = {c.lower(): c for c in df.columns}
    for target, alt in candidates.items():
        if target in df.columns:
            continue
        for c in alt:
            if c.lower() in lower_to_orig:
                rename_map[lower_to_orig[c.lower()]] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)

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
    df = df.rename(columns={gv_col: "gvkey", year_col: "year"})
    return df


def merge_outcomes(
    union_fy: pd.DataFrame,
    gd: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    print_banner("Merge Union with Glassdoor (Contemporaneous and Next-Year)")

    current = union_fy.copy()
    current["outcome_year"] = current["election_year"]
    current["outcome_source"] = "contemporaneous"

    next_year = union_fy.copy()
    next_year["outcome_year"] = next_year["election_year"] + 1
    next_year["outcome_source"] = "next_year"

    stack = pd.concat([current, next_year], ignore_index=True)

    merged = stack.merge(
        gd,
        left_on=["gvkey", "outcome_year"],
        right_on=["gvkey", "year"],
        how="left",
        indicator="merge_union_glassdoor",
    )

    merged["post_election_year"] = (merged["outcome_year"] > merged["election_year"]).astype("Int8")
    merged["event_year"] = (merged["outcome_year"] - merged["election_year"]).astype("Int64")

    merged = merged.merge(
        controls,
        left_on=["gvkey", "outcome_year"],
        right_on=["gvkey", "year"],
        how="left",
        indicator="merge_controls",
        suffixes=("", "_ctrl"),
    )

    # Keep one row per gvkey-outcome_year-outcome_source-election record.
    key = [c for c in ["gvkey", "outcome_year", "outcome_source", "election_id", "case_number"] if c in merged.columns]
    if key:
        merged = merged.drop_duplicates(key, keep="first").copy()

    return merged


def summarize_final(df: pd.DataFrame) -> None:
    print_banner("Final Validation")
    print(f"Final shape: {df.shape}")
    print(f"Unique firms: {df['gvkey'].nunique():,}")
    if "outcome_year" in df.columns:
        print(f"Year range: {int(df['outcome_year'].min())} to {int(df['outcome_year'].max())}")

    if "win_union" in df.columns:
        print("\nwin_union distribution:")
        print(df["win_union"].value_counts(dropna=False).sort_index())

    if "union_margin" in df.columns:
        print("\nunion_margin summary:")
        print(df["union_margin"].describe(percentiles=[0.01, 0.5, 0.99]))

    dup_n = int(df.duplicated(["gvkey", "outcome_year", "outcome_source"]).sum())
    print(f"Duplicate gvkey-year-source count: {dup_n:,}")

    major = [
        c
        for c in [
            *MAIN_OUTCOMES,
            *REVIEW_VOLUME_CONTROLS,
            "union_margin",
            "win_union",
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
        print("\nMissingness report for major outcomes/controls:")
        print(miss.round(4))

    if "merge_union_glassdoor" in df.columns:
        print("\nUnion-Glassdoor merge diagnostics:")
        print(df["merge_union_glassdoor"].value_counts(dropna=False))
    if "merge_controls" in df.columns:
        print("\nControls merge diagnostics:")
        print(df["merge_controls"].value_counts(dropna=False))


def make_stata_compatible(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype("Int8")
        elif pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")

    # Drop columns with nested objects that cannot be exported to Stata.
    drop_cols = []
    for col in out.columns:
        if out[col].dtype == "object":
            sample = out[col].dropna().head(20)
            if sample.map(lambda x: isinstance(x, (list, dict, tuple, set))).any():
                drop_cols.append(col)
    if drop_cols:
        print(f"Dropping unsupported object columns for Stata: {drop_cols}")
        out = out.drop(columns=drop_cols)

    # Stata variable name constraints: <= 32 chars, [a-zA-Z_][a-zA-Z0-9_]*
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

    # Only continuous variables used in regressions; excludes IDs, dummies, years,
    # vote counts, running variable union_margin, and event-time/day variables.
    candidate_vars = [
        *MAIN_OUTCOMES,
        "pct_current",
        *RAW_CONTROL_VARS,
        *LAG_CONTROL_VARS,
        "close_election_abs_margin",
    ]
    winsor_vars = [c for c in candidate_vars if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]

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
        "winsorization": "1st and 99th percentiles",
        "winsorized_variables": winsor_vars,
        "excluded_by_design_examples": [
            "identifiers",
            "fixed-effect indexing variables",
            "binary treatment variables",
            "union_margin",
            "event-time variables",
            "count variables that should remain integers",
        ],
    }
    OUT_WINSOR_LOG.write_text(json.dumps(winsor_meta, indent=2), encoding="utf-8")
    print(f"Saved winsorization log: {OUT_WINSOR_LOG}")


def main() -> None:
    print_banner("Build Union x Glassdoor Firm-Year Regression Dataset")
    union_raw, gd_raw, controls_raw = load_inputs()

    union_clean = clean_union(union_raw)
    union_one = resolve_firm_year_elections(union_clean)

    gd_clean = clean_glassdoor_firm_year(gd_raw)
    controls = prepare_controls(controls_raw)

    merged = merge_outcomes(union_one, gd_clean, controls)

    # Keep one row per firm-year outcome observation in final file.
    key = [c for c in ["gvkey", "outcome_year", "outcome_source", "election_id", "case_number"] if c in merged.columns]
    if key:
        merged = merged.drop_duplicates(key, keep="first").copy()

    summarize_final(merged)
    export_outputs(merged)


if __name__ == "__main__":
    main()
