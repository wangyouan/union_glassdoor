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
GLASSDOOR_FIRM_YEAR_PATH = Path(
    "/data/disk4/workspace/projects/glassdoor/outputs/firm_year_glassdoor_union.parquet"
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
OUT_STATA_MAP = OUT_DIR / "union_glassdoor_firm_year_regression_stata_varname_map.csv"
OUT_WIN_STATA_MAP = OUT_DIR / "union_glassdoor_firm_year_regression_winsor_1_99_stata_varname_map.csv"

MAIN_OUTCOMES = [
    "GD_rating",
    "GD_outlook",
    "GD_career_opp",
    "GD_ceo",
    "GD_comp_benefit",
    "GD_senior_mgmt",
    "GD_wlb",
    "GD_culture",
    "GD_diversity",
    "GD_recommend",
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

GLASSDOOR_MERGE_SHIFTS = [
    (-1, "lag1"),
    (0, ""),
    (1, "for1"),
]

RATING_BASE_SUFFIXES = [
    "GD_rating",
    "GD_outlook",
    "GD_career_opp",
    "GD_ceo",
    "GD_comp_benefit",
    "GD_senior_mgmt",
    "GD_wlb",
    "GD_culture",
    "GD_diversity",
    "GD_recommend",
]

ROLE_PREFIX_RENAME = {
    "role_likely_unionizable": "mayu",
    "role_likely_excluded_from_union": "notu",
    "role_ambiguous_union_status": "ambu",
}

STATA_TOKEN_MAP = {
    "role_likely_unionizable": "mayu",
    "role_likely_excluded_from_union": "notu",
    "role_ambiguous_union_status": "ambu",
    "gd_rating": "gdrat",
    "gd_outlook": "gdout",
    "gd_career_opp": "gdcar",
    "gd_ceo": "gdceo",
    "gd_comp_benefit": "gdcomp",
    "gd_senior_mgmt": "gdsen",
    "gd_wlb": "gdwlb",
    "gd_culture": "gdcult",
    "gd_diversity": "gddiv",
    "gd_recommend": "gdrec",
    "sdsic2": "sds2",
    "sdff48": "sdf48",
    "lag1": "l1",
    "for1": "f1",
}


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


def find_curr_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.lower().endswith("_curr")]


def rename_role_group_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df.columns:
        new_col = col
        for old, new in ROLE_PREFIX_RENAME.items():
            new_col = new_col.replace(old, new)
        if new_col != col:
            rename_map[col] = new_col

    if rename_map:
        print(f"Renaming role subgroup columns to short prefixes: {len(rename_map):,}")
        examples = list(rename_map.items())[:12]
        print(f"Role subgroup rename examples: {examples}")
        out = df.rename(columns=rename_map)
        dups = out.columns[out.columns.duplicated()].tolist()
        if dups:
            raise ValueError(f"Column collision after role prefix rename: {dups[:10]}")
        return out
    return df


def detect_rating_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    suffixes = []
    for base in RATING_BASE_SUFFIXES:
        suffixes.extend([base, f"{base}_lag1", f"{base}_for1"])

    for col in df.columns:
        if any(col.endswith(s) for s in suffixes):
            if pd.api.types.is_numeric_dtype(df[col]):
                candidates.append(col)

    return sorted(set(candidates))


def _standardize_within_groups(
    df: pd.DataFrame,
    col: str,
    group_keys: Sequence[str],
) -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce")
    grp = vals.groupby([df[k] for k in group_keys])

    mean = grp.transform("mean")
    std = grp.transform("std")
    count = grp.transform("count")

    out = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = vals.notna() & mean.notna() & std.notna() & (std != 0) & (count >= 2)
    out.loc[valid] = (vals.loc[valid] - mean.loc[valid]) / std.loc[valid]
    return out


def add_industry_year_standardized_ratings(df: pd.DataFrame) -> pd.DataFrame:
    print_banner("Add Industry-Year Standardized Ratings")
    out = df.copy()

    rating_cols = detect_rating_columns(out)
    print(f"Detected rating variables for standardization: {len(rating_cols):,}")

    if "sic" in out.columns:
        print(f"Non-missing sic in merged data: {int(out['sic'].notna().sum()):,}")
    else:
        print("WARNING: sic not present in merged data.")
    if "sic2" in out.columns:
        print(f"Non-missing sic2 in merged data: {int(out['sic2'].notna().sum()):,}")
    else:
        print("WARNING: sic2 not present in merged data; skip _sdsic2.")
    if "ff48" in out.columns:
        print(f"Non-missing ff48 in merged data: {int(out['ff48'].notna().sum()):,}")
    else:
        print("WARNING: ff48 not present in merged data; skip _sdff48.")

    created_sic2: List[str] = []
    created_ff48: List[str] = []

    has_sic2_keys = all(k in out.columns for k in ["election_year", "sic2"])
    has_ff48_keys = all(k in out.columns for k in ["election_year", "ff48"])

    if has_sic2_keys:
        sic2_cell_n = int(
            out.loc[out["election_year"].notna() & out["sic2"].notna(), ["election_year", "sic2"]]
            .drop_duplicates()
            .shape[0]
        )
        print(f"Industry-year cells for sic2: {sic2_cell_n:,}")
        for c in rating_cols:
            new_col = f"{c}_sdsic2"
            out[new_col] = _standardize_within_groups(out, c, ["election_year", "sic2"])
            created_sic2.append(new_col)
    else:
        print("WARNING: missing election_year or sic2; no _sdsic2 variables created.")

    if has_ff48_keys:
        ff48_cell_n = int(
            out.loc[out["election_year"].notna() & out["ff48"].notna(), ["election_year", "ff48"]]
            .drop_duplicates()
            .shape[0]
        )
        print(f"Industry-year cells for ff48: {ff48_cell_n:,}")
        for c in rating_cols:
            new_col = f"{c}_sdff48"
            out[new_col] = _standardize_within_groups(out, c, ["election_year", "ff48"])
            created_ff48.append(new_col)
    else:
        print("WARNING: missing election_year or ff48; no _sdff48 variables created.")

    created_all = [*created_sic2, *created_ff48]
    print(f"Created _sdsic2 variables: {len(created_sic2):,}")
    print(f"Created _sdff48 variables: {len(created_ff48):,}")
    if created_all:
        print(f"Example standardized vars (first 10): {created_all[:10]}")
        miss = out[created_all].isna().mean()
        print(
            "Standardized vars missingness (mean/min/max): "
            f"{miss.mean():.4f} / {miss.min():.4f} / {miss.max():.4f}"
        )

    return out


def load_glassdoor_firm_year(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    lower = path.name.lower()
    if lower.endswith(".csv"):
        print(f"Reading firm-year file: {path}")
        return pd.read_csv(path)
    if lower.endswith(".parquet"):
        print(f"Reading firm-year file: {path}")
        return pd.read_parquet(path)
    if lower.endswith(".pkl") or lower.endswith(".pickle"):
        print(f"Reading firm-year file: {path}")
        return pd.read_pickle(path)
    if not lower.endswith(".zip"):
        raise ValueError(f"Unsupported Glassdoor firm-year file type: {path}")

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
    gd = load_glassdoor_firm_year(GLASSDOOR_FIRM_YEAR_PATH)
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
    year_col = detect_column(df, ["year", "review_year", "fyear", "calendar_year"])

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

    # Backward compatibility: convert long role-group variable names to short prefixes.
    df = rename_role_group_prefixes(df)

    curr_cols = find_curr_columns(df)
    print(f"Detected _curr columns: {len(curr_cols):,}")

    return df


def build_glassdoor_period_frames(gd: pd.DataFrame) -> dict[str, pd.DataFrame]:
    gd_cols = [c for c in gd.columns if c not in {"gvkey", "year"}]
    out: dict[str, pd.DataFrame] = {}

    for shift, suffix in GLASSDOOR_MERGE_SHIFTS:
        target_year_col = {
            -1: "gd_year_lag1",
            0: "gd_year",
            1: "gd_year_for1",
        }[shift]

        frame = gd[["gvkey", "year", *gd_cols]].copy()
        frame = frame.rename(columns={"year": target_year_col})

        rename_map = {}
        for col in gd_cols:
            rename_map[col] = f"{col}_{suffix}" if suffix else col
        frame = frame.rename(columns=rename_map)
        out[suffix or "curr"] = frame

    return out


def prepare_controls(controls: pd.DataFrame) -> pd.DataFrame:
    print_banner("Prepare Compustat Controls")
    df = controls.copy()
    gv_col = detect_column(df, ["gvkey"])
    year_col = detect_column(df, ["fyear", "year"])

    # Print all columns so we can see what industry fields are available.
    print(f"Controls file columns ({len(df.columns)}): {list(df.columns)}")
    industry_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in ["sic", "ff", "fama", "industry", "naics", "gind", "ffi"])
    ]
    print(f"Industry-related columns found: {industry_cols}")

    # Rename alternative SIC column names to 'sic'.
    sic_alt_names = ["siccd", "sich", "sic_compustat", "sic_code"]
    if "sic" not in df.columns:
        for alt in sic_alt_names:
            if alt in df.columns:
                df = df.rename(columns={alt: "sic"})
                print(f"Renamed '{alt}' -> 'sic'")
                break

    # Rename alternative ff48 column names to 'ff48'.
    ff48_alt_names = ["ff_48", "ffi48", "fama_french_48", "ff_ind48", "ff48_ind", "ffi_48"]
    if "ff48" not in df.columns:
        for alt in ff48_alt_names:
            if alt in df.columns:
                df = df.rename(columns={alt: "ff48"})
                print(f"Renamed '{alt}' -> 'ff48'")
                break

    has_sic = "sic" in df.columns
    has_sic2 = "sic2" in df.columns
    has_ff48 = "ff48" in df.columns
    print(f"Controls has sic: {has_sic}")
    print(f"Controls has sic2: {has_sic2}")
    print(f"Controls has ff48: {has_ff48}")
    if not has_ff48:
        print("WARNING: ff48 not found in controls; proceeding without ff48 standardization.")

    df[gv_col] = standardize_gvkey(df[gv_col])
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")

    if has_sic:
        df["sic"] = pd.to_numeric(df["sic"], errors="coerce")
    if has_sic2:
        df["sic2"] = pd.to_numeric(df["sic2"], errors="coerce")
    elif has_sic:
        df["sic2"] = np.floor(df["sic"] / 100.0)
        print("Constructed sic2 from sic using floor(sic / 100).")
    else:
        print("WARNING: both sic and sic2 are missing in controls; skip sic2 standardization.")

    if "sic2" in df.columns:
        df["sic2"] = pd.to_numeric(df["sic2"], errors="coerce").astype("Int64")
    if has_ff48:
        df["ff48"] = pd.to_numeric(df["ff48"], errors="coerce").astype("Int64")
    df = df[df[gv_col].notna() & df[year_col].notna()].copy()

    keep_cols = [
        c
        for c in [
            gv_col,
            year_col,
            "conm",
            "tic",
            "cik",
            "sic",
            "sic2",
            "ff48",
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
    print_banner("Merge Union with Glassdoor (t-1, t, t+1)")

    merged = union_fy.copy()
    merged["gd_year_lag1"] = merged["election_year"] - 1
    merged["gd_year"] = merged["election_year"]
    merged["gd_year_for1"] = merged["election_year"] + 1

    gd_frames = build_glassdoor_period_frames(gd)

    merged = merged.merge(
        gd_frames["lag1"],
        left_on=["gvkey", "gd_year_lag1"],
        right_on=["gvkey", "gd_year_lag1"],
        how="left",
        indicator="merge_union_glassdoor_lag1",
    )

    merged = merged.merge(
        gd_frames["curr"],
        left_on=["gvkey", "gd_year"],
        right_on=["gvkey", "gd_year"],
        how="left",
        indicator="merge_union_glassdoor_curr",
    )

    merged = merged.merge(
        gd_frames["for1"],
        left_on=["gvkey", "gd_year_for1"],
        right_on=["gvkey", "gd_year_for1"],
        how="left",
        indicator="merge_union_glassdoor_for1",
    )

    merged["control_year"] = merged["election_year"]
    merged = merged.merge(
        controls,
        left_on=["gvkey", "control_year"],
        right_on=["gvkey", "year"],
        how="left",
        indicator="merge_controls",
        suffixes=("", "_ctrl"),
    )

    key = [c for c in ["gvkey", "election_year", "election_id", "case_number"] if c in merged.columns]
    if key:
        merged = merged.drop_duplicates(key, keep="first").copy()

    return merged


def summarize_final(df: pd.DataFrame) -> None:
    print_banner("Final Validation")
    print(f"Final shape: {df.shape}")
    print(f"Unique firms: {df['gvkey'].nunique():,}")
    if "election_year" in df.columns:
        print(f"Election year range: {int(df['election_year'].min())} to {int(df['election_year'].max())}")

    if "win_union" in df.columns:
        print("\nwin_union distribution:")
        print(df["win_union"].value_counts(dropna=False).sort_index())

    if "union_margin" in df.columns:
        print("\nunion_margin summary:")
        print(df["union_margin"].describe(percentiles=[0.01, 0.5, 0.99]))

    dup_key = [c for c in ["gvkey", "election_year", "election_id", "case_number"] if c in df.columns]
    dup_n = int(df.duplicated(dup_key).sum()) if dup_key else 0
    print(f"Duplicate final-key count: {dup_n:,}")

    curr_cols = find_curr_columns(df)
    if curr_cols:
        print(f"_curr columns included in final sample: {len(curr_cols):,}")

    period_outcomes = []
    period_review_controls = []
    for _, suffix in GLASSDOOR_MERGE_SHIFTS:
        for col in MAIN_OUTCOMES:
            name = f"{col}_{suffix}" if suffix else col
            if name in df.columns:
                period_outcomes.append(name)
        for col in REVIEW_VOLUME_CONTROLS:
            name = f"{col}_{suffix}" if suffix else col
            if name in df.columns:
                period_review_controls.append(name)

    major = [
        c
        for c in [
            *period_outcomes,
            *period_review_controls,
            "union_margin",
            "win_union",
            *LAG_CONTROL_VARS,
        ]
        if c in df.columns
    ]
    major += [c for c in curr_cols if c not in major]
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

    for merge_col in [
        "merge_union_glassdoor_lag1",
        "merge_union_glassdoor_curr",
        "merge_union_glassdoor_for1",
    ]:
        if merge_col in df.columns:
            print(f"\n{merge_col} diagnostics:")
            print(df[merge_col].value_counts(dropna=False))
    if "merge_controls" in df.columns:
        print("\nControls merge diagnostics:")
        print(df["merge_controls"].value_counts(dropna=False))


def make_stata_compatible(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    # Stata variable name constraints: <= 32 chars, [a-zA-Z_][a-zA-Z0-9_]*
    rename = {}
    rows = []
    used = set()
    for c in out.columns:
        new = c.lower()
        for old_tok, new_tok in STATA_TOKEN_MAP.items():
            new = new.replace(old_tok, new_tok)
        new = re.sub(r"[^a-z0-9_]", "_", new)
        new = re.sub(r"_+", "_", new).strip("_")
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
        rows.append({"original_name": c, "stata_name": new})
    if rename:
        out = out.rename(columns=rename)

    rename_map_df = pd.DataFrame(rows)
    return out, rename_map_df


def winsorize_for_regression(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()

    # Only winsorize financial / Compustat control variables.
    # Glassdoor rating variables (GD_*), review volume counts, and all other
    # non-financial columns are left unchanged.
    candidate_vars = [
        *RAW_CONTROL_VARS,
        *LAG_CONTROL_VARS,
    ]

    winsor_vars = [
        c
        for c in candidate_vars
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
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

    stata_df, stata_map = make_stata_compatible(df)
    stata_df.to_stata(OUT_DTA, write_index=False, version=118)
    stata_map.to_csv(OUT_STATA_MAP, index=False)
    print(f"Saved Stata: {OUT_DTA}")
    print(f"Saved Stata varname map: {OUT_STATA_MAP}")

    win_df, winsor_vars = winsorize_for_regression(df)
    win_df.to_parquet(OUT_WIN_PARQUET, index=False)
    print(f"Saved winsorized parquet: {OUT_WIN_PARQUET}")

    win_stata, win_stata_map = make_stata_compatible(win_df)
    win_stata.to_stata(OUT_WIN_DTA, write_index=False, version=118)
    win_stata_map.to_csv(OUT_WIN_STATA_MAP, index=False)
    print(f"Saved winsorized Stata: {OUT_WIN_DTA}")
    print(f"Saved winsorized Stata varname map: {OUT_WIN_STATA_MAP}")

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
    if "mayu_GD_rating" not in gd_clean.columns:
        raise RuntimeError(
            "Firm-year union input is missing mayu_GD_rating. "
            "Re-run build_firm_year_aggregates.py to create role subgroup rating outcomes with short names."
        )
    controls = prepare_controls(controls_raw)

    merged = merge_outcomes(union_one, gd_clean, controls)

    # Keep one row per election record in final file.
    key = [c for c in ["gvkey", "election_year", "election_id", "case_number"] if c in merged.columns]
    if key:
        merged = merged.drop_duplicates(key, keep="first").copy()

    merged = add_industry_year_standardized_ratings(merged)

    summarize_final(merged)
    export_outputs(merged)

    # --- Final validation: fail loudly if required outputs are missing ---
    errors: list[str] = []
    pq_df = pd.read_parquet(OUT_PARQUET, columns=None)
    if "sic" not in pq_df.columns:
        errors.append("FAIL: 'sic' missing from final parquet. Re-run build_compustat_controls.py to add industry variables.")
    if "sic2" not in pq_df.columns:
        errors.append("FAIL: 'sic2' missing from final parquet.")
    sdsic2_vars = [c for c in pq_df.columns if c.endswith("_sdsic2")]
    if not sdsic2_vars:
        errors.append("FAIL: No _sdsic2 standardized variables created. Check sic/sic2 availability in controls.")
    if "mayu_GD_rating" not in pq_df.columns:
        errors.append("FAIL: mayu_GD_rating missing in final parquet.")
    if not any(c.startswith("mayu_GD") for c in pq_df.columns):
        errors.append("FAIL: no mayu_GD* variables in final parquet.")
    if any(c.startswith("role_likely_unionizable_GD") for c in pq_df.columns):
        errors.append("FAIL: old prefix role_likely_unionizable_GD still present in final parquet.")
    if any(c.startswith("role_likely_excluded_from_union_GD") for c in pq_df.columns):
        errors.append("FAIL: old prefix role_likely_excluded_from_union_GD still present in final parquet.")
    if any(c.startswith("role_ambiguous_union_status_GD") for c in pq_df.columns):
        errors.append("FAIL: old prefix role_ambiguous_union_status_GD still present in final parquet.")

    if not OUT_STATA_MAP.exists() or not OUT_WIN_STATA_MAP.exists():
        errors.append("FAIL: Stata variable name mapping CSV files were not generated.")
    else:
        map_df = pd.read_csv(OUT_STATA_MAP)
        bad_patterns = [
            r"role_ambiguous_union_status_g_",
            r"role_likely_excluded_from_uni_",
            r"role_likely_unionizable_gd_",
        ]
        for pat in bad_patterns:
            bad = map_df[map_df["stata_name"].str.contains(pat, regex=True, na=False)]
            if len(bad) > 0:
                errors.append(f"FAIL: bad unreadable Stata names still present for pattern {pat}.")
    if errors:
        print("\n" + "=" * 88)
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  {e}")
        print("=" * 88)
        raise RuntimeError("Final regression dataset is missing required variables. See validation errors above.")
    else:
        print("\nValidation passed: sic, sic2, and _sdsic2 variables are all present.")


if __name__ == "__main__":
    main()
