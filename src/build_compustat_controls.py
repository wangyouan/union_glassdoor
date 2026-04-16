from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import wrds


OUTPUT_PATH = Path(
    "/data/disk4/workspace/projects/union_glassdoor/outputs/compustat_firm_controls.parquet"
)

FUNDA_FILTERS = {
    "indfmt": "INDL",
    "datafmt": "STD",
    "consol": "C",
    "popsrc": "D",
}

CORE_FUNDA_VARS = [
    "gvkey",
    "datadate",
    "fyear",
    "sale",
    "at",
    "ceq",
    "seq",
    "lt",
    "dltt",
    "dlc",
    "che",
    "capx",
    "xrd",
    "ni",
    "oibdp",
    "ppent",
    "emp",
    "csho",
    "prcc_f",
    "txditc",
    "pstkrv",
    "pstkl",
    "pstk",
]

NAMES_VARS = ["gvkey", "conm", "tic", "cik"]

BASE_CONTROL_VARS = [
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


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def log_positive(x: pd.Series) -> pd.Series:
    x_num = pd.to_numeric(x, errors="coerce").astype("float64")
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    mask = x_num > 0
    out.loc[mask] = np.log(x_num.loc[mask])
    return out


def get_table_columns(conn: wrds.Connection, schema: str, table: str) -> List[str]:
    q = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = '{table}'
        ORDER BY ordinal_position
    """
    cols = conn.raw_sql(q)["column_name"].tolist()
    return cols


def fetch_funda(conn: wrds.Connection, filters: Dict[str, str]) -> pd.DataFrame:
    funda_cols = get_table_columns(conn, "comp", "funda")
    selected = [c for c in CORE_FUNDA_VARS if c in funda_cols]
    required = {"gvkey", "datadate", "fyear", "at"}
    missing_required = sorted(required - set(selected))
    if missing_required:
        raise ValueError(f"Missing required funda columns: {missing_required}")

    where_parts = []
    for k, v in filters.items():
        if k in funda_cols:
            where_parts.append(f"{k} = '{v}'")
    where_sql = " AND ".join(where_parts) if where_parts else "1=1"

    query = f"""
        SELECT {', '.join(selected)}
        FROM comp.funda
        WHERE {where_sql}
          AND gvkey IS NOT NULL
          AND datadate IS NOT NULL
          AND fyear IS NOT NULL
    """
    df = conn.raw_sql(query, date_cols=["datadate"])
    print(f"Fetched comp.funda rows: {len(df):,}")
    return df


def fetch_names(conn: wrds.Connection) -> pd.DataFrame:
    names_cols = get_table_columns(conn, "comp", "names")
    selected = [c for c in NAMES_VARS if c in names_cols]
    if "gvkey" not in selected:
        raise ValueError("comp.names does not contain gvkey")

    query = f"""
        SELECT {', '.join(selected)}
        FROM comp.names
        WHERE gvkey IS NOT NULL
    """
    names = conn.raw_sql(query)
    names = names.sort_values(by=[c for c in ["gvkey", "conm", "tic", "cik"] if c in names.columns])
    names = names.drop_duplicates(subset=["gvkey"], keep="first")
    print(f"Fetched comp.names rows (dedup to gvkey): {len(names):,}")
    return names


def resolve_gvkey_fyear_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dup_count = int(df.duplicated(subset=["gvkey", "fyear"]).sum())
    print(f"Initial duplicate gvkey-fyear rows: {dup_count:,}")
    if dup_count == 0:
        return df

    # Keep the latest datadate within gvkey-fyear; if ties remain, keep first.
    out = (
        df.sort_values(["gvkey", "fyear", "datadate"])
        .drop_duplicates(subset=["gvkey", "fyear"], keep="last")
        .copy()
    )
    final_dup = int(out.duplicated(subset=["gvkey", "fyear"]).sum())
    print(f"Duplicate gvkey-fyear rows after resolution: {final_dup:,}")
    return out


def construct_controls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "sale",
        "at",
        "ceq",
        "seq",
        "lt",
        "dltt",
        "dlc",
        "che",
        "capx",
        "xrd",
        "ni",
        "oibdp",
        "ppent",
        "emp",
        "csho",
        "prcc_f",
        "txditc",
        "pstkrv",
        "pstkl",
        "pstk",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    at = out.get("at", pd.Series(np.nan, index=out.index))
    dltt = out.get("dltt", pd.Series(np.nan, index=out.index)).fillna(0)
    dlc = out.get("dlc", pd.Series(np.nan, index=out.index)).fillna(0)

    out["market_equity"] = out.get("prcc_f", np.nan) * out.get("csho", np.nan)

    pref_stock = (
        out.get("pstkrv", np.nan)
        .combine_first(out.get("pstkl", np.nan))
        .combine_first(out.get("pstk", np.nan))
        .fillna(0)
    )
    seq = out.get("seq", pd.Series(np.nan, index=out.index))
    ceq = out.get("ceq", pd.Series(np.nan, index=out.index))
    txditc = out.get("txditc", pd.Series(np.nan, index=out.index)).fillna(0)

    book_equity = seq.combine_first(ceq) + txditc - pref_stock
    out["book_equity"] = book_equity

    out["size"] = log_positive(at)
    out["log_me"] = log_positive(out["market_equity"])
    out["leverage"] = safe_ratio(dltt + dlc, at)
    out["cash_ratio"] = safe_ratio(out.get("che", np.nan), at)
    out["roa"] = safe_ratio(out.get("ni", np.nan), at)
    out["profitability"] = safe_ratio(out.get("oibdp", np.nan), at)
    out["tangibility"] = safe_ratio(out.get("ppent", np.nan), at)
    out["capx_at"] = safe_ratio(out.get("capx", np.nan), at)
    out["rd_at"] = safe_ratio(out.get("xrd", np.nan), at)
    out["book_to_market"] = safe_ratio(out["book_equity"], out["market_equity"])
    out["log_emp"] = log_positive(out.get("emp", np.nan))

    out = out.sort_values(["gvkey", "datadate"]).copy()
    out["lag_sale"] = out.groupby("gvkey", dropna=False)["sale"].shift(1)
    out["sales_growth"] = safe_ratio(out["sale"], out["lag_sale"]) - 1

    for var in BASE_CONTROL_VARS:
        out[f"L_{var}"] = out.groupby("gvkey", dropna=False)[var].shift(1)

    return out


def validation_report(df: pd.DataFrame, final_vars: Iterable[str]) -> None:
    print_banner("Validation")
    print(f"Shape: {df.shape}")
    if "fyear" in df.columns:
        print(f"Year range: {int(df['fyear'].min())} to {int(df['fyear'].max())}")
    print(f"Unique gvkeys: {df['gvkey'].nunique():,}")
    print(f"Duplicate gvkey-fyear count: {int(df.duplicated(['gvkey', 'fyear']).sum()):,}")

    existing_final = [v for v in final_vars if v in df.columns]
    if existing_final:
        print("\nSummary stats for constructed controls:")
        print(df[existing_final].describe(percentiles=[0.01, 0.5, 0.99]).T.round(4))

        miss = (
            df[existing_final]
            .isna()
            .mean()
            .rename("missing_share")
            .sort_values(ascending=False)
            .to_frame()
        )
        print("\nMissingness report (share missing):")
        print(miss.round(4))


def main() -> None:
    print_banner("Build Compustat Firm Controls")
    conn = wrds.Connection(wrds_username='wangyouan')

    try:
        funda = fetch_funda(conn, FUNDA_FILTERS)
        names = fetch_names(conn)
    finally:
        conn.close()

    print_banner("Merge and Resolve Duplicates")
    df = funda.merge(names, on="gvkey", how="left", validate="m:1")
    df["gvkey"] = df["gvkey"].astype(str).str.strip()
    df["fyear"] = pd.to_numeric(df["fyear"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["gvkey", "fyear", "datadate"]).copy()
    df = resolve_gvkey_fyear_duplicates(df)

    print_banner("Construct Controls")
    df = construct_controls(df)

    keep_cols = [
        c
        for c in [
            "gvkey",
            "datadate",
            "fyear",
            "conm",
            "tic",
            "cik",
            "sale",
            "at",
            "ceq",
            "seq",
            "lt",
            "dltt",
            "dlc",
            "che",
            "capx",
            "xrd",
            "ni",
            "oibdp",
            "ppent",
            "emp",
            "csho",
            "prcc_f",
            "market_equity",
            "book_equity",
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
        if c in df.columns
    ]

    lag_cols = [f"L_{v}" for v in BASE_CONTROL_VARS if f"L_{v}" in df.columns]
    out = df[keep_cols + lag_cols].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")

    validation_report(out, BASE_CONTROL_VARS + lag_cols)


if __name__ == "__main__":
    main()
