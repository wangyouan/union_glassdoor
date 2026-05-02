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
    "sich",
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

# Fama-French 48 industry boundaries (lower SIC code for each group).
# Each tuple is (ff48_code, sic_lo, sic_hi).  Codes match Ken French's groupings.
_FF48_SIC_RANGES: list[tuple[int, int, int]] = [
    (1, 100, 199), (1, 200, 299), (1, 700, 799), (1, 910, 919), (1, 2048, 2048),
    (2, 2000, 2009), (2, 2010, 2019), (2, 2020, 2029), (2, 2030, 2039), (2, 2040, 2046),
    (2, 2050, 2059), (2, 2060, 2063), (2, 2070, 2079), (2, 2090, 2092), (2, 2095, 2095),
    (2, 2098, 2099),
    (3, 2064, 2068), (3, 2086, 2086), (3, 2087, 2087), (3, 2096, 2096), (3, 2097, 2097),
    (4, 2080, 2080), (4, 2082, 2082), (4, 2083, 2083), (4, 2084, 2084), (4, 2085, 2085),
    (5, 2100, 2199),
    (6, 920, 999), (6, 3650, 3651), (6, 3652, 3652), (6, 3732, 3732), (6, 3930, 3931),
    (6, 3940, 3949),
    (7, 7800, 7819), (7, 7820, 7829),
    (8, 7830, 7833),
    (9, 7840, 7841),
    (10, 7900, 7900), (10, 7911, 7911), (10, 7920, 7929), (10, 7933, 7933),
    (10, 7940, 7949), (10, 7980, 7980), (10, 7990, 7999),
    (11, 7010, 7019), (11, 7040, 7049), (11, 7213, 7213),
    (12, 5900, 5999), (12, 7200, 7200), (12, 7201, 7201), (12, 7202, 7202),
    (12, 7203, 7203), (12, 7204, 7204), (12, 7205, 7205), (12, 7206, 7206),
    (12, 7207, 7207), (12, 7208, 7208), (12, 7209, 7209), (12, 7210, 7210),
    (12, 7211, 7211), (12, 7214, 7214), (12, 7215, 7215), (12, 7216, 7216),
    (12, 7217, 7217), (12, 7218, 7218), (12, 7219, 7219), (12, 7220, 7220),
    (12, 7221, 7221), (12, 7230, 7231), (12, 7240, 7241), (12, 7250, 7251),
    (12, 7260, 7269), (12, 7270, 7290), (12, 7291, 7291), (12, 7292, 7299),
    (12, 7300, 7300), (12, 7389, 7389), (12, 7395, 7395), (12, 7500, 7500),
    (12, 7520, 7529), (12, 7530, 7539), (12, 7540, 7549), (12, 7600, 7600),
    (12, 7620, 7620), (12, 7622, 7622), (12, 7623, 7623), (12, 7629, 7629),
    (12, 7630, 7631), (12, 7640, 7641), (12, 7690, 7699), (12, 8100, 8199),
    (12, 8200, 8299), (12, 8300, 8399), (12, 8400, 8499), (12, 8600, 8699),
    (12, 8800, 8899), (12, 7374, 7374),
    (13, 7372, 7372), (13, 7371, 7371), (13, 7373, 7373),
    (14, 7374, 7374),
    (15, 8000, 8099),
    (16, 4800, 4899),
    (17, 4900, 4949), (17, 4950, 4959), (17, 4960, 4969), (17, 4970, 4979),
    (18, 5000, 5099), (18, 5110, 5113), (18, 5120, 5122), (18, 5130, 5139),
    (18, 5140, 5149), (18, 5150, 5159), (18, 5160, 5169), (18, 5170, 5172),
    (18, 5180, 5182), (18, 5190, 5199),
    (19, 5200, 5299),
    (20, 5300, 5399), (20, 5945, 5945), (20, 5960, 5969), (20, 5970, 5979),
    (20, 5990, 5990),
    (21, 5400, 5499),
    (22, 5500, 5599),
    (23, 5600, 5699),
    (24, 5700, 5736), (24, 5750, 5799),
    (25, 5800, 5819),
    (26, 5820, 5829),
    (27, 5900, 5900), (27, 5912, 5912), (27, 5940, 5940), (27, 5941, 5941),
    (27, 5942, 5942), (27, 5943, 5943), (27, 5944, 5944), (27, 5946, 5946),
    (27, 5947, 5947), (27, 5948, 5948), (27, 5949, 5949), (27, 5950, 5959),
    (27, 5980, 5989), (27, 5991, 5999),
    (28, 3860, 3861), (28, 3870, 3879),
    (29, 3840, 3849), (29, 3850, 3851),
    (30, 3826, 3826), (30, 3827, 3827), (30, 3829, 3829),
    (31, 3820, 3820), (31, 3821, 3821), (31, 3822, 3822), (31, 3823, 3823),
    (31, 3824, 3824), (31, 3825, 3825), (31, 3828, 3828),
    (32, 3559, 3559), (32, 3562, 3562), (32, 3563, 3563), (32, 3564, 3564),
    (32, 3567, 3567), (32, 3590, 3590), (32, 3599, 3599),
    (33, 3443, 3443), (33, 3460, 3469), (33, 3490, 3499),
    (34, 3310, 3317), (34, 3320, 3325), (34, 3330, 3339), (34, 3340, 3341),
    (34, 3350, 3357), (34, 3360, 3369), (34, 3390, 3399),
    (35, 1040, 1049),
    (36, 1000, 1009), (36, 1010, 1019), (36, 1020, 1029), (36, 1030, 1039),
    (36, 1050, 1059), (36, 1060, 1069), (36, 1070, 1079), (36, 1080, 1089),
    (36, 1090, 1099), (36, 1100, 1119), (36, 1400, 1499),
    (37, 1500, 1511), (37, 1520, 1531), (37, 1540, 1549), (37, 1600, 1699),
    (37, 1700, 1799),
    (38, 1300, 1300), (38, 1310, 1319), (38, 1320, 1329), (38, 1380, 1389),
    (38, 1390, 1399), (38, 2900, 2911), (38, 2990, 2999),
    (39, 2910, 2919), (39, 2950, 2959), (39, 2992, 2992), (39, 2999, 2999),
    (40, 2910, 2910), (40, 2911, 2911), (40, 2990, 2990),
    (41, 2800, 2809), (41, 2810, 2819), (41, 2820, 2823), (41, 2824, 2824),
    (41, 2825, 2829), (41, 2860, 2869), (41, 2870, 2879), (41, 2890, 2891),
    (41, 2892, 2899),
    (42, 2830, 2830), (42, 2831, 2831), (42, 2833, 2836),
    (43, 2840, 2843), (43, 2844, 2844),
    (44, 2200, 2269), (44, 2270, 2279), (44, 2280, 2284), (44, 2290, 2295),
    (44, 2297, 2297), (44, 2298, 2298), (44, 2299, 2299), (44, 2393, 2395),
    (44, 2397, 2399),
    (45, 2300, 2390), (45, 3020, 3021), (45, 3100, 3111), (45, 3130, 3131),
    (45, 3140, 3149), (45, 3150, 3151), (45, 3963, 3965),
    (46, 2400, 2439), (46, 2450, 2459), (46, 2490, 2499), (46, 2660, 2661),
    (46, 2950, 2952), (46, 2990, 2991), (46, 3200, 3200),
    (46, 3210, 3211), (46, 3240, 3241), (46, 3250, 3259), (46, 3261, 3261),
    (46, 3264, 3264), (46, 3270, 3275), (46, 3280, 3281), (46, 3290, 3293),
    (46, 3295, 3299), (46, 3420, 3442), (46, 3446, 3446), (46, 3448, 3452),
    (46, 3490, 3499), (46, 3559, 3559), (46, 3760, 3769), (46, 3842, 3842),
    (46, 3990, 3999),
    (47, 2500, 2519), (47, 2590, 2599),
    (48, 3630, 3639), (48, 3640, 3649), (48, 3660, 3660), (48, 3690, 3699),
    (48, 3714, 3714), (48, 3716, 3716), (48, 3750, 3751), (48, 3792, 3792),
    (48, 3900, 3900), (48, 3910, 3911), (48, 3914, 3914), (48, 3915, 3915),
    (48, 3960, 3962), (48, 3991, 3991), (48, 3993, 3993),
]


def sic_to_ff48(sic_series: pd.Series) -> pd.Series:
    """Map a numeric SIC series to Fama-French 48 industry codes."""
    sic = pd.to_numeric(sic_series, errors="coerce")
    ff48 = pd.Series(pd.NA, index=sic.index, dtype="Int64")
    for ff_code, lo, hi in _FF48_SIC_RANGES:
        mask = sic.between(lo, hi, inclusive="both")
        ff48 = ff48.where(~mask, other=ff_code)
    return ff48


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

    # SIC code: standardize sich → sic, construct sic2, map to ff48
    if "sich" in out.columns:
        out["sic"] = pd.to_numeric(out["sich"], errors="coerce").astype("Int64")
    elif "sic" in out.columns:
        out["sic"] = pd.to_numeric(out["sic"], errors="coerce").astype("Int64")
    if "sic" in out.columns:
        out["sic2"] = (out["sic"] // 100).astype("Int64")
        out["ff48"] = sic_to_ff48(out["sic"])
    else:
        print("WARNING: sich/sic not found in funda; sic/sic2/ff48 will not be available.")

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
            "sich",
            "sic",
            "sic2",
            "ff48",
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
