from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "/data/disk4/workspace/projects/glassdoor/outputs/job_title_standardized_universe.csv"
)

OUT_DIR = Path("/data/disk4/workspace/projects/union_glassdoor/outputs")
OUT_CLASSIFIED = OUT_DIR / "union_classified_title_universe.csv"
OUT_DIAG = OUT_DIR / "union_title_classification_diagnostics.json"
OUT_PROTOCOL = OUT_DIR / "union_title_classification_protocol.md"
OUT_AMBIG = OUT_DIR / "union_ambiguous_title_examples.csv"
OUT_LOW_INFO = OUT_DIR / "union_low_information_title_examples.csv"
OUT_TOP = OUT_DIR / "union_top_titles_by_reviews.csv"


LOW_INFO_TOKENS = {
    "",
    "anonymous",
    "employee",
    "unemployed",
    "none",
    "na",
    "n a",
    "n/a",
    "unknown",
    "test",
    "other",
    "non",
    "dy",
    "spring",
    "material",
    "student",
}

AMBIGUOUS_STANDALONE = {
    "analyst",
    "associate",
    "consultant",
    "specialist",
    "support",
    "lead",
    "advisor",
    "agent",
    "employee",
    "worker",
    "anonymous",
    "unemployed",
    "student",
    "intern",
    "assistant",
}

# Retail / Store Workers
RETAIL_UNIONIZABLE_KWS = [
    "retail sales associate",
    "sales floor associate",
    "stock associate",
    "shelf stocker",
    "store associate",
    "store clerk",
    "grocery clerk",
    "warehouse associate",
    "fulfillment associate",
    "picker",
    "packer",
    "cashier",
    "sales associate",  # EXPLICIT: common retail title
]

# Food Service
FOOD_SERVICE_UNIONIZABLE_KWS = [
    "barista",
    "crew member",
    "sandwich artist",
    "line cook",
    "prep cook",
    "dishwasher",
    "server",
    "waiter",
    "waitress",
    "food service worker",
    "busser",
    "kitchen staff",
]

# Logistics / Transportation
LOGISTICS_UNIONIZABLE_KWS = [
    "package handler",
    "delivery driver",
    "warehouse worker",
    "dock worker",
    "loader",
    "forklift operator",
    "material handler",
    "driver",
]

# Manufacturing
MANUFACTURING_UNIONIZABLE_KWS = [
    "machinist",
    "welder",
    "assembler",
    "technician",
    "mechanic",
    "operator",
    "laborer",
    "production worker",
    "maintenance worker",
    "field technician",
    "installer",
    "maintenance technician",
]

# Healthcare Support
HEALTHCARE_SUPPORT_UNIONIZABLE_KWS = [
    "nursing assistant",
    "medical assistant",
    "caregiver",
    "home health aide",
    "patient care technician",
    "nurse aide",
]

# Aviation / Hospitality
AVIATION_HOSPITALITY_UNIONIZABLE_KWS = [
    "flight attendant",
    "housekeeper",
    "room attendant",
    "front desk",
]

# Banking
BANKING_UNIONIZABLE_KWS = [
    "bank teller",
]

# Frontline / Generalist
FRONTLINE_UNIONIZABLE_KWS = [
    "customer service",
    "customer support",
    "call center",
    "contact center",
    "service representative",
    "clerk",
    "receptionist",
    "security guard",
    "janitor",
    "cleaner",
    "hourly",
    "line worker",
    "logistics",
    "fulfillment",
    "warehouse",
    "delivery",
    "maintenance",
    "production",
    "manufacturing",
]

# Executive / C-Suite
EXECUTIVE_EXCLUDED_KWS = [
    "ceo",
    "cfo",
    "coo",
    "cto",
    "cio",
    "chief",
    "president",
    "vice president",
    "vp",
    "chief of staff",
]

# Managerial
MANAGERIAL_EXCLUDED_KWS = [
    "managing director",
    "general manager",
    "store manager",
    "regional manager",
    "district manager",
    "operations manager",
    "director",
    "manager",
    "superintendent",
]

# Supervisory / Leadership
SUPERVISORY_EXCLUDED_KWS = [
    "supervisor",
    "foreman",
    "lead supervisor",
    "head of",
    "principal",
]

# Legal
LEGAL_EXCLUDED_KWS = [
    "attorney",
    "lawyer",
    "legal",
    "counsel",
    "general counsel",
    "corporate counsel",
    "litigation",
    "paralegal",
    "law clerk",
    "legal assistant",
]

# HR / Employee Relations / Labor Relations
HR_EXCLUDED_KWS = [
    "human resources",
    "hr",
    "people operations",
    "people partner",
    "hrbp",
    "recruiter",
    "recruiting",
    "talent acquisition",
    "employee relations",
    "labor relations",
    "industrial relations",
    "compensation",
    "benefits manager",
    "payroll manager",
]

# Strategy / Corporate
STRATEGY_EXCLUDED_KWS = [
    "strategy",
    "corporate development",
    "business strategy",
    "management consultant",
    "internal consultant",
    "transformation",
    "strategic initiatives",
    "corporate planning",
]

# Owner / Self-Employed
OWNER_EXCLUDED_KWS = [
    "founder",
    "co founder",
    "owner",
    "partner",
    "managing partner",
    "independent contractor",
    "contractor",
    "freelancer",
    "self employed",
]

# Multilingual Keywords (Spanish)
SPANISH_UNIONIZABLE_KWS = [
    "vendedor",  # sales associate
    "cajero",    # cashier
    "operador",  # operator
    "tecnico",   # technician
    "conductor", # driver
    "cocinero",  # cook
    "mozo",      # laborer
    "ayudante",  # helper
    "empleado",  # employee (frontline)
]

# Multilingual Keywords (Portuguese)
PORTUGUESE_UNIONIZABLE_KWS = [
    "atendente",    # attendant
    "desenvolvedor", # developer
    "engenheiro",   # engineer
    "operador",     # operator
    "motorista",    # driver
    "cozinheiro",   # cook
    "auxiliar",     # assistant
    "tecnico",      # technician
]

# Multilingual Keywords (French)
FRENCH_UNIONIZABLE_KWS = [
    "caissier",  # cashier
    "caissiere", # cashier (female)
    "technicien", # technician
    "operateur", # operator
    "vendeur",   # sales
]

UNIONIZABLE_KEYWORDS = (
    RETAIL_UNIONIZABLE_KWS +
    FOOD_SERVICE_UNIONIZABLE_KWS +
    LOGISTICS_UNIONIZABLE_KWS +
    MANUFACTURING_UNIONIZABLE_KWS +
    HEALTHCARE_SUPPORT_UNIONIZABLE_KWS +
    AVIATION_HOSPITALITY_UNIONIZABLE_KWS +
    BANKING_UNIONIZABLE_KWS +
    FRONTLINE_UNIONIZABLE_KWS +
    SPANISH_UNIONIZABLE_KWS +
    PORTUGUESE_UNIONIZABLE_KWS +
    FRENCH_UNIONIZABLE_KWS
)

EXCLUDED_KEYWORDS = [
    "manager",
    "director",
    "executive",
    "vice president",
    "vp",
    "president",
    "chief",
    "ceo",
    "cfo",
    "coo",
    "cto",
    "cio",
    "head of",
    "managing director",
    "general manager",
    "store manager",
    "operations manager",
    "supervisor",
    "team lead",
    "shift lead",
    "technical lead",
    "foreman",
    "principal",
    "senior leadership",
    "attorney",
    "lawyer",
    "legal",
    "counsel",
    "general counsel",
    "corporate counsel",
    "litigation",
    "paralegal",
    "law clerk",
    "legal assistant",
    "human resources",
    "hr",
    "people operations",
    "people partner",
    "hrbp",
    "recruiter",
    "recruiting",
    "talent acquisition",
    "employee relations",
    "labor relations",
    "industrial relations",
    "compensation",
    "benefits manager",
    "payroll manager",
    "strategy",
    "corporate development",
    "business strategy",
    "management consultant",
    "internal consultant",
    "chief of staff",
    "transformation",
    "strategic initiatives",
    "corporate planning",
    "founder",
    "co founder",
    "owner",
    "partner",
    "managing partner",
    "independent contractor",
    "contractor",
    "freelancer",
    "self employed",
]

EXCLUDED_STRONG = [
    "attorney",
    "lawyer",
    "legal",
    "counsel",
    "human resources",
    "labor relations",
    "employee relations",
    "strategy",
    "corporate development",
    "founder",
    "owner",
    "partner",
    "independent contractor",
    "contractor",
    "freelancer",
    "self employed",
]

HIGH_LEVEL_PROFESSIONAL_KWS = [
    "software engineer",
    "software developer",
    "developer",
    "engineer",
    "architect",
    "data scientist",
    "machine learning engineer",
    "product manager",
    "product owner",
    "portfolio manager",
    "trader",
    "quant",
    "quantitative",
    "financial analyst",
    "investment banker",
]

OC_MANAGEMENT_KWS = [
    "manager",
    "director",
    "vice president",
    "vp",
    "president",
    "chief",
    "head of",
    "supervisor",
    "team lead",
    "shift lead",
    "operations manager",
    "project manager",
    "program manager",
    "product manager",
]

# Technical / Engineering
OC_TECH_KWS = [
    "software engineer",
    "senior software engineer",
    "software developer",
    "senior software developer",
    "developer",
    "programmer",
    "engineer",
    "devops",
    "data scientist",
    "senior data scientist",
    "data engineer",
    "machine learning engineer",
    "systems engineer",
    "solutions architect",
    "systems analyst",
    "cybersecurity",
    "information security",
    "network engineer",
    "database administrator",
    "it specialist",
    "information technology",
    "technical support engineer",
    "systems administrator",
]

# Analytics / Business Intelligence
OC_ANALYTICS_KWS = [
    "business analyst",
    "data analyst",
    "financial analyst",
    "quantitative analyst",
    "research analyst",
    "business intelligence",
    "analytics",
]

# Product / Design
OC_PRODUCT_DESIGN_KWS = [
    "product designer",
    "ux designer",
    "ui designer",
    "creative director",
    "design lead",
    "product manager",
    "product owner",
]

# Research
OC_RESEARCH_KWS = [
    "scientist",
    "researcher",
    "economist",
    "statistician",
    "research scientist",
    "principal scientist",
    "senior researcher",
]

OC_TECH_KWS = OC_TECH_KWS + OC_ANALYTICS_KWS + OC_RESEARCH_KWS

OC_AMBIGUOUS_KWS = {
    "analyst",
    "associate",
    "consultant",
    "specialist",
    "assistant",
    "support",
}

def normalize_title_text(value: object) -> str:
    """
    Normalize title text for keyword matching.
    - Lowercase
    - Remove accents and diacritics
    - Convert special characters to spaces
    - Collapse repeated spaces
    """
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip().lower()
    # Decompose accents and diacritics
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # Replace common special characters with spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _phrase_to_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def matched_keywords(text: str, keywords: Iterable[str]) -> List[str]:
    hits: List[str] = []
    for kw in keywords:
        if _phrase_to_pattern(kw).search(text):
            hits.append(kw)
    return hits


def infer_primary_title_column(df: pd.DataFrame) -> str:
    candidates = [
        "title_standardized",
        "title_for_classification",
        "title_original",
        "job_title_clean",
        "job_title_raw",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("Could not infer a standardized title column.")


def is_low_information_title(title: str, row: pd.Series) -> tuple[bool, str]:
    if not title:
        return True, "empty_title"

    if "low_information_title" in row.index and pd.notna(row["low_information_title"]):
        if int(row["low_information_title"]) == 1:
            reason = str(row.get("low_information_reason", "low_information_flagged")).strip()
            return True, reason or "low_information_flagged"

    tokens = title.split()
    if title in LOW_INFO_TOKENS:
        return True, "known_low_information_token"
    if len(tokens) == 1 and (tokens[0] in LOW_INFO_TOKENS or len(tokens[0]) <= 2):
        return True, "single_token_low_information"

    if "is_suspicious" in row.index and pd.notna(row["is_suspicious"]):
        if int(row["is_suspicious"]) == 1:
            reason = str(row.get("suspicious_reason", "suspicious_flagged")).strip()
            return True, reason or "suspicious_flagged"

    return False, ""


def classify_union_dimension(title: str, row: pd.Series) -> dict[str, object]:
    low_info, low_reason = is_low_information_title(title, row)
    if low_info:
        return {
            "union_likely_unionizable": 0,
            "union_likely_excluded": 0,
            "union_ambiguous": 1,
            "union_classification": "ambiguous",
            "union_confidence": "low",
            "union_reason": f"low_information:{low_reason}",
        }

    union_hits = matched_keywords(title, UNIONIZABLE_KEYWORDS)
    multilingual_union = matched_keywords(title, SPANISH_UNIONIZABLE_KWS + PORTUGUESE_UNIONIZABLE_KWS + FRENCH_UNIONIZABLE_KWS)
    if multilingual_union:
        union_hits.extend(multilingual_union)

    excluded_hits = matched_keywords(title, EXCLUDED_KEYWORDS)
    excluded_strong_hits = matched_keywords(title, EXCLUDED_STRONG)

    # Strong excluded signals
    if excluded_strong_hits:
        return {
            "union_likely_unionizable": 0,
            "union_likely_excluded": 1,
            "union_ambiguous": 0,
            "union_classification": "likely_excluded",
            "union_confidence": "high",
            "union_reason": f"excluded_strong:{'|'.join(sorted(set(excluded_strong_hits)))}",
        }

    n_u = len(set(union_hits))
    n_e = len(set(excluded_hits))

    # Conflict resolution
    if n_u > 0 and n_e > 0:
        if n_e >= n_u + 2:
            return {
                "union_likely_unionizable": 0,
                "union_likely_excluded": 1,
                "union_ambiguous": 0,
                "union_classification": "likely_excluded",
                "union_confidence": "medium",
                "union_reason": f"excluded_dominates:{'|'.join(sorted(set(excluded_hits)))}",
            }
        if n_u >= n_e + 2:
            return {
                "union_likely_unionizable": 1,
                "union_likely_excluded": 0,
                "union_ambiguous": 0,
                "union_classification": "likely_unionizable",
                "union_confidence": "medium",
                "union_reason": f"unionizable_dominates:{'|'.join(sorted(set(union_hits)))}",
            }
        return {
            "union_likely_unionizable": 0,
            "union_likely_excluded": 0,
            "union_ambiguous": 1,
            "union_classification": "ambiguous",
            "union_confidence": "low",
            "union_reason": (
                "conflicting_signals:"
                f"u={','.join(sorted(set(union_hits)))};e={','.join(sorted(set(excluded_hits)))}"
            ),
        }

    # Clear excluded signal
    if n_e > 0:
        conf = "high"
        reason = f"excluded_keywords:{'|'.join(sorted(set(excluded_hits)))}"
        # Generic managerial titles get medium confidence
        if title in ["manager", "director", "supervisor"]:
            conf = "medium"
            reason = f"excluded_generic_managerial:{title}"
        return {
            "union_likely_unionizable": 0,
            "union_likely_excluded": 1,
            "union_ambiguous": 0,
            "union_classification": "likely_excluded",
            "union_confidence": conf,
            "union_reason": reason,
        }

    # Clear unionizable signal
    if n_u > 0:
        return {
            "union_likely_unionizable": 1,
            "union_likely_excluded": 0,
            "union_ambiguous": 0,
            "union_classification": "likely_unionizable",
            "union_confidence": "high",
            "union_reason": f"unionizable_keywords:{'|'.join(sorted(set(union_hits)))}",
        }

    # Default: ambiguous
    return {
        "union_likely_unionizable": 0,
        "union_likely_excluded": 0,
        "union_ambiguous": 1,
        "union_classification": "ambiguous",
        "union_confidence": "low",
        "union_reason": "no_clear_union_signal",
    }


def classify_oc_dimension(title: str, row: pd.Series) -> dict[str, object]:
    """
    Classify as Organizational Capital (OC) likely if the role primarily involves:
    1. Management of people / products / operations
    2. Technical/engineering work
    3. Creative/product design
    4. Research/scientific work
    """
    low_info, low_reason = is_low_information_title(title, row)
    if low_info:
        return {
            "oc_likely": 0,
            "oc_management": 0,
            "oc_technical_engineering": 0,
            "oc_creative_product": 0,
            "oc_ambiguous": 1,
            "oc_reason": f"low_information:{low_reason}",
        }

    mgmt_hits = matched_keywords(title, OC_MANAGEMENT_KWS)
    tech_hits = matched_keywords(title, OC_TECH_KWS)
    creative_hits = matched_keywords(title, OC_PRODUCT_DESIGN_KWS)

    non_oc_frontline_hits = matched_keywords(
        title,
        [
            "warehouse",
            "package handler",
            "dock worker",
            "loader",
            "forklift",
            "material handler",
            "driver",
            "delivery",
            "cashier",
            "barista",
            "call center",
            "contact center",
            "cleaner",
            "janitor",
            "housekeeper",
            "room attendant",
            "server",
            "cook",
            "dishwasher",
            "kitchen",
            "food service",
            "receptionist",
            "customer service",
            "customer support",
            "store associate",
            "retail associate",
            "bank teller",
            "flight attendant",
            "front desk",
        ],
    )

    if title in OC_AMBIGUOUS_KWS:
        # But check for strong technical/product/research context
        if any(ctx in title for ctx in ["data", "business", "product", "research", "science"]):
            return {
                "oc_likely": 1,
                "oc_management": 0,
                "oc_technical_engineering": 1,
                "oc_creative_product": 0,
                "oc_ambiguous": 0,
                "oc_reason": "oc_technical_context",
            }
        return {
            "oc_likely": 0,
            "oc_management": 0,
            "oc_technical_engineering": 0,
            "oc_creative_product": 0,
            "oc_ambiguous": 1,
            "oc_reason": f"ambiguous_oc_standalone:{title}",
        }

    oc_management = int(len(mgmt_hits) > 0)
    oc_technical = int(len(tech_hits) > 0)
    oc_creative = int(len(creative_hits) > 0)

    oc_likely = int((oc_management + oc_technical + oc_creative) > 0)
    oc_ambiguous = int(oc_likely == 0)

    if oc_likely == 0 and non_oc_frontline_hits:
        return {
            "oc_likely": 0,
            "oc_management": 0,
            "oc_technical_engineering": 0,
            "oc_creative_product": 0,
            "oc_ambiguous": 0,
            "oc_reason": f"non_oc_frontline:{'|'.join(sorted(set(non_oc_frontline_hits)))}",
        }

    if oc_likely == 1:
        reasons: List[str] = []
        if oc_management:
            reasons.append("management")
        if oc_technical:
            reasons.append("technical_engineering")
        if oc_creative:
            reasons.append("creative_product")
        return {
            "oc_likely": 1,
            "oc_management": oc_management,
            "oc_technical_engineering": oc_technical,
            "oc_creative_product": oc_creative,
            "oc_ambiguous": 0,
            "oc_reason": f"oc_signals:{'|'.join(reasons)}",
        }

    return {
        "oc_likely": 0,
        "oc_management": 0,
        "oc_technical_engineering": 0,
        "oc_creative_product": 0,
        "oc_ambiguous": 1,
        "oc_reason": "no_clear_oc_signal",
    }


def _top_titles(df: pd.DataFrame, mask: pd.Series, n: int = 30) -> list[dict]:
    cols = [
        "title_standardized",
        "n_reviews",
        "n_firms",
        "n_gvkeys",
        "union_classification",
        "union_confidence",
        "union_reason",
        "oc_likely",
        "oc_reason",
    ]
    use_cols = [c for c in cols if c in df.columns]
    tmp = df.loc[mask, use_cols].copy()
    if "n_reviews" in tmp.columns:
        tmp = tmp.sort_values(["n_reviews", "title_standardized"], ascending=[False, True])
    else:
        tmp = tmp.sort_values(["title_standardized"], ascending=[True])
    return tmp.head(n).to_dict(orient="records")


def generate_diagnostics(df: pd.DataFrame) -> dict:
    total_titles = int(len(df))
    has_reviews = "n_reviews" in df.columns
    total_reviews = float(df["n_reviews"].sum()) if has_reviews else None

    class_counts = df["union_classification"].value_counts(dropna=False).to_dict()
    class_shares = {k: (v / total_titles if total_titles else None) for k, v in class_counts.items()}

    weighted = {}
    if has_reviews and total_reviews and total_reviews > 0:
        for cls in ["likely_unionizable", "likely_excluded", "ambiguous"]:
            w = float(df.loc[df["union_classification"] == cls, "n_reviews"].sum())
            weighted[cls] = w / total_reviews

    oc_count = int((df["oc_likely"] == 1).sum())
    oc_share = oc_count / total_titles if total_titles else None
    oc_weighted_share = None
    if has_reviews and total_reviews and total_reviews > 0:
        oc_weighted_share = float(df.loc[df["oc_likely"] == 1, "n_reviews"].sum()) / total_reviews

    low_info_count = int(df["low_information_title"].fillna(0).astype(int).sum()) if "low_information_title" in df.columns else None
    suspicious_count = int(df["is_suspicious"].fillna(0).astype(int).sum()) if "is_suspicious" in df.columns else None

    diagnostics = {
        "total_titles": total_titles,
        "total_reviews_represented": total_reviews,
        "union_classification_counts": class_counts,
        "union_classification_shares": class_shares,
        "union_classification_review_weighted_shares": weighted,
        "oc_likely_count": oc_count,
        "oc_likely_share": oc_share,
        "oc_likely_review_weighted_share": oc_weighted_share,
        "low_information_title_count": low_info_count,
        "is_suspicious_count": suspicious_count,
        "top_30_titles_by_union_classification": {
            cls: _top_titles(df, df["union_classification"] == cls, 30)
            for cls in ["likely_unionizable", "likely_excluded", "ambiguous"]
        },
        "top_30_ambiguous_titles_by_reviews": _top_titles(df, df["union_classification"] == "ambiguous", 30),
        "top_30_union_excluded_and_oc_likely": _top_titles(
            df,
            (df["union_likely_excluded"] == 1) & (df["oc_likely"] == 1),
            30,
        ),
    }
    return diagnostics


def build_protocol_markdown(diagnostics: dict) -> str:
    c = diagnostics["union_classification_counts"]
    s = diagnostics["union_classification_shares"]
    oc_count = diagnostics["oc_likely_count"]
    oc_share = diagnostics["oc_likely_share"]

    return "\n".join(
        [
            "# Union-Only Title Classification Protocol",
            "",
            "## 1. Objective",
            "This protocol creates a union-specific title classification for the Union Election x Glassdoor pipeline.",
            "",
            "## 2. Two Separate Dimensions",
            "- Union bargaining-unit relevance: likely_unionizable / likely_excluded / ambiguous.",
            "- Organizational-capital status: OC-likely roles based on management, technical/engineering, and creative/product content.",
            "These dimensions are intentionally separate and not collapsed.",
            "",
            "## 3. Main Classification Rules",
            "- Deterministic, rule-based keyword matching with conservative word-boundary patterns.",
            "- Low-information or suspicious titles are forced to union ambiguous and OC ambiguous.",
            "- Legal/HR/labor-relations/strategy/owner-contractor signals map to union likely_excluded.",
            "- Rank-and-file operational/frontline signals map to union likely_unionizable.",
            "- Conflicting excluded and unionizable signals map to ambiguous unless one side clearly dominates.",
            "",
            "## 4. Ambiguous Cases",
            "- Generic tokens (for example, analyst, associate, specialist, lead, assistant) are ambiguous without context.",
            "- Conflicting role signals are kept ambiguous to avoid overconfident misclassification.",
            "",
            "## 5. Output Variables",
            "- union_likely_unionizable, union_likely_excluded, union_ambiguous",
            "- union_classification, union_confidence, union_reason",
            "- oc_likely, oc_management, oc_technical_engineering, oc_creative_product, oc_ambiguous, oc_reason",
            "",
            "## 6. Diagnostics Summary",
            f"- Total titles: {diagnostics['total_titles']:,}",
            f"- Union likely_unionizable: {c.get('likely_unionizable', 0):,} ({s.get('likely_unionizable', 0.0):.4f})",
            f"- Union likely_excluded: {c.get('likely_excluded', 0):,} ({s.get('likely_excluded', 0.0):.4f})",
            f"- Union ambiguous: {c.get('ambiguous', 0):,} ({s.get('ambiguous', 0.0):.4f})",
            f"- OC likely count/share: {oc_count:,} / {oc_share:.4f}",
            "",
            "## 7. Limitations And Next Review Queue",
            "- Title-only rules cannot capture full job context and firm-specific role definitions.",
            "- Generic titles remain ambiguous by design and should be manually/LLM reviewed in later steps.",
            "- Use ambiguous and low-information output tables as the first review queue.",
            "",
        ]
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    primary_title_col = infer_primary_title_column(df)

    df["title_standardized"] = (
        df[primary_title_col].astype("string").fillna("").map(normalize_title_text)
    )

    union_rows = []
    oc_rows = []
    for _, row in df.iterrows():
        title = str(row["title_standardized"])
        union_rows.append(classify_union_dimension(title, row))
        oc_rows.append(classify_oc_dimension(title, row))

    union_df = pd.DataFrame(union_rows)
    oc_df = pd.DataFrame(oc_rows)
    out = pd.concat([df.reset_index(drop=True), union_df, oc_df], axis=1)

    out.to_csv(OUT_CLASSIFIED, index=False)

    ambiguous_df = out[out["union_classification"] == "ambiguous"].copy()
    if "n_reviews" in ambiguous_df.columns:
        ambiguous_df = ambiguous_df.sort_values(["n_reviews", "title_standardized"], ascending=[False, True])
    ambiguous_df.head(500).to_csv(OUT_AMBIG, index=False)

    low_mask = pd.Series(False, index=out.index)
    if "low_information_title" in out.columns:
        low_mask = low_mask | (out["low_information_title"].fillna(0).astype(int) == 1)
    if "is_suspicious" in out.columns:
        low_mask = low_mask | (out["is_suspicious"].fillna(0).astype(int) == 1)
    low_df = out[low_mask].copy()
    if "n_reviews" in low_df.columns:
        low_df = low_df.sort_values(["n_reviews", "title_standardized"], ascending=[False, True])
    low_df.head(500).to_csv(OUT_LOW_INFO, index=False)

    top_df = out.copy()
    if "n_reviews" in top_df.columns:
        top_df = top_df.sort_values(["n_reviews", "title_standardized"], ascending=[False, True])
    top_df.head(1000).to_csv(OUT_TOP, index=False)

    diagnostics = generate_diagnostics(out)
    OUT_DIAG.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    protocol = build_protocol_markdown(diagnostics)
    OUT_PROTOCOL.write_text(protocol, encoding="utf-8")

    print("Union title classification complete.")
    print(f"Input: {INPUT_PATH}")
    print(f"Primary title column: {primary_title_col}")
    print(f"Rows: {len(out):,}")
    print("Union classification counts:")
    print(out["union_classification"].value_counts(dropna=False).to_string())
    oc_count = int((out["oc_likely"] == 1).sum())
    oc_share = oc_count / len(out) if len(out) else np.nan
    print(f"oc_likely count/share: {oc_count:,} / {oc_share:.4f}")
    print("Wrote outputs:")
    print(f"- {OUT_CLASSIFIED}")
    print(f"- {OUT_DIAG}")
    print(f"- {OUT_PROTOCOL}")
    print(f"- {OUT_AMBIG}")
    print(f"- {OUT_LOW_INFO}")
    print(f"- {OUT_TOP}")


if __name__ == "__main__":
    main()
