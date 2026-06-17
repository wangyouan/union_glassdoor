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
OUT_PRE_STEP1C = OUT_DIR / "union_classified_title_universe_pre_step1c.csv"
OUT_DIAG = OUT_DIR / "union_title_classification_diagnostics.json"
OUT_PROTOCOL = OUT_DIR / "union_title_classification_protocol.md"
OUT_AMBIG = OUT_DIR / "union_ambiguous_title_examples.csv"
OUT_LOW_INFO = OUT_DIR / "union_low_information_title_examples.csv"
OUT_TOP = OUT_DIR / "union_top_titles_by_reviews.csv"
OUT_STEP1C_RECLASSIFIED = OUT_DIR / "STEP1C_RECLASSIFIED_EXAMPLES.csv"
OUT_STEP1C_SUMMARY = OUT_DIR / "STEP1C_SUMMARY.md"


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

AMBIGUOUS_ROLE_PHRASES = [
    "sales associate",
    "sales",
    "sales representative",
    "sales consultant",
    "sales assistant",
    "sales advisor",
    "inside sales",
    "outside sales representative",
    "sales specialist",
    "business development",
    "business development representative",
    "account executive",
    "account manager",
    "consultant",
    "analyst",
    "associate",
    "specialist",
    "assistant",
    "coordinator",
    "advisor",
    "agent",
    "officer",
    "representative",
]

WEAK_SUPERVISORY_AMBIGUOUS = [
    "assistant manager",
    "shift supervisor",
    "shift leader",
    "team lead",
    "team leader",
    "lead",
]

CONFLICT_MANAGER_AMBIGUOUS_PHRASES = [
    "production manager",
    "customer service manager",
    "service delivery manager",
    "delivery manager",
    # STEP1D Fix 2 & 3: roles miscaught by EXCLUDED_STRONG
    "product owner",          # Scrum IC role, not NLRA supervisor
    "legal assistant",        # support staff, typically unionizable
    "legal secretary",        # support staff, typically unionizable
    "legal clerk",            # support staff, typically unionizable
    "legal receptionist",
    "legal coordinator",
]

# Retail / Store Workers
RETAIL_UNIONIZABLE_KWS = [
    "retail sales associate",
    "sales floor associate",
    "seasonal sales associate",
    "part time sales associate",
    "part-time sales associate",
    "stock associate",
    "stocker",
    "overnight stocker",
    "shelf stocker",
    "store associate",
    "store clerk",
    "grocery clerk",
    "retail assistant",
    "shop assistant",
    "beauty advisor",
    "key holder",
    "keyholder",
    "team member",
    "cashier",
]

# Food Service
FOOD_SERVICE_UNIONIZABLE_KWS = [
    "barista",
    "crew member",
    "crew",
    "sandwich artist",
    "line cook",
    "prep cook",
    "cook",
    "dishwasher",
    "server",
    "waiter",
    "waitress",
    "host",
    "hostess",
    "bartender",
    "baker",
    "food service worker",
    "busser",
    "kitchen staff",
]

# Logistics / Transportation
LOGISTICS_UNIONIZABLE_KWS = [
    "package handler",
    "part time package handler",
    "part-time package handler",
    "material handler",
    "picker",
    "packer",
    "picker packer",
    "order picker",
    "warehouse worker",
    "warehouse associate",
    "fulfillment associate",
    "sortation associate",
    "dock worker",
    "loader",
    "forklift operator",
    "delivery driver",
    "driver",
    "truck driver",
    "courier",
    "ramp agent",
    "postman",
]

# Manufacturing
MANUFACTURING_UNIONIZABLE_KWS = [
    "machine operator",
    "operator",
    "production",
    "production worker",
    "assembler",
    "laborer",
    "general laborer",
    "welder",
    "machinist",
    "mechanic",
    "maintenance worker",
    "maintenance technician",
    "field technician",
    "installer",
    "electrician",
    "laborer",
]

# Healthcare Support
HEALTHCARE_SUPPORT_UNIONIZABLE_KWS = [
    "medical assistant",
    "certified nursing assistant",
    "nursing assistant",
    "cna",
    "caregiver",
    "home health aide",
    "patient care technician",
    "phlebotomist",
]

# Aviation / Hospitality
AVIATION_HOSPITALITY_UNIONIZABLE_KWS = [
    "housekeeper",
    "room attendant",
    "front desk agent",
    "front desk receptionist",
    "receptionist",
    "concierge",
]

# Banking
BANKING_UNIONIZABLE_KWS = [
    "bank teller",
    "teller",
]

# Frontline / Generalist
FRONTLINE_UNIONIZABLE_KWS = [
    "security guard",
    "security officer",
    "customer service representative",
    "customer service associate",
    "customer service agent",
    "customer care representative",
    "technical support representative",
    "call center representative",
    "contact center representative",
    "service representative",
    "clerk",
    "janitor",
    "cleaner",
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
    "head of",
    "general manager",
    "managing director",
]

# Managerial
MANAGERIAL_EXCLUDED_KWS = [
    "regional manager",
    "district manager",
    "director",
]

# Supervisory / Leadership
SUPERVISORY_EXCLUDED_KWS = [
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
    "principal",
]

# Multilingual Keywords (Spanish)
SPANISH_UNIONIZABLE_KWS = [
    "vendedor",
    "cajero",
    "operador",
    "operador de caixa",
    "operador de producao",
    "operador de maquinas",
    "motorista",
    "recepcionista",
    "auxiliar de producao",
    "auxiliar de logistica",
    "tecnico",
]

# Multilingual Keywords (Portuguese)
PORTUGUESE_UNIONIZABLE_KWS = [
    "atendente",
    "operador",
    "motorista",
    "tecnico",
]

# Multilingual Keywords (French)
FRENCH_UNIONIZABLE_KWS = [
    "caissier",
    "caissiere",
    "technicien",
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
    "director",
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
    "principal",
    "attorney",
    "lawyer",
    "legal",
    "counsel",
    "general counsel",
    "corporate counsel",
    "litigation",
    "human resources",
    "hrbp",
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
    "founder",
    "co founder",
    "owner",
    "partner",
]

EXCLUDED_STRONG = [
    "attorney",
    "lawyer",
    "legal",
    "counsel",
    "human resources",
    "labor relations",
    "employee relations",
    "ceo",
    "cfo",
    "coo",
    "cto",
    "cio",
    "chief",
    "vice president",
    "vp",
    "head of",
    "managing director",
    "general manager",
    "regional manager",
    "district manager",
    "director",
    "strategy",
    "corporate development",
    "founder",
    "owner",
    "partner",
    "principal",
]

# STEP1D Fix 1: principal + IC role = ambiguous (not excluded)
PRINCIPAL_IC_OVERRIDE_KWS = [
    "engineer", "scientist", "architect", "developer", "designer",
    "analyst", "researcher", "consultant", "specialist", "technician",
    "cashier", "cajero",
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
    # STEP1D Fix 4: "team lead" and "shift lead" removed — frontline supervisory, not OC
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
    "data scientist",
    "data engineer",
    "machine learning engineer",
    "systems engineer",
    "systems analyst",
    "system administrator",
    "systems administrator",
    "database administrator",
    "network engineer",
    "cybersecurity analyst",
    "information security analyst",
    "it analyst",
    "it specialist",
    "information technology",
    "business analyst",
    "data analyst",
    "financial analyst",
    "quantitative analyst",
    "research analyst",
    "research scientist",
    "scientist",
    "researcher",
    "economist",
    "statistician",
    "technical writer",
    "devops",
    "desenvolvedor",
    "engenheiro",
    "ingenieur",
    "ingeniero",
]

# Analytics / Business Intelligence
OC_ANALYTICS_KWS = [
    "business intelligence",
    "analytics",
]

# Product / Design
OC_PRODUCT_DESIGN_KWS = [
    "product manager",
    "product designer",
    "ux designer",
    "ui designer",
    "graphic designer",
    "designer",
]

# Research
OC_RESEARCH_KWS = [
    "research scientist",
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


def _mainly_contains_phrase(title: str, phrase: str, max_tokens: int = 5) -> bool:
    if not _phrase_to_pattern(phrase).search(title):
        return False
    return len(title.split()) <= max_tokens

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

    union_hits = sorted(set(matched_keywords(title, UNIONIZABLE_KEYWORDS)))
    excluded_hits = sorted(set(matched_keywords(title, EXCLUDED_KEYWORDS)))
    excluded_strong_hits = sorted(set(matched_keywords(title, EXCLUDED_STRONG)))
    ambiguous_role_hits = sorted(set(matched_keywords(title, AMBIGUOUS_ROLE_PHRASES)))
    weak_supervisory_hits = sorted(set(matched_keywords(title, WEAK_SUPERVISORY_AMBIGUOUS)))
    conflict_manager_hits = sorted(set(matched_keywords(title, CONFLICT_MANAGER_AMBIGUOUS_PHRASES)))

    # Strong excluded signals
    if excluded_strong_hits:
        # STEP1D Fix 1: principal + IC role override
        if "principal" in excluded_strong_hits:
            if any(kw in title for kw in PRINCIPAL_IC_OVERRIDE_KWS):
                return {
                    "union_likely_unionizable": 0,
                    "union_likely_excluded": 0,
                    "union_ambiguous": 1,
                    "union_classification": "ambiguous",
                    "union_confidence": "medium",
                    "union_reason": f"principal_ic_override:{'|'.join(excluded_strong_hits)}",
                }
        # STEP1D Fix 2+3: CONFLICT_MANAGER_AMBIGUOUS overrides EXCLUDED_STRONG
        if any(kw in title for kw in CONFLICT_MANAGER_AMBIGUOUS_PHRASES):
            return {
                "union_likely_unionizable": 0,
                "union_likely_excluded": 0,
                "union_ambiguous": 1,
                "union_classification": "ambiguous",
                "union_confidence": "medium",
                "union_reason": f"conflict_ambiguous_override:{'|'.join(excluded_strong_hits)}",
            }
        return {
            "union_likely_unionizable": 0,
            "union_likely_excluded": 1,
            "union_ambiguous": 0,
            "union_classification": "likely_excluded",
            "union_confidence": "high",
            "union_reason": f"excluded_strong:{'|'.join(excluded_strong_hits)}",
        }

    n_u = len(union_hits)
    n_e = len(excluded_hits)

    # Weak supervisory titles are kept ambiguous when they are the main title signal.
    if any(_mainly_contains_phrase(title, kw) for kw in WEAK_SUPERVISORY_AMBIGUOUS):
        return {
            "union_likely_unionizable": 0,
            "union_likely_excluded": 0,
            "union_ambiguous": 1,
            "union_classification": "ambiguous",
            "union_confidence": "low",
            "union_reason": f"ambiguous_weak_supervisory:{'|'.join(weak_supervisory_hits)}",
        }

    if n_u == 0 and n_e == 0 and ambiguous_role_hits:
        return {
            "union_likely_unionizable": 0,
            "union_likely_excluded": 0,
            "union_ambiguous": 1,
            "union_classification": "ambiguous",
            "union_confidence": "low",
            "union_reason": f"ambiguous_generic_role:{'|'.join(ambiguous_role_hits)}",
        }

    # Conflict resolution
    if n_u > 0 and n_e > 0:
        if weak_supervisory_hits or conflict_manager_hits:
            return {
                "union_likely_unionizable": 0,
                "union_likely_excluded": 0,
                "union_ambiguous": 1,
                "union_classification": "ambiguous",
                "union_confidence": "low",
                "union_reason": (
                    "conflict_weak_supervisory_or_manager:"
                    f"u={','.join(union_hits)};e={','.join(excluded_hits)}"
                ),
            }

        if n_e >= 3 and n_u <= 1:
            return {
                "union_likely_unionizable": 0,
                "union_likely_excluded": 1,
                "union_ambiguous": 0,
                "union_classification": "likely_excluded",
                "union_confidence": "medium",
                "union_reason": f"excluded_dominates_conflict:{'|'.join(excluded_hits)}",
            }
        return {
            "union_likely_unionizable": 0,
            "union_likely_excluded": 0,
            "union_ambiguous": 1,
            "union_classification": "ambiguous",
            "union_confidence": "low",
            "union_reason": (
                "conflicting_signals:"
                f"u={','.join(union_hits)};e={','.join(excluded_hits)}"
            ),
        }

    if conflict_manager_hits:
        return {
            "union_likely_unionizable": 0,
            "union_likely_excluded": 0,
            "union_ambiguous": 1,
            "union_classification": "ambiguous",
            "union_confidence": "low",
            "union_reason": f"ambiguous_managerial_context:{'|'.join(conflict_manager_hits)}",
        }

    # Clear excluded signal
    if n_e > 0:
        conf = "high"
        reason = f"excluded_keywords:{'|'.join(excluded_hits)}"
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
            "union_reason": f"unionizable_keywords:{'|'.join(union_hits)}",
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


def _class_counts_and_weighted(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, float]]:
    counts = df["union_classification"].value_counts(dropna=False).to_dict()
    weighted: dict[str, float] = {}
    if "n_reviews" in df.columns:
        total_reviews = float(df["n_reviews"].sum())
        if total_reviews > 0:
            for cls in ["likely_unionizable", "likely_excluded", "ambiguous"]:
                weighted[cls] = float(df.loc[df["union_classification"] == cls, "n_reviews"].sum()) / total_reviews
    return counts, weighted


def _format_markdown_table(df: pd.DataFrame, columns: list[str], n: int = 50) -> str:
    if df.empty:
        return "- None"

    tmp = df.copy()
    if "n_reviews" in tmp.columns:
        tmp = tmp.sort_values(["n_reviews", "title_standardized"], ascending=[False, True])
    else:
        tmp = tmp.sort_values(["title_standardized"], ascending=[True])
    tmp = tmp.head(n)

    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, r in tmp.iterrows():
        vals = []
        for c in columns:
            v = r.get(c, "")
            if pd.isna(v):
                vals.append("")
            else:
                vals.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_step1c_summary(old_df: pd.DataFrame, new_df: pd.DataFrame, merged: pd.DataFrame) -> str:
    old_counts, old_weighted = _class_counts_and_weighted(old_df)
    new_counts, new_weighted = _class_counts_and_weighted(new_df)

    amb_to_union = merged[
        (merged["old_union_classification"] == "ambiguous")
        & (merged["new_union_classification"] == "likely_unionizable")
    ][["title_standardized", "n_reviews", "old_union_classification", "new_union_classification"]]

    excl_to_amb = merged[
        (merged["old_union_classification"] == "likely_excluded")
        & (merged["new_union_classification"] == "ambiguous")
    ][["title_standardized", "n_reviews", "old_union_classification", "new_union_classification"]]

    newly_oc = merged[
        (merged["old_oc_likely"] == 0)
        & (merged["new_oc_likely"] == 1)
    ][["title_standardized", "n_reviews", "old_oc_likely", "new_oc_likely", "new_oc_reason"]]

    remaining_amb = new_df[new_df["union_classification"] == "ambiguous"][
        ["title_standardized", "n_reviews", "union_reason", "oc_likely", "oc_reason"]
    ]

    return "\n".join(
        [
            "# STEP1C Summary",
            "",
            "## Rule Changes",
            "- Removed broad unionizable signals for generic sales and technical/developer terms.",
            "- Added narrow frontline unionizable phrase rules across retail, food service, logistics, trades, healthcare support, hospitality, and customer service.",
            "- Added multilingual frontline rules and prevented Portuguese/Spanish technical-professional terms from being unionizable by default.",
            "- Kept weak supervisory titles (assistant manager, shift supervisor, team lead, etc.) as ambiguous when they are the main title signal.",
            "- Conflict handling now avoids unionizable-by-token-count behavior; strong excluded signals can win, otherwise conflict stays ambiguous.",
            "",
            "## Classification Counts Before/After",
            f"- Before: likely_unionizable={old_counts.get('likely_unionizable', 0):,}, likely_excluded={old_counts.get('likely_excluded', 0):,}, ambiguous={old_counts.get('ambiguous', 0):,}",
            f"- After: likely_unionizable={new_counts.get('likely_unionizable', 0):,}, likely_excluded={new_counts.get('likely_excluded', 0):,}, ambiguous={new_counts.get('ambiguous', 0):,}",
            "",
            "## Review-Weighted Shares Before/After",
            f"- Before: likely_unionizable={old_weighted.get('likely_unionizable', 0.0):.4f}, likely_excluded={old_weighted.get('likely_excluded', 0.0):.4f}, ambiguous={old_weighted.get('ambiguous', 0.0):.4f}",
            f"- After: likely_unionizable={new_weighted.get('likely_unionizable', 0.0):.4f}, likely_excluded={new_weighted.get('likely_excluded', 0.0):.4f}, ambiguous={new_weighted.get('ambiguous', 0.0):.4f}",
            "",
            "## Top 50 Ambiguous -> Likely_Unionizable",
            _format_markdown_table(
                amb_to_union,
                ["title_standardized", "n_reviews", "old_union_classification", "new_union_classification"],
                n=50,
            ),
            "",
            "## Top 50 Likely_Excluded -> Ambiguous",
            _format_markdown_table(
                excl_to_amb,
                ["title_standardized", "n_reviews", "old_union_classification", "new_union_classification"],
                n=50,
            ),
            "",
            "## Top 50 Newly OC_Likely",
            _format_markdown_table(
                newly_oc,
                ["title_standardized", "n_reviews", "old_oc_likely", "new_oc_likely", "new_oc_reason"],
                n=50,
            ),
            "",
            "## High-Review Remaining Ambiguous Titles",
            _format_markdown_table(
                remaining_amb,
                ["title_standardized", "n_reviews", "union_reason", "oc_likely", "oc_reason"],
                n=50,
            ),
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

    old_df = None
    if OUT_PRE_STEP1C.exists():
        old_df = pd.read_csv(OUT_PRE_STEP1C)

    out.to_csv(OUT_CLASSIFIED, index=False)

    if old_df is not None and "title_standardized" in old_df.columns:
        old_keep_cols = [
            "title_standardized",
            "union_classification",
            "union_reason",
            "oc_likely",
            "oc_reason",
        ]
        old_keep_cols = [c for c in old_keep_cols if c in old_df.columns]
        old_cmp = old_df[old_keep_cols].copy().rename(
            columns={
                "union_classification": "old_union_classification",
                "union_reason": "old_union_reason",
                "oc_likely": "old_oc_likely",
                "oc_reason": "old_oc_reason",
            }
        )

        new_cmp = out[
            [
                "title_standardized",
                "n_reviews",
                "union_classification",
                "union_reason",
                "oc_likely",
                "oc_reason",
            ]
        ].copy().rename(
            columns={
                "union_classification": "new_union_classification",
                "union_reason": "new_union_reason",
                "oc_likely": "new_oc_likely",
                "oc_reason": "new_oc_reason",
            }
        )

        merged = new_cmp.merge(old_cmp, on="title_standardized", how="left")

        changed_mask = (
            (merged["old_union_classification"].fillna("") != merged["new_union_classification"].fillna(""))
            | (merged["old_union_reason"].fillna("") != merged["new_union_reason"].fillna(""))
            | (merged["old_oc_likely"].fillna(-1).astype(int) != merged["new_oc_likely"].fillna(-1).astype(int))
            | (merged["old_oc_reason"].fillna("") != merged["new_oc_reason"].fillna(""))
        )

        step1c_examples = merged.loc[
            changed_mask,
            [
                "title_standardized",
                "n_reviews",
                "old_union_classification",
                "new_union_classification",
                "old_union_reason",
                "new_union_reason",
                "old_oc_likely",
                "new_oc_likely",
                "old_oc_reason",
                "new_oc_reason",
            ],
        ].copy()

        if "n_reviews" in step1c_examples.columns:
            step1c_examples = step1c_examples.sort_values(["n_reviews", "title_standardized"], ascending=[False, True])
        step1c_examples.to_csv(OUT_STEP1C_RECLASSIFIED, index=False)

        step1c_summary = build_step1c_summary(old_df=old_df, new_df=out, merged=merged)
        OUT_STEP1C_SUMMARY.write_text(step1c_summary, encoding="utf-8")

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
    if old_df is not None and "title_standardized" in old_df.columns:
        print(f"- {OUT_STEP1C_RECLASSIFIED}")
        print(f"- {OUT_STEP1C_SUMMARY}")


if __name__ == "__main__":
    main()
