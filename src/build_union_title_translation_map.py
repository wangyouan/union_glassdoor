from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


INPUT_PATH = Path(
    "/data/disk4/workspace/projects/glassdoor/outputs/job_title_standardized_universe.csv"
)

OUT_DIR = Path("/data/disk4/workspace/projects/union_glassdoor/outputs")
OUT_TRANSLATION_MAP = OUT_DIR / "union_title_translation_map.csv"
OUT_NORMALIZED = OUT_DIR / "union_title_universe_normalized.csv"
OUT_DIAG = OUT_DIR / "union_title_translation_diagnostics.json"
OUT_REVIEW_QUEUE = OUT_DIR / "union_title_translation_review_queue.csv"


LANGUAGE_SIGNALS: dict[str, set[str]] = {
    "portuguese": {
        "gerente",
        "coordenador",
        "analista",
        "vendedor",
        "vendedora",
        "atendente",
        "recepcionista",
        "auxiliar",
        "producao",
        "logistica",
        "operador",
        "motorista",
        "tecnico",
        "enfermeiro",
        "cozinheiro",
        "garcom",
        "caixa",
        "estagiario",
        "estagiaria",
        "estagio",
        "jovem aprendiz",
        "engenheiro",
        "desenvolvedor",
        "recursos humanos",
    },
    "spanish": {
        "practicante",
        "becario",
        "cajero",
        "vendedor",
        "consultor de ventas",
        "promotor de ventas",
        "enfermera",
        "cocinero",
        "ingeniero",
        "analista",
        "gerente",
        "recepcionista",
    },
    "french": {
        "stagiaire",
        "caissier",
        "caissiere",
        "conseiller clientele",
        "vendeur",
        "technicien",
        "serveur",
        "equipier polyvalent",
        "chef de projet",
        "operateur",
        "ingenieur",
    },
    "german": {
        "werkstudent",
    },
}

NON_ENGLISH_SIGNALS = set().union(*LANGUAGE_SIGNALS.values())


# Priority: exact phrase rules are applied before token-level substitutions.
EXACT_PHRASE_MAP: dict[str, str] = {
    # Intern / trainee
    "estagiario": "intern",
    "estagiaria": "intern",
    "estagio": "intern",
    "stagiaire": "intern",
    "practicante": "intern",
    "becario": "intern",
    "jovem aprendiz": "apprentice",
    "werkstudent": "working student",
    # Manager
    "gerente": "manager",
    "gerente comercial": "sales manager",
    "gerente de projetos": "project manager",
    "gerente de vendas": "sales manager",
    "coordenador": "coordinator",
    "coordenador de vendas": "sales coordinator",
    "chef de projet": "project manager",
    # Analyst
    "analista": "analyst",
    "analista financeiro": "financial analyst",
    "analista de sistemas": "systems analyst",
    "analista administrativo": "administrative analyst",
    "analista de recursos humanos": "human resources analyst",
    "analista de marketing": "marketing analyst",
    "analista pleno": "analyst",
    "analista junior": "junior analyst",
    "analista senior": "senior analyst",
    "analista de recursos humanos rh": "human resources analyst",
    "analista de sistemas senior": "senior systems analyst",
    # Sales / retail
    "vendedor": "salesperson",
    "vendedora": "salesperson",
    "consultor de vendas": "sales consultant",
    "promotor de vendas": "sales promoter",
    "cajero": "cashier",
    "caissier": "cashier",
    "caissiere": "cashier",
    "operador de caixa": "cashier",
    "atendente": "attendant",
    "agente de atendimento": "customer service agent",
    "conseiller clientele": "customer advisor",
    "vendeur": "salesperson",
    "caixa": "cashier",
    "vendedor comercial": "salesperson",
    "vendeur conseil": "sales advisor",
    "vendeur polyvalent": "sales associate",
    "vendeur conseil en magasin": "store sales advisor",
    "vendeur magasinier": "sales stock associate",
    "vendeur stockiste": "stock associate",
    "vendeur boutique": "boutique sales associate",
    "vendeur produits techniques": "technical product salesperson",
    "conseiller vendeur": "sales advisor",
    "magasinier vendeur": "stockroom sales associate",
    # Frontline / support
    "recepcionista": "receptionist",
    "assistente administrativo": "administrative assistant",
    "auxiliar administrativo": "administrative assistant",
    "auxiliar de producao": "production assistant",
    "auxiliar de logistica": "logistics assistant",
    "operador": "operator",
    "operador de producao": "production operator",
    "operador de maquinas": "machine operator",
    "operador de telemarketing": "call center operator",
    "motorista": "driver",
    "tecnico": "technician",
    "technicien": "technician",
    "ingenieur": "engineer",
    "ingeniero": "engineer",
    "engenheiro": "engineer",
    "desenvolvedor": "developer",
    "atendente de call center": "call center attendant",
    "jovem aprendiz administrativo": "administrative apprentice",
    "estagiario de engenharia": "engineering intern",
    "supervisor coordenador": "supervisor coordinator",
    "operadora de caixa": "cashier",
    "op de caixa": "cashier",
    "fiscal de caixa": "cashier supervisor",
    "frente de caixa": "front-end cashier",
    "caixa bancario": "bank teller",
    "caixa executivo": "bank teller",
    "caixa treinador": "cashier trainer",
    "lider de caixa": "cashier lead",
    "agente de negocios caixa": "bank service agent",
    "agente de negocios caixa agencia": "bank branch service agent",
    "fiscal de caixa i": "cashier supervisor",
    "fiscal de caixa op de caixa": "cashier supervisor cashier",
    "operadora de caixa i": "cashier",
    "operateur de production": "production operator",
    "operateur production": "production operator",
    "operateur de fabrication": "manufacturing operator",
    "operateur polyvalent": "general operator",
    "operateur logistique": "logistics operator",
    "operateur de saisie": "data entry operator",
    "operateur salle blanche": "cleanroom operator",
    "operateur animateur attraction": "attraction operator",
    # Healthcare
    "enfermeiro": "nurse",
    "enfermera": "nurse",
    "tecnico de enfermagem": "nursing technician",
    "auxiliar de enfermagem": "nursing assistant",
    # Other
    "consultor": "consultant",
    "especialista": "specialist",
    "administrativo": "administrative",
    "cozinheiro": "cook",
    "cocinero": "cook",
    "garcom": "waiter",
    "serveur": "waiter",
    "equipier polyvalent": "crew member",
    "operateur": "operator",
    "ingenieur d etudes": "engineer",
    # HR / human resources
    "recursos humanos": "human resources",
    "jefe de recursos humanos": "human resources manager",
    "jefe de recursos humanos rrhh": "human resources manager",
    "coordinador de recursos humanos": "human resources coordinator",
    "coordinador de recursos humanos rh": "human resources coordinator",
    "coordinador de recursos humanos rrhh": "human resources coordinator",
    "generalista de recursos humanos": "human resources generalist",
    "generalista de recursos humanos rrhh": "human resources generalist",
    "asistente de recursos humanos": "human resources assistant",
    "asistente de recursos humanos rrhh": "human resources assistant",
    "pasante de recursos humanos": "human resources intern",
    "pasante de recursos humanos rrhh": "human resources intern",
    "consultora de recursos humanos": "human resources consultant",
    "consultora de recursos humanos rh": "human resources consultant",
    "supervisor de recursos humanos": "human resources supervisor",
    "supervisor de recursos humanos rh": "human resources supervisor",
    "director de recursos humanos": "human resources director",
    "director de recursos humanos rrhh": "human resources director",
    "director recursos humanos": "human resources director",
    "diretor de recursos humanos": "human resources director",
    "diretor de recursos humanos rh": "human resources director",
    "gestor de recursos humanos": "human resources manager",
    "gestor de recursos humanos rh": "human resources manager",
    "psicologa recursos humanos": "human resources psychologist",
    "recursos humanos reclutamiento y seleccion": "human resources recruiting and selection",
}


TOKEN_MAP: dict[str, str] = {
    "gerente": "manager",
    "coordenador": "coordinator",
    "analista": "analyst",
    "vendedor": "salesperson",
    "vendedora": "salesperson",
    "cajero": "cashier",
    "caissier": "cashier",
    "caissiere": "cashier",
    "atendente": "attendant",
    "recepcionista": "receptionist",
    "assistente": "assistant",
    "auxiliar": "assistant",
    "operador": "operator",
    "motorista": "driver",
    "tecnico": "technician",
    "technicien": "technician",
    "ingenieur": "engineer",
    "ingeniero": "engineer",
    "engenheiro": "engineer",
    "desenvolvedor": "developer",
    "enfermeiro": "nurse",
    "enfermera": "nurse",
    "consultor": "consultant",
    "especialista": "specialist",
    "administrativo": "administrative",
    "cozinheiro": "cook",
    "cocinero": "cook",
    "garcom": "waiter",
    "serveur": "waiter",
    "stagiaire": "intern",
    "practicante": "intern",
    "becario": "intern",
    "estagiario": "intern",
    "estagiaria": "intern",
    "estagio": "intern",
    "werkstudent": "working student",
    "producao": "production",
    "logistica": "logistics",
    "vendas": "sales",
    "projetos": "projects",
    "sistemas": "systems",
    "financeiro": "financial",
    "marketing": "marketing",
}

MODIFIER_TOKENS = {
    "senior",
    "sr",
    "junior",
    "jr",
    "pleno",
    "trainee",
    "lead",
    "principal",
}

STOPWORDS = {
    "de",
    "do",
    "da",
    "del",
    "di",
    "du",
    "des",
    "la",
    "le",
    "el",
    "y",
    "e",
}


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _phrase_to_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def contains_any_phrase(text: str, phrases: Iterable[str]) -> bool:
    for p in phrases:
        if _phrase_to_pattern(p).search(text):
            return True
    return False


def detect_language_or_origin(normalized_title: str) -> str:
    if not normalized_title:
        return "unknown"

    hits: dict[str, int] = {k: 0 for k in LANGUAGE_SIGNALS.keys()}
    for lang, signals in LANGUAGE_SIGNALS.items():
        for sig in signals:
            if _phrase_to_pattern(sig).search(normalized_title):
                hits[lang] += 1

    best_lang = max(hits, key=hits.get)
    if hits[best_lang] > 0:
        return best_lang

    ascii_alpha = re.fullmatch(r"[a-z0-9\s]+", normalized_title) is not None
    if ascii_alpha:
        return "english_or_unknown"
    return "unknown"


def _reorder_tokens(tokens: list[str]) -> list[str]:
    modifiers = [t for t in tokens if t in MODIFIER_TOKENS]
    core = [t for t in tokens if t not in MODIFIER_TOKENS]
    return modifiers + core


def translate_title(normalized_title: str) -> dict[str, str]:
    if not normalized_title:
        return {
            "title_canonical_en": "",
            "translation_source": "untranslated",
            "translation_confidence": "low",
            "translation_note": "empty_title",
        }

    if normalized_title in EXACT_PHRASE_MAP:
        return {
            "title_canonical_en": EXACT_PHRASE_MAP[normalized_title],
            "translation_source": "dictionary",
            "translation_confidence": "high",
            "translation_note": "exact_phrase_mapping",
        }

    # Phrase-level substitution first for multi-word entries.
    phrase_matches: list[tuple[str, str]] = []
    interim = normalized_title
    for phrase in sorted(EXACT_PHRASE_MAP.keys(), key=lambda x: len(x.split()), reverse=True):
        if len(phrase.split()) <= 1:
            continue
        patt = _phrase_to_pattern(phrase)
        if patt.search(interim):
            phrase_matches.append((phrase, EXACT_PHRASE_MAP[phrase]))
            interim = patt.sub(EXACT_PHRASE_MAP[phrase], interim)

    tokens = [t for t in interim.split() if t]
    mapped_tokens: list[str] = []
    mapped_any_token = False
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        if tok in TOKEN_MAP:
            mapped_tokens.append(TOKEN_MAP[tok])
            mapped_any_token = True
        else:
            mapped_tokens.append(tok)

    mapped_tokens = _reorder_tokens(mapped_tokens)
    candidate = re.sub(r"\s+", " ", " ".join(mapped_tokens)).strip()

    if phrase_matches:
        changed = candidate != normalized_title
        return {
            "title_canonical_en": candidate if candidate else normalized_title,
            "translation_source": "dictionary",
            "translation_confidence": "high" if changed else "medium",
            "translation_note": "phrase_mapping",
        }

    if mapped_any_token:
        changed = candidate != normalized_title
        return {
            "title_canonical_en": candidate if candidate else normalized_title,
            "translation_source": "dictionary",
            "translation_confidence": "medium" if changed else "low",
            "translation_note": "token_mapping",
        }

    looks_non_english = contains_any_phrase(normalized_title, NON_ENGLISH_SIGNALS)
    if looks_non_english:
        return {
            "title_canonical_en": normalized_title,
            "translation_source": "untranslated",
            "translation_confidence": "low",
            "translation_note": "non_english_signal_no_mapping",
        }

    return {
        "title_canonical_en": normalized_title,
        "translation_source": "already_english_or_unchanged",
        "translation_confidence": "high",
        "translation_note": "already_english_or_unchanged",
    }


def top_rows(df: pd.DataFrame, mask: pd.Series, n: int = 100) -> list[dict]:
    cols = [
        "title_standardized",
        "title_canonical_en",
        "n_reviews",
        "detected_language_or_origin",
        "translation_source",
        "translation_confidence",
        "translation_note",
    ]
    use_cols = [c for c in cols if c in df.columns]
    out = df.loc[mask, use_cols].copy()
    if "n_reviews" in out.columns:
        out = out.sort_values(["n_reviews", "title_standardized"], ascending=[False, True])
    else:
        out = out.sort_values(["title_standardized"], ascending=[True])
    return out.head(n).to_dict(orient="records")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    if "title_standardized" not in df.columns:
        raise KeyError("Input must include title_standardized.")

    df["title_standardized"] = df["title_standardized"].astype("string").fillna("")
    df["title_normalized"] = df["title_standardized"].map(normalize_text)

    detected = []
    translated_rows = []
    for title in df["title_normalized"]:
        detected.append(detect_language_or_origin(title))
        translated_rows.append(translate_title(title))

    df["detected_language_or_origin"] = detected
    trans_df = pd.DataFrame(translated_rows)
    out = pd.concat([df.reset_index(drop=True), trans_df], axis=1)

    # If unchanged and language is unclear/english, mark as unchanged source.
    unchanged_mask = out["title_canonical_en"] == out["title_normalized"]
    englishish_mask = out["detected_language_or_origin"].isin(["english_or_unknown", "unknown"])
    out.loc[
        unchanged_mask & englishish_mask,
        ["translation_source", "translation_confidence", "translation_note"],
    ] = ["already_english_or_unchanged", "high", "already_english_or_unchanged"]

    changed_mask = out["title_canonical_en"] != out["title_normalized"]

    non_english_signal_mask = out["title_normalized"].map(lambda x: contains_any_phrase(x, NON_ENGLISH_SIGNALS))

    review_mask = (
        (out["translation_confidence"] == "low")
        | ((out["title_canonical_en"] == out["title_normalized"]) & non_english_signal_mask)
        | (non_english_signal_mask & (out["translation_note"] == "non_english_signal_no_mapping"))
    )

    map_cols = [
        "title_standardized",
        "title_normalized",
        "title_canonical_en",
        "detected_language_or_origin",
        "translation_source",
        "translation_confidence",
        "translation_note",
    ]
    if "n_reviews" in out.columns:
        map_cols.append("n_reviews")

    translation_map = out[map_cols].copy()
    if "n_reviews" in translation_map.columns:
        translation_map = translation_map.sort_values(["n_reviews", "title_standardized"], ascending=[False, True])

    review_cols = [
        "title_standardized",
        "title_canonical_en",
        "n_reviews",
        "example_companies",
        "detected_language_or_origin",
        "translation_source",
        "translation_confidence",
        "translation_note",
    ]
    review_cols = [c for c in review_cols if c in out.columns]
    review_queue = out.loc[review_mask, review_cols].copy()
    if "n_reviews" in review_queue.columns:
        review_queue = review_queue.sort_values(["n_reviews", "title_standardized"], ascending=[False, True])

    translation_map.to_csv(OUT_TRANSLATION_MAP, index=False)
    out.to_csv(OUT_NORMALIZED, index=False)
    review_queue.to_csv(OUT_REVIEW_QUEUE, index=False)

    total_titles = int(len(out))
    total_reviews = float(out["n_reviews"].sum()) if "n_reviews" in out.columns else None
    changed_titles = int(changed_mask.sum())
    changed_share = changed_titles / total_titles if total_titles else None

    changed_review_share = None
    if "n_reviews" in out.columns and total_reviews and total_reviews > 0:
        changed_review_share = float(out.loc[changed_mask, "n_reviews"].sum()) / total_reviews

    diagnostics = {
        "total_titles": total_titles,
        "total_reviews_represented": total_reviews,
        "titles_changed_by_translation": changed_titles,
        "titles_changed_share": changed_share,
        "changed_review_weighted_share": changed_review_share,
        "top_100_changed_titles_by_reviews": top_rows(out, changed_mask, n=100),
        "top_100_low_confidence_untranslated_titles_by_reviews": top_rows(
            out,
            (out["translation_confidence"] == "low") & (out["translation_source"] == "untranslated"),
            n=100,
        ),
        "counts_by_detected_language_or_origin": out["detected_language_or_origin"].value_counts(dropna=False).to_dict(),
        "counts_by_translation_source": out["translation_source"].value_counts(dropna=False).to_dict(),
    }

    OUT_DIAG.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print("Union title translation map build complete.")
    print(f"Input: {INPUT_PATH}")
    print(f"Rows: {len(out):,}")
    print(f"Changed titles: {changed_titles:,}")
    if changed_review_share is not None:
        print(f"Changed review-weighted share: {changed_review_share:.4f}")
    print("Wrote outputs:")
    print(f"- {OUT_TRANSLATION_MAP}")
    print(f"- {OUT_NORMALIZED}")
    print(f"- {OUT_DIAG}")
    print(f"- {OUT_REVIEW_QUEUE}")


if __name__ == "__main__":
    main()
