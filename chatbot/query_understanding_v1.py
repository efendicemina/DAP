"""
Task 1 - Analiza korisnickog upita i ekstrakcija kljucnih informacija.

Ovaj modul je dopuna intent klasifikatora. Intent model daje glavnu klasu upita,
a ovaj sloj iz korisnickog pitanja izdvaja strukturirane informacije koje su korisne
za dalje rutiranje retrieval sistema:
- tip trazene informacije,
- tip dokumenta,
- proceduru ili pravnu oblast,
- instituciju/nadleznost,
- domenske kljucne rijeci.

Modul je namjerno lagan i interpretabilan jer se koristi u pravnoj domeni gdje je
bitno znati zasto je pitanje rutirano na odredjenu oblast.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional


DIACRITIC_TRANSLATION = str.maketrans(
    {
        "č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "dj",
        "Č": "c", "Ć": "c", "Š": "s", "Ž": "z", "Đ": "dj",
    }
)

STOPWORDS = {
    "a", "ako", "ali", "bi", "bih", "bila", "bilo", "biti", "da", "do",
    "ga", "gdje", "i", "ili", "ima", "je", "jer", "kako", "ko", "koja",
    "koje", "koji", "kome", "mi", "mogu", "moze", "na", "ne", "neki",
    "o", "od", "ovo", "po", "pod", "sam", "se", "sta", "su", "taj",
    "te", "treba", "trebam", "u", "za",
}

ACTION_TYPE_PATTERNS: Dict[str, List[str]] = {
    "kontakt": [r"\bkontakt\b", r"\btelefon\b", r"\bemail\b", r"\be-mail\b", r"\badres", r"kome se obratiti", r"koga da kontaktiram"],
    "takse": [r"koliko\s+kosta", r"\btaks", r"\bnaknad", r"\buplat", r"\biznos", r"\bcijena"],
    "obrazac": [r"\bobrazac\b", r"\bformular\b", r"\bprijav", r"\bzahtjev"],
    "dokumentacija": [r"dokument", r"sta mi treba", r"koji.*treb", r"sta.*pred", r"prilog", r"dostavit"],
    "procedura": [r"\bkako\b", r"\bprocedur", r"\bpostup", r"\bkorac", r"pokren", r"registrovat", r"osnovat"],
    "pravni_akt": [r"\bzakon\b", r"\bpravilnik\b", r"\bpropis", r"\bodluka\b", r"\bkonvenc"],
    "registar": [r"\bregistar\b", r"\blista\b", r"\bspisak\b", r"\bizvod\b", r"evidenc"],
    "termin": [r"\btermin\b", r"\braspored\b", r"\bkada\b", r"\bdatum\b", r"objavljen"],
}

DOCUMENT_TYPE_PATTERNS: Dict[str, List[str]] = {
    "zakon": [r"\bzakon\b", r"zakona", r"zakons"],
    "pravilnik": [r"\bpravilnik\b", r"pravilnika"],
    "obrazac": [r"\bobrazac\b", r"formular", r"prijavni obrazac"],
    "zahtjev": [r"\bzahtjev\b", r"zahtjeva", r"molba"],
    "rjesenje": [r"\brjesenje\b", r"rjesenja"],
    "registar": [r"\bregistar\b", r"evidencija", r"lista", r"spisak"],
    "izvod": [r"\bizvod\b"],
    "takse": [r"\btaks", r"naknada", r"uplata", r"iznos"],
    "kontakt": [r"telefon", r"email", r"e-mail", r"adresa", r"kontakt"],
}

PROCEDURE_PATTERNS: Dict[str, List[str]] = {
    "registracija_udruzenja": [r"registr.*udruzenj", r"osniv.*udruzenj", r"upis.*udruzenj", r"\bnvo\b", r"nevladin"],
    "registracija_fondacije": [r"registr.*fondacij", r"osniv.*fondacij", r"upis.*fondacij"],
    "promjena_podataka": [r"promjen.*podat", r"izmjen.*podat", r"promjen.*adres", r"promjen.*zastup", r"preregistr"],
    "brisanje_iz_registra": [r"bris.*regist", r"prestanak rada", r"likvidacij", r"odjav"],
    "pravosudni_ispit": [r"pravosudni\s+ispit", r"polag.*pravosud"],
    "strucni_upravni_ispit": [r"strucni\s+upravni\s+ispit", r"upravni\s+ispit"],
    "besplatna_pravna_pomoc": [r"besplat.*pravn.*pomoc", r"pravna pomoc"],
    "medjunarodna_pravna_pomoc": [r"medjunarod.*pravn.*pomoc", r"pravna saradnja", r"predmet.*inostran"],
    "alimentacija": [r"alimentacij", r"izdrzavanje"],
    "otmica_djece": [r"otmic.*djec", r"povrat.*djet", r"hask.*konvenc", r"odvodjenj.*djet"],
    "registar_udruzenja": [r"registar.*udruzenj", r"provjer.*udruzenj", r"izvod.*udruzenj"],
    "registar_notara": [r"registar.*notar", r"lista.*notar", r"spisak.*notar"],
    "registar_tumaca": [r"registar.*tumac", r"sudsk.*tumac", r"ovlasten.*tumac"],
    "zospi": [r"zospi", r"slobod.*pristup.*informacij", r"pristup informacij"],
}

LEGAL_TOPIC_PATTERNS: Dict[str, List[str]] = {
    "udruzenja_i_fondacije": [r"udruzenj", r"fondacij", r"\bnvo\b", r"nevladin"],
    "ispiti": [r"pravosudni ispit", r"strucni upravni ispit", r"\bispit\b"],
    "pravna_pomoc": [r"pravna pomoc", r"alimentacij", r"otmic.*djec", r"medjunarod.*pomoc"],
    "registri": [r"registar", r"notar", r"tumac", r"lista", r"spisak"],
    "zakoni_i_propisi": [r"zakon", r"pravilnik", r"propis", r"konvenc"],
    "kontakt_i_nadleznosti": [r"kontakt", r"nadlezn", r"telefon", r"email", r"adresa", r"sektor"],
}

INSTITUTION_PATTERNS: Dict[str, List[str]] = {
    "ministarstvo_pravde_bih": [r"ministarstvo pravde", r"\bmpr\b", r"ministar"],
    "sud": [r"\bsud\b", r"suda", r"sudski"],
    "notar": [r"notar"],
    "centar_za_socijalni_rad": [r"centar za socijalni rad", r"socijalni rad"],
    "organ_uprave": [r"organ uprave", r"upravni organ", r"institucij"],
}

EXPECTED_PROCEDURE_BY_SUBINTENT = {
    "registracija_udruzenja": "registracija_udruzenja",
    "registracija_fondacije": "registracija_fondacije",
    "promjena_podataka": "promjena_podataka",
    "brisanje_iz_registra": "brisanje_iz_registra",
    "obrazac_registracija": "registracija_udruzenja",
    "obrazac_promjena": "promjena_podataka",
    "obrazac_ispiti": "pravosudni_ispit",
    "obrazac_zospi": "zospi",
    "zakoni_udruzenja": "registracija_udruzenja",
    "zakoni_ispiti": "pravosudni_ispit",
    "zakoni_zospi": "zospi",
    "besplatna_pravna_pomoc": "besplatna_pravna_pomoc",
    "medjunarodna_pravna_pomoc": "medjunarodna_pravna_pomoc",
    "alimentacija": "alimentacija",
    "otmica_djece": "otmica_djece",
    "pravosudni_ispit": "pravosudni_ispit",
    "strucni_upravni_ispit": "strucni_upravni_ispit",
    "kontakt_ispiti": "pravosudni_ispit",
    "termini_ispita": "pravosudni_ispit",
    "registar_udruzenja": "registar_udruzenja",
    "registar_notara": "registar_notara",
    "registar_tumaca": "registar_tumaca",
}


@dataclass
class QueryUnderstandingResult:
    original_query: str
    normalized_query: str
    action_type: str
    document_types: List[str]
    procedures: List[str]
    legal_topics: List[str]
    institutions: List[str]
    keywords: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def normalize_text(text: str) -> str:
    text = text.translate(DIACRITIC_TRANSLATION).lower()
    text = re.sub(r"[^a-z0-9\s\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_pattern_labels(normalized_text: str, patterns: Dict[str, List[str]]) -> List[str]:
    hits: List[str] = []
    for label, regexes in patterns.items():
        if any(re.search(regex, normalized_text) for regex in regexes):
            hits.append(label)
    return hits


def detect_action_type(normalized_text: str) -> str:
    for action_type, regexes in ACTION_TYPE_PATTERNS.items():
        if any(re.search(regex, normalized_text) for regex in regexes):
            return action_type
    return "opce_pitanje"


def extract_keywords(normalized_text: str, max_keywords: int = 8) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", normalized_text)
    candidates = [token for token in tokens if len(token) > 2 and token not in STOPWORDS]
    keywords: List[str] = []
    seen = set()
    for token in candidates:
        if token not in seen:
            keywords.append(token)
            seen.add(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def extract_key_information(query: str) -> QueryUnderstandingResult:
    normalized = normalize_text(query)
    return QueryUnderstandingResult(
        original_query=query,
        normalized_query=normalized,
        action_type=detect_action_type(normalized),
        document_types=_find_pattern_labels(normalized, DOCUMENT_TYPE_PATTERNS),
        procedures=_find_pattern_labels(normalized, PROCEDURE_PATTERNS),
        legal_topics=_find_pattern_labels(normalized, LEGAL_TOPIC_PATTERNS),
        institutions=_find_pattern_labels(normalized, INSTITUTION_PATTERNS),
        keywords=extract_keywords(normalized),
    )


def expected_procedure_from_subintent(subintent: str) -> Optional[str]:
    return EXPECTED_PROCEDURE_BY_SUBINTENT.get(str(subintent))


def evaluate_procedure_extraction(rows: Iterable[Dict[str, object]]) -> Dict[str, float]:
    total = 0
    correct = 0
    no_expected = 0
    for row in rows:
        text = str(row.get("text", ""))
        expected = expected_procedure_from_subintent(str(row.get("subintent", "")))
        if expected is None:
            no_expected += 1
            continue
        total += 1
        extracted = extract_key_information(text).procedures
        if expected in extracted:
            correct += 1
    return {
        "evaluated_examples": float(total),
        "correct_examples": float(correct),
        "skipped_without_weak_label": float(no_expected),
        "procedure_extraction_accuracy": correct / total if total else 0.0,
    }


def demo() -> None:
    examples = [
        "Kako registrovati udruženje građana?",
        "Treba mi obrazac za prijavu pravosudnog ispita.",
        "Koliko košta stručni upravni ispit?",
        "Gdje mogu pronaći registar sudskih tumača?",
        "Kome se obratiti za međunarodnu otmicu djeteta?",
    ]
    for example in examples:
        print(json.dumps(extract_key_information(example).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()
