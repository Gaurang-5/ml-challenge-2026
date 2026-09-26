"""Unified text cleaning, Indic squashed transliteration, and field normalization.

Zero external lookups. Operates purely in-memory using Python stdlib + text_unidecode.
"""
from __future__ import annotations

import re
import unicodedata
import text_unidecode

LEGAL_SUFFIXES = frozenset({
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "llc", "llp", "plc", "pvt", "private", "sa", "sarl", "sas",
    "eurl", "sci", "snc", "ei", "sasu", "sarlu", "gie", "earl", "selarl",
    "sca", "scs", "services", "center", "enterprises", "solutions",
    "consulting", "group", "holdings", "associates", "partners", "international"
})

ADDRESS_STOP_WORDS = frozenset({
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "drive",
    "dr", "near", "the", "and", "of", "at", "behind", "opposite", "building",
    "floor", "unit", "block", "sector", "flat", "door", "no", "opp", "nr",
    "bh", "rue", "r", "av", "bd", "boulevard", "all", "allee", "pl", "place",
    "chem", "chemin", "pobox", "po", "box", "hno", "plot", "doorno",
    "de", "la", "le", "les", "du", "des", "d", "l", "cours", "crs", "impasse", "imp",
    "passage", "route", "rte"
})

ADDRESS_EXPANSIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle",
    "pkwy": "parkway", "hwy": "highway", "ter": "terrace", "ste": "suite",
    "apt": "apartment", "bldg": "building", "fl": "floor",
    "opp": "opposite", "nr": "near", "bh": "behind", "col": "colony",
    "sec": "sector", "h.no": "hno", "h-no": "hno", "p.o.": "pobox",
    "r.": "rue", "av.": "avenue", "bd.": "boulevard", "pl.": "place",
    "crs": "cours", "imp": "impasse", "rte": "route", "fbg": "faubourg",
    "r": "rue", "av": "avenue", "bd": "boulevard"
}


def clean_text(s: str) -> str:
    """ASCII transliteration, repeated-character compression, punctuation stripping."""
    if not s:
        return ""
    # Transliterate Unicode scripts (Devanagari, Tamil, Telugu, Kannada, accents)
    s = text_unidecode.unidecode(s).lower()
    # Strip URL domain suffixes (e.g. .com, .org, .co.in, .in)
    s = re.sub(r"\.(com|org|net|in|co|us|gov|io|fr)\b", " ", s)
    # Strip DBA / C/O trading prefixes
    s = re.sub(r"\b(d\.?b\.?a\.?|c/o|t/a)\b", " ", s)
    # Compress repeated consonants and vowels (letters only, preserving house numbers and PIN codes)
    s = re.sub(r"([a-zA-Z])\1+", r"\1", s)
    # Strip non-alphanumeric
    s = re.sub(r"[^\w]+", " ", s).strip()
    return s


def clean_business_name(name: str) -> tuple[str, list[str]]:
    """Returns normalized full name string and list of non-legal tokens."""
    cleaned = clean_text(name)
    tokens = [w for w in cleaned.split() if len(w) >= 2 and w not in LEGAL_SUFFIXES]
    normalized_name = " ".join(tokens)
    return normalized_name, tokens


def clean_address(addr: str) -> tuple[str, list[str], list[str]]:
    """Returns normalized address string, address tokens, and extracted numbers."""
    if not addr or addr.strip().lower() in ("null", "<null>", "none"):
        return "", [], []
    cleaned = clean_text(addr)
    raw_tokens = cleaned.split()
    tokens = []
    for w in raw_tokens:
        exp = ADDRESS_EXPANSIONS.get(w, w)
        if len(exp) >= 2 and exp not in ADDRESS_STOP_WORDS:
            tokens.append(exp)
    numbers = re.findall(r"\d+", cleaned)
    normalized_addr = " ".join(tokens)
    return normalized_addr, tokens, numbers
