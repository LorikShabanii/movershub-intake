"""Rule-based checks that run BEFORE and AFTER the model.

Pre-model checks look only at the raw text and can stop the pipeline before any tokens are spent
(emergencies, blocked cargo) or mark the text as untrusted (prompt-injection patterns).

Post-model checks look at the model's output and demote anything the text does not support.
They never raise confidence, only lower it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Optional

from .schema import ALL_VALUE_FIELDS, Extraction, Lang

MAX_INPUT_CHARS = 4000
MAX_DAYS_AHEAD = 365


@dataclass(frozen=True)
class Flag:
    code: str  # machine-readable, e.g. "emergency", "date_in_past"
    stage: str  # "pre" or "post"
    detail: str  # human-readable, shown to the reviewer


# --------------------------------------------------------------------------- pre-model
EMERGENCY_TERMS = {
    "en": ["fire", "injured", "hurt", "ambulance", "bleeding", "police", "emergency"],
    # 'gjak' (blood) alone is NOT in this list: it matched the city Gjakovë and escalated normal requests.
    "sq": ["zjarr", "i lënduar", "e lënduar", "ambulanc", "rrjedh gjak", "gjakderdhje", "polici", "urgjenc"],
}

BLOCKED_TERMS = {
    "weapons": ["gun", "rifle", "pistol", "ammunition", "armë", "pushkë", "pistolet", "municion"],
    "drugs": ["drugs", "cocaine", "heroin", "drogë", "kokain"],
    "hazardous": [
        "gas cylinder", "gas cylinders", "propane", "explosive", "fireworks", "fuel cans",
        "bombol", "eksploziv", "fishekzjarr", "karburant",
    ],
}

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |the )?(previous|prior|above) ",
    r"(system|developer) prompt",
    r"you are now (a|an|the) ",
    r"(print|reveal|show|output|repeat) (your|the) (instructions|prompt|rules|configuration)",
    r"</?\s*(system|assistant|instructions?)\s*>",
    r"\bassistant\s*:",
    r"\b(for|to|dear|hey|attention) the (ai|assistant|model|bot|llm|system)\b",
    r"\bif you are an? (ai|assistant|model|bot|llm)\b",
    r"injoro (të gjitha )?(udhëzimet|rregullat)",
    r"(shfaq|trego|printo) (promptin|udhëzimet|rregullat)",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def _find_terms(text: str, terms: Iterable[str]) -> Optional[str]:
    """Word-start match so 'begun' never trips 'gun', while 'bombol' still matches 'bombola'."""
    low = text.lower()
    for term in terms:
        if re.search(r"\b" + re.escape(term), low):
            return term
    return None


def check_length(text: str) -> Optional[Flag]:
    if len(text) > MAX_INPUT_CHARS:
        return Flag("input_too_long", "pre", f"Request is {len(text)} characters; limit is {MAX_INPUT_CHARS}.")
    if not text.strip():
        return Flag("input_empty", "pre", "Request is empty.")
    return None


def detect_emergency(text: str) -> Optional[Flag]:
    for lang, terms in EMERGENCY_TERMS.items():
        hit = _find_terms(text, terms)
        if hit:
            return Flag("emergency", "pre", f"Possible emergency ({lang}: '{hit}'). Route to a person immediately.")
    return None


def detect_blocked_topics(text: str) -> Optional[Flag]:
    for category, terms in BLOCKED_TERMS.items():
        hit = _find_terms(text, terms)
        if hit:
            return Flag("blocked_topic", "pre", f"Blocked cargo category '{category}' ('{hit}').")
    return None


def detect_injection(text: str) -> Optional[Flag]:
    for rx in _INJECTION_RE:
        m = rx.search(text)
        if m:
            return Flag("prompt_injection", "pre", f"Instruction-like text in customer input: '{m.group(0)}'.")
    return None


def pre_checks(text: str) -> list[Flag]:
    flags = []
    for fn in (check_length, detect_emergency, detect_blocked_topics, detect_injection):
        f = fn(text)
        if f:
            flags.append(f)
    return flags


# --------------------------------------------------------------------------- language heuristic
_STOPWORDS = {
    Lang.sq: {"dhe", "për", "nga", "në", "jam", "dua", "shtëpi", "banesë", "banesën", "transport", "kati", "ju", "lutem", "më"},
    Lang.en: {"and", "for", "from", "to", "the", "we", "apartment", "move", "would", "please", "floor", "need", "room"},
}
_WORD_RE = re.compile(r"[a-zA-Zëç]+")


def guess_language(text: str) -> Lang:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    scores = {lang: sum(1 for w in words if w in sw) for lang, sw in _STOPWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else Lang.other


# --------------------------------------------------------------------------- post-model
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


_PHONE_RE = re.compile(r"\d")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _demote(field, flags: list[Flag], code: str, detail: str, factor: float = 0.0) -> None:
    field.confidence = round(field.confidence * factor, 3)
    flags.append(Flag(code, "post", detail))


def evidence_check(ex: Extraction, text: str, flags: list[Flag]) -> None:
    """A value whose evidence span is not actually in the text gets confidence 0 (hallucination guard)."""
    norm_text = _norm(text)
    for name in ALL_VALUE_FIELDS:
        f = getattr(ex, name)
        if f.value is None:
            f.confidence = 0.0
            continue
        if not f.evidence:
            _demote(f, flags, "missing_evidence", f"{name}: value without supporting quote.", factor=0.5)
            continue
        if _norm(f.evidence) not in norm_text:
            _demote(f, flags, "unsupported_evidence", f"{name}: quote '{f.evidence}' not found in request.")


def date_check(ex: Extraction, today: date, flags: list[Flag]) -> None:
    f = ex.move_date
    if f.value is None:
        return
    try:
        d = date.fromisoformat(f.value)
    except ValueError:
        _demote(f, flags, "date_unparseable", f"move_date '{f.value}' is not ISO YYYY-MM-DD.")
        return
    if d < today:
        _demote(f, flags, "date_in_past", f"move_date {d.isoformat()} is before today ({today.isoformat()}).")
    elif d > today + timedelta(days=MAX_DAYS_AHEAD):
        _demote(f, flags, "date_too_far", f"move_date {d.isoformat()} is more than {MAX_DAYS_AHEAD} days ahead.", 0.5)


def contact_check(ex: Extraction, flags: list[Flag]) -> None:
    p = ex.contact_phone
    if p.value is not None:
        digits = len(_PHONE_RE.findall(p.value))
        if not 7 <= digits <= 15:
            _demote(p, flags, "phone_implausible", f"contact_phone '{p.value}' has {digits} digits.")
    e = ex.contact_email
    if e.value is not None and not _EMAIL_RE.match(e.value.strip()):
        _demote(e, flags, "email_implausible", f"contact_email '{e.value}' is not a valid address.")


def floor_check(ex: Extraction, flags: list[Flag]) -> None:
    for name in ("pickup_floor", "dropoff_floor"):
        f = getattr(ex, name)
        if f.value is not None and not 0 <= f.value <= 50:
            _demote(f, flags, "floor_implausible", f"{name}={f.value} is outside 0..50.")


def language_check(ex: Extraction, text: str, flags: list[Flag]) -> None:
    guess = guess_language(text)
    if guess is not Lang.other and ex.language is not guess:
        flags.append(Flag("language_mismatch", "post", f"Model said '{ex.language.value}', heuristic says '{guess.value}'."))


def value_injection_check(ex: Extraction, flags: list[Flag]) -> None:
    """Extracted values must be data, never instructions."""
    for name in ALL_VALUE_FIELDS:
        f = getattr(ex, name)
        if isinstance(f.value, str) and detect_injection(f.value):
            _demote(f, flags, "injection_in_output", f"{name} contains instruction-like text.")


def post_checks(ex: Extraction, text: str, today: date) -> list[Flag]:
    flags: list[Flag] = []
    evidence_check(ex, text, flags)
    date_check(ex, today, flags)
    contact_check(ex, flags)
    floor_check(ex, flags)
    language_check(ex, text, flags)
    value_injection_check(ex, flags)
    return flags
