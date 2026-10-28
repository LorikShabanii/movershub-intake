"""Clarification questions in the customer's language.

Deterministic templates, not model output: they are testable, cost nothing, and cannot be
steered by whatever the customer wrote. The model only decides *which* fields are missing.

Languages: Albanian and English. Any other language gets the English questions; adding a
language means adding a reviewed set of templates here, nothing else changes.
"""
from __future__ import annotations

from .schema import Lang

_T = {
    "pickup_address": {
        Lang.en: "Which address are we picking up from? Please give the street, number and city.",
        Lang.sq: "Nga cila adresë do të bëhet marrja? Ju lutem na jepni rrugën, numrin dhe qytetin.",
    },
    "dropoff_address": {
        Lang.en: "Which address are we delivering to? Please give the street, number and city.",
        Lang.sq: "Në cilën adresë do të bëhet dorëzimi? Ju lutem na jepni rrugën, numrin dhe qytetin.",
    },
    "move_date": {
        Lang.en: "On which date would you like to move? (for example 15.11.2026)",
        Lang.sq: "Në cilën datë dëshironi të bëni transportin? (p.sh. 15.11.2026)",
    },
    "date_in_past": {
        Lang.en: "The date you mentioned seems to be in the past. Could you confirm the moving date?",
        Lang.sq: "Data që keni shënuar duket se ka kaluar. A mund ta konfirmoni datën e transportit?",
    },
    "home_size": {
        Lang.en: "How big is the home we are moving? (studio, 1-room, 2-room, 3-room, house)",
        Lang.sq: "Sa e madhe është banesa që po transportojmë? (garsonierë, 1 dhomë, 2 dhoma, 3 dhoma, shtëpi)",
    },
    "contact": {
        Lang.en: "How can we reach you? A phone number or email address is enough.",
        Lang.sq: "Si mund t'ju kontaktojmë? Mjafton një numër telefoni ose një email.",
    },
}

_OPENING = {
    Lang.en: "Thank you for your request. To prepare a quote we still need a few details:",
    Lang.sq: "Faleminderit për kërkesën tuaj. Për të përgatitur një ofertë na duhen edhe disa detaje:",
}


def question(key: str, lang: Lang) -> str:
    table = _T[key]
    return table.get(lang, table[Lang.en])


def opening(lang: Lang) -> str:
    return _OPENING.get(lang, _OPENING[Lang.en])


def build_clarification(missing: list[str], lang: Lang) -> list[str]:
    """Ordered questions for the given missing-field keys, in the customer's language (English fallback)."""
    return [question(k, lang) for k in missing if k in _T]
