"""The intake pipeline: pre-checks -> model -> post-checks -> decision.

Every result ends in front of a person. The decision only says *which* queue it lands in and what,
if anything, we should ask the customer first. Nothing is sent to the customer automatically.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

from . import rules
from .clarify import build_clarification
from .metering import Stopwatch, Usage
from .prompts import SYSTEM_PROMPT, build_user_message
from .providers import LLMProvider
from .schema import CONTACT_FIELDS, REQUIRED_FIELDS, Extraction, Lang

CONFIDENCE_THRESHOLD = 0.7


class Decision(str, Enum):
    ESCALATE_EMERGENCY = "escalate_emergency"  # a person must look now; model not called
    BLOCKED = "blocked"  # cargo we do not move; model not called
    NEEDS_CLARIFICATION = "needs_clarification"  # ask the customer, then review
    READY_FOR_REVIEW = "ready_for_review"  # complete enough for a person to prepare a quote


@dataclass
class IntakeResult:
    request_id: str
    decision: Decision
    language: str
    extraction: Optional[Extraction]
    flags: list[rules.Flag]
    missing: list[str]
    clarification: list[str]
    usage: Usage
    requires_human: bool = True  # always; kept explicit so nobody "optimises" it away

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decision"] = self.decision.value
        d["extraction"] = self.extraction.model_dump(mode="json") if self.extraction else None
        return d


def _request_id(text: str) -> str:
    """Stable id that lets us correlate logs and test results without storing the raw text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _missing_fields(ex: Extraction) -> list[str]:
    missing = [n for n in REQUIRED_FIELDS if getattr(ex, n).value is None or getattr(ex, n).confidence < CONFIDENCE_THRESHOLD]
    has_contact = any(
        getattr(ex, n).value is not None and getattr(ex, n).confidence >= CONFIDENCE_THRESHOLD for n in CONTACT_FIELDS
    )
    if not has_contact:
        missing.append("contact")
    return missing


def run(text: str, provider: LLMProvider, today: Optional[date] = None) -> IntakeResult:
    today = today or date.today()
    rid = _request_id(text)
    usage = Usage()

    # ---- 1. pre-model checks: can stop before spending a token
    pre = rules.pre_checks(text)
    codes = {f.code for f in pre}
    lang_guess = rules.guess_language(text)
    if "emergency" in codes:
        return IntakeResult(rid, Decision.ESCALATE_EMERGENCY, lang_guess.value, None, pre, [], [], usage)
    if "blocked_topic" in codes:
        return IntakeResult(rid, Decision.BLOCKED, lang_guess.value, None, pre, [], [], usage)
    if "input_too_long" in codes or "input_empty" in codes:
        return IntakeResult(rid, Decision.NEEDS_CLARIFICATION, lang_guess.value, None, pre, [], [], usage)

    # ---- 2. model: structured extraction with confidence per field
    with Stopwatch() as sw:
        resp = provider.extract(SYSTEM_PROMPT, build_user_message(text, today))
    extra = {"stop_reason": resp.stop_reason}
    if resp.note:
        extra["note"] = resp.note
    usage = Usage(resp.model, resp.input_tokens, resp.output_tokens, latency_ms=sw.ms, model_called=True,
                  extra=extra).fill_cost()
    ex = resp.extraction

    # ---- 3. post-model checks: demote anything the text does not support
    post = rules.post_checks(ex, text, today)
    flags = pre + post

    # ---- 4. decision + clarification in the customer's language
    missing = _missing_fields(ex)
    if any(f.code == "date_in_past" for f in post) and "move_date" in missing:
        missing[missing.index("move_date")] = "date_in_past"
    lang = ex.language if ex.language is not Lang.other else lang_guess
    clarification = build_clarification(missing, lang)
    decision = Decision.NEEDS_CLARIFICATION if missing else Decision.READY_FOR_REVIEW
    return IntakeResult(rid, decision, ex.language.value, ex, flags, missing, clarification, usage)
