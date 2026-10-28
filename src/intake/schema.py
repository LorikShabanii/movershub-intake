"""Pydantic schema for the structured extraction.

Design notes
------------
* Every extracted field carries its own ``confidence`` and a verbatim ``evidence`` span.
  The post-model checks use the evidence span to catch invented values.
* No numeric/string constraints in the schema on purpose: the structured-output API rejects
  ``minimum``/``maximum``/``minLength``, so ranges are validated locally in ``rules.py``.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Lang(str, Enum):
    sq = "sq"
    en = "en"
    other = "other"  # anything else: extraction still runs, clarification questions fall back to English


class TextField(BaseModel):
    value: Optional[str] = Field(
        None, description="The extracted value, or null if the request does not state it. Never guess."
    )
    confidence: float = Field(
        0.0,
        description="How sure you are that value is correct: 0.0 = pure guess, 1.0 = stated explicitly. "
        "Use 0.0 when value is null.",
    )
    evidence: Optional[str] = Field(
        None, description="Short verbatim quote from the request that supports value. Null when value is null."
    )


class IntField(BaseModel):
    value: Optional[int] = Field(None, description="The extracted integer, or null if not stated.")
    confidence: float = Field(0.0, description="0.0 to 1.0. Use 0.0 when value is null.")
    evidence: Optional[str] = Field(None, description="Verbatim quote supporting value, or null.")


class BoolField(BaseModel):
    value: Optional[bool] = Field(None, description="true / false, or null if not stated.")
    confidence: float = Field(0.0, description="0.0 to 1.0. Use 0.0 when value is null.")
    evidence: Optional[str] = Field(None, description="Verbatim quote supporting value, or null.")


class Extraction(BaseModel):
    """Structured view of one incoming moving request."""

    language: Lang = Field(description="Language the request is written in: sq (Albanian), en (English), or other.")
    pickup_address: TextField = Field(description="Where the goods are collected (street, number, city if given).")
    dropoff_address: TextField = Field(description="Where the goods are delivered.")
    move_date: TextField = Field(
        description="Requested moving date as ISO 8601 YYYY-MM-DD. Resolve relative dates ('next Saturday', "
        "'nesër', 'pasnesër') using the today's-date line in the prompt. Null if no date is given."
    )
    home_size: TextField = Field(
        description="Size of the home being moved, as written (e.g. 'studio', '2-room apartment', "
        "'banesë me 2 dhoma', 'shtëpi', 'house')."
    )
    pickup_floor: IntField = Field(description="Floor number at pickup; ground floor = 0.")
    pickup_elevator: BoolField = Field(description="Whether an elevator is available at pickup.")
    dropoff_floor: IntField = Field(description="Floor number at drop-off; ground floor = 0.")
    dropoff_elevator: BoolField = Field(description="Whether an elevator is available at drop-off.")
    special_items: TextField = Field(
        description="Comma-separated list of bulky or delicate items mentioned (piano, safe, aquarium...). "
        "Null if none mentioned."
    )
    contact_name: TextField = Field(description="Customer's name as written.")
    contact_phone: TextField = Field(description="Phone number as written in the request.")
    contact_email: TextField = Field(description="Email address as written in the request.")


# Fields a human needs before a quote can be prepared. Contact needs phone OR email.
REQUIRED_FIELDS = ("pickup_address", "dropoff_address", "move_date", "home_size")
CONTACT_FIELDS = ("contact_phone", "contact_email")
ALL_VALUE_FIELDS = (
    "pickup_address",
    "dropoff_address",
    "move_date",
    "home_size",
    "pickup_floor",
    "pickup_elevator",
    "dropoff_floor",
    "dropoff_elevator",
    "special_items",
    "contact_name",
    "contact_phone",
    "contact_email",
)


def empty_extraction(language: Lang = Lang.other) -> Extraction:
    """An extraction with every field null — used by the mock provider and as a safe fallback."""
    return Extraction(
        language=language,
        **{name: {"value": None, "confidence": 0.0, "evidence": None} for name in ALL_VALUE_FIELDS},
    )
