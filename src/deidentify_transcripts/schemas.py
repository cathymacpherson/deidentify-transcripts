from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PiiType = Literal[
    "person_name",
    "nickname",
    "family_member",
    "clinician_name",
    "school",
    "place",
    "organisation",
    "address",
    "postcode",
    "date",
    "phone",
    "email",
    "url",
    "id_number",
    "other",
]


class Turn(BaseModel):
    turn_id: int
    speaker: str = "unknown"
    text: str
    anonymised_text: str = ""


class Transcript(BaseModel):
    transcript_id: str
    turns: list[Turn] = Field(default_factory=list)


class PiiMention(BaseModel):
    text: str
    pii_type: PiiType
    confidence: float = 0.9


class PiiMentions(BaseModel):
    mentions: list[PiiMention] = Field(default_factory=list)


class ResidualIdentifiers(BaseModel):
    identifiers: list[str] = Field(default_factory=list)


class PiiSpan(BaseModel):
    turn_id: int
    start: int
    end: int
    text: str
    pii_type: PiiType
    token: str = ""
    confidence: float = 1.0
    source: Literal["llm", "regex"]


class ReviewItem(BaseModel):
    turn_id: int
    text: str
    reason: str
    pii_type: PiiType | None = None
    confidence: float = 1.0


class DeidReport(BaseModel):
    transcript_id: str
    spans: list[PiiSpan] = Field(default_factory=list)
    registry: dict[str, str] = Field(default_factory=dict)
    review_items: list[ReviewItem] = Field(default_factory=list)
    status: Literal["clean", "needs_review"] = "clean"

