from __future__ import annotations

import re

from .schemas import PiiSpan, Transcript

_PREFIX = {
    "person_name": "NAME",
    "nickname": "NAME",
    "family_member": "NAME",
    "clinician_name": "NAME",
    "school": "SCHOOL",
    "place": "PLACE",
    "organisation": "ORG",
    "occupation": "OCCUPATION",
    "address": "ADDRESS",
    "postcode": "POSTCODE",
    "date": "DATE",
    "phone": "PHONE",
    "email": "EMAIL",
    "url": "URL",
    "id_number": "ID",
    "social_media_handle": "HANDLE",
    "other": "PII",
}


class NameRegistry:
    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(text.lower().split())

    def token_for(self, span: PiiSpan) -> str:
        key = self._normalise(span.text)
        if key in self.mapping:
            return self.mapping[key]
        prefix = _PREFIX.get(span.pii_type, "PII")
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        token = f"[{prefix}_{self._counts[prefix]}]"
        self.mapping[key] = token
        return token


def redact(transcript: Transcript, spans: list[PiiSpan], registry: NameRegistry) -> Transcript:
    by_turn: dict[int, list[PiiSpan]] = {}
    for span in spans:
        by_turn.setdefault(span.turn_id, []).append(span)

    for turn in transcript.turns:
        text = turn.text
        for span in sorted(by_turn.get(turn.turn_id, []), key=lambda item: item.start, reverse=True):
            text = text[:span.start] + (span.token or registry.token_for(span)) + text[span.end:]
        turn.anonymised_text = text
    return transcript


def sweep_registry(transcript: Transcript, registry: NameRegistry) -> Transcript:
    """Redact exact residual occurrences of already-known PII surfaces across every turn.

    Used to auto-correct a residual identifier confirmed by the gate: once it's registered with
    a token, this closes the loop so the same surface is also masked anywhere else it appears,
    not just in the turn the gate flagged it in.
    """
    surfaces = sorted(registry.mapping.items(), key=lambda item: len(item[0]), reverse=True)
    for turn in transcript.turns:
        text = turn.anonymised_text
        for surface, token in surfaces:
            if not surface:
                continue
            pattern = re.compile(rf"(?<![\w\[])({re.escape(surface)})(?![\w\]])", re.IGNORECASE)
            text = pattern.sub(token, text)
        turn.anonymised_text = text
    return transcript

