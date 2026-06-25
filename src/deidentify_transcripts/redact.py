from __future__ import annotations

from .schemas import PiiSpan, Transcript

_PREFIX = {
    "person_name": "NAME",
    "nickname": "NAME",
    "family_member": "NAME",
    "clinician_name": "NAME",
    "school": "SCHOOL",
    "place": "PLACE",
    "organisation": "ORG",
    "address": "ADDRESS",
    "postcode": "POSTCODE",
    "date": "DATE",
    "phone": "PHONE",
    "email": "EMAIL",
    "url": "URL",
    "id_number": "ID",
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

