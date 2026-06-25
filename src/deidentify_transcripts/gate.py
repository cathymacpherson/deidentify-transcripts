from __future__ import annotations

import re
from collections.abc import Callable

from .detect import filter_generic_spans, is_bare_generic_identifier, regex_spans
from .schemas import PiiSpan, ResidualIdentifiers, ReviewItem, Transcript

_TOKEN_RE = re.compile(r"\[[A-Z]+_\d+\]")
_TOKEN_ARTIFACT_RE = re.compile(r"^\[?[A-Z]+_\d+\]?$")

RESIDUAL_SYSTEM = (
    "This transcript turn has already been de-identified with [BRACKETED_TOKENS]. Find any remaining "
    "personally identifying information that is not already a token: names, schools, places, "
    "organisations, addresses and postcodes. Do not return generic words such as school, work, home, "
    "doctor, husband or sister by themselves. Never return an existing token or the inside of a "
    "token, for example [NAME_1], NAME_1, [PLACE_2] or PLACE_2. Return exact remaining identifier "
    "strings, or an empty list if none remain."
)


def make_residual_detector(
    model_call: Callable[..., ResidualIdentifiers],
) -> Callable[[int, str], list[str]]:
    def residual(turn_id: int, text: str) -> list[str]:
        result = model_call(system=RESIDUAL_SYSTEM, text=text, output_type=ResidualIdentifiers)
        masked = _TOKEN_RE.sub(" ", text)
        return list(result.identifiers) + [span.text for span in regex_spans(turn_id, masked)]

    return residual


def validate(
    transcript: Transcript,
    spans: list[PiiSpan],
    *,
    low_confidence_threshold: float,
    residual_fn: Callable[[int, str], list[str]],
) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    seen: set[tuple[int, str, str]] = set()

    for turn in transcript.turns:
        for hit in residual_fn(turn.turn_id, turn.anonymised_text):
            if _TOKEN_ARTIFACT_RE.fullmatch(hit.strip()) or is_bare_generic_identifier(hit):
                continue
            key = (turn.turn_id, hit, "residual PII (gate)")
            if key not in seen:
                seen.add(key)
                items.append(
                    ReviewItem(turn_id=turn.turn_id, text=hit, reason="residual PII (gate)")
                )

    for span in filter_generic_spans(spans):
        if span.confidence < low_confidence_threshold:
            key = (span.turn_id, span.text, "low-confidence detection")
            if key not in seen:
                seen.add(key)
                items.append(
                    ReviewItem(
                        turn_id=span.turn_id,
                        text=span.text,
                        reason="low-confidence detection",
                        pii_type=span.pii_type,
                        confidence=span.confidence,
                    )
                )
    return items
