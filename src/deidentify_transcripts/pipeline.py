from __future__ import annotations

from collections.abc import Callable

from .detect import propagate_known_identifiers
from .gate import validate
from .redact import NameRegistry, redact
from .schemas import DeidReport, PiiSpan, Transcript


def deidentify(
    transcript: Transcript,
    *,
    detect_fn: Callable[[int, str], list[PiiSpan]],
    residual_fn: Callable[[int, str], list[str]],
    low_confidence_threshold: float = 0.5,
) -> tuple[Transcript, DeidReport]:
    """Mutate a transcript's anonymised_text fields and return it with its sensitive report."""
    registry = NameRegistry()
    spans: list[PiiSpan] = []

    for turn in transcript.turns:
        spans.extend(detect_fn(turn.turn_id, turn.text))

    spans = propagate_known_identifiers(
        [(turn.turn_id, turn.text) for turn in transcript.turns],
        spans,
    )
    for span in spans:
        span.token = registry.token_for(span)

    redact(transcript, spans, registry)
    review_items = validate(
        transcript,
        spans,
        low_confidence_threshold=low_confidence_threshold,
        residual_fn=residual_fn,
    )
    report = DeidReport(
        transcript_id=transcript.transcript_id,
        spans=spans,
        registry=dict(registry.mapping),
        review_items=review_items,
        status="needs_review" if review_items else "clean",
    )
    return transcript, report
