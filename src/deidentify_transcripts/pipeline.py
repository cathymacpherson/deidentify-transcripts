from __future__ import annotations

from collections.abc import Callable

from .detect import propagate_known_identifiers
from .gate import validate
from .redact import NameRegistry, redact, sweep_registry
from .schemas import DeidReport, PiiSpan, RunMetadata, Transcript


def deidentify(
    transcript: Transcript,
    *,
    detect_fn: Callable[[int, str], list[PiiSpan]],
    residual_fn: Callable[[int, str], list[str]],
    low_confidence_threshold: float = 0.5,
    run_metadata: RunMetadata | None = None,
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

    # A residual identifier confirmed by the gate is auto-redacted (registered + swept) rather
    # than always blocking as a review item — it's kept in review_items with an updated reason so
    # the auto-correction stays auditable. Low-confidence stage-1 detections are NOT auto-corrected:
    # those are genuinely uncertain, not a confirmed miss the way a gate residual is.
    gate_residuals = [item for item in review_items if item.reason == "residual PII (gate)"]
    other_items = [item for item in review_items if item.reason != "residual PII (gate)"]
    if gate_residuals:
        for item in gate_residuals:
            span = PiiSpan(
                turn_id=item.turn_id, start=0, end=0, text=item.text,
                pii_type=item.pii_type or "other", confidence=item.confidence, source="llm",
            )
            registry.token_for(span)
            item.reason = "gate: auto-corrected"
        sweep_registry(transcript, registry)
    review_items = other_items + gate_residuals

    report = DeidReport(
        transcript_id=transcript.transcript_id,
        run_metadata=run_metadata,
        spans=spans,
        registry=dict(registry.mapping),
        review_items=review_items,
        status="needs_review" if other_items else "clean",
    )
    return transcript, report
