"""Per-transcript review lists.

A flat list of the turns the labeller is unsure about: which turn, what it assigned, how confident
it was, why it was flagged, and the text itself. The reviewer has the full transcript, so
surrounding context is not duplicated here — the turn id locates it.

This is a priority list, not a filter. Errors also occur in turns the system was confident about,
and those do not appear here, so the whole transcript still needs reading.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .labelling import TurnVotes
from .schemas import Turn

#: Below this confidence a turn is listed for review.
#:
#: Chosen from measurement, not intuition: on one corpus this listed ~14% of turns and caught ~49%
#: of errors, against ~30% of turns for ~53% under an earlier, blunter scheme. Raising it towards
#: 1.0 lists more turns for modest extra capture; lowering it gives a shorter, denser list.
#: Re-measure on a new corpus with `label-summary --calibration`.
FLAG_THRESHOLD = 0.75

#: Ordered most error-dense first, from measured error rates per reason.
REASONS = [
    "existing label kept - please confirm what it means",
    "no view of this turn",
    "model could not decide",
    "the two passes disagreed",
    "model disagreed with itself across views",
    "independent system disagreed",
    "low confidence",
]


@dataclass(frozen=True)
class ReviewItem:
    turn_id: int
    speaker: str
    confidence: float
    reason: str
    votes: list[str]
    text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "votes": self.votes,
            "text": self.text,
        }


def flag_reason(
    vote: TurnVotes,
    second: str | None = None,
    label: str | None = None,
    roles: Sequence[str] = ("C", "T"),
) -> str:
    """Why this turn needs a look. Order matters — the first match wins."""
    if label is not None and label not in roles and label != "unclear":
        return "existing label kept - please confirm what it means"
    if not vote.votes:
        return "no view of this turn"
    if vote.winner == "unclear":
        return "model could not decide"
    if vote.contradicts_anchor:
        return "the two passes disagreed"
    if len(set(vote.votes)) > 1:
        return "model disagreed with itself across views"
    if second and second != vote.winner:
        return "independent system disagreed"
    return "low confidence"


def build_review(
    transcript_id: str,
    turns: Sequence[Turn],
    labels: Sequence[str],
    votes: Sequence[TurnVotes],
    confidence: Sequence[float],
    second_labels: Sequence[str] | None = None,
    *,
    threshold: float = FLAG_THRESHOLD,
    roles: Sequence[str] = ("C", "T"),
) -> dict[str, object]:
    """Build the review list for one transcript.

    Items are ordered by how error-dense their reason was in testing, then by confidence, so the
    turns most likely to be wrong come first. Consecutive turn ids indicate a run flagged together,
    which is usually one correction rather than several.
    """
    seconds = list(second_labels) if second_labels else [None] * len(turns)
    items: list[ReviewItem] = []
    for i, (turn, vote, conf) in enumerate(zip(turns, votes, confidence)):
        if conf >= threshold:
            continue
        items.append(
            ReviewItem(
                turn_id=turn.turn_id,
                speaker=labels[i] if i < len(labels) else vote.winner,
                confidence=conf,
                reason=flag_reason(
                    vote,
                    seconds[i] if i < len(seconds) else None,
                    labels[i] if i < len(labels) else None,
                    roles,
                ),
                votes=list(vote.votes),
                text=turn.text,
            )
        )

    priority = {reason: n for n, reason in enumerate(REASONS)}
    items.sort(key=lambda it: (priority.get(it.reason, len(REASONS)), it.confidence, it.turn_id))

    return {
        "transcript_id": transcript_id,
        "turns_total": len(turns),
        "turns_flagged": len(items),
        "note": (
            "A priority list, not a filter. Errors also occur in turns not listed here - in "
            "testing roughly a third of them did - so the whole transcript still needs reading. "
            "Turns labelled by hand are never listed. Use turn_id to find each turn in the "
            "transcript for surrounding context."
        ),
        "items": [item.as_dict() for item in items],
    }
