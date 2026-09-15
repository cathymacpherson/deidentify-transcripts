"""Cheap, deterministic speaker-labelling baselines.

These exist to set the bar. A majority-class labeller scores highly whenever one role dominates,
so no approach is worth its cost until it clearly beats these. No model, no network, no training
data required.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .schemas import Turn

#: First-person markers. Clients narrate their own experience; therapists reflect it back.
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself)\b", re.IGNORECASE)
#: Second-person markers, which run the other way.
_SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Prediction:
    labels: list[str]
    confidence: list[float]


def majority_baseline(turns: Sequence[Turn], *, label: str = "C") -> Prediction:
    """Label everything as one role. The floor — and usually a deceptively high one."""
    return Prediction([label] * len(turns), [0.0] * len(turns))


def turn_features(turn: Turn) -> dict[str, float]:
    """Per-turn features. Cheap, interpretable, and computed without any model."""
    text = turn.text or ""
    words = text.split()
    word_count = len(words) or 1
    return {
        "is_question": 1.0 if text.strip().endswith("?") else 0.0,
        "first_person": len(_FIRST_PERSON.findall(text)) / word_count,
        "second_person": len(_SECOND_PERSON.findall(text)) / word_count,
        "word_count": float(len(words)),
    }


def rule_baseline(
    turns: Sequence[Turn],
    *,
    primary: str = "C",
    secondary: str = "T",
) -> Prediction:
    """Score each turn on question-asking and pronoun use, then smooth over runs.

    Questions and second-person address point to the interviewing role; first-person narrative
    points to the disclosing role. Individually weak, but the smoothing pass exploits the fact
    that speakers hold the floor for several turns at a time.
    """
    raw: list[float] = []
    for turn in turns:
        f = turn_features(turn)
        # Positive favours the secondary (interviewing) role, negative the primary.
        score = (
            1.5 * f["is_question"]
            + 3.0 * f["second_person"]
            - 4.0 * f["first_person"]
            - 0.02 * min(f["word_count"], 40)
        )
        raw.append(score)

    smoothed = _smooth(raw)
    labels = [secondary if s > 0 else primary for s in smoothed]
    confidence = [min(abs(s) / 2.0, 1.0) for s in smoothed]
    return Prediction(labels, confidence)


def _smooth(scores: Sequence[float], *, window: int = 2, weight: float = 0.35) -> list[float]:
    """Blend each turn's score with its neighbours.

    Speakers hold the floor across several consecutive turns, so an isolated turn with no signal
    of its own is better predicted by its surroundings than by itself. This is the cheapest
    possible way to use that structure.
    """
    out = []
    for i, score in enumerate(scores):
        lo, hi = max(0, i - window), min(len(scores), i + window + 1)
        neighbours = [s for j, s in enumerate(scores[lo:hi], start=lo) if j != i]
        context = sum(neighbours) / len(neighbours) if neighbours else 0.0
        out.append((1 - weight) * score + weight * context)
    return out


#: Name -> baseline, for reporting them side by side.
BASELINES: dict[str, Callable[[Sequence[Turn]], Prediction]] = {
    "majority": majority_baseline,
    "rule": rule_baseline,
}
