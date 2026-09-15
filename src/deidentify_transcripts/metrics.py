"""Scoring for speaker labelling.

Deliberately reports several numbers rather than one: with a dominant role, accuracy alone is
close to meaningless, and a system can segment a transcript perfectly while assigning both roles
backwards. Nothing here touches a model or a network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClassScore:
    label: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class Score:
    """Result of comparing predicted labels against gold labels."""

    scored_turns: int
    skipped_turns: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassScore] = field(default_factory=dict)
    boundary_f1: float = 0.0
    boundary_support: int = 0
    flipped_macro_f1: float = 0.0

    @property
    def looks_flipped(self) -> bool:
        """True if swapping the two roles would score materially better.

        A system can group turns into speakers correctly and still assign the roles the wrong way
        round. That failure is catastrophic and invisible in a pooled accuracy figure, so it is
        checked explicitly.
        """
        return self.flipped_macro_f1 > self.macro_f1


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _class_score(label: str, gold: Sequence[str], predicted: Sequence[str]) -> ClassScore:
    true_positive = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
    predicted_positive = sum(1 for p in predicted if p == label)
    actual_positive = sum(1 for g in gold if g == label)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    return ClassScore(label, actual_positive, precision, recall, _f1(precision, recall))


def _boundary_f1(gold: Sequence[str], predicted: Sequence[str]) -> tuple[float, int]:
    """Agreement on *where the speaker changes*, independent of which role is which.

    This separates segmentation quality from role assignment: a transcript labelled with the roles
    inverted still scores perfectly here, which is exactly the point.
    """
    if len(gold) < 2:
        return 0.0, 0
    gold_changes = [a != b for a, b in zip(gold, gold[1:])]
    pred_changes = [a != b for a, b in zip(predicted, predicted[1:])]
    true_positive = sum(1 for g, p in zip(gold_changes, pred_changes) if g and p)
    precision = true_positive / sum(pred_changes) if any(pred_changes) else 0.0
    recall = true_positive / sum(gold_changes) if any(gold_changes) else 0.0
    return _f1(precision, recall), sum(gold_changes)


def score_labels(
    gold: Sequence[str | None],
    predicted: Sequence[str | None],
    *,
    labels: Sequence[str] = ("C", "T"),
) -> Score:
    """Compare predicted against gold, skipping turns with no usable gold label.

    A turn is skipped when its gold label is None or outside ``labels`` — an unlabelled turn, a
    merged-speaker mark, or an out-of-vocabulary value. Those carry no ground truth to score
    against, and including them would penalise correct behaviour.
    """
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must be the same length")

    pairs = [(g, p) for g, p in zip(gold, predicted) if g in labels]
    skipped = len(gold) - len(pairs)
    if not pairs:
        return Score(scored_turns=0, skipped_turns=skipped, accuracy=0.0, macro_f1=0.0)

    gold_kept = [g for g, _ in pairs]
    pred_kept = [p if p is not None else "" for _, p in pairs]

    per_class = {label: _class_score(label, gold_kept, pred_kept) for label in labels}
    macro_f1 = sum(c.f1 for c in per_class.values()) / len(per_class)
    accuracy = sum(1 for g, p in zip(gold_kept, pred_kept) if g == p) / len(pairs)
    boundary, boundary_support = _boundary_f1(gold_kept, pred_kept)

    swap = {labels[0]: labels[1], labels[1]: labels[0]} if len(labels) == 2 else {}
    flipped_pred = [swap.get(p, p) for p in pred_kept]
    flipped_classes = [_class_score(label, gold_kept, flipped_pred) for label in labels]
    flipped_macro = sum(c.f1 for c in flipped_classes) / len(flipped_classes)

    return Score(
        scored_turns=len(pairs),
        skipped_turns=skipped,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        boundary_f1=boundary,
        boundary_support=boundary_support,
        flipped_macro_f1=flipped_macro,
    )


@dataclass(frozen=True)
class BurdenPoint:
    coverage: float      # proportion of turns the system labelled automatically
    reviewed: int        # turns sent to a human
    accuracy: float      # accuracy on the turns it kept


def review_burden_curve(
    gold: Sequence[str | None],
    predicted: Sequence[str | None],
    confidence: Sequence[float],
    *,
    labels: Sequence[str] = ("C", "T"),
    steps: Sequence[float] = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5),
) -> list[BurdenPoint]:
    """Accuracy against how much a human has to check.

    The system is allowed to abstain, so the deliverable is an operating point rather than a single
    score: at each coverage level, how accurate are the turns it kept? Turns are given up
    least-confident first.
    """
    triples = [
        (g, p, c) for g, p, c in zip(gold, predicted, confidence) if g in labels
    ]
    if not triples:
        return []
    ordered = sorted(triples, key=lambda t: t[2], reverse=True)
    points = []
    for coverage in steps:
        keep = max(1, round(len(ordered) * coverage))
        kept = ordered[:keep]
        correct = sum(1 for g, p, _ in kept if g == p)
        points.append(
            BurdenPoint(
                coverage=keep / len(ordered),
                reviewed=len(ordered) - keep,
                accuracy=correct / keep,
            )
        )
    return points


@dataclass(frozen=True)
class ReviewEfficiency:
    """How much a reviewer gains for a given amount of checking.

    When a human reviews everything anyway, accuracy is the wrong headline. What matters is
    whether the system's mistakes end up in the part the reviewer is looking hardest at, and how
    many mistakes survive in the part they skim.
    """

    budget: float             # fraction of turns flagged for close review
    reviewed: int             # turns flagged
    errors_total: int
    errors_caught: int        # errors inside the flagged set
    errors_missed: int        # errors the reviewer must catch unaided
    error_capture: float      # caught / total — the number to optimise
    unflagged_error_rate: float   # missed errors per unflagged turn


def error_capture_curve(
    gold: Sequence[str | None],
    predicted: Sequence[str | None],
    confidence: Sequence[float],
    *,
    labels: Sequence[str] = ("C", "T"),
    budgets: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> list[ReviewEfficiency]:
    """At each review budget, what share of the system's errors are flagged?

    Turns are flagged least-confident first. A system whose confidence is informative concentrates
    its errors in a small flagged set; one whose confidence is noise scatters them, and the curve
    stays close to the diagonal.
    """
    triples = [(g, p, c) for g, p, c in zip(gold, predicted, confidence) if g in labels]
    if not triples:
        return []
    ordered = sorted(triples, key=lambda t: t[2])  # least confident first
    errors_total = sum(1 for g, p, _ in ordered if g != p)

    out = []
    for budget in budgets:
        reviewed = max(1, round(len(ordered) * budget))
        flagged, rest = ordered[:reviewed], ordered[reviewed:]
        caught = sum(1 for g, p, _ in flagged if g != p)
        missed = errors_total - caught
        out.append(
            ReviewEfficiency(
                budget=reviewed / len(ordered),
                reviewed=reviewed,
                errors_total=errors_total,
                errors_caught=caught,
                errors_missed=missed,
                error_capture=caught / errors_total if errors_total else 1.0,
                unflagged_error_rate=missed / len(rest) if rest else 0.0,
            )
        )
    return out


def edit_actions(
    gold: Sequence[str | None],
    predicted: Sequence[str | None],
    *,
    labels: Sequence[str] = ("C", "T"),
) -> int:
    """Contiguous blocks of wrong labels — a proxy for corrections the reviewer must make.

    Counting wrong *turns* overstates the work: a whole run mislabelled in one block is a single
    correction, whereas the same number of errors scattered individually is many. Two systems with
    identical accuracy can differ several-fold in what they actually cost a reviewer.
    """
    pairs = [(g, p) for g, p in zip(gold, predicted) if g in labels]
    blocks = 0
    in_block = False
    for g, p in pairs:
        if g != p:
            if not in_block:
                blocks += 1
                in_block = True
        else:
            in_block = False
    return blocks
