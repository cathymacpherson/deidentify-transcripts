"""A trained sequence labeller for speaker roles.

Two parts, both learned from labelled transcripts rather than hand-tuned:

* an **emission model** — logistic regression over per-turn features, scoring how much a turn on
  its own looks like each role;
* a **transition model** — how often each role follows each role, which is the corpus's run-length
  structure learned directly instead of approximated by a smoothing constant.

Viterbi decoding then picks the label sequence that best fits both. This is what the rule
baseline's fixed smoothing weight was standing in for, and it is aimed squarely at boundary
detection, the weakest part of that baseline.

Pure Python and dependency-free: training is a few seconds per epoch on a corpus of this size.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .baselines import Prediction
from .schemas import Turn

_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself)\b", re.IGNORECASE)
_SECOND_PERSON = re.compile(r"\b(you|you're|you've|your|yours|yourself)\b", re.IGNORECASE)
_PLURAL = re.compile(r"\b(we|we're|us|our|ours)\b", re.IGNORECASE)
_QUESTION_WORD = re.compile(r"^\s*(what|how|why|when|where|who|which|do|does|did|can|could|would|"
                            r"have|has|are|is|tell)\b", re.IGNORECASE)
_REFLECTIVE = re.compile(
    r"(it sounds like|sounds like|so you|you're saying|i wonder|what i'm hearing|"
    r"it seems like|that makes sense|i hear you)", re.IGNORECASE
)
_PROCESS = re.compile(
    r"(our time|next week|last (week|time|session)|this session|today we|"
    r"before we (finish|end|start)|homework|check in)", re.IGNORECASE
)
_HEDGE = re.compile(r"\b(um|uh|like|kinda|sorta|i guess|i mean|you know)\b", re.IGNORECASE)


def base_features(turn: Turn) -> dict[str, float]:
    """Features of one turn considered alone. Sparse: zero-valued keys are omitted."""
    text = turn.text or ""
    words = text.split()
    n = len(words) or 1
    f: dict[str, float] = {"bias": 1.0}

    if text.rstrip().endswith("?"):
        f["ends_question"] = 1.0
    if _QUESTION_WORD.search(text):
        f["opens_question_word"] = 1.0

    for name, pattern in (
        ("first_person", _FIRST_PERSON),
        ("second_person", _SECOND_PERSON),
        ("plural", _PLURAL),
        ("hedge", _HEDGE),
    ):
        hits = len(pattern.findall(text))
        if hits:
            f[f"{name}_rate"] = hits / n
            f[f"has_{name}"] = 1.0

    if _REFLECTIVE.search(text):
        f["reflective"] = 1.0
    if _PROCESS.search(text):
        f["process_talk"] = 1.0

    # Length, bucketed rather than raw: a linear weight on word count would be dominated by a
    # handful of very long turns.
    if len(words) <= 3:
        f["very_short"] = 1.0
    elif len(words) <= 8:
        f["short"] = 1.0
    elif len(words) >= 20:
        f["long"] = 1.0
    f["log_len"] = math.log1p(len(words)) / 4.0
    return f


#: How far either side to look. Neighbouring turns are what a per-turn rule cannot use.
CONTEXT = 2


def turn_features(turns: Sequence[Turn], index: int) -> dict[str, float]:
    """Features of a turn plus those of its neighbours, and its position in the transcript."""
    f = dict(base_features(turns[index]))
    for offset in range(-CONTEXT, CONTEXT + 1):
        if offset == 0:
            continue
        j = index + offset
        if 0 <= j < len(turns):
            for key, value in base_features(turns[j]).items():
                if key != "bias":
                    f[f"{offset:+d}:{key}"] = value
        else:
            f[f"{offset:+d}:absent"] = 1.0
    f["position"] = index / max(len(turns) - 1, 1)
    return f


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class LogisticModel:
    """Binary logistic regression over sparse named features, trained by SGD."""

    weights: dict[str, float] = field(default_factory=dict)

    def score(self, features: dict[str, float]) -> float:
        return sum(self.weights.get(k, 0.0) * v for k, v in features.items())

    def probability(self, features: dict[str, float]) -> float:
        return _sigmoid(self.score(features))

    def fit(
        self,
        rows: Sequence[dict[str, float]],
        targets: Sequence[int],
        *,
        epochs: int = 12,
        learning_rate: float = 0.25,
        l2: float = 1e-6,
        seed: int = 0,
    ) -> "LogisticModel":
        order = list(range(len(rows)))
        rng = random.Random(seed)
        for epoch in range(epochs):
            rng.shuffle(order)
            rate = learning_rate / (1.0 + epoch)
            for i in order:
                features, target = rows[i], targets[i]
                error = self.probability(features) - target
                for key, value in features.items():
                    current = self.weights.get(key, 0.0)
                    self.weights[key] = current - rate * (error * value + l2 * current)
        return self

    def top_features(self, n: int = 15) -> list[tuple[str, float]]:
        """Largest-magnitude weights — what the model actually keyed on."""
        return sorted(self.weights.items(), key=lambda kv: -abs(kv[1]))[:n]


@dataclass
class SequenceLabeller:
    """Emission model plus learned transitions, decoded with Viterbi."""

    model: LogisticModel
    #: log P(next | current), indexed [current][next]; index 1 is the positive class
    transitions: list[list[float]]
    start: list[float]
    primary: str = "C"
    secondary: str = "T"

    def _emissions(self, turns: Sequence[Turn]) -> list[tuple[float, float]]:
        out = []
        for i in range(len(turns)):
            p = self.model.probability(turn_features(turns, i))
            p = min(max(p, 1e-9), 1 - 1e-9)
            out.append((math.log(1 - p), math.log(p)))
        return out

    def predict(self, turns: Sequence[Turn], *, decode: str = "marginal") -> Prediction:
        """Label a transcript.

        Defaults to ``"marginal"``, which measured better on this corpus for macro-F1, boundary
        F1 and error capture alike.

        ``decode="viterbi"`` picks the single most likely label *sequence*. That maximises
        path likelihood, which on a corpus where most boundaries are "no change" biases towards
        suppressing switches — good for token accuracy, bad for finding changeovers.

        ``decode="marginal"`` labels each turn by its own posterior instead. It ignores path
        consistency but does not inherit that bias, so it usually recovers boundary recall.
        Which is better depends on whether token accuracy or changeover detection matters more.
        """
        if not turns:
            return Prediction([], [])
        emissions = self._emissions(turns)
        if decode == "marginal":
            posteriors = self._posteriors(emissions)
            return Prediction(
                [self.secondary if p > 0.5 else self.primary for p in posteriors],
                [abs(2 * p - 1) for p in posteriors],
            )
        if decode != "viterbi":
            raise ValueError("decode must be 'viterbi' or 'marginal'")

        # Viterbi for the labels: best whole-sequence path, which is what respects run structure.
        best = [[self.start[k] + emissions[0][k] for k in (0, 1)]]
        back: list[list[int]] = [[0, 0]]
        for t in range(1, len(turns)):
            row, pointers = [0.0, 0.0], [0, 0]
            for k in (0, 1):
                options = [best[t - 1][j] + self.transitions[j][k] for j in (0, 1)]
                pointers[k] = 0 if options[0] >= options[1] else 1
                row[k] = options[pointers[k]] + emissions[t][k]
            best.append(row)
            back.append(pointers)

        last = 0 if best[-1][0] >= best[-1][1] else 1
        path = [last]
        for t in range(len(turns) - 1, 0, -1):
            last = back[t][last]
            path.append(last)
        path.reverse()

        posteriors = self._posteriors(emissions)
        labels = [self.secondary if k else self.primary for k in path]
        confidence = [abs(2 * p - 1) for p in posteriors]
        return Prediction(labels, confidence)

    def _posteriors(self, emissions: list[tuple[float, float]]) -> list[float]:
        """Forward-backward marginals, used for confidence rather than for decoding.

        Viterbi gives the best path but no sense of how close the alternative was; the marginal
        probability at each position does, and that is what the review queue needs.
        """
        n = len(emissions)
        forward = [[self.start[k] + emissions[0][k] for k in (0, 1)]]
        for t in range(1, n):
            forward.append([
                _logsumexp([forward[t - 1][j] + self.transitions[j][k] for j in (0, 1)])
                + emissions[t][k]
                for k in (0, 1)
            ])
        backward = [[0.0, 0.0] for _ in range(n)]
        for t in range(n - 2, -1, -1):
            backward[t] = [
                _logsumexp([
                    self.transitions[k][j] + emissions[t + 1][j] + backward[t + 1][j]
                    for j in (0, 1)
                ])
                for k in (0, 1)
            ]
        out = []
        for t in range(n):
            scores = [forward[t][k] + backward[t][k] for k in (0, 1)]
            total = _logsumexp(scores)
            out.append(math.exp(scores[1] - total))
        return out


def _logsumexp(values: Sequence[float]) -> float:
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(sum(math.exp(v - top) for v in values))


def train(
    sequences: Sequence[tuple[Sequence[Turn], Sequence[str]]],
    *,
    primary: str = "C",
    secondary: str = "T",
    epochs: int = 12,
    seed: int = 0,
) -> SequenceLabeller:
    """Fit emissions and transitions from labelled transcripts.

    ``sequences`` pairs each transcript's turns with its gold labels. Turns whose gold label is
    neither role are skipped for emission training but still break the transition chain, so a
    merged or unlabelled turn never invents a transition that did not occur.
    """
    rows: list[dict[str, float]] = []
    targets: list[int] = []
    transition_counts = [[1.0, 1.0], [1.0, 1.0]]  # add-one smoothing
    start_counts = [1.0, 1.0]

    for turns, gold in sequences:
        if len(turns) != len(gold):
            raise ValueError("each sequence needs one gold label per turn")
        previous: int | None = None
        first = True
        for i, (turn, label) in enumerate(zip(turns, gold)):
            if label not in (primary, secondary):
                previous = None  # chain broken: do not fabricate a transition across a gap
                continue
            k = 1 if label == secondary else 0
            rows.append(turn_features(turns, i))
            targets.append(k)
            if first:
                start_counts[k] += 1.0
                first = False
            if previous is not None:
                transition_counts[previous][k] += 1.0
            previous = k

    if not rows:
        raise ValueError("no labelled turns to train on")

    model = LogisticModel().fit(rows, targets, epochs=epochs, seed=seed)
    transitions = [
        [math.log(c / sum(row)) for c in row] for row in transition_counts
    ]
    start = [math.log(c / sum(start_counts)) for c in start_counts]
    return SequenceLabeller(
        model=model, transitions=transitions, start=start,
        primary=primary, secondary=secondary,
    )
