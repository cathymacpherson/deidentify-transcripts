"""Can a transcript's own confidence predict how wrong it is?

Error rate varies several-fold between transcripts, so treating them all the same wastes effort on
the good ones and under-serves the bad. If the share of low-confidence turns in a file predicts its
error rate, that gives per-file triage using only what the system knows at run time — no gold
labels, so it works on unlabelled transcripts too.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileStats:
    name: str
    turns: int
    flagged: int
    errors: int

    @property
    def flag_rate(self) -> float:
        return self.flagged / self.turns if self.turns else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.turns if self.turns else 0.0


def _ranks(values: list[float]) -> list[float]:
    """Ranks, averaging ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation. Suits triage, where the ordering matters and the scale does not."""
    if len(xs) != len(ys):
        raise ValueError("inputs must be the same length")
    n = len(xs)
    if n < 3:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    mean_x, mean_y = sum(rx) / n, sum(ry) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = sum((a - mean_x) ** 2 for a in rx) ** 0.5
    den_y = sum((b - mean_y) ** 2 for b in ry) ** 0.5
    return 0.0 if den_x == 0 or den_y == 0 else num / (den_x * den_y)


@dataclass(frozen=True)
class TriagePoint:
    files_routed: int
    share_of_files: float
    share_of_turns: float
    share_of_errors: float


def triage_curve(stats: list[FileStats]) -> list[TriagePoint]:
    """Route the worst-looking files to full manual coding; what share of error goes with them?

    Files are ranked by flag rate — which is available without gold labels — and the curve shows
    what proportion of all errors would be covered by routing the top N. A curve well above the
    diagonal means the system knows which transcripts it is handling badly.
    """
    if not stats:
        return []
    ranked = sorted(stats, key=lambda s: -s.flag_rate)
    total_turns = sum(s.turns for s in ranked)
    total_errors = sum(s.errors for s in ranked)
    points, turns, errors = [], 0, 0
    for n, item in enumerate(ranked, start=1):
        turns += item.turns
        errors += item.errors
        points.append(
            TriagePoint(
                files_routed=n,
                share_of_files=n / len(ranked),
                share_of_turns=turns / total_turns if total_turns else 0.0,
                share_of_errors=errors / total_errors if total_errors else 0.0,
            )
        )
    return points
