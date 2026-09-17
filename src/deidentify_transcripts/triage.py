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


#: Confidence bands. Once disagreement strength is weighted in, confidence is effectively
#: continuous, so per-value rates are computed on a handful of turns each and are pure noise.
#: Reporting and calibration both work in bands.
CONFIDENCE_BANDS = [
    (0.0, 0.5), (0.5, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.95), (0.95, 0.999),
    (0.999, 1.01),
]


def band_of(confidence: float) -> tuple[float, float]:
    """The band a confidence value falls in."""
    for low, high in CONFIDENCE_BANDS:
        if low <= confidence < high:
            return low, high
    return CONFIDENCE_BANDS[-1]


def band_label(band: tuple[float, float]) -> str:
    low, high = band
    return "1.000 (settled)" if low >= 0.999 else f"{low:.2f} - {high:.2f}"


def banded_calibration(
    rows: list[dict[str, str]], roles=("C", "T")
) -> dict[tuple[float, float], tuple[int, float]]:
    """Turn count and error rate per confidence band, from a scored run."""
    buckets: dict[tuple[float, float], list[int]] = {}
    for row in rows:
        if row.get("gold") not in roles:
            continue
        try:
            conf = float(row["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        buckets.setdefault(band_of(conf), []).append(1 if row.get("error") == "WRONG" else 0)
    return {b: (len(v), sum(v) / len(v)) for b, v in buckets.items() if v}


def calibration_from_report(rows: list[dict[str, str]], roles=("C", "T")) -> dict[float, float]:
    """Measured error rate per confidence value, from a scored run.

    Turns the abstract confidence number into something checkable: how often turns at each level
    were actually wrong, on this corpus, against human labels.
    """
    buckets: dict[float, list[int]] = {}
    for row in rows:
        if row.get("gold") not in roles:
            continue
        try:
            conf = round(float(row["confidence"]), 3)
        except (KeyError, ValueError):
            continue
        buckets.setdefault(conf, []).append(1 if row.get("error") == "WRONG" else 0)
    return {c: sum(v) / len(v) for c, v in buckets.items() if v}


def expected_errors(
    counts: dict[float, int], calibration: dict[float, float]
) -> list[tuple[float, int, float]]:
    """Pair each confidence level with how many errors it is expected to hold.

    A level absent from the calibration falls back to the nearest measured one, so an unseen
    confidence value still gets a sensible estimate rather than being dropped.
    """
    if not calibration:
        return []
    out = []
    for conf in sorted(counts, reverse=True):
        rate = calibration.get(conf)
        if rate is None:
            nearest = min(calibration, key=lambda c: abs(c - conf))
            rate = calibration[nearest]
        out.append((conf, counts[conf], counts[conf] * rate))
    return out


#: Plain-language description of every column in an annotated audit file, written for the person
#: checking the labels rather than for a developer. Kept beside the code that writes those columns
#: so the two cannot drift apart.
CODEBOOK: list[tuple[str, str, str]] = [
    (
        "check",
        "Whether this turn is worth checking. 'strong' means both automatic systems disagree "
        "with the human label - the best evidence that the coding may be wrong. 'yes' means only "
        "the language model disagrees. Blank means nothing flagged it; most turns are blank.",
        "strong / yes / (blank)",
    ),
    (
        "verdict",
        "Blank, for you to fill in. Suggested use: 'coder' if the original label was wrong, "
        "'model' if the automatic label is wrong, 'unclear' if the text cannot settle it.",
        "(you decide)",
    ),
    ("file", "The transcript this turn comes from.", "e.g. 001A_Full.json"),
    (
        "turn_id",
        "Position of the turn in the transcript, counting from 0. Rows are in this order, so the "
        "turns either side of a row are the rows either side.",
        "0, 1, 2, ...",
    ),
    ("gold", "The speaker label assigned by the human coder.", "C or T"),
    ("predicted", "The speaker label assigned automatically.", "C, T, or unclear"),
    (
        "confidence",
        "How sure the automatic labeller was, from 0 to 1. Combines whether its own repeated "
        "readings agreed and whether the second system objected. 1.000 means everything agreed.",
        "0.000 - 1.000",
    ),
    (
        "second_opinion",
        "The label chosen by a second, independent system - a simple statistical model that "
        "counts word patterns rather than reading the conversation. It never changes the label; "
        "it is only used as a cross-check.",
        "C or T",
    ),
    (
        "second_conf",
        "How sure that second system was, from 0 to 1. A confident objection carries more weight "
        "than a marginal one.",
        "0.000 - 1.000",
    ),
    (
        "anchor",
        "If an early pass judged this turn unmistakable, the label it gave. Blank for most turns.",
        "C, T, or (blank)",
    ),
    (
        "anchor_contradicted",
        "'yes' where a later pass disagreed with that early judgement - unusual, and worth a look.",
        "yes / (blank)",
    ),
    (
        "votes",
        "The label from each separate reading of this turn, separated by |. Each turn is read "
        "three or four times with different surrounding context. 'T|T|T' means every reading "
        "agreed; 'T|T|C' means they differed.",
        "e.g. T|T|T",
    ),
    (
        "error",
        "'WRONG' wherever the automatic label differs from the human one, at any confidence. "
        "Wider than 'check', which only marks the confident disagreements.",
        "WRONG / (blank)",
    ),
    ("text", "The words spoken in this turn.", ""),
]


def write_codebook(path) -> None:
    """Write the column descriptions as a CSV beside the audit files."""
    import csv

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["column", "what it means", "values"])
        writer.writerows(CODEBOOK)
