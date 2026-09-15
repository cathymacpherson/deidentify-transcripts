"""LLM speaker labelling: a careful anchor pass, then an overlapping window pass.

Pass 1 marks only turns that are unmistakable, which fixes which role is which and stops the
transcript being labelled consistently backwards. Pass 2 labels every turn in overlapping windows,
so each turn is judged several times against different surrounding context; where those judgements
disagree, the turn is uncertain. That disagreement is the confidence signal — a measurement of the
model's stability, not a number it reported about itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from .schemas import Turn

#: Wire vocabulary. Mapped to the project's own role names at the edge.
WireRole = Literal["C", "T", "other", "unclear"]


class AnchorLabel(BaseModel):
    turn_id: int
    speaker: Literal["C", "T", "other"]
    reason: str = ""


class Anchors(BaseModel):
    anchors: list[AnchorLabel] = Field(default_factory=list)


class WindowLabel(BaseModel):
    turn_id: int
    speaker: WireRole


class WindowLabels(BaseModel):
    labels: list[WindowLabel] = Field(default_factory=list)


ANCHOR_SYSTEM = (
    "You are labelling a therapy transcript. Each line is one turn, numbered. Identify ONLY the "
    "turns whose speaker is beyond any doubt.\n"
    "C = the client (the person seeking help). T = the therapist (the practitioner). "
    "other = someone who is neither, such as a receptionist or research assistant setting up "
    "or ending the session.\n"
    "Mark a turn as T only for something a therapist alone would say: structuring the session "
    "('we're coming to the end of our time'), reflecting the client's words back, psychoeducation, "
    "or asking about the client's experience.\n"
    "Mark a turn as C only for something the client alone would say: sustained first-person "
    "account of their own life, feelings, or history.\n"
    "SKIP anything you are not certain about. Short turns, acknowledgements ('yeah', 'right', "
    "'mm'), and anything that either person could plausibly say must be left out. Returning few "
    "anchors is correct; returning a wrong one is not. Aim for quality, not coverage.\n"
    "Return only the turns you are certain about, with their exact turn numbers."
)

WINDOW_SYSTEM = (
    "You are labelling a therapy transcript. Each line is one turn, numbered, from one continuous "
    "stretch of the conversation.\n"
    "C = the client (the person seeking help). T = the therapist (the practitioner). "
    "other = someone who is neither. unclear = the text genuinely does not show who spoke.\n"
    "Some turns are already labelled and marked FIXED. Those are correct: do not change them, and "
    "use them to work out who is speaking around them.\n"
    "Label EVERY turn in the window, including the fixed ones, returning each turn's exact number. "
    "Use the flow of the conversation: a question is usually followed by that person's answer, and "
    "a speaker normally continues for several turns before the other replies. A short "
    "acknowledgement usually belongs to whoever is listening, not whoever is telling the story.\n"
    "Use 'unclear' only when the surrounding conversation genuinely does not settle it. Do not "
    "guess to avoid saying unclear, and do not use unclear to avoid a judgement the context "
    "supports."
)


def format_window(
    turns: Sequence[Turn], known: dict[int, str], *, mark_fixed: bool = True
) -> str:
    """Render turns for the model: numbered, with any already-known labels marked FIXED."""
    lines = []
    for turn in turns:
        label = known.get(turn.turn_id)
        if label and mark_fixed:
            prefix = f"[{turn.turn_id}] {label} FIXED:"
        elif label:
            prefix = f"[{turn.turn_id}] {label}:"
        else:
            prefix = f"[{turn.turn_id}] ?:"
        lines.append(f"{prefix} {turn.text}")
    return "\n".join(lines)


def windows(total: int, *, size: int, step: int) -> list[tuple[int, int]]:
    """Start/end index pairs covering ``total`` turns, overlapping by ``size - step``.

    Every turn must appear in several windows, because the disagreement between those views is
    the entire confidence signal — a turn seen once has no signal at all.

    Plain sliding windows fail at the edges: the first and last few turns fall in only one window.
    Shorter windows anchored at each end fix that, and give genuinely different context rather
    than a repeat of the same view. The opening of a session matters here, since that is where
    third-party and framing turns appear.
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")
    if total <= 0:
        return []
    if total <= size:
        return [(0, total)]

    spans = []
    start = 0
    while start + size < total:
        spans.append((start, start + size))
        start += step
    spans.append((total - size, total))

    for fraction in (0.4, 0.7):
        length = max(2, int(size * fraction))
        if length < total:
            spans.append((0, length))
            spans.append((total - length, total))
    return sorted(set(spans))


@dataclass
class TurnVotes:
    turn_id: int
    votes: list[str] = field(default_factory=list)
    anchor: str | None = None

    @property
    def winner(self) -> str:
        """Most-voted label. An anchor wins outright — pass 1 is deliberately conservative."""
        if self.anchor:
            return self.anchor
        if not self.votes:
            return "unclear"
        counts: dict[str, int] = {}
        for vote in self.votes:
            counts[vote] = counts.get(vote, 0) + 1
        best = max(counts.values())
        # Ties resolve to unclear rather than to whichever label sorts first.
        winners = [label for label, n in counts.items() if n == best]
        return winners[0] if len(winners) == 1 else "unclear"

    @property
    def agreement(self) -> float:
        """Share of votes going to the winner — the confidence signal.

        An anchor does NOT get free confidence. Pass 1 can be wrong, and a wrong anchor pinned at
        maximum confidence is the worst possible failure: an error the reviewer is told to trust.
        Where the windows contradict an anchor, confidence falls accordingly.
        """
        if not self.votes:
            return 1.0 if self.anchor else 0.0
        winner = self.winner
        share = self.votes.count(winner) / len(self.votes)
        if self.anchor and self.contradicts_anchor:
            # The windows saw something the anchor pass did not. Trust neither.
            return min(share, self.votes.count(self.anchor) / len(self.votes))
        return share

    @property
    def contradicts_anchor(self) -> bool:
        """True when the windows disagreed with a pass-1 anchor — worth a human's attention."""
        if not self.anchor or not self.votes:
            return False
        return any(vote != self.anchor for vote in self.votes)


@dataclass
class LabelledTranscript:
    labels: list[str]
    confidence: list[float]
    votes: list[TurnVotes]
    anchor_count: int
    window_count: int
    manual_count: int = 0
    preserved_count: int = 0
    labels_expected: int = 0
    labels_returned: int = 0

    @property
    def unclear_count(self) -> int:
        return sum(1 for label in self.labels if label == "unclear")

    @property
    def coverage(self) -> float:
        """Share of requested labels the model actually returned. Below 1.0 means dropped turns."""
        if not self.labels_expected:
            return 1.0
        return self.labels_returned / self.labels_expected

    @property
    def min_views(self) -> int:
        return min((len(v.votes) for v in self.votes), default=0)

    @property
    def unvoted_turns(self) -> int:
        """Turns no window ever labelled — no signal at all."""
        return sum(1 for v in self.votes if not v.votes)


def anchor_pass(
    turns: Sequence[Turn],
    model_call: Callable[..., Anchors],
    *,
    batch: int = 120,
) -> dict[int, str]:
    """Pass 1. Returns turn_id -> label for turns the model is certain about."""
    found: dict[int, str] = {}
    valid = {turn.turn_id for turn in turns}
    for start in range(0, len(turns), batch):
        chunk = turns[start : start + batch]
        result = model_call(
            system=ANCHOR_SYSTEM,
            text=format_window(chunk, {}, mark_fixed=False),
            output_type=Anchors,
        )
        for item in result.anchors:
            # The model can return a turn number that is not in the transcript; ignore rather
            # than trust, since a bad anchor propagates through everything built on it.
            if item.turn_id in valid:
                found[item.turn_id] = item.speaker
    return found


def window_pass(
    turns: Sequence[Turn],
    anchors: dict[int, str],
    model_call: Callable[..., WindowLabels],
    *,
    size: int = 40,
    step: int = 13,
) -> tuple[list[TurnVotes], int, int]:
    """Pass 2. Each turn is labelled once per window it appears in.

    Also returns how many labels were asked for and how many came back. A model given a long
    window may silently return fewer labels than it was shown — dropping turns, renumbering, or
    losing track near the end. Those turns quietly lose a vote and their confidence degrades with
    no error raised, so the shortfall has to be measured rather than assumed absent.
    """
    votes = {turn.turn_id: TurnVotes(turn_id=turn.turn_id, anchor=anchors.get(turn.turn_id))
             for turn in turns}
    expected = returned = 0
    for start, end in windows(len(turns), size=size, step=step):
        chunk = turns[start:end]
        allowed = {turn.turn_id for turn in chunk}
        expected += len(chunk)
        result = model_call(
            system=WINDOW_SYSTEM,
            text=format_window(chunk, anchors),
            output_type=WindowLabels,
        )
        seen = set()
        for item in result.labels:
            if item.turn_id in allowed and item.turn_id not in seen:
                votes[item.turn_id].votes.append(item.speaker)
                seen.add(item.turn_id)
        returned += len(seen)
    return [votes[turn.turn_id] for turn in turns], expected, returned


def existing_labels(turns: Sequence[Turn], roles: Sequence[str]) -> dict[int, str]:
    """Manual role labels already on the transcript, as turn_id -> canonical label.

    These are human judgements and are treated as fact: never overwritten, and used as anchors,
    which is strictly better than anchors the model guesses at. A partially coded transcript is
    therefore the *easiest* case, not an awkward one.

    Recognition is delegated to ``inventory.classify_speaker`` so there is ONE definition of what
    counts as a role label across the codebase. An earlier version matched exactly against
    ``("C", "T")`` and silently failed on a lowercase ``c``, treating a human's label as an
    unrecognised value.

    The canonical spelling is returned, so a transcript coded with mixed case comes out
    consistent. That is a change of spelling, not of judgement — unlike a mistyped or unrecognised
    label, which is preserved exactly by ``preserved_labels``.
    """
    from .inventory import DEFAULT_VOCABULARY, classify_speaker

    primary, secondary = roles[0], roles[1]
    found: dict[int, str] = {}
    for turn in turns:
        value = (turn.speaker or "").strip()
        if not value:
            continue
        kind = classify_speaker(value, DEFAULT_VOCABULARY)
        if kind == DEFAULT_VOCABULARY.primary_name:
            found[turn.turn_id] = primary
        elif kind == DEFAULT_VOCABULARY.secondary_name:
            found[turn.turn_id] = secondary
    return found


def preserved_labels(turns: Sequence[Turn], roles: Sequence[str]) -> dict[int, str]:
    """Human labels that are neither a known role nor a recognised blank.

    A merged-speaker mark, a mistyped label, a convention this tool does not recognise — a person
    wrote each of these deliberately. Replacing one with a model guess destroys information and
    silently resolves something a human may have marked as unresolvable. They are kept exactly as
    written and flagged for review instead.
    """
    from .inventory import DEFAULT_VOCABULARY, UNLABELLED_VALUES, classify_speaker

    out = {}
    for turn in turns:
        value = (turn.speaker or "").strip()
        if not value or value.casefold() in UNLABELLED_VALUES:
            continue
        # Same recognition rule as existing_labels: anything it accepts as a role is not
        # "unrecognised", whatever its spelling.
        if classify_speaker(value, DEFAULT_VOCABULARY) in (
            DEFAULT_VOCABULARY.primary_name, DEFAULT_VOCABULARY.secondary_name
        ):
            continue
        out[turn.turn_id] = value
    return out


def label_transcript(
    turns: Sequence[Turn],
    anchor_call: Callable[..., Anchors],
    window_call: Callable[..., WindowLabels],
    *,
    size: int = 40,
    step: int = 13,
    anchor_batch: int = 120,
    roles: Sequence[str] = ("C", "T"),
    use_existing: bool = True,
) -> LabelledTranscript:
    """Run both passes and combine them into labels with a confidence per turn.

    With ``use_existing`` (the default, for real labelling), turns that already carry a manual
    label keep it and seed the anchor set.

    Evaluation MUST pass ``use_existing=False``. The gold labels live in the same field, so
    leaving this on hands the model the answers and reports near-perfect accuracy that means
    nothing.
    """
    if not turns:
        return LabelledTranscript([], [], [], 0, 0)
    manual = existing_labels(turns, roles) if use_existing else {}
    preserved = preserved_labels(turns, roles) if use_existing else {}
    anchors = anchor_pass(turns, anchor_call, batch=anchor_batch)
    # A human label always wins over a model anchor.
    anchors.update(manual)
    votes, expected, returned = window_pass(
        turns, anchors, window_call, size=size, step=step
    )
    return LabelledTranscript(
        labels=[
            manual.get(v.turn_id) or preserved.get(v.turn_id) or v.winner for v in votes
        ],
        # Preserved values get zero confidence so they sort to the top of the review list: the
        # label stands, but a human has to say what it means.
        confidence=[
            1.0 if v.turn_id in manual else (0.0 if v.turn_id in preserved else v.agreement)
            for v in votes
        ],
        votes=votes,
        manual_count=len(manual),
        preserved_count=len(preserved),
        anchor_count=len(anchors),
        window_count=len(windows(len(turns), size=size, step=step)),
        labels_expected=expected,
        labels_returned=returned,
    )


#: How much a disagreeing second opinion discounts confidence. Chosen so that a turn with split
#: votes still ranks as more suspicious than a unanimous turn a second system merely disputes,
#: matching the measured error rates for each case.
SECOND_OPINION_PENALTY = 0.85


def apply_second_opinion(
    votes: Sequence[TurnVotes],
    second_labels: Sequence[str],
    *,
    penalty: float = SECOND_OPINION_PENALTY,
) -> list[float]:
    """Adjust per-turn confidence using an independent system's labels.

    The LLM's own votes only detect *instability* — asking one model three times cannot reveal an
    error it makes consistently, and most of its errors are consistent. An independent system
    fails on different turns, so disagreement between the two is evidence the LLM's votes cannot
    provide however many times they are taken.

    The second opinion never changes a label: it is used purely to rank turns for review. A worse
    system overruling a better one would cost accuracy to buy confidence.
    """
    if len(votes) != len(second_labels):
        raise ValueError("need one second-opinion label per turn")
    out = []
    for vote, other in zip(votes, second_labels):
        confidence = vote.agreement
        if other and vote.winner != "unclear" and other != vote.winner:
            confidence *= penalty
        out.append(confidence)
    return out
