"""Classify transcripts by whether they already carry manual C/T speaker labels.

This is a deterministic metadata check: it reads each transcript's ``speaker`` fields and
counts which label values appear. No model, no network, and no transcript text is inspected,
reported, or written anywhere.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .io import load_transcript

#: Default speaker values counted as a manual role label, normalised casefold-and-stripped.
#: These defaults suit a clinical transcript coded C/T; a project using another scheme (for
#: example interviewer/participant) overrides them via the environment - see RoleVocabulary.
CLIENT_LABELS = frozenset({"c", "client"})
THERAPIST_LABELS = frozenset({"t", "th", "therapist", "clinician", "counsellor", "counselor"})

#: Values meaning "no label was recorded", rather than a label someone chose.
UNLABELLED_VALUES = {"", "unknown", "none", "n/a", "na", "-"}

TRANSCRIPT_SUFFIXES = (".txt", ".json", ".xlsx", ".xls")

#: Separators an annotator might use to mark a turn covering more than one speaker,
#: e.g. "C/T", "C+T", "C & T", "C and T".
_ROLE_SEPARATOR = re.compile(r"\s*(?:[/\\&+,;]|\band\b)\s*", re.IGNORECASE)

#: Fraction of turns that must carry a C/T label for a transcript to count as labelled.
DEFAULT_THRESHOLD = 0.5

#: Proportion of *unrecognised* speaker values a transcript may contain before the file itself is
#: treated as a vocabulary problem. Merged-speaker marks such as "C/T" are deliberately excluded:
#: they are a known annotation with defined per-turn handling, whereas an unrecognised value means
#: the file's vocabulary is not understood at all.
DEFAULT_ANOMALY_TOLERANCE = 0.02

@dataclass(frozen=True)
class RoleVocabulary:
    """Which raw speaker values count as which role.

    ``primary`` and ``secondary`` are the two conversational roles. The names are deliberately
    neutral at this layer: nothing here assumes a clinical setting. The defaults are the
    client/therapist scheme; ``from_env`` lets a project substitute its own without code changes.
    """

    primary: frozenset[str] = CLIENT_LABELS
    secondary: frozenset[str] = THERAPIST_LABELS
    primary_name: str = "client"
    secondary_name: str = "therapist"

    @classmethod
    def from_env(cls) -> "RoleVocabulary":
        """Read overrides from DEID_PRIMARY_LABELS / DEID_SECONDARY_LABELS (comma separated)."""
        import os

        def parse(name: str, fallback: frozenset[str]) -> frozenset[str]:
            raw = os.getenv(name)
            if not raw:
                return fallback
            return frozenset(part.strip().casefold() for part in raw.split(",") if part.strip())

        return cls(
            primary=parse("DEID_PRIMARY_LABELS", CLIENT_LABELS),
            secondary=parse("DEID_SECONDARY_LABELS", THERAPIST_LABELS),
            primary_name=os.getenv("DEID_PRIMARY_NAME", "client"),
            secondary_name=os.getenv("DEID_SECONDARY_NAME", "therapist"),
        )


DEFAULT_VOCABULARY = RoleVocabulary()

Bucket = str
LABELLED: Bucket = "labelled"
PARTIAL: Bucket = "partial"
UNLABELLED: Bucket = "unlabelled"
REVIEW: Bucket = "review"


def split_roles(value: str, vocabulary: RoleVocabulary = DEFAULT_VOCABULARY) -> list[str]:
    """Split a speaker value on multi-speaker separators and return the recognised roles.

    Returns the role kinds found, in order, e.g. ``["client", "therapist"]`` for "C/T".
    A value that does not split, or whose parts are not all recognised roles, returns
    fewer than two entries and is not treated as a merged turn.
    """
    parts = [p for p in _ROLE_SEPARATOR.split(value.strip()) if p]
    if len(parts) < 2:
        return []
    kinds = []
    for part in parts:
        normalised = part.strip().casefold()
        if normalised in vocabulary.primary:
            kinds.append(vocabulary.primary_name)
        elif normalised in vocabulary.secondary:
            kinds.append(vocabulary.secondary_name)
        else:
            return []  # an unrecognised part means this is not a clean multi-role value
    return kinds


def is_multi_role(value: str, vocabulary: RoleVocabulary = DEFAULT_VOCABULARY) -> bool:
    """True if the speaker value marks a turn covering more than one speaker."""
    return len(split_roles(value, vocabulary)) >= 2


def classify_speaker(value: str, vocabulary: RoleVocabulary = DEFAULT_VOCABULARY) -> str:
    """Return the primary role name, the secondary role name, 'unlabelled' or 'other'."""
    normalised = value.strip().casefold()
    if normalised in vocabulary.primary:
        return vocabulary.primary_name
    if normalised in vocabulary.secondary:
        return vocabulary.secondary_name
    if normalised in UNLABELLED_VALUES:
        return "unlabelled"
    return "other"


@dataclass
class TranscriptInventory:
    path: Path
    total_turns: int = 0
    client: int = 0
    therapist: int = 0
    unlabelled: int = 0
    other: int = 0
    merged: int = 0
    other_values: set[str] = field(default_factory=set)
    merged_values: set[str] = field(default_factory=set)
    error: str | None = None

    @property
    def role_labelled(self) -> int:
        return self.client + self.therapist

    @property
    def completeness(self) -> float:
        """Proportion of turns carrying a C/T label. 0.0 for an empty or unreadable file."""
        if not self.total_turns:
            return 0.0
        return self.role_labelled / self.total_turns

    @property
    def anomalous(self) -> int:
        """Turns whose speaker value is neither a clean role label nor a recognised blank."""
        return self.other + self.merged

    @property
    def unrecognised_rate(self) -> float:
        """Proportion of turns whose speaker value is not understood at all.

        Merged-speaker turns are excluded — those are recognised and handled per turn.
        """
        if not self.total_turns:
            return 0.0
        return self.other / self.total_turns

    @property
    def residual_unlabelled(self) -> int:
        """Turns still needing a label. Non-zero even for a transcript counted as labelled."""
        return self.unlabelled

    def bucket(
        self,
        threshold: float,
        anomaly_tolerance: float = DEFAULT_ANOMALY_TOLERANCE,
    ) -> Bucket:
        """Each bucket names what happens to the transcript next.

        - ``labelled``   role labels on at least ``threshold`` of turns; may still have gaps
        - ``partial``    some role labels, below threshold: label the gaps, keep the human labels
        - ``unlabelled`` no role labels at all: label from scratch
        - ``review``     a human must decide; not automatically processable

        A few anomalous turns — a mistyped label, a turn marked as covering both speakers — do
        not send an otherwise well-coded file to ``review``. Those turns are handled individually
        by the review queue. Only when anomalies exceed ``anomaly_tolerance`` is the file itself
        treated as having a vocabulary problem.
        """
        if self.error is not None:
            return REVIEW
        if self.unrecognised_rate > anomaly_tolerance:
            # Enough unrecognised values that the file's vocabulary is not understood - it may be
            # coded in a scheme this tool does not know, so it must not be treated as unlabelled.
            return REVIEW
        if self.role_labelled == 0 and self.merged:
            # Coded, but only ever as merged turns: nothing usable and nothing to fill.
            return REVIEW
        if self.role_labelled == 0:
            # No role labels. A few stray values under tolerance do not change that; they are
            # flagged individually rather than holding back the whole file.
            return UNLABELLED
        if self.completeness >= threshold:
            return LABELLED
        return PARTIAL


def inspect_transcript(
    path: Path, vocabulary: RoleVocabulary = DEFAULT_VOCABULARY
) -> TranscriptInventory:
    record = TranscriptInventory(path=path)
    try:
        transcript = load_transcript(path)
    except Exception as exc:  # noqa: BLE001 - a bad file is a finding, not a crash
        record.error = f"{type(exc).__name__}: {exc}"
        return record

    record.total_turns = len(transcript.turns)
    for turn in transcript.turns:
        value = turn.speaker.strip()
        if is_multi_role(value, vocabulary):
            record.merged += 1
            record.merged_values.add(value)
            continue
        kind = classify_speaker(value, vocabulary)
        if kind == vocabulary.primary_name:
            record.client += 1
        elif kind == vocabulary.secondary_name:
            record.therapist += 1
        elif kind == "unlabelled":
            record.unlabelled += 1
        else:
            record.other += 1
            record.other_values.add(value)
    return record


def discover_transcripts(input_dir: Path) -> list[Path]:
    """Transcripts directly inside ``input_dir``.

    Deliberately non-recursive: the destination folders this command creates sit inside
    ``input_dir``, so a recursive scan would re-sort files that have already been filed.
    """
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in TRANSCRIPT_SUFFIXES
    )


def take_inventory(
    input_dir: Path, vocabulary: RoleVocabulary = DEFAULT_VOCABULARY
) -> list[TranscriptInventory]:
    return [inspect_transcript(path, vocabulary) for path in discover_transcripts(input_dir)]


@dataclass
class SpeakerAudit:
    """Distribution of raw speaker values across a set of transcripts."""

    files_scanned: int = 0
    files_failed: int = 0
    total_turns: int = 0
    #: raw speaker value -> number of turns carrying it
    turns_by_value: Counter[str] = field(default_factory=Counter)
    #: raw speaker value -> number of files it appears in
    files_by_value: Counter[str] = field(default_factory=Counter)
    #: raw speaker value -> total words in turns carrying it
    words_by_value: Counter[str] = field(default_factory=Counter)
    #: files containing at least one merged-speaker turn
    multi_role_files: int = 0

    #: vocabulary the audit was taken with, needed to interpret the values
    vocabulary: RoleVocabulary = DEFAULT_VOCABULARY

    @property
    def multi_role_values(self) -> list[str]:
        return sorted(v for v in self.turns_by_value if is_multi_role(v, self.vocabulary))

    @property
    def multi_role_turns(self) -> int:
        return sum(self.turns_by_value[v] for v in self.multi_role_values)

    def mean_words(self, value: str) -> float:
        turns = self.turns_by_value[value]
        return self.words_by_value[value] / turns if turns else 0.0


def audit_speaker_values(
    paths: list[Path], vocabulary: RoleVocabulary = DEFAULT_VOCABULARY
) -> SpeakerAudit:
    """Survey raw speaker values across transcripts, exactly as the loader produced them.

    Values are not normalised, so whitespace and Unicode variants of the same label appear as
    separate rows. Reads turn text only to count words; no text is retained or reported.
    """
    audit = SpeakerAudit(vocabulary=vocabulary)
    files_with_multi: set[Path] = set()
    for path in paths:
        try:
            transcript = load_transcript(path)
        except Exception:  # noqa: BLE001 - counted, not raised
            audit.files_failed += 1
            continue
        audit.files_scanned += 1
        seen: set[str] = set()
        for turn in transcript.turns:
            # Deliberately NOT stripped: this is a diagnostic, and a trailing space or a
            # non-breaking space is precisely the kind of thing it exists to surface.
            value = turn.speaker
            audit.total_turns += 1
            audit.turns_by_value[value] += 1
            audit.words_by_value[value] += len(turn.text.split())
            seen.add(value)
            if is_multi_role(value, vocabulary):
                files_with_multi.add(path)
        for value in seen:
            audit.files_by_value[value] += 1
    audit.multi_role_files = len(files_with_multi)
    return audit


def discover_transcripts_recursive(root: Path) -> list[Path]:
    """Every transcript under ``root``. A path to a single file returns just that file."""
    if root.is_file():
        return [root]
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TRANSCRIPT_SUFFIXES
    )


def display_speaker_value(value: str) -> str:
    """Render a speaker value for a human, exposing anything invisible.

    A value with surrounding whitespace or non-ASCII characters is shown as its repr, so a
    trailing non-breaking space or a smart quote is visible rather than silently identical
    to the value next to it.
    """
    if value == "":
        return "(empty)"
    if value != value.strip() or not value.isascii() or not value.isprintable():
        return repr(value)
    return value


def review_reason(
    record: TranscriptInventory,
    threshold: float,
    anomaly_tolerance: float = DEFAULT_ANOMALY_TOLERANCE,
) -> str:
    """Why this transcript was sent to review. Empty string if it was not."""
    if record.bucket(threshold, anomaly_tolerance) != REVIEW:
        return ""
    if record.error is not None:
        return f"could not be read - {record.error}"
    if record.unrecognised_rate > anomaly_tolerance:
        values = ", ".join(repr(v) for v in sorted(record.other_values)[:8])
        return (
            f"{record.other}/{record.total_turns} turns "
            f"({record.unrecognised_rate:.1%}) use unrecognised speaker value(s): {values}"
        )
    if record.role_labelled == 0 and record.merged:
        return (
            f"no single-speaker labels at all; {record.merged} of {record.total_turns} turn(s) "
            "are merged-speaker marks only"
        )
    return "held back for an unrecognised reason - please report this"


def tolerated_anomaly_note(record: TranscriptInventory) -> str:
    """Describe odd turns in a file that was filed normally anyway. Empty if there are none."""
    if not record.anomalous:
        return ""
    parts = []
    if record.other:
        values = ", ".join(repr(v) for v in sorted(record.other_values)[:8])
        parts.append(f"{record.other} unrecognised ({values})")
    if record.merged:
        values = ", ".join(repr(v) for v in sorted(record.merged_values)[:8])
        parts.append(f"{record.merged} merged ({values})")
    return f"{record.total_turns} turns; " + "; ".join(parts)
