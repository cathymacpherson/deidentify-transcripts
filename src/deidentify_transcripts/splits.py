"""Leakage-free evaluation folds.

Speakers recur across a corpus, so a random split puts the same person on both sides and measures
memorisation of that person's speech style rather than generalisation. Folds are therefore built
over speaker groups, never over transcripts.

Nothing here reads a transcript; it works purely from an identity mapping.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Mapping:
    """Which group each member belongs to, e.g. client -> clinician."""

    #: member id -> group id
    group_of: dict[str, str]

    @property
    def members_by_group(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for member, group in self.group_of.items():
            out[group].append(member)
        return {g: sorted(m) for g, m in sorted(out.items())}

    @property
    def group_sizes(self) -> list[tuple[str, int]]:
        """Groups with their member counts, largest first."""
        return sorted(
            ((g, len(m)) for g, m in self.members_by_group.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )


#: Values meaning "no group recorded"; rows carrying one are dropped.
MISSING_GROUP = {"", "n/a", "na", "none", "null", "-", "unknown"}


def load_mapping(
    path: Path, *, member_column: str = "session", group_column: str = "therapist"
) -> Mapping:
    """Read a two-column identity mapping. Rows with a missing group are skipped.

    A duplicated member is an error rather than a last-one-wins: it means the mapping disagrees
    with itself about who someone belongs to, and silently picking one would corrupt every fold.
    """
    group_of: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {member_column, group_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"mapping is missing column(s): {', '.join(sorted(missing))}")
        for row in reader:
            member = (row.get(member_column) or "").strip()
            group = (row.get(group_column) or "").strip()
            if not member or group.casefold() in MISSING_GROUP:
                continue
            if member in group_of and group_of[member] != group:
                raise ValueError(
                    f"member {member!r} is mapped to more than one group; "
                    "the mapping must be resolved before splitting"
                )
            group_of[member] = group
    return Mapping(group_of=group_of)


@dataclass
class Fold:
    index: int
    groups: list[str] = field(default_factory=list)
    members: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


def grouped_k_fold(mapping: Mapping, k: int = 5) -> list[Fold]:
    """Assign whole groups to ``k`` folds, greedily balancing member count.

    Greedy largest-first assignment keeps folds close in size even when group sizes are very
    skewed, which leave-one-group-out cannot do. A group is never split across folds — that would
    reintroduce exactly the leakage this exists to prevent.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    sizes = mapping.group_sizes
    if not sizes:
        raise ValueError("mapping contains no usable rows")
    if len(sizes) < k:
        raise ValueError(
            f"only {len(sizes)} group(s) available; cannot build {k} disjoint folds. "
            "Use a smaller k, or accept that cross-group generalisation cannot be measured."
        )

    folds = [Fold(index=i) for i in range(k)]
    members_by_group = mapping.members_by_group
    for group, _ in sizes:
        target = min(folds, key=lambda f: (f.size, f.index))
        target.groups.append(group)
        target.members.extend(members_by_group[group])
    for fold in folds:
        fold.groups.sort()
        fold.members.sort()
    return folds


def fold_balance(folds: list[Fold]) -> tuple[int, int, float]:
    """Smallest fold, largest fold, and largest/smallest ratio — how uneven the split is."""
    sizes = [f.size for f in folds]
    smallest, largest = min(sizes), max(sizes)
    return smallest, largest, (largest / smallest if smallest else float("inf"))


def dominant_group_share(fold: Fold, mapping: Mapping) -> float:
    """Share of a fold contributed by its largest single group.

    A fold dominated by one group produces a score about that group, not about the corpus, which
    is why per-fold results are reported rather than only a mean.
    """
    if not fold.size:
        return 0.0
    members_by_group = mapping.members_by_group
    return max(len(members_by_group[g]) for g in fold.groups) / fold.size


def member_id_from_filename(path: Path) -> str | None:
    """Derive the mapping's member id from a transcript filename.

    Mapping keys are usually unpadded integers while filenames are zero-padded and carry a
    session suffix, so the leading digits are taken and leading zeros dropped. Returns None when
    the filename has no leading digits, which the caller should treat as an error rather than
    silently skip.
    """
    match = re.match(r"^0*(\d+)", path.stem)
    return match.group(1) if match else None


def assign_transcripts(
    paths: list[Path], folds: list[Fold]
) -> tuple[dict[int, list[Path]], list[Path]]:
    """Place each transcript in its member's fold.

    Returns the per-fold transcripts and any transcripts that could not be matched. Unmatched
    files are returned rather than dropped: silently excluding data from an evaluation is how a
    score stops meaning what it claims to.
    """
    fold_of_member = {m: f.index for f in folds for m in f.members}
    by_fold: dict[int, list[Path]] = {f.index: [] for f in folds}
    unmatched: list[Path] = []
    for path in paths:
        member = member_id_from_filename(path)
        index = fold_of_member.get(member) if member else None
        if index is None:
            unmatched.append(path)
        else:
            by_fold[index].append(path)
    return by_fold, unmatched
