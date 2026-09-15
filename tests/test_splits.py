import pytest

from deidentify_transcripts.splits import (
    Mapping,
    dominant_group_share,
    fold_balance,
    grouped_k_fold,
    load_mapping,
)


def write_mapping(path, rows, header="session,therapist"):
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_load_mapping_reads_rows(tmp_path):
    path = write_mapping(tmp_path / "m.csv", ["1,A", "2,A", "3,B"])
    mapping = load_mapping(path)
    assert mapping.group_of == {"1": "A", "2": "A", "3": "B"}
    assert mapping.members_by_group == {"A": ["1", "2"], "B": ["3"]}


def test_load_mapping_skips_missing_groups(tmp_path):
    path = write_mapping(tmp_path / "m.csv", ["1,A", "2,n/a", "3,", "4,B"])
    mapping = load_mapping(path)
    assert set(mapping.group_of) == {"1", "4"}


def test_load_mapping_strips_whitespace(tmp_path):
    path = write_mapping(tmp_path / "m.csv", ["1, A", " 2,B "])
    mapping = load_mapping(path)
    assert mapping.group_of == {"1": "A", "2": "B"}


def test_load_mapping_rejects_a_member_in_two_groups(tmp_path):
    path = write_mapping(tmp_path / "m.csv", ["1,A", "1,B"])
    with pytest.raises(ValueError, match="more than one group"):
        load_mapping(path)


def test_load_mapping_requires_expected_columns(tmp_path):
    path = write_mapping(tmp_path / "m.csv", ["1,A"], header="client,clinician")
    with pytest.raises(ValueError, match="missing column"):
        load_mapping(path)


def test_load_mapping_accepts_custom_column_names(tmp_path):
    path = write_mapping(tmp_path / "m.csv", ["1,A"], header="client,clinician")
    mapping = load_mapping(path, member_column="client", group_column="clinician")
    assert mapping.group_of == {"1": "A"}


def test_grouped_k_fold_never_splits_a_group():
    mapping = Mapping(group_of={str(i): f"G{i // 4}" for i in range(40)})
    folds = grouped_k_fold(mapping, k=5)
    seen: set[str] = set()
    for fold in folds:
        assert not seen & set(fold.groups)
        seen |= set(fold.groups)
    assert sum(f.size for f in folds) == 40


def test_grouped_k_fold_balances_what_it_can_around_an_unsplittable_group():
    # Half the corpus is one group, so no k=4 split can be even: the largest fold can never be
    # smaller than that group. The rest should still be distributed evenly around it.
    group_of = {f"big{i}": "BIG" for i in range(20)}
    group_of.update({f"s{i}": f"S{i}" for i in range(20)})
    mapping = Mapping(group_of=group_of)
    folds = grouped_k_fold(mapping, k=4)

    _, largest, _ = fold_balance(folds)
    assert largest == 20  # the floor imposed by the unsplittable group
    others = sorted(f.size for f in folds if f.size != 20)
    assert max(others) - min(others) <= 1


def test_grouped_k_fold_balances_evenly_when_groups_allow_it():
    mapping = Mapping(group_of={str(i): f"G{i // 2}" for i in range(40)})
    folds = grouped_k_fold(mapping, k=4)
    smallest, largest, ratio = fold_balance(folds)
    assert ratio == 1.0
    assert smallest == largest == 10


def test_grouped_k_fold_rejects_too_few_groups():
    mapping = Mapping(group_of={"1": "A", "2": "B"})
    with pytest.raises(ValueError, match="cannot build 5 disjoint folds"):
        grouped_k_fold(mapping, k=5)


def test_grouped_k_fold_rejects_empty_mapping():
    with pytest.raises(ValueError, match="no usable rows"):
        grouped_k_fold(Mapping(group_of={}), k=2)


def test_dominant_group_share_flags_a_fold_owned_by_one_group():
    group_of = {f"b{i}": "BIG" for i in range(10)}
    group_of["s"] = "SMALL"
    mapping = Mapping(group_of=group_of)
    folds = grouped_k_fold(mapping, k=2)
    big_fold = max(folds, key=lambda f: f.size)
    assert dominant_group_share(big_fold, mapping) == 1.0


def test_member_id_strips_padding_and_suffix(tmp_path):
    from pathlib import Path

    from deidentify_transcripts.splits import member_id_from_filename

    # Invented filenames that exercise the padding/suffix shapes, not real transcript ids.
    assert member_id_from_filename(Path("007B_sample.json")) == "7"
    assert member_id_from_filename(Path("512C_longsuffix.json")) == "512"
    assert member_id_from_filename(Path("42.json")) == "42"
    assert member_id_from_filename(Path("notes.json")) is None


def test_assign_transcripts_places_files_in_their_members_fold(tmp_path):
    from pathlib import Path

    from deidentify_transcripts.splits import assign_transcripts

    mapping = Mapping(group_of={"1": "A", "2": "A", "3": "B", "4": "B"})
    folds = grouped_k_fold(mapping, k=2)
    paths = [Path(f"00{i}B_sample.json") for i in (1, 2, 3, 4)]

    by_fold, unmatched = assign_transcripts(paths, folds)

    assert not unmatched
    assert sum(len(v) for v in by_fold.values()) == 4
    # Members of the same group must land together.
    fold_of = {p.name: i for i, ps in by_fold.items() for p in ps}
    assert fold_of["001B_sample.json"] == fold_of["002B_sample.json"]


def test_assign_transcripts_reports_unmatched_rather_than_dropping(tmp_path):
    from pathlib import Path

    from deidentify_transcripts.splits import assign_transcripts

    mapping = Mapping(group_of={"1": "A", "2": "B"})
    folds = grouped_k_fold(mapping, k=2)
    by_fold, unmatched = assign_transcripts(
        [Path("001B.json"), Path("999B.json"), Path("notes.json")], folds
    )
    assert {p.name for p in unmatched} == {"999B.json", "notes.json"}
