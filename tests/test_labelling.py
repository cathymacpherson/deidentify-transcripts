"""Tests for the LLM labelling passes. A stub model stands in for the server: no network."""

import pytest

from deidentify_transcripts.labelling import (
    Anchors,
    TurnVotes,
    WindowLabel,
    WindowLabels,
    anchor_pass,
    format_window,
    label_transcript,
    window_pass,
    windows,
)
from deidentify_transcripts.schemas import Turn


def make_turns(n, text="some words here"):
    return [Turn(turn_id=i, text=f"{text} {i}") for i in range(n)]


class StubModel:
    """Replays canned structured responses and records what it was asked."""

    def __init__(self, anchors=None, window_labeller=None):
        self._anchors = anchors or []
        self._window_labeller = window_labeller or (lambda ids: {i: "C" for i in ids})
        self.calls = []

    def __call__(self, *, system, text, output_type):
        self.calls.append(text)
        ids = [int(line.split("]")[0][1:]) for line in text.strip().split("\n")]
        if output_type is Anchors:
            return Anchors(anchors=[a for a in self._anchors if a.turn_id in ids])
        assigned = self._window_labeller(ids)
        return WindowLabels(labels=[WindowLabel(turn_id=i, speaker=s) for i, s in assigned.items()])


def test_windows_cover_every_turn():
    spans = windows(100, size=40, step=13)
    covered = set()
    for start, end in spans:
        covered |= set(range(start, end))
    assert covered == set(range(100))


def _view_counts(total, size, step):
    counts = {i: 0 for i in range(total)}
    for start, end in windows(total, size=size, step=step):
        for i in range(start, end):
            counts[i] += 1
    return counts


def test_every_turn_including_the_edges_gets_several_views():
    # A turn seen once has no confidence signal, so the edges matter as much as the middle.
    counts = _view_counts(100, 40, 13)
    assert min(counts.values()) >= 3
    assert counts[0] >= 3
    assert counts[99] >= 3


def test_edge_coverage_holds_for_a_long_transcript():
    counts = _view_counts(1200, 40, 13)
    assert min(counts.values()) >= 3


def test_short_transcript_is_one_window():
    assert windows(10, size=40, step=13) == [(0, 10)]


def test_no_turns_no_windows():
    assert windows(0, size=40, step=13) == []


def test_windows_reject_bad_parameters():
    with pytest.raises(ValueError):
        windows(10, size=0, step=5)


def test_format_window_numbers_turns_and_marks_fixed():
    turns = make_turns(3)
    text = format_window(turns, {1: "T"})
    assert "[0] ?:" in text
    assert "[1] T FIXED:" in text
    assert "[2] ?:" in text


def test_anchor_pass_keeps_only_valid_turn_ids():
    from deidentify_transcripts.labelling import AnchorLabel

    turns = make_turns(5)
    model = StubModel(anchors=[
        AnchorLabel(turn_id=2, speaker="T"),
        AnchorLabel(turn_id=99, speaker="C"),  # not in the transcript
    ])
    found = anchor_pass(turns, model)
    assert found == {2: "T"}


def test_window_pass_ignores_labels_outside_the_window():
    turns = make_turns(50)
    model = StubModel(window_labeller=lambda ids: {**{i: "C" for i in ids}, 9999: "T"})
    votes, _, _ = window_pass(turns, {}, model, size=20, step=10)
    assert all(all(v == "C" for v in tv.votes) for tv in votes)


def test_votes_agree_gives_full_confidence():
    tv = TurnVotes(turn_id=1, votes=["C", "C", "C"])
    assert tv.winner == "C"
    assert tv.agreement == 1.0


def test_split_votes_lower_confidence():
    tv = TurnVotes(turn_id=1, votes=["C", "C", "T"])
    assert tv.winner == "C"
    assert tv.agreement == pytest.approx(2 / 3)


def test_tied_votes_resolve_to_unclear():
    tv = TurnVotes(turn_id=1, votes=["C", "T"])
    assert tv.winner == "unclear"


def test_anchor_overrides_votes_but_disagreement_is_recorded():
    tv = TurnVotes(turn_id=1, votes=["C", "C"], anchor="T")
    assert tv.winner == "T"              # the anchor decides the label
    assert tv.agreement < 1.0            # but it no longer claims full confidence
    assert tv.contradicts_anchor


def test_anchor_agreeing_with_votes_is_not_flagged():
    tv = TurnVotes(turn_id=1, votes=["T", "T"], anchor="T")
    assert not tv.contradicts_anchor


def test_turn_with_no_votes_is_unclear():
    tv = TurnVotes(turn_id=1, votes=[])
    assert tv.winner == "unclear"
    assert tv.agreement == 0.0


def test_label_transcript_runs_both_passes():
    from deidentify_transcripts.labelling import AnchorLabel

    turns = make_turns(60)
    anchor_model = StubModel(anchors=[AnchorLabel(turn_id=5, speaker="T")])
    window_model = StubModel(window_labeller=lambda ids: {i: "C" for i in ids})

    result = label_transcript(turns, anchor_model, window_model, size=20, step=7)

    assert len(result.labels) == 60
    assert result.labels[5] == "T"          # anchor held
    assert result.anchor_count == 1
    assert result.window_count > 1
    assert all(0.0 <= c <= 1.0 for c in result.confidence)


def test_label_transcript_handles_empty_input():
    result = label_transcript([], StubModel(), StubModel())
    assert result.labels == []
    assert result.window_count == 0


def test_unclear_labels_are_counted():
    turns = make_turns(20)
    window_model = StubModel(window_labeller=lambda ids: {i: "unclear" for i in ids})
    result = label_transcript(turns, StubModel(), window_model, size=20, step=7)
    assert result.unclear_count == 20


def test_label_eval_cli_scores_against_gold(tmp_path, monkeypatch):
    """End-to-end through the CLI with the server stubbed out."""
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts import cli
    from deidentify_transcripts.labelling import AnchorLabel

    rows = []
    for i in range(60):
        speaker = "T" if i % 5 == 0 else "C"
        rows.append({"speaker": speaker, "text": f"turn text {i}"})
    (tmp_path / "a.json").write_text(json.dumps({"turns": rows}), encoding="utf-8")

    def fake_structured(*, system, text, output_type):
        ids = [int(line.split("]")[0][1:]) for line in text.strip().split("\n")]
        if output_type is Anchors:
            return Anchors(anchors=[AnchorLabel(turn_id=i, speaker="T") for i in ids if i % 5 == 0])
        return WindowLabels(
            labels=[
                WindowLabel(turn_id=i, speaker="T" if i % 5 == 0 else "C") for i in ids
            ]
        )

    class FakeModel:
        def __init__(self, settings):
            self.structured = fake_structured

    monkeypatch.setattr(cli, "LocalModel", FakeModel)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: object()))

    result = CliRunner().invoke(cli.app, ["label-eval", str(tmp_path), "--limit", "1"])

    assert result.exit_code == 0, result.stdout
    assert "macro-F1 1.000" in result.stdout
    assert "reviewer view" in result.stdout
    assert "a.json" not in result.stdout


def test_label_eval_warns_when_roles_are_inverted(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts import cli

    rows = [{"speaker": "T" if i % 4 == 0 else "C", "text": f"t{i}"} for i in range(40)]
    (tmp_path / "a.json").write_text(json.dumps({"turns": rows}), encoding="utf-8")

    def inverted(*, system, text, output_type):
        ids = [int(line.split("]")[0][1:]) for line in text.strip().split("\n")]
        if output_type is Anchors:
            return Anchors(anchors=[])
        return WindowLabels(
            labels=[WindowLabel(turn_id=i, speaker="C" if i % 4 == 0 else "T") for i in ids]
        )

    class FakeModel:
        def __init__(self, settings):
            self.structured = inverted

    monkeypatch.setattr(cli, "LocalModel", FakeModel)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: object()))

    result = CliRunner().invoke(cli.app, ["label-eval", str(tmp_path), "--limit", "1"])

    assert "roles appear inverted" in result.stdout


def test_anchor_does_not_get_free_confidence_when_windows_disagree():
    # A wrong anchor pinned at 1.0 is an error the reviewer is told to trust - the worst case.
    tv = TurnVotes(turn_id=1, votes=["C", "C", "C"], anchor="T")
    assert tv.winner == "T"                 # the anchor still decides the label
    assert tv.contradicts_anchor
    assert tv.agreement == 0.0              # ...but confidence collapses, so it gets flagged


def test_anchor_partially_contradicted_has_reduced_confidence():
    tv = TurnVotes(turn_id=1, votes=["T", "T", "C"], anchor="T")
    assert tv.agreement < 1.0


def test_anchor_supported_by_windows_keeps_full_confidence():
    tv = TurnVotes(turn_id=1, votes=["T", "T", "T"], anchor="T")
    assert tv.agreement == 1.0


def test_anchor_with_no_votes_keeps_confidence():
    tv = TurnVotes(turn_id=1, votes=[], anchor="T")
    assert tv.agreement == 1.0


def test_label_eval_writes_a_per_turn_report(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts import cli
    from deidentify_transcripts.labelling import AnchorLabel

    rows = [{"speaker": "T" if i % 5 == 0 else "C", "text": f"turn text {i}"} for i in range(60)]
    (tmp_path / "a.json").write_text(json.dumps({"turns": rows}), encoding="utf-8")

    def fake(*, system, text, output_type):
        ids = [int(line.split("]")[0][1:]) for line in text.strip().split("\n")]
        if output_type is Anchors:
            return Anchors(anchors=[AnchorLabel(turn_id=i, speaker="T") for i in ids if i % 5 == 0])
        # Windows disagree with the anchor on turn 0, so its confidence must fall.
        return WindowLabels(labels=[
            WindowLabel(turn_id=i, speaker="C" if i == 0 else ("T" if i % 5 == 0 else "C"))
            for i in ids
        ])

    class FakeModel:
        def __init__(self, settings):
            self.structured = fake

    monkeypatch.setattr(cli, "LocalModel", FakeModel)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda c: object()))

    report = tmp_path / "report.csv"
    result = CliRunner().invoke(
        cli.app, ["label-eval", str(tmp_path), "--limit", "1", "-o", str(report)]
    )

    assert result.exit_code == 0, result.stdout
    lines = report.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith(
        "file,turn_id,gold,predicted,confidence,second_opinion,second_conf,anchor"
    )
    assert len(lines) == 61
    # The contradicted anchor must be visible and no longer at full confidence.
    turn_zero = [l for l in lines if l.split(",")[1] == "0"][0]
    assert "yes" in turn_zero
    assert float(turn_zero.split(",")[4]) < 1.0


def test_second_opinion_lowers_confidence_on_disagreement():
    from deidentify_transcripts.labelling import apply_second_opinion

    votes = [TurnVotes(turn_id=0, votes=["C", "C", "C"])]
    agreed = apply_second_opinion(votes, ["C"])
    disputed = apply_second_opinion(votes, ["T"])
    assert agreed[0] == 1.0
    assert disputed[0] < agreed[0]


def test_second_opinion_never_changes_the_label():
    from deidentify_transcripts.labelling import apply_second_opinion

    votes = [TurnVotes(turn_id=0, votes=["C", "C", "C"])]
    apply_second_opinion(votes, ["T"])
    assert votes[0].winner == "C"  # unchanged


def test_split_votes_still_outrank_a_weakly_held_disagreement():
    from deidentify_transcripts.labelling import apply_second_opinion

    split = TurnVotes(turn_id=0, votes=["C", "C", "T"])
    unanimous = TurnVotes(turn_id=1, votes=["C", "C", "C"])
    # The second system disputes the unanimous turn, but is barely sure of itself.
    confidences = apply_second_opinion([split, unanimous], ["C", "T"], [1.0, 0.1])
    assert confidences[0] < confidences[1]


def test_disagreement_strength_spreads_turns_across_the_range():
    """The whole point: a flat discount put every disagreement at one value."""
    from deidentify_transcripts.labelling import apply_second_opinion

    votes = [TurnVotes(turn_id=i, votes=["C", "C", "C"]) for i in range(3)]
    out = apply_second_opinion(votes, ["T", "T", "T"], [1.0, 0.5, 0.05])
    assert out[0] < out[1] < out[2]          # strongest disagreement is most suspicious
    assert out[2] > 0.95                     # a barely-held one hardly counts
    assert len(set(out)) == 3                # and they no longer collapse together


def test_a_strongly_held_disagreement_ranks_alongside_a_split_vote():
    """Whether it should outrank one is unmeasured - see SECOND_OPINION_WEIGHT."""
    from deidentify_transcripts.labelling import apply_second_opinion

    split = TurnVotes(turn_id=0, votes=["C", "C", "C", "T"])      # 3 of 4 agree
    unanimous = TurnVotes(turn_id=1, votes=["C", "C", "C"])        # all agree, but disputed
    out = apply_second_opinion([split, unanimous], ["C", "T"], [1.0, 1.0])
    assert out[0] == 0.75
    assert 0.6 < out[1] < 0.8      # lands in the same region, not in a bucket of its own


def test_second_opinion_confidence_length_is_checked():
    from deidentify_transcripts.labelling import apply_second_opinion

    with pytest.raises(ValueError, match="one second-opinion confidence per turn"):
        apply_second_opinion([TurnVotes(turn_id=0, votes=["C"])], ["C"], [1.0, 1.0])


def test_both_signals_compound():
    from deidentify_transcripts.labelling import apply_second_opinion

    split = TurnVotes(turn_id=0, votes=["C", "C", "T"])
    agreed = apply_second_opinion([split], ["C"])[0]
    disputed = apply_second_opinion([split], ["T"])[0]
    assert disputed < agreed


def test_unclear_turns_are_not_penalised_further():
    from deidentify_transcripts.labelling import apply_second_opinion

    tied = TurnVotes(turn_id=0, votes=["C", "T"])
    assert tied.winner == "unclear"
    assert apply_second_opinion([tied], ["C"])[0] == tied.agreement


def test_second_opinion_requires_matching_lengths():
    from deidentify_transcripts.labelling import apply_second_opinion

    with pytest.raises(ValueError, match="one second-opinion label per turn"):
        apply_second_opinion([TurnVotes(turn_id=0, votes=["C"])], ["C", "T"])


def test_label_eval_second_opinion_is_trained_without_the_speaker(tmp_path, monkeypatch):
    """The trained model must never have seen the clinician whose transcript it is judging."""
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts import cli

    data = tmp_path / "labelled"
    data.mkdir()
    rows_map = []
    for client in range(1, 9):
        turns = [
            {"speaker": "T" if i % 4 == 0 else "C", "text": f"words for turn {i}"}
            for i in range(40)
        ]
        (data / f"{client:03d}B_sample.json").write_text(
            json.dumps({"turns": turns}), encoding="utf-8"
        )
        rows_map.append(f"{client},TH{client % 3}")
    mapping = tmp_path / "map.csv"
    mapping.write_text("session,therapist\n" + "\n".join(rows_map) + "\n", encoding="utf-8")

    def fake(*, system, text, output_type):
        ids = [int(line.split("]")[0][1:]) for line in text.strip().split("\n")]
        if output_type is Anchors:
            return Anchors(anchors=[])
        return WindowLabels(labels=[
            WindowLabel(turn_id=i, speaker="T" if i % 4 == 0 else "C") for i in ids
        ])

    class FakeModel:
        def __init__(self, settings):
            self.structured = fake

    monkeypatch.setattr(cli, "LocalModel", FakeModel)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda c: object()))

    report = tmp_path / "r.csv"
    result = CliRunner().invoke(
        cli.app,
        ["label-eval", str(data), "--limit", "1", "--mapping", str(mapping), "-o", str(report)],
    )

    assert result.exit_code == 0, result.stdout
    text = report.read_text(encoding="utf-8")
    header = text.splitlines()[0].split(",")
    assert "second_opinion" in header
    # The column is populated, i.e. a model really was trained and applied.
    column = header.index("second_opinion")
    values = {line.split(",")[column] for line in text.splitlines()[1:]}
    assert values & {"C", "T"}


def test_label_eval_without_mapping_leaves_second_opinion_blank(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts import cli

    rows = [{"speaker": "C", "text": f"t{i}"} for i in range(30)]
    (tmp_path / "a.json").write_text(json.dumps({"turns": rows}), encoding="utf-8")

    def fake(*, system, text, output_type):
        ids = [int(line.split("]")[0][1:]) for line in text.strip().split("\n")]
        if output_type is Anchors:
            return Anchors(anchors=[])
        return WindowLabels(labels=[WindowLabel(turn_id=i, speaker="C") for i in ids])

    class FakeModel:
        def __init__(self, settings):
            self.structured = fake

    monkeypatch.setattr(cli, "LocalModel", FakeModel)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda c: object()))

    report = tmp_path / "r.csv"
    CliRunner().invoke(cli.app, ["label-eval", str(tmp_path), "--limit", "1", "-o", str(report)])

    header = report.read_text(encoding="utf-8").splitlines()[0].split(",")
    column = header.index("second_opinion")
    values = {line.split(",")[column] for line in report.read_text().splitlines()[1:]}
    assert values == {""}


def test_coverage_detects_a_model_dropping_turns():
    """A long window where the model returns fewer labels than it was shown."""
    turns = make_turns(60)
    # Return labels for only the first half of whatever it is given.
    model = StubModel(window_labeller=lambda ids: {i: "C" for i in ids[: len(ids) // 2]})
    result = label_transcript(turns, StubModel(), model, size=20, step=7)
    assert result.coverage < 0.6
    assert result.min_views < 3


def test_full_coverage_when_the_model_answers_completely():
    turns = make_turns(60)
    model = StubModel(window_labeller=lambda ids: {i: "C" for i in ids})
    result = label_transcript(turns, StubModel(), model, size=20, step=7)
    assert result.coverage == 1.0
    assert result.unvoted_turns == 0


def test_duplicate_labels_for_one_turn_count_once():
    turns = make_turns(20)

    class Duplicating(StubModel):
        def __call__(self, *, system, text, output_type):
            result = super().__call__(system=system, text=text, output_type=output_type)
            if output_type is WindowLabels:
                return WindowLabels(labels=list(result.labels) + list(result.labels))
            return result

    model = Duplicating(window_labeller=lambda ids: {i: "C" for i in ids})
    result = label_transcript(turns, StubModel(), model, size=20, step=7)
    assert result.coverage == 1.0
    assert all(len(v.votes) <= result.window_count for v in result.votes)


def test_label_eval_writes_a_report_without_being_asked(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts import cli

    (tmp_path / "a.json").write_text(
        json.dumps({"turns": [{"speaker": "C", "text": f"t{i}"} for i in range(20)]}),
        encoding="utf-8",
    )

    def fake(*, system, text, output_type):
        ids = [int(line.split("]")[0][1:]) for line in text.strip().split("\n")]
        if output_type is Anchors:
            return Anchors(anchors=[])
        return WindowLabels(labels=[WindowLabel(turn_id=i, speaker="C") for i in ids])

    class FakeModel:
        def __init__(self, settings):
            self.structured = fake

    monkeypatch.setattr(cli, "LocalModel", FakeModel)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda c: object()))

    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = CliRunner().invoke(cli.app, ["label-eval", str(tmp_path), "--limit", "1"])
        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "label-report.csv").exists()
        assert "per-turn report" in result.stdout
    finally:
        os.chdir(cwd)


def test_label_eval_warns_before_overwriting_an_existing_report(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts import cli

    (tmp_path / "a.json").write_text(
        json.dumps({"turns": [{"speaker": "C", "text": f"t{i}"} for i in range(20)]}),
        encoding="utf-8",
    )
    existing = tmp_path / "old.csv"
    existing.write_text("stale", encoding="utf-8")

    def fake(*, system, text, output_type):
        ids = [int(line.split("]")[0][1:]) for line in text.strip().split("\n")]
        if output_type is Anchors:
            return Anchors(anchors=[])
        return WindowLabels(labels=[WindowLabel(turn_id=i, speaker="C") for i in ids])

    class FakeModel:
        def __init__(self, settings):
            self.structured = fake

    monkeypatch.setattr(cli, "LocalModel", FakeModel)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda c: object()))

    result = CliRunner().invoke(
        cli.app, ["label-eval", str(tmp_path), "--limit", "1", "-o", str(existing)]
    )
    assert "will be overwritten" in result.stdout


def test_existing_manual_labels_are_kept_and_used_as_anchors():
    from deidentify_transcripts.labelling import existing_labels

    turns = [
        Turn(turn_id=0, speaker="T", text="a"),
        Turn(turn_id=1, speaker="unknown", text="b"),
        Turn(turn_id=2, speaker="C", text="c"),
    ]
    assert existing_labels(turns, ("C", "T")) == {0: "T", 2: "C"}

    model = StubModel(window_labeller=lambda ids: {i: "C" for i in ids})
    result = label_transcript(turns, StubModel(), model, size=3, step=2)
    assert result.labels[0] == "T"        # manual label kept despite the window saying C
    assert result.labels[2] == "C"
    assert result.confidence[0] == 1.0
    assert result.manual_count == 2


def test_evaluation_mode_ignores_existing_labels():
    """Gold labels share the speaker field, so evaluation must not read them as anchors."""
    turns = [Turn(turn_id=i, speaker="T", text=f"t{i}") for i in range(3)]
    model = StubModel(window_labeller=lambda ids: {i: "C" for i in ids})
    result = label_transcript(turns, StubModel(), model, size=3, step=2, use_existing=False)
    assert result.labels == ["C", "C", "C"]   # the model's own answer, not the gold label
    assert result.manual_count == 0


def test_unlabelled_transcript_is_unaffected_by_the_setting():
    turns = [Turn(turn_id=i, speaker="unknown", text=f"t{i}") for i in range(3)]
    model = StubModel(window_labeller=lambda ids: {i: "C" for i in ids})
    with_existing = label_transcript(turns, StubModel(), model, size=3, step=2)
    without = label_transcript(turns, StubModel(), model, size=3, step=2, use_existing=False)
    assert with_existing.labels == without.labels


def test_out_of_vocabulary_labels_are_preserved_not_overwritten():
    """C/T, X, CC — a human wrote each one deliberately."""
    from deidentify_transcripts.labelling import preserved_labels

    turns = [
        Turn(turn_id=0, speaker="C", text="a"),
        Turn(turn_id=1, speaker="unknown", text="b"),
        Turn(turn_id=2, speaker="C/T", text="c"),
        Turn(turn_id=3, speaker="X", text="d"),
        Turn(turn_id=4, speaker="CC", text="e"),
        Turn(turn_id=5, speaker="", text="f"),
        Turn(turn_id=6, speaker="n/a", text="g"),
    ]
    assert preserved_labels(turns, ("C", "T")) == {2: "C/T", 3: "X", 4: "CC"}

    model = StubModel(window_labeller=lambda ids: {i: "T" for i in ids})
    result = label_transcript(turns, StubModel(), model, size=7, step=4)

    assert result.labels[0] == "C"      # manual role kept
    assert result.labels[1] == "T"      # unknown filled by the model
    assert result.labels[2] == "C/T"    # merged mark untouched
    assert result.labels[3] == "X"
    assert result.labels[4] == "CC"
    assert result.labels[5] == "T"      # blank filled
    assert result.labels[6] == "T"      # n/a filled
    assert result.preserved_count == 3


def test_preserved_labels_are_flagged_for_review():
    turns = [
        Turn(turn_id=0, speaker="C", text="a"),
        Turn(turn_id=1, speaker="C/T", text="b"),
    ]
    model = StubModel(window_labeller=lambda ids: {i: "C" for i in ids})
    result = label_transcript(turns, StubModel(), model, size=2, step=1)
    assert result.confidence[0] == 1.0   # settled
    assert result.confidence[1] == 0.0   # top of the review list


def test_evaluation_mode_preserves_nothing():
    turns = [Turn(turn_id=0, speaker="C/T", text="a")]
    model = StubModel(window_labeller=lambda ids: {i: "C" for i in ids})
    result = label_transcript(turns, StubModel(), model, size=2, step=1, use_existing=False)
    assert result.labels[0] == "C"
    assert result.preserved_count == 0


def test_role_labels_are_recognised_whatever_their_case():
    """A lowercase 'c' is a client label, not an unrecognised value."""
    from deidentify_transcripts.labelling import existing_labels, preserved_labels

    turns = [
        Turn(turn_id=0, speaker="c", text="a"),
        Turn(turn_id=1, speaker="T", text="b"),
        Turn(turn_id=2, speaker=" t ", text="c"),
        Turn(turn_id=3, speaker="Client", text="d"),
        Turn(turn_id=4, speaker="THERAPIST", text="e"),
        Turn(turn_id=5, speaker="unknown", text="f"),
    ]
    assert existing_labels(turns, ("C", "T")) == {0: "C", 1: "T", 2: "T", 3: "C", 4: "T"}
    assert preserved_labels(turns, ("C", "T")) == {}


def test_lowercase_labels_are_kept_as_anchors_not_flagged():
    turns = [
        Turn(turn_id=0, speaker="c", text="a"),
        Turn(turn_id=1, speaker="unknown", text="b"),
    ]
    model = StubModel(window_labeller=lambda ids: {i: "T" for i in ids})
    result = label_transcript(turns, StubModel(), model, size=2, step=1)

    assert result.labels[0] == "C"        # canonical spelling, model's "T" ignored
    assert result.confidence[0] == 1.0    # settled, not flagged
    assert result.manual_count == 1
    assert result.preserved_count == 0


def test_genuinely_unrecognised_labels_are_still_preserved():
    from deidentify_transcripts.labelling import preserved_labels

    turns = [
        Turn(turn_id=0, speaker="CC", text="a"),
        Turn(turn_id=1, speaker="X", text="b"),
        Turn(turn_id=2, speaker="C/T", text="c"),
    ]
    assert preserved_labels(turns, ("C", "T")) == {0: "CC", 1: "X", 2: "C/T"}
