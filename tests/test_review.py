from deidentify_transcripts.labelling import TurnVotes
from deidentify_transcripts.review import find_flagged, render_report
from deidentify_transcripts.schemas import Turn


def turns(n):
    return [Turn(turn_id=i, text=f"turn text {i}") for i in range(n)]


def votes_for(specs):
    """specs: list of (votes, anchor)."""
    return [TurnVotes(turn_id=i, votes=v, anchor=a) for i, (v, a) in enumerate(specs)]


def test_confident_turns_are_not_flagged():
    v = votes_for([(["C", "C", "C"], None)] * 5)
    assert find_flagged(v, [1.0] * 5) == []


def test_split_votes_are_flagged():
    v = votes_for([(["C", "C", "C"], None), (["C", "T", "T"], None)])
    blocks = find_flagged(v, [1.0, 0.67])
    assert len(blocks) == 1
    assert blocks[0].turn_ids == [1]
    assert "disagreed with itself" in blocks[0].reason


def test_adjacent_flagged_turns_group_into_one_block():
    # A mislabelled run is one correction, not five.
    v = votes_for([(["C", "T", "T"], None)] * 4)
    blocks = find_flagged(v, [0.67] * 4)
    assert len(blocks) == 1
    assert blocks[0].turn_ids == [0, 1, 2, 3]


def test_different_reasons_do_not_merge():
    v = votes_for([(["C", "T", "T"], None), (["C", "C", "C"], None)])
    blocks = find_flagged(v, [0.67, 0.85], second_labels=["C", "T"])
    assert len(blocks) == 2
    assert blocks[1].reason == "independent system disagreed"


def test_second_opinion_disagreement_is_its_own_reason():
    v = votes_for([(["C", "C", "C"], None)])
    blocks = find_flagged(v, [0.85], second_labels=["T"])
    assert blocks[0].reason == "independent system disagreed"


def test_unclear_turns_are_top_priority():
    v = votes_for([(["C", "T"], None)])
    blocks = find_flagged(v, [0.5])
    assert blocks[0].reason == "model could not decide"


def test_turn_with_no_views_is_flagged():
    v = votes_for([([], None)])
    blocks = find_flagged(v, [0.0])
    assert blocks[0].reason == "no view of this turn"


def test_report_shows_flagged_turns_in_context():
    t = turns(10)
    v = votes_for([(["C", "C", "C"], None)] * 10)
    conf = [1.0] * 10
    conf[5] = 0.67
    v[5] = TurnVotes(turn_id=5, votes=["C", "T", "T"])
    report = render_report("demo", t, ["C"] * 10, v, conf)

    assert "# Review: demo" in report
    assert "turn 5" in report
    assert "turn text 4" in report   # context before
    assert "turn text 6" in report   # context after
    assert ">>" in report            # the flagged turn is marked


def test_report_tells_the_reviewer_not_to_stop_at_the_flags():
    t = turns(5)
    v = votes_for([(["C", "T", "T"], None)] + [(["C", "C", "C"], None)] * 4)
    report = render_report("demo", t, ["C"] * 5, v, [0.67] + [1.0] * 4)
    assert "not a" in report.lower() and "finished" in report.lower()


def test_report_orders_by_priority():
    t = turns(6)
    v = [
        TurnVotes(turn_id=0, votes=["C", "C", "C"]),   # second opinion disagrees
        TurnVotes(turn_id=1, votes=["C", "C", "C"]),
        TurnVotes(turn_id=2, votes=["C", "T"]),        # unclear - top priority
        TurnVotes(turn_id=3, votes=["C", "C", "C"]),
        TurnVotes(turn_id=4, votes=["C", "C", "C"]),
        TurnVotes(turn_id=5, votes=["C", "C", "C"]),
    ]
    conf = [0.85, 1.0, 0.5, 1.0, 1.0, 1.0]
    report = render_report("demo", t, ["C"] * 6, v, conf, ["T", "C", "C", "C", "C", "C"])
    could_not = report.index("model could not decide")
    independent = report.index("independent system disagreed")
    assert could_not < independent


def test_report_with_nothing_flagged_says_so():
    t = turns(4)
    v = votes_for([(["C", "C", "C"], None)] * 4)
    report = render_report("demo", t, ["C"] * 4, v, [1.0] * 4)
    assert "Nothing was flagged" in report


def test_report_handles_an_empty_transcript():
    report = render_report("demo", [], [], [], [])
    assert "no turns" in report


def test_label_command_writes_transcript_and_review(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts import cli
    from deidentify_transcripts.labelling import Anchors, WindowLabel, WindowLabels

    src = tmp_path / "in"
    src.mkdir()
    rows = [{"speaker": "unknown", "text": f"turn text {i}"} for i in range(50)]
    (src / "t1.json").write_text(
        json.dumps({"transcript_id": "t1", "turns": rows}), encoding="utf-8"
    )

    def fake(*, system, text, output_type):
        ids = [int(line.split("]")[0][1:]) for line in text.strip().split("\n")]
        if output_type is Anchors:
            return Anchors(anchors=[])
        # Turn 7 gets an inconsistent label so something is flagged.
        return WindowLabels(labels=[
            WindowLabel(turn_id=i, speaker=("T" if (i == 7 and len(ids) > 20) else "C"))
            for i in ids
        ])

    class FakeModel:
        def __init__(self, settings):
            self.structured = fake

    monkeypatch.setattr(cli, "LocalModel", FakeModel)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda c: object()))

    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli.app, ["label", str(src), "--output-dir", str(out), "--window", "20", "--step", "7"]
    )

    assert result.exit_code == 0, result.stdout
    labelled = json.loads((out / "labelled" / "t1.json").read_text(encoding="utf-8"))
    assert len(labelled["turns"]) == 50
    assert labelled["turns"][0]["speaker"] in ("C", "T", "unclear")
    assert labelled["turns"][0]["speaker_source"] == "model"
    assert "speaker_confidence" in labelled["turns"][0]

    review = (out / "review" / "t1.review.md").read_text(encoding="utf-8")
    assert review.startswith("# Review: t1")
    assert "flagged" in review


def test_label_command_reports_a_failing_file(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from deidentify_transcripts import cli

    src = tmp_path / "in"
    src.mkdir()
    (src / "bad.json").write_text("{oops", encoding="utf-8")

    class FakeModel:
        def __init__(self, settings):
            self.structured = lambda **kw: None

    monkeypatch.setattr(cli, "LocalModel", FakeModel)
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda c: object()))

    result = CliRunner().invoke(
        cli.app, ["label", str(src), "--output-dir", str(tmp_path / "out")]
    )
    assert result.exit_code == 1
    assert "FAILED bad.json" in result.stdout + str(result.stderr)


def test_preserved_labels_get_their_own_review_reason():
    t = turns(5)
    v = votes_for([(["C", "C", "C"], None)] * 5)
    conf = [1.0, 1.0, 0.0, 1.0, 1.0]
    labels = ["C", "C", "C/T", "C", "C"]
    report = render_report("demo", t, labels, v, conf)
    assert "existing label kept" in report
    assert "confirm what it means" in report
