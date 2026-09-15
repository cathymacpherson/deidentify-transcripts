import json

from deidentify_transcripts.labelling import TurnVotes
from deidentify_transcripts.review import build_review, flag_reason
from deidentify_transcripts.schemas import Turn


def turns(n):
    return [Turn(turn_id=i, text=f"turn text {i}") for i in range(n)]


def test_confident_turns_are_not_listed():
    t = turns(5)
    v = [TurnVotes(turn_id=i, votes=["C", "C", "C"]) for i in range(5)]
    review = build_review("demo", t, ["C"] * 5, v, [1.0] * 5)
    assert review["items"] == []
    assert review["turns_flagged"] == 0
    assert review["turns_total"] == 5


def test_flagged_turn_carries_what_a_reviewer_needs():
    t = turns(3)
    v = [
        TurnVotes(turn_id=0, votes=["C", "C", "C"]),
        TurnVotes(turn_id=1, votes=["C", "T", "T"]),
        TurnVotes(turn_id=2, votes=["C", "C", "C"]),
    ]
    review = build_review("demo", t, ["C", "T", "C"], v, [1.0, 0.667, 1.0])

    assert len(review["items"]) == 1
    item = review["items"][0]
    assert item["turn_id"] == 1
    assert item["speaker"] == "T"
    assert item["confidence"] == 0.667
    assert item["votes"] == ["C", "T", "T"]
    assert item["text"] == "turn text 1"
    assert "disagreed with itself" in item["reason"]


def test_items_are_ordered_most_suspicious_first():
    t = turns(4)
    v = [
        TurnVotes(turn_id=0, votes=["C", "C", "C"]),   # second opinion disagrees
        TurnVotes(turn_id=1, votes=["C", "T"]),        # could not decide
        TurnVotes(turn_id=2, votes=[]),                # no view at all
        TurnVotes(turn_id=3, votes=["C", "C", "T"]),   # disagreed with itself
    ]
    review = build_review(
        "demo", t, ["C", "unclear", "unclear", "C"], v,
        [0.85, 0.5, 0.0, 0.667], ["T", "C", "C", "C"],
    )
    reasons = [i["reason"] for i in review["items"]]
    assert reasons[0] == "no view of this turn"
    assert reasons[1] == "model could not decide"
    assert reasons[-1] == "independent system disagreed"


def test_preserved_labels_come_first_and_are_explained():
    t = turns(2)
    v = [TurnVotes(turn_id=i, votes=["C", "C", "C"]) for i in range(2)]
    review = build_review("demo", t, ["C/T", "C"], v, [0.0, 1.0])
    assert review["items"][0]["reason"].startswith("existing label kept")
    assert review["items"][0]["speaker"] == "C/T"


def test_note_warns_the_list_is_not_a_filter():
    review = build_review("demo", turns(1), ["C"], [TurnVotes(turn_id=0, votes=["C"])], [0.5])
    assert "not a filter" in review["note"]
    assert "whole transcript" in review["note"]


def test_review_is_json_serialisable():
    t = turns(2)
    v = [TurnVotes(turn_id=0, votes=["C", "T"]), TurnVotes(turn_id=1, votes=["C", "C"])]
    review = build_review("demo", t, ["unclear", "C"], v, [0.5, 1.0])
    assert json.loads(json.dumps(review))["transcript_id"] == "demo"


def test_empty_transcript_produces_an_empty_list():
    review = build_review("demo", [], [], [], [])
    assert review["turns_total"] == 0
    assert review["items"] == []


def test_flag_reason_prefers_the_most_specific_cause():
    assert flag_reason(TurnVotes(turn_id=0, votes=[])) == "no view of this turn"
    assert flag_reason(TurnVotes(turn_id=0, votes=["C", "T"])) == "model could not decide"
    assert flag_reason(
        TurnVotes(turn_id=0, votes=["C", "C"], anchor="T")
    ) == "the two passes disagreed"
    assert flag_reason(
        TurnVotes(turn_id=0, votes=["C", "C", "C"]), second="T"
    ) == "independent system disagreed"


def test_label_command_writes_json_review(tmp_path, monkeypatch):
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
        return WindowLabels(labels=[
            WindowLabel(turn_id=i, speaker=("T" if (i == 7 and len(ids) > 15) else "C"))
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
    review = json.loads((out / "review" / "t1.review.json").read_text(encoding="utf-8"))
    assert review["transcript_id"] == "t1"
    assert review["turns_total"] == 50
    assert isinstance(review["items"], list)
    for item in review["items"]:
        assert set(item) == {"turn_id", "speaker", "confidence", "reason", "votes", "text"}


def test_label_summary_reconciles_flagged_against_unclear(tmp_path):
    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    turns_out = []
    for i in range(20):
        if i < 2:
            conf, speaker, source = 0.0, "unclear", "model"
        elif i < 6:
            conf, speaker, source = 0.85, "C", "model"
        elif i < 8:
            conf, speaker, source = 1.0, "T", "manual"
        else:
            conf, speaker, source = 1.0, "C", "model"
        turns_out.append({
            "turn_id": i, "speaker": speaker, "speaker_confidence": conf,
            "speaker_source": source, "text": f"t{i}",
        })
    path = tmp_path / "t1.json"
    path.write_text(json.dumps({"transcript_id": "t1", "turns": turns_out}), encoding="utf-8")

    result = CliRunner().invoke(app, ["label-summary", str(path)])

    assert result.exit_code == 0, result.stdout
    assert "6 flagged" in result.stdout      # 2 unclear + 4 disputed
    assert "2 manual" in result.stdout
    assert "2 unclear" in result.stdout
    assert "confidence breakdown" in result.stdout
    assert "t0" not in result.stdout          # no transcript text


def test_label_summary_skips_review_files(tmp_path):
    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    (tmp_path / "t1.json").write_text(json.dumps({"turns": [
        {"turn_id": 0, "speaker": "C", "speaker_confidence": 1.0, "speaker_source": "model",
         "text": "x"}]}), encoding="utf-8")
    (tmp_path / "t1.review.json").write_text(json.dumps({"items": []}), encoding="utf-8")

    result = CliRunner().invoke(app, ["label-summary", str(tmp_path)])
    assert result.exit_code == 0
    assert "1 turns" in result.stdout


def test_label_runs_without_a_second_opinion_when_asked(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from deidentify_transcripts import cli
    from deidentify_transcripts.labelling import Anchors, WindowLabel, WindowLabels

    src = tmp_path / "in"
    src.mkdir()
    (src / "t.json").write_text(json.dumps({"transcript_id": "t", "turns": [
        {"speaker": "unknown", "text": f"x{i}"} for i in range(20)]}), encoding="utf-8")

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
        cli.app, ["label", str(src), "--output-dir", str(tmp_path / "out"),
                  "--reference", "none"]
    )
    assert result.exit_code == 0, result.stdout
    assert "no existing labels found" in result.stdout


def test_label_trains_one_second_opinion_from_labelled_transcripts(tmp_path, monkeypatch):
    """No mapping needed: production labelling has no gold labels to leak."""
    from typer.testing import CliRunner

    from deidentify_transcripts import cli
    from deidentify_transcripts.labelling import Anchors, WindowLabel, WindowLabels

    reference = tmp_path / "labelled"
    reference.mkdir()
    for n in range(3):
        rows = [
            {"speaker": "T" if i % 4 == 0 else "C", "text": f"words here {i}"}
            for i in range(60)
        ]
        (reference / f"ref{n}.json").write_text(json.dumps({"turns": rows}), encoding="utf-8")

    src = tmp_path / "in"
    src.mkdir()
    (src / "new.json").write_text(json.dumps({"transcript_id": "new", "turns": [
        {"speaker": "unknown", "text": f"words here {i}"} for i in range(30)]}), encoding="utf-8")

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

    out = tmp_path / "out"
    result = CliRunner().invoke(cli.app, [
        "label", str(src), "--output-dir", str(out),
        "--reference", str(reference), "--window", "15", "--step", "5",
    ])

    assert result.exit_code == 0, result.stdout
    assert "second opinion trained on 3 labelled transcript(s)" in result.stdout


def test_label_notes_when_the_reference_directory_is_missing(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from deidentify_transcripts import cli
    from deidentify_transcripts.labelling import Anchors, WindowLabel, WindowLabels

    src = tmp_path / "in"
    src.mkdir()
    (src / "t.json").write_text(json.dumps({"transcript_id": "t", "turns": [
        {"speaker": "unknown", "text": f"x{i}"} for i in range(20)]}), encoding="utf-8")

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

    result = CliRunner().invoke(cli.app, [
        "label", str(src), "--output-dir", str(tmp_path / "out"),
        "--reference", str(tmp_path / "nonexistent"),
    ])
    assert result.exit_code == 0
    assert "Running without a second opinion" in result.stdout


def test_label_works_with_no_labelled_corpus_at_all(tmp_path, monkeypatch):
    """A project with nothing labelled is a supported case, not a misconfiguration."""
    from typer.testing import CliRunner

    from deidentify_transcripts import cli
    from deidentify_transcripts.labelling import Anchors, WindowLabel, WindowLabels

    src = tmp_path / "in"
    src.mkdir()
    (src / "t.json").write_text(json.dumps({"transcript_id": "t", "turns": [
        {"speaker": "unknown", "text": f"words here {i}"} for i in range(40)]}), encoding="utf-8")

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

    out = tmp_path / "out"
    result = CliRunner().invoke(cli.app, [
        "label", str(src), "--output-dir", str(out),
        "--reference", str(tmp_path / "nothing-here"), "--window", "15", "--step", "5",
    ])

    assert result.exit_code == 0, result.stdout
    assert "Running without a second opinion" in result.stdout
    assert "This is fine" in result.stdout
    # Labelling still completed and produced both outputs.
    labelled = json.loads((out / "labelled" / "t.json").read_text(encoding="utf-8"))
    assert len(labelled["turns"]) == 40
    assert all(t["speaker"] in ("C", "T", "unclear") for t in labelled["turns"])
    review = json.loads((out / "review" / "t.review.json").read_text(encoding="utf-8"))
    assert review["turns_total"] == 40
