from deidentify_transcripts.baselines import (
    majority_baseline,
    rule_baseline,
    turn_features,
)
from deidentify_transcripts.metrics import score_labels
from deidentify_transcripts.schemas import Turn


def turns(*texts):
    return [Turn(turn_id=i, text=t) for i, t in enumerate(texts)]


def test_majority_baseline_labels_everything_one_way():
    prediction = majority_baseline(turns("a", "b", "c"))
    assert prediction.labels == ["C", "C", "C"]


def test_turn_features_detect_questions_and_pronouns():
    f = turn_features(Turn(turn_id=0, text="How did that make you feel?"))
    assert f["is_question"] == 1.0
    assert f["second_person"] > 0
    assert f["first_person"] == 0.0

    g = turn_features(Turn(turn_id=1, text="I felt like I had lost my footing"))
    assert g["is_question"] == 0.0
    assert g["first_person"] > 0


def test_rule_baseline_separates_a_clear_exchange():
    sample = turns(
        "I just felt so overwhelmed by all of it",
        "and I could not stop thinking about my mother",
        "I kept going over it in my head",
        "What was that like for you?",
        "I think I felt ashamed more than anything",
        "and I did not want to tell my friends",
    )
    gold = ["C", "C", "C", "T", "C", "C"]
    prediction = rule_baseline(sample)
    result = score_labels(gold, prediction.labels)
    assert result.accuracy >= 0.8
    assert prediction.labels[3] == "T"


def test_rule_baseline_beats_majority_on_a_balanced_exchange():
    sample = turns(
        "What brought you here today?",
        "I have been struggling with my sleep for months",
        "How long has that been going on for you?",
        "I would say since I lost my job",
        "And how are you feeling about that now?",
        "I feel like I am slowly getting back on my feet",
    )
    gold = ["T", "C", "T", "C", "T", "C"]

    rule = score_labels(gold, rule_baseline(sample).labels)
    majority = score_labels(gold, majority_baseline(sample).labels)
    assert rule.macro_f1 > majority.macro_f1


def test_rule_baseline_returns_confidence_per_turn():
    prediction = rule_baseline(turns("I felt awful", "What happened?"))
    assert len(prediction.confidence) == 2
    assert all(0.0 <= c <= 1.0 for c in prediction.confidence)


def test_baselines_handle_empty_input():
    assert majority_baseline([]).labels == []
    assert rule_baseline([]).labels == []


def test_evaluate_cli_reports_baselines(tmp_path):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    # A synthetic exchange: therapist asks, client narrates at length.
    rows = []
    for _ in range(10):
        rows.append({"speaker": "T", "text": "And how did that feel for you?"})
        for _ in range(4):
            rows.append({"speaker": "C", "text": "I kept thinking about it and I felt awful"})
    (tmp_path / "a.json").write_text(json.dumps({"turns": rows}), encoding="utf-8")

    result = CliRunner().invoke(app, ["evaluate", str(tmp_path)])

    assert result.exit_code == 0
    assert "majority" in result.stdout
    assert "rule" in result.stdout
    assert "Bar to beat" in result.stdout
    assert "reviewer view" in result.stdout
    assert "a.json" not in result.stdout


def test_evaluate_cli_fails_clearly_when_nothing_is_labelled(tmp_path):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    (tmp_path / "a.json").write_text(
        json.dumps({"turns": [{"speaker": "unknown", "text": "x"}]}), encoding="utf-8"
    )

    result = CliRunner().invoke(app, ["evaluate", str(tmp_path)])

    assert result.exit_code == 1
    assert "carry C/T labels" in result.stdout + str(result.stderr)
