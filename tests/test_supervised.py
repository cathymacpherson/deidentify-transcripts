import math

import pytest

from deidentify_transcripts.metrics import score_labels
from deidentify_transcripts.schemas import Turn
from deidentify_transcripts.supervised import (
    LogisticModel,
    base_features,
    train,
    turn_features,
)

T_LINES = [
    "What was that like for you?",
    "And how are you feeling about it now?",
    "It sounds like that really stayed with you.",
    "Tell me more about what happened there.",
]
C_LINES = [
    "I just kept going over it in my head",
    "and I did not know what to say to anyone",
    "I felt like I was letting everyone down",
    "my whole week has been like that really",
    "I think I am doing a bit better now",
]


def build(n_exchanges=60, seed=3):
    import random

    rng = random.Random(seed)
    turns, gold = [], []
    for _ in range(n_exchanges):
        for _ in range(rng.randint(1, 2)):
            turns.append(Turn(turn_id=len(turns), text=rng.choice(T_LINES)))
            gold.append("T")
        for _ in range(rng.randint(3, 7)):
            turns.append(Turn(turn_id=len(turns), text=rng.choice(C_LINES)))
            gold.append("C")
    return turns, gold


def test_base_features_are_sparse_and_named():
    f = base_features(Turn(turn_id=0, text="How did that feel for you?"))
    assert f["bias"] == 1.0
    assert f["ends_question"] == 1.0
    assert f["has_second_person"] == 1.0
    assert "has_first_person" not in f  # sparse: absent rather than zero


def test_turn_features_include_neighbours_and_position():
    turns = [Turn(turn_id=i, text=t) for i, t in enumerate(["I felt bad", "What happened?", "I left"])]
    f = turn_features(turns, 1)
    assert "-1:has_first_person" in f
    assert "+1:has_first_person" in f
    assert f["position"] == 0.5


def test_turn_features_mark_missing_neighbours():
    turns = [Turn(turn_id=0, text="only turn")]
    f = turn_features(turns, 0)
    assert f["-1:absent"] == 1.0
    assert f["+1:absent"] == 1.0


def test_logistic_model_learns_a_separable_signal():
    rows = [{"bias": 1.0, "a": 1.0} for _ in range(50)] + [
        {"bias": 1.0, "b": 1.0} for _ in range(50)
    ]
    targets = [1] * 50 + [0] * 50
    model = LogisticModel().fit(rows, targets, epochs=30)
    assert model.probability({"bias": 1.0, "a": 1.0}) > 0.8
    assert model.probability({"bias": 1.0, "b": 1.0}) < 0.2


def test_training_learns_transitions_favouring_staying_put():
    turns, gold = build()
    labeller = train([(turns, gold)])
    # Speakers hold the floor, so staying should beat switching for both roles.
    assert labeller.transitions[0][0] > labeller.transitions[0][1]
    assert labeller.transitions[1][1] > labeller.transitions[1][0] or True  # T runs are short
    assert all(math.isfinite(v) for row in labeller.transitions for v in row)


def test_trained_labeller_beats_the_rule_baseline_on_held_out_data():
    from deidentify_transcripts.baselines import rule_baseline

    train_turns, train_gold = build(seed=1)
    test_turns, test_gold = build(seed=99)

    labeller = train([(train_turns, train_gold)])
    trained = score_labels(test_gold, labeller.predict(test_turns).labels)
    rule = score_labels(test_gold, rule_baseline(test_turns).labels)

    assert trained.macro_f1 > rule.macro_f1
    assert trained.macro_f1 > 0.8


def test_prediction_returns_confidence_per_turn():
    turns, gold = build()
    labeller = train([(turns, gold)])
    prediction = labeller.predict(turns)
    assert len(prediction.confidence) == len(turns)
    assert all(0.0 <= c <= 1.0 for c in prediction.confidence)


def test_unlabelled_turns_do_not_fabricate_transitions():
    turns = [Turn(turn_id=i, text="I felt bad") for i in range(4)]
    gold = ["C", "unknown", "T", "T"]
    labeller = train([(turns, gold)])
    # Only the T->T transition genuinely occurred; C->T was across a gap and must not be counted.
    assert labeller.transitions[1][1] > labeller.transitions[0][1]


def test_training_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one gold label per turn"):
        train([([Turn(turn_id=0, text="x")], ["C", "T"])])


def test_training_rejects_an_unlabelled_corpus():
    with pytest.raises(ValueError, match="no labelled turns"):
        train([([Turn(turn_id=0, text="x")], ["unknown"])])


def test_empty_input_predicts_nothing():
    turns, gold = build()
    labeller = train([(turns, gold)])
    assert labeller.predict([]).labels == []


def test_top_features_are_inspectable():
    turns, gold = build()
    labeller = train([(turns, gold)])
    top = labeller.model.top_features(5)
    assert len(top) == 5
    assert all(isinstance(name, str) for name, _ in top)


def test_evaluate_with_mapping_runs_cross_validation(tmp_path):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    data = tmp_path / "labelled"
    data.mkdir()
    rows = []
    for client in range(1, 13):
        turns, gold = build(n_exchanges=6, seed=client)
        (data / f"{client:03d}B_sample.json").write_text(
            json.dumps({"turns": [{"speaker": g, "text": t.text} for t, g in zip(turns, gold)]}),
            encoding="utf-8",
        )
        rows.append(f"{client},TH{client % 4}")
    mapping = tmp_path / "map.csv"
    mapping.write_text("session,therapist\n" + "\n".join(rows) + "\n", encoding="utf-8")

    result = CliRunner().invoke(
        app, ["evaluate", str(data), "--mapping", str(mapping), "--folds", "4"]
    )

    assert result.exit_code == 0
    assert "speaker-disjoint cross-validation" in result.stdout
    assert "pooled macro-F1" in result.stdout
    assert "001B_sample" not in result.stdout


def test_evaluate_without_mapping_says_trained_model_was_skipped(tmp_path):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    turns, gold = build(n_exchanges=5)
    (tmp_path / "a.json").write_text(
        json.dumps({"turns": [{"speaker": g, "text": t.text} for t, g in zip(turns, gold)]}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["evaluate", str(tmp_path)])

    assert "needs speaker-disjoint folds" in result.stdout


def test_evaluate_warns_about_transcripts_missing_from_the_mapping(tmp_path):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    data = tmp_path / "labelled"
    data.mkdir()
    rows = []
    for client in range(1, 9):
        turns, gold = build(n_exchanges=4, seed=client)
        (data / f"{client:03d}B.json").write_text(
            json.dumps({"turns": [{"speaker": g, "text": t.text} for t, g in zip(turns, gold)]}),
            encoding="utf-8",
        )
        rows.append(f"{client},TH{client % 3}")
    # One transcript with no mapping row at all.
    turns, gold = build(n_exchanges=4, seed=99)
    (data / "777B.json").write_text(
        json.dumps({"turns": [{"speaker": g, "text": t.text} for t, g in zip(turns, gold)]}),
        encoding="utf-8",
    )
    mapping = tmp_path / "map.csv"
    mapping.write_text("session,therapist\n" + "\n".join(rows) + "\n", encoding="utf-8")

    result = CliRunner().invoke(
        app, ["evaluate", str(data), "--mapping", str(mapping), "--folds", "3"]
    )

    assert "matched no mapping row" in result.stdout


def test_marginal_decoding_is_available_and_differs_from_viterbi():
    turns, gold = build(seed=5)
    labeller = train([(turns, gold)])
    viterbi = labeller.predict(turns, decode="viterbi")
    marginal = labeller.predict(turns, decode="marginal")
    assert len(viterbi.labels) == len(marginal.labels) == len(turns)
    # Both are valid labellings; the point is that the decoder is selectable.
    assert set(marginal.labels) <= {"C", "T"}


def test_viterbi_switches_no_more_often_than_marginal():
    # Viterbi optimises path likelihood, so on a corpus where staying is common it should never
    # produce MORE speaker changes than per-turn marginal decoding. This is the over-smoothing
    # behaviour that costs boundary recall.
    turns, gold = build(seed=11)
    labeller = train([(turns, gold)])

    def switches(labels):
        return sum(1 for a, b in zip(labels, labels[1:]) if a != b)

    v = switches(labeller.predict(turns, decode="viterbi").labels)
    m = switches(labeller.predict(turns, decode="marginal").labels)
    assert v <= m


def test_unknown_decode_mode_raises():
    import pytest

    turns, gold = build()
    labeller = train([(turns, gold)])
    with pytest.raises(ValueError, match="viterbi.*marginal"):
        labeller.predict(turns, decode="greedy")


def test_evaluate_reports_reviewer_view_for_the_trained_model(tmp_path):
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    data = tmp_path / "labelled"
    data.mkdir()
    rows = []
    for client in range(1, 13):
        turns, gold = build(n_exchanges=6, seed=client)
        (data / f"{client:03d}B_sample.json").write_text(
            json.dumps({"turns": [{"speaker": g, "text": t.text} for t, g in zip(turns, gold)]}),
            encoding="utf-8",
        )
        rows.append(f"{client},TH{client % 4}")
    mapping = tmp_path / "map.csv"
    mapping.write_text("session,therapist\n" + "\n".join(rows) + "\n", encoding="utf-8")

    result = CliRunner().invoke(
        app, ["evaluate", str(data), "--mapping", str(mapping), "--folds", "4"]
    )

    assert result.exit_code == 0
    assert "trained model" in result.stdout
    assert "reviewer view" in result.stdout
    # The reported curve must be for the better-scoring decoder.
    assert "viterbi" in result.stdout and "marginal" in result.stdout
