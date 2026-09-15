from deidentify_transcripts.metrics import review_burden_curve, score_labels


def test_perfect_prediction_scores_one():
    gold = ["C", "C", "T", "C"]
    result = score_labels(gold, list(gold))
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.boundary_f1 == 1.0
    assert not result.looks_flipped


def test_majority_class_has_high_accuracy_but_poor_macro_f1():
    # The whole reason accuracy is not reported alone.
    gold = ["C"] * 80 + ["T"] * 20
    predicted = ["C"] * 100
    result = score_labels(gold, predicted)
    assert result.accuracy == 0.8
    assert result.macro_f1 < 0.5
    assert result.per_class["T"].recall == 0.0


def test_inverted_roles_are_detected():
    gold = ["C", "C", "T", "T", "C"]
    predicted = ["T", "T", "C", "C", "T"]
    result = score_labels(gold, predicted)
    assert result.looks_flipped
    assert result.flipped_macro_f1 == 1.0


def test_boundary_f1_ignores_which_role_is_which():
    # Segmentation is perfect; role assignment is inverted.
    gold = ["C", "C", "T", "C"]
    predicted = ["T", "T", "C", "T"]
    result = score_labels(gold, predicted)
    assert result.boundary_f1 == 1.0
    assert result.macro_f1 == 0.0


def test_boundary_f1_penalises_missed_changes():
    gold = ["C", "T", "C", "T"]
    predicted = ["C", "C", "C", "C"]
    result = score_labels(gold, predicted)
    assert result.boundary_f1 == 0.0
    assert result.boundary_support == 3


def test_turns_without_usable_gold_are_skipped():
    gold = ["C", None, "C/T", "X", "T"]
    predicted = ["C", "C", "C", "C", "T"]
    result = score_labels(gold, predicted)
    assert result.scored_turns == 2
    assert result.skipped_turns == 3
    assert result.accuracy == 1.0


def test_all_gold_unusable_returns_empty_score():
    result = score_labels([None, None], ["C", "T"])
    assert result.scored_turns == 0
    assert result.macro_f1 == 0.0


def test_mismatched_lengths_raise():
    import pytest

    with pytest.raises(ValueError):
        score_labels(["C"], ["C", "T"])


def test_review_burden_trades_coverage_for_accuracy():
    gold = ["C", "C", "T", "T"]
    predicted = ["C", "C", "T", "C"]     # last one wrong
    confidence = [0.9, 0.9, 0.9, 0.1]    # and least confident
    points = review_burden_curve(gold, predicted, confidence, steps=(1.0, 0.75))
    assert points[0].accuracy == 0.75
    assert points[0].reviewed == 0
    assert points[1].accuracy == 1.0     # giving up the uncertain turn fixes it
    assert points[1].reviewed == 1


def test_review_burden_skips_unusable_gold():
    points = review_burden_curve(["C", None], ["C", "C"], [0.9, 0.9], steps=(1.0,))
    assert points[0].reviewed == 0
    assert points[0].accuracy == 1.0


def test_error_capture_rewards_informative_confidence():
    from deidentify_transcripts.metrics import error_capture_curve

    gold = ["C"] * 90 + ["T"] * 10
    predicted = ["C"] * 100                       # every T is wrong
    # Confidence that knows where it is weak: the errors are the least confident turns.
    good = [0.9] * 90 + [0.05] * 10
    curve = error_capture_curve(gold, predicted, good, budgets=(0.1,))
    assert curve[0].error_capture == 1.0          # all 10 errors flagged within a 10% budget
    assert curve[0].errors_missed == 0


def test_error_capture_punishes_uninformative_confidence():
    from deidentify_transcripts.metrics import error_capture_curve

    gold = ["C"] * 90 + ["T"] * 10
    predicted = ["C"] * 100
    flat = [0.9] * 100                            # confidence tells the reviewer nothing
    curve = error_capture_curve(gold, predicted, flat, budgets=(0.1,))
    assert curve[0].error_capture < 1.0
    assert curve[0].errors_missed > 0


def test_unflagged_error_rate_is_what_survives_review():
    from deidentify_transcripts.metrics import error_capture_curve

    gold = ["C"] * 98 + ["T", "T"]
    predicted = ["C"] * 100
    confidence = [0.9] * 98 + [0.1, 0.9]          # one error flagged, one not
    curve = error_capture_curve(gold, predicted, confidence, budgets=(0.05,))
    assert curve[0].errors_caught == 1
    assert curve[0].errors_missed == 1
    assert curve[0].unflagged_error_rate > 0


def test_edit_actions_counts_blocks_not_turns():
    from deidentify_transcripts.metrics import edit_actions

    gold = ["C"] * 10
    one_run_wrong = ["T"] * 5 + ["C"] * 5         # 5 wrong turns, 1 correction
    scattered = ["T", "C", "T", "C", "T", "C", "C", "C", "C", "C"]  # 3 wrong turns, 3 corrections
    assert edit_actions(gold, one_run_wrong) == 1
    assert edit_actions(gold, scattered) == 3


def test_edit_actions_is_zero_when_perfect():
    from deidentify_transcripts.metrics import edit_actions

    assert edit_actions(["C", "T", "C"], ["C", "T", "C"]) == 0


def test_edit_actions_skips_unusable_gold():
    from deidentify_transcripts.metrics import edit_actions

    assert edit_actions(["C", None, "C"], ["T", "T", "T"]) == 1
