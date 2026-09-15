import pytest

from deidentify_transcripts.triage import FileStats, spearman, triage_curve


def test_spearman_detects_a_perfect_relationship():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_zero_for_no_relationship():
    assert abs(spearman([1, 2, 3, 4], [1, 1, 1, 1])) < 1e-9


def test_spearman_handles_ties():
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


def test_spearman_needs_enough_points():
    assert spearman([1, 2], [2, 1]) == 0.0


def test_spearman_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


def test_triage_puts_the_worst_looking_files_first():
    stats = [
        FileStats("clean", turns=100, flagged=5, errors=1),
        FileStats("messy", turns=100, flagged=60, errors=30),
        FileStats("middling", turns=100, flagged=20, errors=10),
    ]
    curve = triage_curve(stats)
    # Routing one file should capture the messy one's errors.
    assert curve[0].files_routed == 1
    assert curve[0].share_of_errors == pytest.approx(30 / 41)


def test_triage_curve_ends_at_everything():
    stats = [FileStats(f"f{i}", turns=10, flagged=i, errors=i) for i in range(1, 5)]
    curve = triage_curve(stats)
    assert curve[-1].share_of_files == 1.0
    assert curve[-1].share_of_errors == pytest.approx(1.0)


def test_triage_handles_no_errors():
    stats = [FileStats("a", turns=10, flagged=1, errors=0)]
    assert triage_curve(stats)[0].share_of_errors == 0.0


def test_triage_handles_empty_input():
    assert triage_curve([]) == []


def test_high_rho_can_still_mean_low_lift():
    """Ordering can be perfect while the files barely differ — rho alone is misleading."""
    stats = [FileStats(f"f{i}", turns=100, flagged=10 + i, errors=10 + i) for i in range(8)]
    rho = spearman([s.flag_rate for s in stats], [s.error_rate for s in stats])
    curve = triage_curve(stats)
    quarter = curve[len(curve) // 4 - 1]
    lift = quarter.share_of_errors / quarter.share_of_turns
    assert rho == pytest.approx(1.0)      # ranking is perfect
    assert lift < 1.5                     # but routing on it gains little


def test_calibration_measures_error_rate_per_confidence():
    from deidentify_transcripts.triage import calibration_from_report

    rows = (
        [{"gold": "C", "confidence": "1.000", "error": ""} for _ in range(90)]
        + [{"gold": "C", "confidence": "1.000", "error": "WRONG"} for _ in range(10)]
        + [{"gold": "C", "confidence": "0.667", "error": "WRONG"} for _ in range(5)]
        + [{"gold": "C", "confidence": "0.667", "error": ""} for _ in range(5)]
    )
    cal = calibration_from_report(rows)
    assert cal[1.0] == pytest.approx(0.10)
    assert cal[0.667] == pytest.approx(0.50)


def test_calibration_ignores_turns_with_no_gold_label():
    from deidentify_transcripts.triage import calibration_from_report

    rows = [
        {"gold": "unknown", "confidence": "1.000", "error": ""},
        {"gold": "C", "confidence": "1.000", "error": "WRONG"},
    ]
    assert calibration_from_report(rows) == {1.0: 1.0}


def test_expected_errors_applies_measured_rates():
    from deidentify_transcripts.triage import expected_errors

    out = expected_errors({1.0: 700, 0.667: 30}, {1.0: 0.04, 0.667: 0.29})
    assert out[0] == (1.0, 700, pytest.approx(28.0))
    assert out[1] == (0.667, 30, pytest.approx(8.7))


def test_expected_errors_falls_back_to_the_nearest_measured_level():
    from deidentify_transcripts.triage import expected_errors

    out = expected_errors({0.85: 100}, {0.667: 0.30, 1.0: 0.04})
    assert out[0][2] == pytest.approx(4.0)  # 0.85 is nearer 1.0 than 0.667


def test_expected_errors_with_no_calibration_is_empty():
    from deidentify_transcripts.triage import expected_errors

    assert expected_errors({1.0: 10}, {}) == []


def test_label_summary_shows_expected_errors(tmp_path):
    import csv
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    turns_out = [
        {"turn_id": i, "speaker": "C",
         "speaker_confidence": 1.0 if i >= 20 else 0.667,
         "speaker_source": "model", "text": f"t{i}"}
        for i in range(100)
    ]
    labelled = tmp_path / "a.json"
    labelled.write_text(json.dumps({"transcript_id": "a", "turns": turns_out}), encoding="utf-8")

    cal = tmp_path / "cal.csv"
    with cal.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["gold", "confidence", "error"])
        w.writeheader()
        for _ in range(96):
            w.writerow({"gold": "C", "confidence": "1.000", "error": ""})
        for _ in range(4):
            w.writerow({"gold": "C", "confidence": "1.000", "error": "WRONG"})
        for _ in range(7):
            w.writerow({"gold": "C", "confidence": "0.667", "error": "WRONG"})
        for _ in range(13):
            w.writerow({"gold": "C", "confidence": "0.667", "error": ""})

    result = CliRunner().invoke(
        app, ["label-summary", str(labelled), "--calibration", str(cal)]
    )

    assert result.exit_code == 0, result.stdout
    assert "expected errors" in result.stdout
    assert "errors expected in 100 turns" in result.stdout
