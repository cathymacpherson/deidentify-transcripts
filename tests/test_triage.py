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
