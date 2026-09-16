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


def test_diagnose_reports_whether_disagreement_strength_matters(tmp_path):
    """The specific question: do stronger disagreements mean likelier errors?"""
    import csv

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    report = tmp_path / "r.csv"
    fields = ["file", "turn_id", "gold", "predicted", "confidence", "second_opinion",
              "second_conf", "anchor", "anchor_contradicted", "votes", "error", "text"]
    with report.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        n = 0

        def row(conf, second, second_conf, wrong):
            nonlocal n
            w.writerow({
                "file": "f.json", "turn_id": n, "gold": "C",
                "predicted": "T" if wrong else "C", "confidence": f"{conf:.3f}",
                "second_opinion": second, "second_conf": f"{second_conf:.3f}",
                "anchor": "", "anchor_contradicted": "", "votes": "C|C|C",
                "error": "WRONG" if wrong else "", "text": "x",
            })
            n += 1

        # Strong disagreements mostly wrong; weak ones mostly right.
        for i in range(40):
            row(0.66, "T", 0.98, wrong=i < 20)
        for i in range(40):
            row(0.96, "T", 0.10, wrong=i < 2)
        for i in range(200):
            row(1.0, "C", 0.90, wrong=i < 8)

    result = CliRunner().invoke(app, ["label-diagnose", str(report)])

    assert result.exit_code == 0, result.stdout
    assert "does a STRONGER disagreement mean a likelier error?" in result.stdout
    assert "YES" in result.stdout


def test_diagnose_says_no_when_strength_does_not_matter(tmp_path):
    import csv

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    report = tmp_path / "r.csv"
    fields = ["file", "turn_id", "gold", "predicted", "confidence", "second_opinion",
              "second_conf", "anchor", "anchor_contradicted", "votes", "error", "text"]
    with report.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for i in range(120):
            strong = i % 2 == 0
            w.writerow({
                "file": "f.json", "turn_id": i, "gold": "C",
                "predicted": "T" if i % 10 == 0 else "C",
                "confidence": "0.800", "second_opinion": "T",
                "second_conf": "0.98" if strong else "0.10",
                "anchor": "", "anchor_contradicted": "", "votes": "C|C|C",
                "error": "WRONG" if i % 10 == 0 else "", "text": "x",
            })

    result = CliRunner().invoke(app, ["label-diagnose", str(report)])
    assert "NO - error rate varies by only" in result.stdout


def test_diagnose_refuses_a_misaligned_report(tmp_path):
    """A shifted column must fail loudly, not report zero errors."""
    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    report = tmp_path / "bad.csv"
    # Header has 12 columns; rows have 11 - the exact failure that produced "0 wrong".
    report.write_text(
        "file,turn_id,gold,predicted,confidence,second_opinion,second_conf,anchor,"
        "anchor_contradicted,votes,error,text\n"
        + "\n".join(f"f.json,{i},C,C,1.000,T,,,C|C|C,,x" for i in range(200)) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["label-diagnose", str(report)])
    assert result.exit_code == 1
    assert "misaligned" in result.stdout + str(result.stderr)


def test_diagnose_accepts_a_well_formed_report(tmp_path):
    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    report = tmp_path / "good.csv"
    rows = []
    for i in range(200):
        wrong = "WRONG" if i % 20 == 0 else ""
        rows.append(f"f.json,{i},C,{'T' if wrong else 'C'},1.000,C,0.900,,,C|C|C,{wrong},x")
    report.write_text(
        "file,turn_id,gold,predicted,confidence,second_opinion,second_conf,anchor,"
        "anchor_contradicted,votes,error,text\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["label-diagnose", str(report)])
    assert result.exit_code == 0, result.stdout
    assert "10 wrong" in result.stdout


def _write_labelled(directory, client, n_turns=80, seed=0):
    import json
    import random

    rng = random.Random(seed)
    T = ["What was that like for you?", "And how do you feel now?", "Tell me more."]
    C = ["I kept going over it", "I felt awful", "my week was rough", "yeah"]
    turns, gold = [], []
    while len(turns) < n_turns:
        for _ in range(rng.randint(1, 2)):
            turns.append(rng.choice(T)); gold.append("T")
        for _ in range(rng.randint(2, 5)):
            turns.append(rng.choice(C)); gold.append("C")
    (directory / f"{client:03d}B.json").write_text(
        json.dumps({"turns": [{"speaker": g, "text": t} for t, g in zip(turns, gold)]}),
        encoding="utf-8",
    )
    return turns, gold


def test_rescore_repairs_a_report_missing_second_conf(tmp_path):
    import csv

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    reference = tmp_path / "labelled"
    reference.mkdir()
    mapping_rows = []
    old_rows = []
    for client in range(1, 7):
        turns, gold = _write_labelled(reference, client, seed=client)
        mapping_rows.append(f"{client},TH{client % 3}")
        for i, (text, g) in enumerate(zip(turns, gold)):
            # An 11-column row: the shape the bug produced.
            old_rows.append([
                f"{client:03d}B.json", i, g, g, "1.000", "C", "", "", "C|C|C", "", text,
            ])
    mapping = tmp_path / "map.csv"
    mapping.write_text("session,therapist\n" + "\n".join(mapping_rows) + "\n", encoding="utf-8")

    old = tmp_path / "old.csv"
    with old.open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["file", "turn_id", "gold", "predicted", "confidence", "second_opinion",
                    "second_conf", "anchor", "anchor_contradicted", "votes", "error", "text"])
        w.writerows(old_rows)

    new = tmp_path / "new.csv"
    result = CliRunner().invoke(app, [
        "label-rescore", str(old), "-o", str(new),
        "--mapping", str(mapping), "--reference", str(reference),
    ])

    assert result.exit_code == 0, result.stdout
    assert "realigning" in result.stdout
    with new.open(encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    assert len(rows) == len(old_rows)
    assert all(len(r) == 12 for r in rows)
    # The column that was missing is now populated.
    assert any(r["second_conf"] for r in rows)
    # And it is a usable report.
    assert CliRunner().invoke(app, ["label-diagnose", str(new)]).exit_code == 0


def test_rescore_makes_no_server_calls(tmp_path, monkeypatch):
    """Rescoring must be entirely local - that is the whole point."""
    import csv

    from typer.testing import CliRunner

    from deidentify_transcripts import cli

    def explode(*a, **k):
        raise AssertionError("rescoring must not contact the server")

    monkeypatch.setattr(cli, "LocalModel", explode)

    reference = tmp_path / "labelled"
    reference.mkdir()
    rows, mapping_rows = [], []
    for client in range(1, 5):
        turns, gold = _write_labelled(reference, client, seed=client)
        mapping_rows.append(f"{client},TH{client % 2}")
        for i, (text, g) in enumerate(zip(turns, gold)):
            rows.append([f"{client:03d}B.json", i, g, g, "1.000", "C", "0.9",
                         "", "", "C|C|C", "", text])
    mapping = tmp_path / "map.csv"
    mapping.write_text("session,therapist\n" + "\n".join(mapping_rows) + "\n", encoding="utf-8")
    old = tmp_path / "old.csv"
    with old.open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(cli.REPORT_COLUMNS)
        w.writerows(rows)

    result = CliRunner().invoke(cli.app, [
        "label-rescore", str(old), "-o", str(tmp_path / "new.csv"),
        "--mapping", str(mapping), "--reference", str(reference),
    ])
    assert result.exit_code == 0, result.stdout


def test_audit_lists_confident_disagreements_strongest_first(tmp_path):
    import csv

    from typer.testing import CliRunner

    from deidentify_transcripts import cli

    report = tmp_path / "r.csv"
    with report.open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(cli.REPORT_COLUMNS)
        # Confident, both systems agree against the human - strongest evidence.
        w.writerow(["f.json", 1, "C", "T", "1.000", "T", "0.99", "", "", "T|T|T", "WRONG", "aa"])
        # Confident, but the second system sided with the human.
        w.writerow(["f.json", 2, "C", "T", "0.980", "C", "0.90", "", "", "T|T|T", "WRONG", "bb"])
        # Disagrees but the labeller was unsure - not evidence about the human.
        w.writerow(["f.json", 3, "C", "T", "0.600", "T", "0.50", "", "", "C|T|T", "WRONG", "cc"])
        # Agrees with the human - not a candidate.
        w.writerow(["f.json", 4, "C", "C", "1.000", "C", "0.99", "", "", "C|C|C", "", "dd"])

    out = tmp_path / "audit.csv"
    result = CliRunner().invoke(cli.app, ["label-audit", str(report), "-o", str(out)])

    assert result.exit_code == 0, result.stdout
    with out.open(encoding="utf-8", newline="") as h:
        rows = list(csv.DictReader(h))
    assert [r["turn_id"] for r in rows] == ["1", "2"]     # the unsure one is excluded
    assert rows[0]["second_system_agrees"] == "yes"       # strongest evidence first
    assert rows[0]["human_label"] == "C"
    assert rows[0]["suggested_label"] == "T"
    assert rows[0]["verdict"] == ""
    assert "1 confident disagreement" in result.stdout or "2 confident" in result.stdout
    assert "floor on the true label error rate" in result.stdout


def test_audit_rejects_a_report_without_gold_labels(tmp_path):
    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    report = tmp_path / "r.csv"
    report.write_text("turn_id,speaker\n1,C\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["label-audit", str(report), "-o", str(tmp_path / "o.csv")])
    assert result.exit_code == 1
    assert "no gold labels" in result.stdout + str(result.stderr)


def test_summary_bins_continuous_confidence(tmp_path):
    """Weighted confidence is effectively continuous; a row per value is unreadable."""
    import csv
    import json
    import random

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    rng = random.Random(0)
    turns_out = [
        {"turn_id": i, "speaker": "C",
         "speaker_confidence": round(rng.uniform(0.4, 1.0), 3),
         "speaker_source": "model", "text": f"t{i}"}
        for i in range(600)
    ]
    labelled = tmp_path / "a.json"
    labelled.write_text(json.dumps({"transcript_id": "a", "turns": turns_out}), encoding="utf-8")

    cal = tmp_path / "cal.csv"
    with cal.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["gold", "confidence", "error"])
        w.writeheader()
        for _ in range(400):
            conf = rng.uniform(0.4, 1.0)
            w.writerow({"gold": "C", "confidence": f"{conf:.3f}",
                        "error": "WRONG" if rng.random() > conf else ""})

    result = CliRunner().invoke(
        app, ["label-summary", str(labelled), "--calibration", str(cal)]
    )

    assert result.exit_code == 0, result.stdout
    lines = result.stdout.splitlines()
    # Bands, not one row per distinct value: the output must stay short.
    assert len(lines) < 40, f"output is {len(lines)} lines - not binned"
    assert "0.65 - 0.75" in result.stdout
    assert "errors expected in 600 turns" in result.stdout


def test_summary_marks_bands_with_too_few_samples(tmp_path):
    import csv
    import json

    from typer.testing import CliRunner

    from deidentify_transcripts.cli import app

    labelled = tmp_path / "a.json"
    labelled.write_text(json.dumps({"transcript_id": "a", "turns": [
        {"turn_id": i, "speaker": "C", "speaker_confidence": 0.7,
         "speaker_source": "model", "text": "x"} for i in range(50)]}), encoding="utf-8")

    cal = tmp_path / "cal.csv"
    with cal.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["gold", "confidence", "error"])
        w.writeheader()
        for i in range(5):        # a band measured on only five turns
            w.writerow({"gold": "C", "confidence": "0.700", "error": "WRONG" if i < 2 else ""})

    result = CliRunner().invoke(
        app, ["label-summary", str(labelled), "--calibration", str(cal)]
    )
    assert "few samples" in result.stdout
