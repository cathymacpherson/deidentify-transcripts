import json
from pathlib import Path

from typer.testing import CliRunner

from deidentify_transcripts.cli import app
from deidentify_transcripts.inventory import (
    LABELLED,
    PARTIAL,
    REVIEW,
    UNLABELLED,
    classify_speaker,
    inspect_transcript,
)

runner = CliRunner()


def write_json(path: Path, speakers: list[str]) -> Path:
    path.write_text(
        json.dumps(
            {
                "transcript_id": path.stem,
                "turns": [{"speaker": s, "text": f"line {i}"} for i, s in enumerate(speakers)],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_classify_speaker_variants():
    assert classify_speaker("C") == "client"
    assert classify_speaker(" t ") == "therapist"
    assert classify_speaker("Therapist") == "therapist"
    assert classify_speaker("unknown") == "unlabelled"
    assert classify_speaker("") == "unlabelled"
    assert classify_speaker("CC") == "other"


def test_fully_labelled_transcript_is_labelled(tmp_path):
    record = inspect_transcript(write_json(tmp_path / "a.json", ["C", "T", "C", "T"]))
    assert record.completeness == 1.0
    assert record.bucket(0.5) == LABELLED


def test_wholly_unlabelled_transcript_is_unlabelled(tmp_path):
    record = inspect_transcript(write_json(tmp_path / "b.json", ["unknown"] * 5))
    assert record.role_labelled == 0
    assert record.bucket(0.5) == UNLABELLED


def test_partially_labelled_is_processable_not_review(tmp_path):
    # 2 of 10 turns labelled - the labeller fills the gaps and keeps the two human labels.
    record = inspect_transcript(write_json(tmp_path / "c.json", ["C", "T"] + ["unknown"] * 8))
    assert record.completeness == 0.2
    assert record.bucket(0.5) == PARTIAL
    assert record.bucket(0.1) == LABELLED


def test_partial_with_unrecognised_values_still_needs_review(tmp_path):
    # Real labels plus an unrecognised vocabulary is a human decision, not a gap to fill.
    record = inspect_transcript(
        write_json(tmp_path / "d.json", ["C", "Interviewer"] + ["unknown"] * 8)
    )
    assert record.bucket(0.5) == REVIEW


def test_labelled_transcript_can_still_report_residual_gaps(tmp_path):
    record = inspect_transcript(write_json(tmp_path / "e.json", ["C"] * 9 + ["unknown"]))
    assert record.bucket(0.5) == LABELLED
    assert record.residual_unlabelled == 1


def test_mostly_labelled_with_a_few_blanks_still_counts_as_labelled(tmp_path):
    record = inspect_transcript(write_json(tmp_path / "d.json", ["C"] * 9 + ["unknown"]))
    assert record.bucket(0.5) == LABELLED


def test_unrecognised_speaker_values_go_to_review(tmp_path):
    record = inspect_transcript(write_json(tmp_path / "e.json", ["Interviewer", "Participant"]))
    assert record.role_labelled == 0
    assert record.other == 2
    assert record.bucket(0.5) == REVIEW  # not silently filed as unlabelled
    assert record.other_values == {"Interviewer", "Participant"}


def test_unreadable_file_is_reported_not_raised(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    record = inspect_transcript(bad)
    assert record.error is not None
    assert record.bucket(0.5) == REVIEW


def test_cli_dry_run_moves_nothing(tmp_path):
    write_json(tmp_path / "a.json", ["C", "T"])
    write_json(tmp_path / "b.json", ["unknown", "unknown"])

    result = runner.invoke(app, ["inventory", str(tmp_path)])

    assert result.exit_code == 0
    assert "dry run" in result.stdout
    assert (tmp_path / "a.json").exists()
    assert not (tmp_path / "labelled").exists()


def test_cli_apply_sorts_into_folders(tmp_path):
    write_json(tmp_path / "a.json", ["C", "T"])
    write_json(tmp_path / "b.json", ["unknown", "unknown"])
    write_json(tmp_path / "c.json", ["C"] + ["unknown"] * 9)

    result = runner.invoke(app, ["inventory", str(tmp_path), "--apply"])

    assert result.exit_code == 0
    assert (tmp_path / "labelled" / "a.json").exists()
    assert (tmp_path / "unlabelled" / "b.json").exists()
    assert (tmp_path / "partial" / "c.json").exists()


def test_cli_reports_residual_gaps_in_labelled_files(tmp_path):
    write_json(tmp_path / "a.json", ["C"] * 9 + ["unknown"])

    result = runner.invoke(app, ["inventory", str(tmp_path)])

    assert "still contain unlabelled turns" in result.stdout
    assert "1 turns in total" in result.stdout


def test_cli_output_contains_no_filenames_or_content(tmp_path):
    write_json(tmp_path / "secret-participant-007.json", ["C", "T"])

    result = runner.invoke(app, ["inventory", str(tmp_path)])

    assert "secret-participant-007" not in result.stdout
    assert "line 0" not in result.stdout


def test_cli_rerun_does_not_rescan_destination_folders(tmp_path):
    write_json(tmp_path / "a.json", ["C", "T"])
    runner.invoke(app, ["inventory", str(tmp_path), "--apply"])

    result = runner.invoke(app, ["inventory", str(tmp_path)])

    assert result.exit_code == 1
    assert "no transcripts found" in result.stdout + result.stderr


def test_manifest_records_per_file_detail(tmp_path):
    write_json(tmp_path / "a.json", ["C", "T"])
    manifest = tmp_path / "manifest.csv"

    runner.invoke(app, ["inventory", str(tmp_path), "--manifest", str(manifest)])

    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("file,bucket,turns")
    assert "a.json" in rows[1]


def test_split_roles_detects_merged_turns():
    from deidentify_transcripts.inventory import is_multi_role, split_roles

    assert split_roles("C/T") == ["client", "therapist"]
    assert split_roles("T/C") == ["therapist", "client"]
    assert split_roles("C + T") == ["client", "therapist"]
    assert split_roles("C and T") == ["client", "therapist"]
    assert split_roles("Client/Therapist") == ["client", "therapist"]
    assert is_multi_role("C/T")


def test_split_roles_ignores_non_role_and_single_values():
    from deidentify_transcripts.inventory import is_multi_role, split_roles

    assert split_roles("C") == []
    assert split_roles("CC") == []
    assert split_roles("unknown") == []
    assert split_roles("C/Interviewer") == []  # unrecognised part - not a clean merge
    assert not is_multi_role("C")


def test_audit_counts_merged_turns_across_files(tmp_path):
    from deidentify_transcripts.inventory import audit_speaker_values

    write_json(tmp_path / "a.json", ["C", "T", "C/T", "C"])
    write_json(tmp_path / "b.json", ["C", "T"])
    write_json(tmp_path / "c.json", ["C/T", "C/T"])

    audit = audit_speaker_values(sorted(tmp_path.glob("*.json")))

    assert audit.files_scanned == 3
    assert audit.total_turns == 8
    assert audit.multi_role_turns == 3
    assert audit.multi_role_files == 2
    assert audit.multi_role_values == ["C/T"]
    assert audit.files_by_value["C"] == 2


def test_speaker_audit_cli_reports_merged_and_hides_filenames(tmp_path):
    write_json(tmp_path / "participant-0042.json", ["C", "T", "C/T"])

    result = runner.invoke(app, ["speaker-audit", str(tmp_path)])

    assert result.exit_code == 0
    assert "MERGED" in result.stdout
    assert "1 turn(s) across 1 file(s)" in result.stdout
    assert "participant-0042" not in result.stdout


def test_speaker_audit_cli_reports_clean_corpus(tmp_path):
    write_json(tmp_path / "a.json", ["C", "T"])

    result = runner.invoke(app, ["speaker-audit", str(tmp_path)])

    assert "No merged-speaker turns found." in result.stdout


def test_speaker_audit_manifest_lists_values_per_file(tmp_path):
    write_json(tmp_path / "a.json", ["C", "C/T"])
    manifest = tmp_path / "audit.csv"

    runner.invoke(app, ["speaker-audit", str(tmp_path), "--manifest", str(manifest)])

    text = manifest.read_text(encoding="utf-8")
    assert "file,speaker_value,turns,kind" in text
    assert "MERGED" in text


def test_rare_typo_does_not_block_an_otherwise_coded_file(tmp_path):
    # One mistyped label in 100 turns is a turn-level problem, not a file-level one.
    record = inspect_transcript(write_json(tmp_path / "a.json", ["C"] * 99 + ["CC"]))
    assert record.other == 1
    assert record.bucket(0.5) == LABELLED


def test_rare_merged_turn_does_not_block_a_partial_file(tmp_path):
    speakers = ["C", "T", "C/T"] + ["unknown"] * 97
    record = inspect_transcript(write_json(tmp_path / "b.json", speakers))
    assert record.merged == 1
    assert record.merged_values == {"C/T"}
    assert record.bucket(0.5) == PARTIAL


def test_pervasive_odd_values_still_send_a_file_to_review(tmp_path):
    record = inspect_transcript(write_json(tmp_path / "c.json", ["C"] * 50 + ["CC"] * 50))
    assert record.unrecognised_rate == 0.5
    assert record.bucket(0.5) == REVIEW


def test_anomaly_tolerance_is_adjustable(tmp_path):
    record = inspect_transcript(write_json(tmp_path / "d.json", ["C"] * 95 + ["CC"] * 5))
    assert record.bucket(0.5, anomaly_tolerance=0.02) == REVIEW
    assert record.bucket(0.5, anomaly_tolerance=0.10) == LABELLED


def test_merged_turns_are_not_counted_as_role_labels(tmp_path):
    record = inspect_transcript(write_json(tmp_path / "e.json", ["C/T"] * 10))
    assert record.role_labelled == 0
    assert record.merged == 10
    assert record.bucket(0.5) == REVIEW  # wholly merged is a genuine vocabulary problem


def test_cli_reports_tolerated_anomalies(tmp_path):
    write_json(tmp_path / "a.json", ["C"] * 98 + ["CC"] + ["C/T"])

    result = runner.invoke(app, ["inventory", str(tmp_path)])

    assert "filed normally despite containing odd turns" in result.stdout
    assert "1 mistyped label(s), 1 merged-speaker turn(s)" in result.stdout


def test_merged_turns_alone_never_trigger_file_review(tmp_path):
    # Many merged turns alongside real labels: still processable, handled turn by turn.
    speakers = ["C"] * 40 + ["T"] * 40 + ["C/T"] * 20
    record = inspect_transcript(write_json(tmp_path / "a.json", speakers))
    assert record.merged == 20
    assert record.unrecognised_rate == 0.0
    assert record.bucket(0.5) == LABELLED


def test_partial_file_with_several_merged_turns_is_partial(tmp_path):
    speakers = ["C", "T"] + ["C/T"] * 3 + ["unknown"] * 95
    record = inspect_transcript(write_json(tmp_path / "b.json", speakers))
    assert record.bucket(0.5) == PARTIAL


def test_only_unrecognised_values_trigger_review(tmp_path):
    record = inspect_transcript(write_json(tmp_path / "c.json", ["C"] * 90 + ["CC"] * 10))
    assert record.unrecognised_rate == 0.1
    assert record.bucket(0.5) == REVIEW


def test_review_reason_explains_unrecognised_values(tmp_path):
    from deidentify_transcripts.inventory import review_reason

    record = inspect_transcript(write_json(tmp_path / "d.json", ["00"] * 10))
    reason = review_reason(record, 0.5)
    assert "unrecognised speaker value" in reason
    assert "'00'" in reason


def test_review_reason_explains_unreadable_file(tmp_path):
    from deidentify_transcripts.inventory import review_reason

    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert "could not be read" in review_reason(inspect_transcript(bad), 0.5)


def test_explain_flag_prints_reasons(tmp_path):
    write_json(tmp_path / "odd.json", ["Interviewer"] * 10)

    result = runner.invoke(app, ["inventory", str(tmp_path), "--explain"])

    assert "why each review/ file was held back" in result.stdout
    assert "odd.json" in result.stdout
    assert "Interviewer" in result.stdout


def test_blank_file_with_a_few_stray_values_is_unlabelled_not_review(tmp_path):
    # No role labels, one stray value under tolerance: needs labelling from scratch.
    record = inspect_transcript(write_json(tmp_path / "a.json", ["unknown"] * 99 + ["00"]))
    assert record.role_labelled == 0
    assert record.other == 1
    assert record.bucket(0.5) == UNLABELLED


def test_blank_file_with_many_stray_values_still_reviews(tmp_path):
    record = inspect_transcript(write_json(tmp_path / "b.json", ["unknown"] * 50 + ["00"] * 50))
    assert record.bucket(0.5) == REVIEW


def test_merged_only_file_reviews_with_a_clear_reason(tmp_path):
    from deidentify_transcripts.inventory import review_reason

    record = inspect_transcript(write_json(tmp_path / "c.json", ["C/T"] * 10))
    assert record.bucket(0.5) == REVIEW
    assert "merged-speaker marks only" in review_reason(record, 0.5)


def test_review_reason_never_returns_unclassified(tmp_path):
    from deidentify_transcripts.inventory import review_reason

    cases = [
        ["C/T"] * 10,
        ["Interviewer"] * 10,
        ["unknown"] * 50 + ["00"] * 50,
    ]
    for i, speakers in enumerate(cases):
        record = inspect_transcript(write_json(tmp_path / f"r{i}.json", speakers))
        reason = review_reason(record, 0.5)
        assert reason and "unrecognised reason" not in reason


def test_explain_lists_tolerated_odd_turns_with_values(tmp_path):
    write_json(tmp_path / "a.json", ["unknown"] * 99 + ["00"])

    result = runner.invoke(app, ["inventory", str(tmp_path), "--explain"])

    assert "odd turns tolerated in files that were filed normally" in result.stdout
    assert "'00'" in result.stdout


def test_display_speaker_value_exposes_hidden_characters():
    from deidentify_transcripts.inventory import display_speaker_value

    assert display_speaker_value("C") == "C"
    assert display_speaker_value("") == "(empty)"
    assert display_speaker_value("unknown ") == "'unknown '"
    assert display_speaker_value("unknown ") == repr("unknown ")
    assert display_speaker_value("–") == repr("–")


def test_speaker_audit_accepts_a_single_file(tmp_path):
    target = write_json(tmp_path / "one.json", ["C", "T", "C/T"])

    result = runner.invoke(app, ["speaker-audit", str(target)])

    assert result.exit_code == 0
    assert "1 file(s)" in result.stdout
    assert "MERGED" in result.stdout


def test_speaker_audit_shows_whitespace_variants_distinctly(tmp_path):
    target = write_json(tmp_path / "ws.json", ["unknown", "unknown ", "unknown "])

    result = runner.invoke(app, ["speaker-audit", str(target)])

    assert "3 distinct speaker value(s)" in result.stdout
    assert "'unknown '" in result.stdout


def test_speaker_audit_accepts_several_directories(tmp_path):
    (tmp_path / "labelled").mkdir()
    (tmp_path / "partial").mkdir()
    (tmp_path / "unlabelled").mkdir()
    write_json(tmp_path / "labelled" / "a.json", ["C", "T"])
    write_json(tmp_path / "partial" / "b.json", ["C", "C/T"] + ["unknown"] * 18)
    write_json(tmp_path / "unlabelled" / "c.json", ["unknown"] * 10)
    manifest = tmp_path / "audit.csv"

    result = runner.invoke(
        app,
        ["speaker-audit", str(tmp_path / "labelled"), str(tmp_path / "partial"),
         "--manifest", str(manifest)],
    )

    assert result.exit_code == 0
    assert "2 file(s)" in result.stdout
    text = manifest.read_text(encoding="utf-8")
    assert "a.json" in text and "b.json" in text
    assert "c.json" not in text  # unlabelled/ was not asked for
