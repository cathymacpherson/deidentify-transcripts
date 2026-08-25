import json
from datetime import time
from pathlib import Path

import pytest
from openpyxl import Workbook

from deidentify_transcripts.io import load_transcript, save_outputs
from deidentify_transcripts.schemas import DeidReport, RunMetadata, Transcript, Turn

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_load_plain_text_and_save_without_raw_text(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("Interviewer: Hello\nParticipant: My name is Alex\n", encoding="utf-8")

    transcript = load_transcript(source, "p1")
    transcript.turns[1].anonymised_text = "My name is [NAME_1]"
    transcript.turns[0].anonymised_text = "Hello"
    report = DeidReport(
        transcript_id="p1",
        run_metadata=RunMetadata(
            model="gemma4:12b",
            model_digest="abc123",
            pipeline_version="0.1.0",
            started_at_utc="2026-06-26T00:00:00Z",
        ),
        registry={"alex": "[NAME_1]"},
    )

    anonymised_path, report_path, queue_path = save_outputs(transcript, report, tmp_path / "out")
    saved = json.loads(anonymised_path.read_text(encoding="utf-8"))
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert saved["turns"][1]["text"] == "My name is [NAME_1]"
    assert "Alex" not in anonymised_path.read_text(encoding="utf-8")
    assert "Alex" not in report_path.read_text(encoding="utf-8")
    assert saved_report["run_metadata"]["model"] == "gemma4:12b"
    assert saved_report["run_metadata"]["model_digest"] == "abc123"
    assert queue_path.exists()


def _write_xlsx(path, rows):
    workbook = Workbook()
    sheet = workbook.active
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def test_load_xlsx_transcript(tmp_path):
    source = tmp_path / "input.xlsx"
    _write_xlsx(
        source,
        [
            ["Start Time", "Stop Time", "Transcript", "Speaker"],
            [time(0, 0, 1), time(0, 0, 5), "Hello", "Interviewer"],
            [time(0, 0, 6), time(0, 0, 12), "My name is Alex", "Participant"],
            [time(0, 0, 13), time(0, 0, 13), "", "Participant"],
        ],
    )

    transcript = load_transcript(source, "p1")

    assert transcript.transcript_id == "p1"
    assert len(transcript.turns) == 2
    assert transcript.turns[0].speaker == "Interviewer"
    assert transcript.turns[0].text == "Hello"
    assert transcript.turns[0].start_time == "00:00:01"
    assert transcript.turns[0].stop_time == "00:00:05"
    assert transcript.turns[1].text == "My name is Alex"


def test_load_legacy_xls_transcript():
    transcript = load_transcript(EXAMPLES_DIR / "sample-transcript.xls")

    assert transcript.transcript_id == "sample-transcript"
    assert len(transcript.turns) == 3
    assert transcript.turns[0].speaker == "Interviewer"
    assert transcript.turns[0].text == "Could you tell me where you went to school?"
    assert transcript.turns[0].start_time == "00:00:00"
    assert transcript.turns[0].stop_time == "00:00:04"
    assert transcript.turns[1].speaker == "Participant"
    assert "415-555-0134" in transcript.turns[1].text


def test_load_xlsx_missing_required_column_raises(tmp_path):
    source = tmp_path / "input.xlsx"
    _write_xlsx(source, [["Start Time", "Stop Time", "Transcript"], [None, None, "Hello"]])

    with pytest.raises(ValueError, match="speaker"):
        load_transcript(source, "p1")


def test_anonymised_output_includes_times_only_when_present(tmp_path):
    transcript = Transcript(
        transcript_id="p1",
        turns=[
            Turn(turn_id=0, speaker="Interviewer", text="Hello", anonymised_text="Hello"),
            Turn(
                turn_id=1,
                speaker="Participant",
                text="Hi",
                anonymised_text="Hi",
                start_time="00:00:06",
                stop_time="00:00:12",
            ),
        ],
    )
    report = DeidReport(transcript_id="p1")

    anonymised_path, _, _ = save_outputs(transcript, report, tmp_path / "out")
    saved = json.loads(anonymised_path.read_text(encoding="utf-8"))

    assert "start_time" not in saved["turns"][0]
    assert saved["turns"][1]["start_time"] == "00:00:06"
    assert saved["turns"][1]["stop_time"] == "00:00:12"
