import json

from deidentify_transcripts.io import load_transcript, save_outputs
from deidentify_transcripts.schemas import DeidReport, RunMetadata, Transcript, Turn


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
