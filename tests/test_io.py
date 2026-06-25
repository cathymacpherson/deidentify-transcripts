import json

from deidentify_transcripts.io import load_transcript, save_outputs
from deidentify_transcripts.schemas import DeidReport, Transcript, Turn


def test_load_plain_text_and_save_without_raw_text(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("Interviewer: Hello\nParticipant: My name is Alex\n", encoding="utf-8")

    transcript = load_transcript(source, "p1")
    transcript.turns[1].anonymised_text = "My name is [NAME_1]"
    transcript.turns[0].anonymised_text = "Hello"
    report = DeidReport(transcript_id="p1", registry={"alex": "[NAME_1]"})

    anonymised_path, report_path, queue_path = save_outputs(transcript, report, tmp_path / "out")
    saved = json.loads(anonymised_path.read_text(encoding="utf-8"))

    assert saved["turns"][1]["text"] == "My name is [NAME_1]"
    assert "Alex" not in anonymised_path.read_text(encoding="utf-8")
    assert "Alex" not in report_path.read_text(encoding="utf-8")
    assert queue_path.exists()

