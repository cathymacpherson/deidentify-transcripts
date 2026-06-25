from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import DeidReport, Transcript, Turn


def load_transcript(path: Path, transcript_id: str | None = None) -> Transcript:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if transcript_id:
            data["transcript_id"] = transcript_id
        turns = [
            Turn(
                turn_id=index,
                speaker=turn.get("speaker", "unknown"),
                text=turn.get("text", turn.get("raw_text", "")),
            )
            for index, turn in enumerate(data["turns"])
        ]
        return Transcript(
            transcript_id=data.get("transcript_id", transcript_id or path.stem),
            turns=turns,
        )

    turns: list[Turn] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([^:]{1,80}):\s*(.*)$", line)
        speaker, text = (match.group(1).strip(), match.group(2)) if match else ("unknown", line)
        turns.append(Turn(turn_id=len(turns), speaker=speaker, text=text))
    return Transcript(transcript_id=transcript_id or path.stem, turns=turns)


def save_outputs(
    transcript: Transcript,
    report: DeidReport,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    anonymised_dir = output_dir / "anonymised"
    sensitive_dir = output_dir / "sensitive"
    anonymised_dir.mkdir(parents=True, exist_ok=True)
    sensitive_dir.mkdir(parents=True, exist_ok=True)

    anonymised_path = anonymised_dir / f"{transcript.transcript_id}.json"
    anonymised_path.write_text(
        json.dumps(
            {
                "transcript_id": transcript.transcript_id,
                "turns": [
                    {
                        "turn_id": turn.turn_id,
                        "speaker": turn.speaker,
                        "text": turn.anonymised_text,
                    }
                    for turn in transcript.turns
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_path = sensitive_dir / f"{transcript.transcript_id}.deid-report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    queue_path = sensitive_dir / f"{transcript.transcript_id}.review-queue.jsonl"
    with queue_path.open("w", encoding="utf-8") as handle:
        for item in report.review_items:
            handle.write(item.model_dump_json() + "\n")

    return anonymised_path, report_path, queue_path

