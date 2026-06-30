from deidentify_transcripts.pipeline import deidentify
from deidentify_transcripts.schemas import PiiSpan, RunMetadata, Transcript, Turn


def test_pipeline_uses_stable_tokens_and_reports_clean():
    transcript = Transcript(
        transcript_id="t1",
        turns=[
            Turn(turn_id=0, speaker="participant", text="Alex attended Oak Park School."),
            Turn(turn_id=1, speaker="interviewer", text="Did Alex enjoy it?"),
        ],
    )

    def detect(turn_id, text):
        spans = []
        for value, pii_type in [("Alex", "person_name"), ("Oak Park School", "school")]:
            start = 0
            while (index := text.find(value, start)) >= 0:
                spans.append(
                    PiiSpan(
                        turn_id=turn_id,
                        start=index,
                        end=index + len(value),
                        text=value,
                        pii_type=pii_type,
                        source="llm",
                    )
                )
                start = index + len(value)
        return spans

    output, report = deidentify(
        transcript,
        detect_fn=detect,
        residual_fn=lambda turn_id, text: [],
    )

    assert output.turns[0].anonymised_text == "[NAME_1] attended [SCHOOL_1]."
    assert output.turns[1].anonymised_text == "Did [NAME_1] enjoy it?"
    assert report.status == "clean"
    assert report.registry["alex"] == "[NAME_1]"


def test_pipeline_includes_supplied_run_metadata():
    transcript = Transcript(
        transcript_id="t1",
        turns=[Turn(turn_id=0, speaker="participant", text="No identifiers here.")],
    )
    metadata = RunMetadata(
        model="gemma4:12b",
        model_digest="abc123",
        pipeline_version="0.1.0",
        started_at_utc="2026-06-26T00:00:00Z",
    )

    _, report = deidentify(
        transcript,
        detect_fn=lambda turn_id, text: [],
        residual_fn=lambda turn_id, text: [],
        run_metadata=metadata,
    )

    assert report.run_metadata == metadata


def test_pipeline_propagates_identifier_missed_in_a_later_turn():
    transcript = Transcript(
        transcript_id="t1",
        turns=[
            Turn(turn_id=0, speaker="participant", text="Alex attended school."),
            Turn(turn_id=1, speaker="interviewer", text="Does alex still see Dr Smith?"),
        ],
    )

    def detect(turn_id, text):
        if turn_id == 0:
            return [
                PiiSpan(
                    turn_id=0,
                    start=0,
                    end=4,
                    text="Alex",
                    pii_type="person_name",
                    source="llm",
                )
            ]
        smith_start = text.index("Dr Smith")
        return [
            PiiSpan(
                turn_id=1,
                start=smith_start,
                end=smith_start + len("Dr Smith"),
                text="Dr Smith",
                pii_type="clinician_name",
                source="llm",
            )
        ]

    output, report = deidentify(
        transcript,
        detect_fn=detect,
        residual_fn=lambda turn_id, text: [],
    )

    assert output.turns[1].anonymised_text == "Does [NAME_1] still see [NAME_2]?"
    assert report.status == "clean"
    assert len([span for span in report.spans if span.text.lower() == "alex"]) == 2


def test_pipeline_requires_review_for_residual_identifier():
    transcript = Transcript(
        transcript_id="t1",
        turns=[Turn(turn_id=0, speaker="participant", text="I saw Smith.")],
    )
    _, report = deidentify(
        transcript,
        detect_fn=lambda turn_id, text: [],
        residual_fn=lambda turn_id, text: ["Smith"],
    )
    assert report.status == "needs_review"
    assert report.review_items[0].text == "Smith"


def test_pipeline_filters_generic_model_false_positives_and_token_artifacts():
    transcript = Transcript(
        transcript_id="t1",
        turns=[
            Turn(
                turn_id=0,
                speaker="participant",
                text="I stayed at home with the kids while I was in undergrad.",
            )
        ],
    )

    false_positives = [
        ("at home", "place"),
        ("kids", "person_name"),
        ("undergrad", "school"),
    ]

    def detect(turn_id, text):
        return [
            PiiSpan(
                turn_id=turn_id,
                start=text.index(value),
                end=text.index(value) + len(value),
                text=value,
                pii_type=pii_type,
                source="llm",
            )
            for value, pii_type in false_positives
        ]

    # The production detector filters these before the pipeline. Exercise the same filter here.
    from deidentify_transcripts.detect import filter_generic_spans

    output, report = deidentify(
        transcript,
        detect_fn=lambda turn_id, text: filter_generic_spans(detect(turn_id, text)),
        residual_fn=lambda turn_id, text: ["PLACE_1", "NAME_2"],
    )

    assert output.turns[0].anonymised_text == transcript.turns[0].text
    assert report.spans == []
    assert report.review_items == []
    assert report.status == "clean"


def test_pipeline_filters_question_phrases_but_keeps_actual_answers():
    transcript = Transcript(
        transcript_id="t1",
        turns=[
            Turn(
                turn_id=0,
                speaker="interviewer",
                text="Could you confirm your name and where you study?",
            ),
            Turn(
                turn_id=1,
                speaker="participant",
                text="I am Maya Patel from Cedar Grove University.",
            ),
        ],
    )

    candidates = {
        0: [
            ("your name", "person_name"),
            ("where you study", "school"),
        ],
        1: [
            ("Maya Patel", "person_name"),
            ("Cedar Grove University", "school"),
        ],
    }

    def detect(turn_id, text):
        from deidentify_transcripts.detect import filter_generic_spans

        spans = [
            PiiSpan(
                turn_id=turn_id,
                start=text.index(value),
                end=text.index(value) + len(value),
                text=value,
                pii_type=pii_type,
                source="llm",
            )
            for value, pii_type in candidates[turn_id]
        ]
        return filter_generic_spans(spans)

    output, report = deidentify(
        transcript,
        detect_fn=detect,
        residual_fn=lambda turn_id, text: [],
    )

    assert output.turns[0].anonymised_text == transcript.turns[0].text
    assert output.turns[1].anonymised_text == "I am [NAME_1] from [SCHOOL_1]."
    assert [span.text for span in report.spans] == ["Maya Patel", "Cedar Grove University"]
