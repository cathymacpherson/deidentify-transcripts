from deidentify_transcripts.detect import (
    filter_generic_spans,
    is_bare_generic_identifier,
    merge_spans,
    regex_spans,
)
from deidentify_transcripts.schemas import PiiSpan


def test_regex_detects_structured_pii():
    text = "Call 0412 345 678, email alex@example.org on 12/03/2024."
    types = {span.pii_type for span in regex_spans(0, text)}
    assert {"phone", "email", "date"} <= types


def test_phone_regex_leaves_sentence_punctuation_outside_span():
    spans = regex_spans(0, "Call 0412 345 678.")
    phone = next(span for span in spans if span.pii_type == "phone")
    assert phone.text == "0412 345 678"


def test_written_dates_are_detected():
    for text, expected in [
        ("It happened in March 2020.", "March 2020"),
        ("Her birthday was March 15, 2020.", "March 15, 2020"),
        ("He starts on 15th March.", "15th March"),
        ("The appointment is on the 3rd of April.", "3rd of April"),
    ]:
        spans = [span for span in regex_spans(0, text) if span.pii_type == "date"]
        assert any(span.text == expected for span in spans), (text, spans)


def test_social_media_handle_is_detected():
    spans = regex_spans(0, "Follow me @alex_smith for updates.")
    handles = [span for span in spans if span.pii_type == "social_media_handle"]
    assert [span.text for span in handles] == ["@alex_smith"]


def test_social_media_handle_does_not_match_email_local_part():
    spans = regex_spans(0, "Email alex@example.org for details.")
    assert not any(span.pii_type == "social_media_handle" for span in spans)


def test_generic_school_is_filtered_but_named_school_is_retained():
    spans = [
        PiiSpan(
            turn_id=0, start=0, end=6, text="school", pii_type="school", source="llm"
        ),
        PiiSpan(
            turn_id=0,
            start=10,
            end=25,
            text="Oak Park School",
            pii_type="school",
            source="llm",
        ),
    ]
    assert [span.text for span in filter_generic_spans(spans)] == ["Oak Park School"]


def test_common_descriptive_phrases_are_not_identifiers():
    for text in [
        "at home",
        "your home",
        "the kids",
        "undergrad",
        "sir",
        "my husband",
        "I",
        "sorry",
        "okay",
        "thanks",
        "your name",
        "where you study",
        "who is your doctor",
        "teacher",
        "front age",
        "another department",
    ]:
        assert is_bare_generic_identifier(text)

    assert not is_bare_generic_identifier("Oak Park School")
    assert not is_bare_generic_identifier("Sarah's sister")
    assert not is_bare_generic_identifier("27 Wattlebird Lane")


def test_merge_spans_keeps_longest_overlapping_span():
    spans = [
        PiiSpan(
            turn_id=0, start=0, end=10, text="0412345678", pii_type="phone", source="regex"
        ),
        PiiSpan(
            turn_id=0, start=4, end=8, text="2345", pii_type="id_number", source="regex"
        ),
    ]
    assert [span.pii_type for span in merge_spans(spans)] == ["phone"]
