import pytest

from deidentify_transcripts.model import parse_structured_content
from deidentify_transcripts.schemas import PiiMentions


def test_structured_parser_accepts_trailing_model_control_token():
    content = (
        '{"mentions": [{"text": "Alex", "pii_type": "person_name", "confidence": 0.9}]}'
        "\n<|tool_response>"
    )

    result = parse_structured_content(content, PiiMentions)

    assert result.mentions[0].text == "Alex"


def test_structured_parser_still_rejects_invalid_json():
    with pytest.raises(RuntimeError, match="invalid structured output"):
        parse_structured_content('{"mentions": [}', PiiMentions)
