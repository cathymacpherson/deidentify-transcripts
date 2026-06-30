from __future__ import annotations

import re
from collections.abc import Callable

from .schemas import PiiMention, PiiMentions, PiiSpan, PiiType

_PATTERNS: list[tuple[PiiType, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("url", re.compile(r"\bhttps?://\S+")),
    (
        "phone",
        # Starts and ends on a digit so sentence punctuation is not swallowed by the token.
        re.compile(r"(?<!\w)\+?\d(?:[\s().-]?\d){7,14}(?!\w)"),
    ),
    ("date", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    ("id_number", re.compile(r"\b\d{7,}\b")),
]

_BARE_GENERIC_PII: dict[str, set[str]] = {
    "school": {
        "school", "college", "university", "uni", "class", "high school", "undergrad",
    },
    "family_member": {
        "husband", "wife", "mother", "father", "mum", "mom", "dad", "sister",
        "brother", "parents", "children", "kids",
    },
    "clinician_name": {"counselor", "counsellor", "doctor", "therapist", "clinician", "gp"},
    "organisation": {
        "work", "job", "office", "department", "company", "organisation", "organization",
    },
    "place": {"home", "house", "store", "shop", "factory"},
}
_BARE_GENERIC_TEXT = set().union(*_BARE_GENERIC_PII.values())
_COMMON_NON_IDENTIFIER_WORDS = {
    # Common transcript/therapy words and generic roles. These are rejected only when an entire
    # candidate is made from common words, so "Oak Park School" and "Sarah's sister" are retained.
    "a", "age", "an", "another", "any", "back", "bottom", "case", "centre", "center", "class",
    "client", "college", "company", "counselor", "counsellor", "department", "doctor", "edge",
    "family", "front", "gp", "group", "home", "hospital", "house", "job", "middle", "office",
    "organisation", "organization", "part", "place", "practice", "program", "programme",
    "provider", "room", "school", "section", "service", "shop", "side", "store", "teacher",
    "therapist", "top", "uni", "university", "ward", "work", "workplace",
}
_GENERIC_TITLES = {"sir", "madam", "ma'am", "mr", "mrs", "ms", "miss"}
_NON_NAME_WORDS = {
    # Pronouns and determiners
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours",
    "they", "them", "their", "theirs", "it", "its", "this", "that",
    # Common acknowledgements, courtesy words and discourse markers
    "yes", "no", "yeah", "yep", "okay", "ok", "thanks", "thank", "sorry",
    "please", "hello", "hi", "goodbye", "bye", "absolutely", "right", "well",
}
_NON_IDENTIFIER_PHRASE_WORDS = {
    # Identifiers should not include conversational pronouns or question words. For example,
    # "your name" and "where you study" describe requested information; they are not the answer.
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "what", "where", "when", "why", "who", "whom", "whose", "which", "how",
}
_GENERIC_ROLE_PREFIXES = {
    "my", "your", "our", "their", "his", "her", "the", "a", "an",
}
_GENERIC_PLACE_PREFIXES = {
    "at", "in", "from", "to", "near", "around", "inside", "outside",
}

DETECTION_SYSTEM = (
    "Find personally identifying information in one transcript turn. Return each mention's EXACT "
    "text and one type from: person_name, nickname, family_member, clinician_name, school, place, "
    "organisation, address, postcode. Do not include phone numbers, email addresses, dates, URLs or "
    "long ID numbers because deterministic patterns handle those. Return genuine identifiers only. "
    "Do not return generic words or ordinary descriptive phrases such as school, undergrad, class, "
    "work, at home, your home, kids, children, sir, doctor, husband or sister. Pronouns and ordinary "
    "conversation words such as I, you, yes, no, okay, thanks and sorry are never names. Questions or "
    "descriptions such as 'your name', 'where you study', 'your school' and 'who is your doctor' are "
    "not identifiers: return only the actual answer, such as Maya Patel, Oak Park School or Dr Smith. "
    "Return generic terms only as part of a specific identifying phrase such as Sarah's sister or a "
    "named workplace. Confidence must be between 0 and 1."
)


def regex_spans(turn_id: int, text: str) -> list[PiiSpan]:
    spans: list[PiiSpan] = []
    for pii_type, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            spans.append(
                PiiSpan(
                    turn_id=turn_id,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    pii_type=pii_type,
                    confidence=1.0,
                    source="regex",
                )
            )
    return spans


def normalise_identifier_text(text: str) -> str:
    return " ".join(text.lower().strip(" .,!?:;\"'()[]{}").split())


def is_bare_generic_identifier(text: str, pii_type: PiiType | None = None) -> bool:
    normalised = normalise_identifier_text(text)
    if (
        normalised in _GENERIC_TITLES
        or normalised in _NON_NAME_WORDS
        or normalised in _BARE_GENERIC_TEXT
    ):
        return True

    words = normalised.split()
    if any(word in _NON_IDENTIFIER_PHRASE_WORDS for word in words):
        return True
    while words and words[0] in _GENERIC_PLACE_PREFIXES:
        words.pop(0)
    while words and words[0] in _GENERIC_ROLE_PREFIXES:
        words.pop(0)
    reduced = " ".join(words)

    # Pronoun/determiner phrases such as "my husband", "the kids", "at home", and
    # "your home" remain generic. A named relation such as "Sarah's sister" is retained.
    if reduced in _BARE_GENERIC_TEXT or reduced in _GENERIC_TITLES:
        return True
    if reduced and all(word in _COMMON_NON_IDENTIFIER_WORDS for word in reduced.split()):
        return True
    if pii_type is None:
        return False
    return reduced in _BARE_GENERIC_PII.get(pii_type, set())


def filter_generic_spans(spans: list[PiiSpan]) -> list[PiiSpan]:
    return [span for span in spans if not is_bare_generic_identifier(span.text, span.pii_type)]


def locate(turn_id: int, text: str, mention: PiiMention) -> list[PiiSpan]:
    spans: list[PiiSpan] = []
    start = 0
    while mention.text:
        index = text.find(mention.text, start)
        if index == -1:
            break
        spans.append(
            PiiSpan(
                turn_id=turn_id,
                start=index,
                end=index + len(mention.text),
                text=mention.text,
                pii_type=mention.pii_type,
                confidence=mention.confidence,
                source="llm",
            )
        )
        start = index + len(mention.text)
    return spans


def merge_spans(spans: list[PiiSpan]) -> list[PiiSpan]:
    kept: list[PiiSpan] = []
    for span in sorted(spans, key=lambda item: (item.turn_id, item.start, -(item.end - item.start))):
        overlaps = any(
            other.turn_id == span.turn_id
            and not (span.end <= other.start or span.start >= other.end)
            for other in kept
        )
        if not overlaps:
            kept.append(span)
    return kept


def propagate_known_identifiers(
    turns: list[tuple[int, str]],
    spans: list[PiiSpan],
) -> list[PiiSpan]:
    """Find exact, case-insensitive repeats of contextual identifiers across all turns.

    The model only sees one turn at a time. Once it has identified a contextual value such as
    ``Alex`` anywhere, deterministic matching ensures later occurrences do not depend on the model
    rediscovering it. Structured regex spans are already detected independently on every turn.
    """
    contextual = {
        (normalise_identifier_text(span.text), span.pii_type): span
        for span in spans
        if span.source == "llm" and len(normalise_identifier_text(span.text)) >= 3
    }
    propagated = list(spans)
    for (_, _), known in contextual.items():
        pattern = re.compile(rf"(?<!\w){re.escape(known.text)}(?!\w)", re.IGNORECASE)
        for turn_id, text in turns:
            for match in pattern.finditer(text):
                propagated.append(
                    PiiSpan(
                        turn_id=turn_id,
                        start=match.start(),
                        end=match.end(),
                        text=match.group(),
                        pii_type=known.pii_type,
                        confidence=known.confidence,
                        source="llm",
                    )
                )
    return merge_spans(propagated)


def make_detector(
    model_call: Callable[..., PiiMentions],
) -> Callable[[int, str], list[PiiSpan]]:
    def detect(turn_id: int, text: str) -> list[PiiSpan]:
        result = model_call(system=DETECTION_SYSTEM, text=text, output_type=PiiMentions)
        llm_spans = filter_generic_spans(
            [span for mention in result.mentions for span in locate(turn_id, text, mention)]
        )
        return merge_spans(llm_spans + regex_spans(turn_id, text))

    return detect
