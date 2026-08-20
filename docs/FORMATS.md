# Transcript format reference

Full parsing rules for the two accepted transcript formats, and details on the included synthetic
test transcripts. See the main [README](../README.md) for the basic setup and run instructions.

## Plain text

Use one turn per line, preferably with a speaker label:

```text
Interviewer: Could you tell me where you went to school?
Participant: I attended Oak Park School and my GP is Dr Smith.
```

Parsing rules:

- The first colon separates the speaker label from the turn text.
- Blank lines are ignored.
- A line without a colon is accepted with speaker `unknown`.
- Multi-line turns are not joined automatically; every non-empty line becomes a separate turn.
- Speaker labels are retained in the output but are not themselves de-identified.
- Save the file as UTF-8 text.

## JSON

The preferred JSON shape is:

```json
{
  "transcript_id": "participant-001",
  "turns": [
    {
      "speaker": "interviewer",
      "text": "Where did you go to school?"
    },
    {
      "speaker": "participant",
      "text": "I attended Oak Park School."
    }
  ]
}
```

JSON parsing rules:

- A top-level `turns` array is required.
- Every turn must contain either `text` or the legacy field `raw_text`.
- `speaker` is optional and defaults to `unknown`.
- Input `turn_id` values are ignored and regenerated as `0, 1, 2, ...`.
- Other fields such as timestamps and interview metadata are currently discarded.
- A top-level `transcript_id` is used when present.
- If `transcript_id` is absent, the input filename is used.
- A legacy top-level `interview_id` is not currently read. Use `--id` if the desired identifier
  differs from the filename.

Run the included preferred-format JSON example:

```bash
deidentify-transcripts run examples/sample-transcript.json
```

To specify an explicit ID:

```bash
deidentify-transcripts run examples/sample-transcript.json --id study-participant-001
```

Word documents, PDFs, spreadsheets, subtitle files and paragraph-style transcripts must first be
converted to one of the two formats above.

## Customizing the parser

The rules above are current behavior, not a configurable setting — matching a different transcript
shape (a different speaker delimiter, an extra JSON field, preserving source `turn_id` values, etc.)
means editing the parser itself, in `load_transcript()` in
[src/deidentify_transcripts/io.py](../src/deidentify_transcripts/io.py):

- Plain text: the speaker/text split is one regex,
  `re.match(r"^([^:]{1,80}):\s*(.*)$", line)`. Change the delimiter or the 80-character speaker-label
  cap here.
- JSON: the field-name fallback chain, `turn.get("text", turn.get("raw_text", ""))`, is where to add
  support for another field name from a different export tool. `turn.get("speaker", "unknown")` sets
  the default speaker, and `transcript_id` resolution (explicit `--id` → JSON's own `transcript_id` →
  filename) is nearby.
- Input `turn_id` values are always discarded and regenerated (`turn_id=index`); change that line to
  preserve source turn IDs instead.

Whatever `load_transcript()` returns has to fit the `Turn`/`Transcript` models in
[src/deidentify_transcripts/schemas.py](../src/deidentify_transcripts/schemas.py) — check there too
if you need to carry an extra field (like a timestamp) through the pipeline rather than just change
how an existing field is matched.

## Synthetic test transcripts

Three synthetic example files are included:

- `examples/sample-transcript.txt`: a small plain-text installation smoke test.
- `examples/sample-transcript.json`: a preferred-format JSON example.
- `examples/complex-synthetic-transcript.txt`: a harder model-comparison test containing repeated
  names, case variants, nicknames, clinicians, schools, workplaces, addresses, places, contact
  details, a date and an ID number.

The complex example has a human-readable answer key at
`examples/complex-synthetic-expected.json`. Use it to count missed identifiers and unnecessary
replacements. It is not consumed automatically by the application.

Run the complex example in a separate output directory:

```bash
deidentify-transcripts run examples/complex-synthetic-transcript.txt \
  --output-dir output/complex-qwen
```

When comparing another model, change `.env` and use another output directory:

```bash
deidentify-transcripts run examples/complex-synthetic-transcript.txt \
  --output-dir output/complex-gemma
```
