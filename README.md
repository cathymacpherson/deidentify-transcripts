# deidentify-transcripts

Locally anonymise interview transcripts using:

1. deterministic patterns for email addresses, URLs, phone numbers, dates and long ID numbers;
2. a downloaded local language model for contextual identifiers such as names, schools,
   organisations, addresses and places;
3. stable placeholders such as `[NAME_1]` and `[SCHOOL_1]`;
4. an independent second model pass and a mandatory human-review queue.

The transcript is sent only to Ollama on the same computer. It is not sent to a commercial cloud
API.

> De-identification reduces risk; it does not prove that a transcript is anonymous. Validate this
> workflow against representative, manually annotated transcripts and retain human review.

## Recommended computer

- 16 GB RAM: use `qwen3:8b`.
- 24-32 GB RAM: use `qwen3:8b` comfortably, or evaluate `gemma3:12b`.
- 8 GB RAM: a 4B model may run, but is not recommended for sensitive production work.

A dedicated GPU is helpful but not required. CPU-only processing will be slower. Keep approximately
15 GB of disk space free for Ollama, the model and working files.

## 1. Install Ollama and a model

### Ubuntu

```bash
curl -fsSL https://ollama.com/install.sh | sh
systemctl status ollama
```

If the service is not running:

```bash
sudo systemctl enable --now ollama
```

### Windows or macOS

Install Ollama from <https://ollama.com/download>.

### Download a model

For a computer with 16 GB RAM:

```bash
ollama pull qwen3:8b
```

For a computer with approximately 24-32 GB RAM:

```bash
ollama pull gemma3:12b
```

Do not select an Ollama model whose name ends in `-cloud`.

Confirm that Ollama is available:

```bash
curl http://localhost:11434/v1/models
```

In Windows PowerShell, use:

```powershell
Invoke-RestMethod http://localhost:11434/v1/models
```

Ollama should listen only on `localhost:11434`.

## 2. Install this application

### Ubuntu, macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

On Ubuntu, if virtual-environment support is missing:

```bash
sudo apt update
sudo apt install python3-venv
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Check the configuration and model connection:

```bash
deidentify-transcripts doctor
```

Expected output resembles:

```text
OK: local endpoint http://localhost:11434/v1; model qwen3:8b
```

## 3. Select or switch models

The selected model is configured in `.env`:

```env
OLLAMA_MODEL=qwen3:8b
```

To use Gemma instead:

```env
OLLAMA_MODEL=gemma3:12b
```

Changing `.env` affects the next run; no code change is required.

Useful checks:

```bash
ollama list
deidentify-transcripts doctor
ollama ps
```

- `ollama list` shows downloaded models.
- `doctor` shows which model this application will request.
- `ollama ps` shows the model currently loaded in memory.

## 4. Prepare a transcript

The application accepts plain-text and JSON transcripts.

### Plain text

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

### JSON

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

### Synthetic test transcripts

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

## 5. Run de-identification

Ubuntu, macOS or Linux:

```bash
deidentify-transcripts run examples/sample-transcript.txt
```

Windows PowerShell:

```powershell
deidentify-transcripts run examples\sample-transcript.txt
```

Optional arguments:

```text
deidentify-transcripts run transcript.txt --id participant-001 --output-dir output
```

Outputs:

```text
output/
  anonymised/<id>.json
  sensitive/<id>.deid-report.json
  sensitive/<id>.review-queue.jsonl
```

- `anonymised/` contains the speaker and anonymised text, but no raw transcript field.
- `sensitive/` contains original identifiers and must remain in approved encrypted storage.
- An exit code of `2` means human review is required.
- Running the same transcript ID again overwrites that ID's previous output files. Use another
  `--output-dir` or copy the previous files when comparing models.

Example anonymised text:

```text
[NAME_1] attended [SCHOOL_1]. Call [PHONE_1].
```

## 6. Review the results

Always inspect:

1. the complete anonymised transcript for identifiers that were missed;
2. `sensitive/<id>.review-queue.jsonl`;
3. replacements that remove clinically meaningful non-identifying text.

The detector processes each turn using the local model. It then deterministically propagates
identifiers already found elsewhere in the transcript, so repeated names receive the same token. A
second local-model pass checks the redacted text for remaining identifiers.

The review queue may contain genuine misses as well as conservative false alarms. Both require a
human decision. The automated status `clean` means only that the configured checks found no
unresolved items; it is not a certification of anonymity.

## 7. Ubuntu notes

- Paths use `/`, for example `examples/sample-transcript.txt`.
- Activate the environment with `source .venv/bin/activate`.
- Ollama normally runs as a `systemd` service.
- A supported NVIDIA GPU may be used automatically when the appropriate driver is installed. Check
  with `nvidia-smi`.
- Inspect memory and processor information with `free -h` and `lscpu`.
- Keep Ollama on localhost. Do not configure `OLLAMA_HOST=0.0.0.0` for sensitive transcripts.
- Restrict local output permissions where appropriate:

```bash
chmod 700 output output/sensitive
chmod 600 .env output/sensitive/*
```

## Privacy checklist

- Obtain ethics, governance and information-security approval for the project.
- Use an institution-managed computer and approved encrypted storage.
- Keep Ollama bound to `localhost`; do not expose port `11434` publicly.
- Do not use cloud-hosted Ollama models.
- Do not commit transcripts, reports, `.env`, `data/`, or `output/`.
- Check institutional backup, antivirus and device-sync policies.
- Delete sensitive temporary and report files according to the approved retention plan.

## Development

The unit tests do not require Ollama or real transcript data:

```bash
pytest
```
