# Transcript Deidentification

De-identify interview or therapy transcripts using:

1. deterministic patterns for email addresses, URLs, phone numbers, numeric and written dates (for
   example `March 2020` or `15th of April`), social media handles and long ID numbers;
2. a local language model for contextual identifiers such as names, schools,
   organisations, occupations, addresses and places;
3. stable placeholders such as `[NAME_1]` and `[SCHOOL_1]`;
4. an independent second local model pass that auto-corrects any residual identifier it confirms,
   plus a human-review queue for anything it can't confirm with confidence.

The default project setup sends transcript text to a secure Macquarie University-managed vLLM server.
That server exposes an OpenAI-compatible API and is intended for project researchers. Collaborators
reach it directly over Tailscale, using the server's tailnet IP address — access is being granted for
a limited time while this setup is tested. The application itself still runs on your own computer,
and transcripts leave your device only to reach that approved server. A fully local Ollama setup is
also available for testing or offline use.

> De-identification reduces risk; it does not prove that a transcript is anonymous. Validate this
> workflow against human review.

## What you'll need

- Tailscale installed, signed in with your own account, with the vLLM server machine shared to you by the project maintainer (instructions below);
- a project-issued inference API key;
- Python 3.11 or newer;
- this repository;
- approved encrypted storage for the input transcripts and `output/sensitive/` reports.

The default server-based setup does not require a local GPU or local LLM installation.

## 1. Connect to the project server

The project server lives on an institutional machine that isn't reachable over the open internet.
Once the project maintainer has shared it with your Tailscale account and you've accepted the
invite, you can reach the server directly at its tailnet IP address — no SSH tunnel or intermediate
machine needed. You only need to set this up once per machine, and Tailscale needs to stay
connected whenever you run the application.

### Install Tailscale and accept the share invite

[Tailscale](https://tailscale.com) creates a private, secure network ("tailnet") between your
computer and the project server, so you can reach it without a full VPN client. Access here works
by device sharing: the project maintainer shares just the vLLM server machine with your account,
and you accept that share by email — you do not need a shared auth key, and there's nothing secret
in the invite email itself, since accepting it requires signing in as you.

1. Install Tailscale for your platform:
   - Windows: <https://tailscale.com/download/windows>
   - macOS: <https://tailscale.com/download/mac> (or via the Mac App Store)
   - Ubuntu or other Linux:

     ```bash
     curl -fsSL https://tailscale.com/install.sh | sh
     ```

2. Open the Tailscale app (or run `tailscale up` on Linux) and sign in with your own account —
   whichever identity provider Tailscale offers (Google, Microsoft, GitHub, passkey, etc.). This
   does not require an auth key.
3. Ask the project maintainer to share the vLLM server machine with the email address tied to that
   account, if they haven't already. You'll receive an email from Tailscale with a link to accept
   the share.
4. Open that link and accept it. The server machine will then appear as a node in your own tailnet
   — you don't gain access to the maintainer's whole tailnet, only that one shared machine.

#### Check the Tailscale connection

Run `tailscale status`. You should see your own machine listed as `Connected`, and the shared
server machine listed among your peers. If you don't see the server listed, the share invite may
not have been accepted yet, or may have expired — check your email or ask the project maintainer
to re-share it.

### Check you can reach the server

The vLLM server's tailnet IP address is `100.127.175.5`. Once Tailscale is connected, confirm you
can reach it:

```bash
curl http://100.127.175.5:4200/v1/models
```

You should get back a JSON response listing the available model(s). The `doctor` command later in
this guide gives a second confirmation that everything can talk to the server.

If the connection fails:

- confirm Tailscale is connected (`tailscale status` should show you as `Connected`);
- confirm the server's tailnet IP with the project maintainer, in case it has changed;
- do not continue with real transcripts until this succeeds.

## 2. Install this application

### Ubuntu, macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
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
```

## 3. Configure the project server

Create the project configuration file:

```bash
deidentify-transcripts init-config
```

This writes `.env` with the approved server settings:

```env
DEID_ALLOW_REMOTE_LLM=true
VLLM_BASE_URL=http://100.127.175.5:4200/v1
VLLM_MODEL=large
VLLM_INFERENCE_HUB_API_KEY=replace-with-issued-key
VLLM_OUTPUT_MODE=native
```

`VLLM_BASE_URL` already points at the server's tailnet IP address, reachable directly once
Tailscale is connected — no further edits needed there. Open `.env` and make one change:

- replace `replace-with-issued-key` with your issued API key.

Notes:

- `DEID_ALLOW_REMOTE_LLM=true` confirms that this project is intentionally sending transcript text
  to the approved institutional server.
- `VLLM_BASE_URL` must point at the server's tailnet IP address; ask the project maintainer if it
  changes.
- `VLLM_MODEL=large` must match the model name exposed by the server.
- `VLLM_INFERENCE_HUB_API_KEY` should be the key issued to the researcher. Do not commit it.
- `.env` is gitignored.
- If `.env` already exists, `init-config` will stop rather than overwrite it. Use
  `deidentify-transcripts init-config --force` only if you intentionally want to replace the
  existing file.

Check the configuration and model connection. Make sure Tailscale is still connected, then run:

```bash
deidentify-transcripts doctor
```

Expected output resembles:

```text
OK: remote vllm endpoint http://100.127.175.5:4200/v1; model large
```

If `doctor` fails:

- confirm Tailscale is connected (`tailscale status` should show you as `Connected`);
- confirm the API key was copied correctly;
- confirm the server's tailnet IP and model name with the project maintainer;
- do not continue with real transcripts until `doctor` succeeds.

## 4. Optional local-only fallback

Use this only for testing/offline work. Local quality and speed depend on the machine and model.

Recommended local models:

- 16 GB RAM: `qwen3:8b`.
- 24-32 GB RAM: `qwen3:8b` comfortably, or evaluate `gemma3:12b`.
- 8 GB RAM: a 4B model may run, but is not recommended for sensitive production work.

Install Ollama from <https://ollama.com/download>, then download a model:

```bash
ollama pull qwen3:8b
```

For a larger local machine:

```bash
ollama pull gemma3:12b
```

Then create a local config:

```bash
deidentify-transcripts init-config --local
```

If `.env` already exists and you intentionally want to replace it:

```bash
deidentify-transcripts init-config --local --force
```

The local config contains:

```env
DEID_ALLOW_REMOTE_LLM=false
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen3:8b
OLLAMA_API_KEY=ollama
```

Do not select an Ollama model whose name ends in `-cloud`. Ollama should listen only on
`localhost:11434`.

Useful local checks:

```bash
ollama list
deidentify-transcripts doctor
ollama ps
```

- `ollama list` shows downloaded models.
- `doctor` shows which model this application will request.
- `ollama ps` shows the model currently loaded in memory.

## 5. Prepare a transcript

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

## 6. Run de-identification

Make sure Tailscale is still connected (see [step 1](#1-connect-to-the-project-server)) before
running the commands below.

Run a small example first:

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

For a real transcript, replace `transcript.txt` with the approved local path to the transcript file.
Use `--id` to assign the participant/study identifier you want in the output filenames.

Outputs:

```text
output/
  anonymised/<id>.json
  sensitive/<id>.deid-report.json
  sensitive/<id>.review-queue.jsonl
```

- `anonymised/` contains the speaker and anonymised text, but no raw transcript field.
- `sensitive/` contains original identifiers and must remain in approved encrypted storage. Each
  de-identification report also records the selected model, the Ollama digest when available, the
  pipeline version and the UTC run-start time.
- An exit code of `2` means the run completed but human review is required — specifically, a
  stage-1 detection the model wasn't confident about. This is common and does not mean the software
  crashed. It does not fire for a residual identifier the second pass independently confirmed and
  auto-corrected; see [7. Review the results](#7-review-the-results).
- Running the same transcript ID again overwrites that ID's previous output files. Use another
  `--output-dir` or copy the previous files when comparing models.

Example anonymised text:

```text
[NAME_1] attended [SCHOOL_1]. Call [PHONE_1].
```

## 7. Review the results

Always inspect:

1. the complete anonymised transcript for identifiers that were missed;
2. `sensitive/<id>.review-queue.jsonl`;
3. replacements that remove clinically meaningful non-identifying text.

The detector processes each turn using the configured model. It then deterministically propagates
identifiers already found elsewhere in the transcript, so repeated names receive the same token. A
second model pass checks the redacted text for anything left over.

The review queue (`review-queue.jsonl`) can contain two different kinds of entry, distinguished by
their `reason` field:

- `reason: "gate: auto-corrected"` — the second pass found and confirmed a residual identifier that
  slipped through stage 1. The pipeline has already redacted it everywhere it appears in the
  transcript and registered it under a token; no action is required, but it's worth spot-checking
  that the auto-redaction looks right. Entries like this do not, by themselves, put the run into
  `needs_review` status or affect the exit code.
- `reason: "low-confidence detection"` — a stage-1 detection the model wasn't confident about. This
  is a genuine judgement call and needs a human decision: confirm it's an identifier (and redact it
  manually) or dismiss it as a false alarm. Entries like this do put the run into `needs_review`
  status and produce exit code `2`.

The automated status `clean` means only that no low-confidence detections were left outstanding —
not that the second pass found nothing (it may have auto-corrected something) and not a
certification of anonymity. Always read the anonymised transcript yourself regardless of status.

## 8. Platform notes

- Paths use `/`, for example `examples/sample-transcript.txt`.
- Activate the environment with `source .venv/bin/activate`.
- Disconnect Tailscale (`tailscale down`) when you're finished for the day; it doesn't need to stay
  connected outside of active sessions.
- Restrict local output permissions where appropriate:

```bash
chmod 700 output output/sensitive
chmod 600 .env output/sensitive/*
```

For Windows PowerShell, use backslashes in paths, for example
`examples\sample-transcript.txt`. If script activation is blocked, run PowerShell as your normal
user and check your execution policy with your local IT support.

For local-only Ollama use on Ubuntu, Ollama normally runs as a `systemd` service. A supported NVIDIA
GPU may be used automatically when the appropriate driver is installed; check with `nvidia-smi`.
Keep Ollama on localhost. Do not configure `OLLAMA_HOST=0.0.0.0` for sensitive transcripts.

## Privacy checklist

- Obtain ethics, governance and information-security approval for the project.
- Use an institution-managed computer and approved encrypted storage.
- Use the project server only from approved accounts and networks.
- Keep your issued API key secret; do not paste it into notebooks, scripts, chat, commits or
  screenshots.
- Don't forward the Tailscale share invite email to anyone else; it's tied to your own account, and
  the project maintainer should share the server directly with any additional collaborator instead.
- If using local Ollama, keep it bound to `localhost`; do not expose port `11434` publicly.
- Use `DEID_ALLOW_REMOTE_LLM=true` only for approved institution-managed endpoints.
- Do not use cloud-hosted Ollama models.
- Do not commit transcripts, reports, `.env`, `data/`, or `output/`.
- Check institutional backup, antivirus and device-sync policies.
- Delete sensitive temporary and report files according to the approved retention plan.

## Development

The unit tests do not require Ollama or real transcript data:

```bash
pytest
```
