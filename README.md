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

> De-identification reduces risk; it does not prove that a transcript is anonymous. Please validate this
> workflow against human review.

## What you'll need

- Tailscale installed, signed in with your own account, with the vLLM server machine shared to you by MQ researchers (instructions below);
- a project-issued inference API key;
- Python 3.11 or newer;
- this repository.

The default server-based setup does not require a local GPU or local LLM installation.

## 1. Connect to the project server

The project server lives on an institutional machine that isn't reachable over the open internet.
Once an MQ researcher has shared it with your Tailscale account and you've accepted the
invite, you can reach the server directly at its tailnet IP address — no SSH tunnel or intermediate
machine needed. You only need to set this up once per machine, and Tailscale needs to stay
connected whenever you run the application.

### Install Tailscale and accept the share invite

[Tailscale](https://tailscale.com) creates a private, secure network ("tailnet") between your
computer and the project server, so you can reach it without a full VPN client. Access here works
by device sharing: MQ researchers will share just the vLLM server machine with your account,
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
3. Ask MQ researchers to share the vLLM server machine with the email address tied to that
   account, if they haven't already. You'll receive an email from Tailscale with a link to accept
   the share.
4. Open that link and accept it. The server machine will then appear as a node in your own tailnet

### Check you can reach the server

Run `tailscale status`. You should see your own machine listed as `Connected`, and the shared
server machine listed among your peers.

The vLLM server's tailnet IP address is `100.127.175.5`. Confirm you can reach it:

```bash
curl http://100.127.175.5:4200/v1/models
```

This endpoint requires the project API key, which you haven't configured yet, so at this stage an
authentication error (for example `401 Unauthorized`) is the expected, successful result — it means
the network path to the server works. A connection timeout or "no route to host" means Tailscale
isn't reaching the server. Once your API key is configured in [step 3](#3-configure-the-project-server),
the `doctor` command gives a full confirmation, including that the key itself works.

If you get a timeout or connection error:

- the share invite may not have been accepted yet, or may have expired — check your email or ask
  MQ researchers to re-share the server;
- confirm the server's tailnet IP with MQ researchers, in case it has changed;
- do not continue with real transcripts until this succeeds.

Disconnect Tailscale (`tailscale down`) when you're finished for the day; it doesn't need to stay
connected outside of active sessions.

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

If script activation is blocked, run PowerShell as your normal user and check your execution
policy with your local IT support.

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
- `VLLM_BASE_URL` must point at the server's tailnet IP address; ask MQ researchers if it changes.
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

If `doctor` fails, re-run the checks from [step 1](#1-connect-to-the-project-server), then confirm
the API key was copied correctly and the model name matches what MQ researchers expect. Do not
continue with real transcripts until `doctor` succeeds.

## 4. Optional local-only fallback

A fully local Ollama setup is available for testing or offline use, instead of the default project
server. It does not require Tailscale, the project server, or an issued API key. See
[docs/LOCAL_OLLAMA.md](docs/LOCAL_OLLAMA.md) for setup, recommended models, and troubleshooting.

## 5. Prepare a transcript

The application accepts plain-text (one turn per line, preferably with a speaker label) and JSON
transcripts:

```text
Interviewer: Could you tell me where you went to school?
Participant: I attended Oak Park School and my GP is Dr Smith.
```

Word documents, PDFs, spreadsheets, subtitle files and paragraph-style transcripts must first be
converted to one of these two formats. Full parsing rules, the JSON shape, and the included
synthetic test transcripts are documented in [docs/FORMATS.md](docs/FORMATS.md).

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

To process every `.txt` and `.json` transcript in a folder in one go, use `batch` instead of `run`:

```bash
deidentify-transcripts batch transcripts/ --output-dir output
```

Each file's transcript ID defaults to its filename stem (there's no `--id` option for `batch`, since
it applies to a single output). `batch` keeps going if one file fails, then exits `1` if any file
failed outright or `2` if any succeeded but need review (matching `run`'s exit codes) — check the
per-file summary lines it prints to see which.

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

Restrict local output permissions where appropriate:

```bash
chmod 700 output output/sensitive
chmod 600 .env output/sensitive/*
```

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
certification of anonymity.


## Development

The unit tests do not require Ollama or real transcript data:

```bash
pytest
```
