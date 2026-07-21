# Transcript Deidentification

De-identify interview or therapy transcripts using:

1. deterministic patterns for email addresses, URLs, phone numbers, dates and long ID numbers;
2. a local language model for contextual identifiers such as names, schools,
   organisations, addresses and places;
3. stable placeholders such as `[NAME_1]` and `[SCHOOL_1]`;
4. an independent second local model pass and a mandatory human-review queue.

The default project setup sends transcript text to a secure Macquarie University-managed vLLM server.
That server exposes an OpenAI-compatible API and is intended for project researchers, but it is only
reachable from within the campus network — not directly from a personal laptop, even over Tailscale.
You reach it by Tailscaling into a campus-based Macquarie machine and opening an SSH tunnel through
it; the application itself still runs on your own computer, and transcripts never leave your device.
A fully local Ollama setup is also available for testing or offline use.

> De-identification reduces risk; it does not prove that a transcript is anonymous. Validate this
> workflow against human review.

## What you'll need

- Tailscale installed and connected, so your computer can reach a campus-based Macquarie University machine that has access to the project server (instructions below);
- SSH access to that campus machine (a username and either a password or an SSH key — ask the project maintainer);
- a project-issued inference API key;
- Python 3.11 or newer;
- this repository;
- approved encrypted storage for the input transcripts and `output/sensitive/` reports.

The default server-based setup does not require a local GPU or local LLM installation.

## 1. Connect to the project server

The project server lives on an institutional machine that isn't reachable over the open internet,
and it isn't reachable directly from your computer even once you're on the tailnet — you first
Tailscale into a campus-based machine that can see the server, then open an SSH tunnel through that
machine. The application still runs on your own computer against `localhost`; the tunnel just
relays traffic to the real server. You only need to set this up once per machine, and you'll need
the tunnel open (see below) whenever you run the application.

### Install Tailscale

[Tailscale](https://tailscale.com) creates a private, secure network ("tailnet") between your
computer and the campus machine, so you can reach it without a full VPN client.

Ask the project maintainer for a Tailscale auth key. Treat it like a password: don't share it,
paste it into chat/email, or commit it anywhere.

#### Windows

1. Download and install Tailscale from <https://tailscale.com/download/windows>.
2. Open PowerShell and run:

   ```powershell
   & "C:\Program Files\Tailscale\tailscale.exe" up --authkey=paste-your-auth-key-here
   ```

3. The Tailscale icon should appear in your system tray, showing you're connected.

#### macOS

1. Download and install Tailscale from <https://tailscale.com/download/mac> (or via the Mac App
   Store).
2. Open Terminal and run:

   ```bash
   sudo tailscale up --authkey=paste-your-auth-key-here
   ```

3. The Tailscale icon should appear in your menu bar, showing you're connected.

#### Ubuntu or other Linux

1. Install Tailscale:

   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   ```

2. Connect using your auth key:

   ```bash
   sudo tailscale up --authkey=paste-your-auth-key-here
   ```

#### Check the Tailscale connection

Run `tailscale status` on any platform. You should see your own machine listed as `Connected`,
along with the campus machine. If `tailscale up` fails or the auth key is rejected, the key may
have expired; ask the project maintainer for a new one.

### Open an SSH tunnel to the campus machine

If you're more used to Remote Desktop, note that SSH here works quite differently: it does not
give you a graphical desktop on the campus machine, and you don't type any commands there. It just
opens a private, secure "pipe" between a port on your own computer and the server, using the campus
machine to relay it. You leave that pipe open in one terminal window, and do all your actual work
(the commands later in this guide) in a second terminal window, on your own computer.

Ask the project maintainer for the campus machine's Tailscale name (or IP) and your SSH username —
shown below as `campus-machine` and `your-username`.

#### Windows

Windows 10/11 include an SSH client already. Check it's there by opening PowerShell and running:

```powershell
ssh -V
```

If that prints a version number, you're set. If it says `ssh` isn't recognized, install it via
**Settings → System → Optional features → Add a feature → OpenSSH Client**, then reopen PowerShell.

Then open the tunnel:

```powershell
ssh -N -L 4200:10.204.35.227:4200 your-username@campus-machine
```

#### macOS or Linux

Open Terminal and run:

```bash
ssh -N -L 4200:10.204.35.227:4200 your-username@campus-machine
```

#### What to expect

- The first time you connect to a given machine, SSH shows a message like `The authenticity of
  host 'campus-machine' can't be established... Are you sure you want to continue connecting
  (yes/no)?`. This is normal for a first connection — type `yes` and press Enter.
- You'll be prompted for a password (or passphrase, if using a key). Type it and press Enter; the
  terminal won't show the characters as you type, which is normal for password prompts.
- After that, the terminal will appear to just sit there with no further output and no prompt.
  That's correct and expected — the `-N` flag tells SSH not to run a remote shell, only to hold the
  tunnel open. Leave this window open and minimized; don't close it or press `Ctrl+C` until you're
  done using the application.
- To stop the tunnel, go back to that window and press `Ctrl+C`.

Leave this SSH window running for as long as you want to use the application — closing it (or
losing the connection) closes the tunnel and the next command you run will fail to reach the
server. Open a separate terminal window for the commands later in this guide, and keep the SSH
window open alongside it.

If you're prompted for a password every time and would rather not be, ask the project maintainer
about setting up an SSH key instead.

Once the tunnel is open, the application (configured in the next step) talks to
`http://localhost:4200`, and the tunnel relays that to the real server. The `doctor` command later
in this guide gives a second confirmation that everything can talk to the server.

If the SSH connection fails:

- confirm Tailscale is connected (`tailscale status` should show you as `Connected`, alongside the
  campus machine);
- confirm your SSH username and password/key with the project maintainer;
- do not continue with real transcripts until the tunnel connects.

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
VLLM_BASE_URL=http://10.204.35.227:4200/v1
VLLM_MODEL=large
VLLM_INFERENCE_HUB_API_KEY=replace-with-issued-key
VLLM_OUTPUT_MODE=native
```

Open `.env` and make two changes:

- replace `replace-with-issued-key` with your issued API key;
- replace `VLLM_BASE_URL` with `http://localhost:4200/v1`. The generated value points at the
  server's real campus address, which is only reachable through the SSH tunnel from
  [step 1](#1-connect-to-the-project-server), not directly from your computer. Pointing at
  `localhost` sends traffic through that tunnel instead.

```env
DEID_ALLOW_REMOTE_LLM=true
VLLM_BASE_URL=http://localhost:4200/v1
VLLM_MODEL=large
VLLM_INFERENCE_HUB_API_KEY=replace-with-issued-key
VLLM_OUTPUT_MODE=native
```

Notes:

- `DEID_ALLOW_REMOTE_LLM=true` confirms that this project is intentionally sending transcript text
  to the approved institutional server.
- `VLLM_BASE_URL` must point at your end of the SSH tunnel (`http://localhost:4200/v1` if you
  followed the example above), not the server's real address.
- `VLLM_MODEL=large` must match the model name exposed by the server.
- `VLLM_INFERENCE_HUB_API_KEY` should be the key issued to the researcher. Do not commit it.
- `.env` is gitignored.
- If `.env` already exists, `init-config` will stop rather than overwrite it. Use
  `deidentify-transcripts init-config --force` only if you intentionally want to replace the
  existing file.

Check the configuration and model connection. Make sure the SSH tunnel from step 1 is still open
in its own terminal, then in another terminal run:

```bash
deidentify-transcripts doctor
```

Expected output resembles:

```text
OK: remote vllm endpoint http://localhost:4200/v1; model large
```

If `doctor` fails:

- confirm the SSH tunnel from step 1 is still open;
- confirm Tailscale is connected (`tailscale status` should show you as `Connected`);
- confirm the API key was copied correctly;
- confirm the server URL and model name with the project maintainer;
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

Make sure your SSH tunnel from [step 1](#1-connect-to-the-project-server) is still open in its own
terminal before running the commands below.

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
- An exit code of `2` means the run completed but human review is required. This is common and does
  not mean the software crashed.
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
second model pass checks the redacted text for remaining identifiers.

The review queue may contain genuine misses as well as conservative false alarms. Both require a
human decision. The automated status `clean` means only that the configured checks found no
unresolved items; it is not a certification of anonymity.

## 8. Platform notes

- Paths use `/`, for example `examples/sample-transcript.txt`.
- Activate the environment with `source .venv/bin/activate`.
- Close the SSH tunnel (`Ctrl+C` in its terminal) when you're finished for the day; it doesn't need
  to stay open outside of active sessions.
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
- Keep API keys, your Tailscale auth key and your SSH credentials secret; do not paste them into
  notebooks, scripts, chat, email, commits or screenshots.
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
