# Optional local-only fallback

Use this only for testing/offline work. Local quality and speed depend on the machine and model.
It does not require Tailscale, the project server, or an issued API key — see the main
[README](../README.md) for that default setup.

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
`localhost:11434` — do not configure `OLLAMA_HOST=0.0.0.0` for sensitive transcripts.

On Ubuntu, Ollama normally runs as a `systemd` service. A supported NVIDIA GPU may be used
automatically when the appropriate driver is installed; check with `nvidia-smi`.

Useful local checks:

```bash
ollama list
deidentify-transcripts doctor
ollama ps
```

- `ollama list` shows downloaded models.
- `doctor` shows which model this application will request.
- `ollama ps` shows the model currently loaded in memory.
