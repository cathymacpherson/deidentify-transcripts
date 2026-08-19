from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ENV_TEMPLATE = """# Default project setup: approved institutional vLLM/OpenAI-compatible endpoint.
# Transcript text is sent to this secure project server for de-identification.
# Reachable directly over Tailscale once your machine has joined the project's tailnet.
DEID_ALLOW_REMOTE_LLM=true
VLLM_BASE_URL=http://100.127.175.5:4200/v1
VLLM_MODEL=large
VLLM_INFERENCE_HUB_API_KEY=replace-with-issued-key
VLLM_OUTPUT_MODE=native

# Stage-one model detections below this confidence are sent for human review.
DEID_LOW_CONFIDENCE_THRESHOLD=0.5

# Per-request timeout. Increase if the server is busy.
DEID_REQUEST_TIMEOUT_SECONDS=180
"""

LOCAL_ENV_TEMPLATE = """# Local-only fallback for testing/offline work.
# Requires Ollama and a downloaded local model.
DEID_ALLOW_REMOTE_LLM=false
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen3:8b
OLLAMA_API_KEY=ollama

# Stage-one model detections below this confidence are sent for human review.
DEID_LOW_CONFIDENCE_THRESHOLD=0.5

# Per-request timeout. Local CPU-only inference can be slow.
DEID_REQUEST_TIMEOUT_SECONDS=180
"""


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    api_key: str
    provider: str
    allow_remote: bool
    low_confidence_threshold: float
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        base_url = _first_env(
            "DEID_LLM_BASE_URL",
            "VLLM_BASE_URL",
            "OLLAMA_BASE_URL",
            default="http://localhost:11434/v1",
        ).rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("LLM base URL must start with http:// or https://")

        provider = os.getenv("DEID_LLM_PROVIDER")
        if not provider:
            provider = "vllm" if os.getenv("VLLM_BASE_URL") else "ollama"
        provider = provider.strip().lower()

        allow_remote = _env_bool("DEID_ALLOW_REMOTE_LLM", default=False)
        is_local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if not is_local and not allow_remote:
            raise ValueError(
                "remote LLM endpoints require DEID_ALLOW_REMOTE_LLM=true because transcripts will leave this machine"
            )

        model = _first_env("DEID_LLM_MODEL", "VLLM_MODEL", "OLLAMA_MODEL", default="qwen3:8b")
        if provider == "ollama" and model.lower().endswith("-cloud"):
            raise ValueError("OLLAMA_MODEL must be a downloaded local model, not an Ollama cloud model")
        return cls(
            base_url=base_url,
            model=model,
            api_key=_first_env(
                "DEID_LLM_API_KEY",
                "VLLM_INFERENCE_HUB_API_KEY",
                "OLLAMA_API_KEY",
                default="ollama",
            ),
            provider=provider,
            allow_remote=allow_remote,
            low_confidence_threshold=float(os.getenv("DEID_LOW_CONFIDENCE_THRESHOLD", "0.5")),
            timeout_seconds=float(os.getenv("DEID_REQUEST_TIMEOUT_SECONDS", "180")),
        )


def _first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_template(*, local: bool = False) -> str:
    return LOCAL_ENV_TEMPLATE if local else PROJECT_ENV_TEMPLATE
