from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    api_key: str
    low_confidence_threshold: float
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError(
                "OLLAMA_BASE_URL must point to localhost for this standalone privacy-preserving setup"
            )
        model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        if model.lower().endswith("-cloud"):
            raise ValueError("OLLAMA_MODEL must be a downloaded local model, not an Ollama cloud model")
        return cls(
            base_url=base_url,
            model=model,
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            low_confidence_threshold=float(os.getenv("DEID_LOW_CONFIDENCE_THRESHOLD", "0.5")),
            timeout_seconds=float(os.getenv("DEID_REQUEST_TIMEOUT_SECONDS", "180")),
        )
