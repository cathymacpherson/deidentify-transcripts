from __future__ import annotations

import json
from typing import TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from .config import Settings

T = TypeVar("T", bound=BaseModel)


def parse_structured_content(content: object, output_type: type[T]) -> T:
    if isinstance(content, dict):
        return output_type.model_validate(content)
    if not isinstance(content, str):
        raise RuntimeError(f"local model returned unsupported structured output: {content!r}")
    try:
        return output_type.model_validate_json(content)
    except (ValueError, json.JSONDecodeError):
        try:
            value, _ = json.JSONDecoder().raw_decode(content.lstrip())
            return output_type.model_validate(value)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"local model returned invalid structured output: {content!r}"
            ) from exc


class LocalModel:
    """Small OpenAI-compatible chat-completions client."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def list_models(self) -> list[str]:
        response = httpx.get(
            f"{self.settings.base_url}/models",
            headers=self._headers(),
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        return [item["id"] for item in response.json().get("data", [])]

    def model_digest(self) -> str | None:
        hostname = urlparse(self.settings.base_url).hostname
        if self.settings.provider != "ollama" or hostname not in {"localhost", "127.0.0.1", "::1"}:
            return None
        response = httpx.get(
            f"{self.settings.base_url.removesuffix('/v1')}/api/tags",
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        for item in response.json().get("models", []):
            if item.get("name") == self.settings.model or item.get("model") == self.settings.model:
                digest = item.get("digest")
                if digest:
                    return str(digest)
        raise RuntimeError(f"model {self.settings.model!r} was not found in Ollama's local registry")

    def structured(self, *, system: str, text: str, output_type: type[T]) -> T:
        schema = output_type.model_json_schema()
        response = httpx.post(
            f"{self.settings.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.settings.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_type.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
            },
            timeout=self.settings.timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return parse_structured_content(content, output_type)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
