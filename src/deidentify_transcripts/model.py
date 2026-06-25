from __future__ import annotations

import json
from typing import TypeVar

import httpx
from pydantic import BaseModel

from .config import Settings

T = TypeVar("T", bound=BaseModel)


class LocalModel:
    """Small OpenAI-compatible client restricted by Settings to localhost."""

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
        if isinstance(content, dict):
            return output_type.model_validate(content)
        try:
            return output_type.model_validate_json(content)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"local model returned invalid structured output: {content!r}") from exc

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

