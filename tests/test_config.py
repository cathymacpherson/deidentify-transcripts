import pytest

from deidentify_transcripts.config import Settings


def test_settings_reject_non_local_endpoint(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://example.org/v1")
    with pytest.raises(ValueError, match="localhost"):
        Settings.from_env()


def test_settings_reject_cloud_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:12b-cloud")
    with pytest.raises(ValueError, match="cloud"):
        Settings.from_env()
