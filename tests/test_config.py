import pytest

from deidentify_transcripts.config import Settings


def test_settings_reject_non_local_endpoint(monkeypatch):
    monkeypatch.setenv("DEID_LLM_BASE_URL", "https://example.org/v1")
    monkeypatch.setenv("DEID_ALLOW_REMOTE_LLM", "false")
    with pytest.raises(ValueError, match="DEID_ALLOW_REMOTE_LLM"):
        Settings.from_env()


def test_settings_allow_remote_vllm_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("DEID_ALLOW_REMOTE_LLM", "true")
    monkeypatch.setenv("DEID_LLM_BASE_URL", "http://192.0.2.10:4200/v1")
    monkeypatch.setenv("DEID_LLM_MODEL", "large")
    monkeypatch.setenv("DEID_LLM_API_KEY", "test-key")
    monkeypatch.setenv("DEID_LLM_PROVIDER", "vllm")

    settings = Settings.from_env()

    assert settings.base_url == "http://192.0.2.10:4200/v1"
    assert settings.model == "large"
    assert settings.api_key == "test-key"
    assert settings.provider == "vllm"
    assert settings.allow_remote is True


def test_settings_reject_cloud_model(monkeypatch):
    monkeypatch.setenv("DEID_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("DEID_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("DEID_LLM_MODEL", "gemma3:12b-cloud")
    with pytest.raises(ValueError, match="cloud"):
        Settings.from_env()
