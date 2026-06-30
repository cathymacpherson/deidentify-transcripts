from typer.testing import CliRunner

from deidentify_transcripts.cli import app


def test_init_config_writes_project_server_env(tmp_path):
    runner = CliRunner()
    env_path = tmp_path / ".env"

    result = runner.invoke(app, ["init-config", "--output", str(env_path)])

    assert result.exit_code == 0
    content = env_path.read_text(encoding="utf-8")
    assert "DEID_ALLOW_REMOTE_LLM=true" in content
    assert "VLLM_BASE_URL=http://10.204.35.227:4200/v1" in content
    assert "VLLM_INFERENCE_HUB_API_KEY=replace-with-issued-key" in content


def test_init_config_writes_local_env(tmp_path):
    runner = CliRunner()
    env_path = tmp_path / ".env"

    result = runner.invoke(app, ["init-config", "--local", "--output", str(env_path)])

    assert result.exit_code == 0
    content = env_path.read_text(encoding="utf-8")
    assert "DEID_ALLOW_REMOTE_LLM=false" in content
    assert "OLLAMA_BASE_URL=http://localhost:11434/v1" in content
    assert "OLLAMA_MODEL=qwen3:8b" in content


def test_init_config_refuses_to_overwrite_without_force(tmp_path):
    runner = CliRunner()
    env_path = tmp_path / ".env"
    env_path.write_text("existing=true\n", encoding="utf-8")

    result = runner.invoke(app, ["init-config", "--output", str(env_path)])

    assert result.exit_code == 1
    assert env_path.read_text(encoding="utf-8") == "existing=true\n"
