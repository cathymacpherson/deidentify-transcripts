from typer.testing import CliRunner

from deidentify_transcripts.cli import _discover_transcripts, app


def test_init_config_writes_project_server_env(tmp_path):
    runner = CliRunner()
    env_path = tmp_path / ".env"

    result = runner.invoke(app, ["init-config", "--output", str(env_path)])

    assert result.exit_code == 0
    content = env_path.read_text(encoding="utf-8")
    assert "DEID_ALLOW_REMOTE_LLM=true" in content
    assert "VLLM_BASE_URL=http://100.127.175.5:4200/v1" in content
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


def test_discover_transcripts_filters_and_sorts(tmp_path):
    (tmp_path / "b.txt").write_text("Speaker: hello\n", encoding="utf-8")
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignore me", encoding="utf-8")
    (tmp_path / "subdir").mkdir()

    found = _discover_transcripts(tmp_path)

    assert found == [tmp_path / "a.json", tmp_path / "b.txt"]


def test_batch_fails_when_no_transcripts_found(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["batch", str(tmp_path)])

    assert result.exit_code == 1
    assert "no .txt or .json transcripts found" in result.output
