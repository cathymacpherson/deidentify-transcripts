from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import httpx
import typer

from .config import Settings, env_template
from .detect import make_detector
from .gate import make_residual_detector
from .io import load_transcript, save_outputs
from .model import LocalModel
from .pipeline import deidentify
from .schemas import RunMetadata

app = typer.Typer(no_args_is_help=True)


def _pipeline_version() -> str:
    try:
        return version("deidentify-transcripts")
    except PackageNotFoundError:
        return "unknown"


def _discover_transcripts(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".json"))


@app.command("init-config")
def init_config(
    output_path: Path = typer.Option(Path(".env"), "--output", "-o", help="Where to write the env file"),
    local: bool = typer.Option(False, "--local", help="Write a local Ollama configuration instead"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing env file"),
) -> None:
    """Create a ready-to-edit .env file."""
    if output_path.exists() and not force:
        typer.echo(
            f"FAILED: {output_path} already exists. Use --force to replace it.",
            err=True,
        )
        raise typer.Exit(code=1)
    output_path.write_text(env_template(local=local), encoding="utf-8")
    if local:
        typer.echo(f"Wrote local Ollama config to {output_path}")
        typer.echo("Next: install Ollama, pull the configured model, then run deidentify-transcripts doctor")
    else:
        typer.echo(f"Wrote project-server config to {output_path}")
        typer.echo("Next: replace VLLM_INFERENCE_HUB_API_KEY with your issued key, then run deidentify-transcripts doctor")


@app.command()
def doctor() -> None:
    """Check LLM endpoint connectivity and the selected model."""
    try:
        settings = Settings.from_env()
        model = LocalModel(settings)
        models = model.list_models()
    except (ValueError, httpx.HTTPError, KeyError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if settings.model not in models:
        typer.echo(
            f"FAILED: model {settings.model!r} is not installed. Available: {', '.join(models) or 'none'}",
            err=True,
        )
        raise typer.Exit(code=1)
    location = "remote" if settings.allow_remote else "local"
    typer.echo(
        f"OK: {location} {settings.provider} endpoint {settings.base_url}; model {settings.model}"
    )


@app.command()
def run(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    transcript_id: str | None = typer.Option(None, "--id", help="Override the transcript identifier"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir"),
) -> None:
    """De-identify a plain-text or JSON transcript using the configured LLM endpoint."""
    try:
        settings = Settings.from_env()
        local_model = LocalModel(settings)
        transcript = load_transcript(input_path, transcript_id)
        if not transcript.turns:
            raise ValueError("the input transcript contains no non-empty turns")
        run_metadata = RunMetadata(
            model=settings.model,
            model_digest=local_model.model_digest(),
            pipeline_version=_pipeline_version(),
            started_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

        output, report = deidentify(
            transcript,
            detect_fn=make_detector(local_model.structured),
            residual_fn=make_residual_detector(local_model.structured),
            low_confidence_threshold=settings.low_confidence_threshold,
            run_metadata=run_metadata,
        )
        anonymised_path, report_path, queue_path = save_outputs(output, report, output_dir)
    except (ValueError, RuntimeError, httpx.HTTPError, KeyError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"{report.transcript_id}: status={report.status}, "
        f"{len(report.spans)} replacements, {len(report.review_items)} review items"
    )
    typer.echo(f"anonymised: {anonymised_path}")
    typer.echo(f"sensitive report: {report_path}")
    typer.echo(f"review queue: {queue_path}")
    if report.status == "needs_review":
        raise typer.Exit(code=2)


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("output"), "--output-dir"),
) -> None:
    """De-identify every plain-text or JSON transcript in a directory."""
    transcripts = _discover_transcripts(input_dir)
    if not transcripts:
        typer.echo(f"FAILED: no .txt or .json transcripts found in {input_dir}", err=True)
        raise typer.Exit(code=1)

    try:
        settings = Settings.from_env()
        local_model = LocalModel(settings)
        model_digest = local_model.model_digest()
    except (ValueError, RuntimeError, httpx.HTTPError, KeyError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    failed = 0
    needs_review = 0
    for path in transcripts:
        try:
            transcript = load_transcript(path)
            if not transcript.turns:
                raise ValueError("the input transcript contains no non-empty turns")
            run_metadata = RunMetadata(
                model=settings.model,
                model_digest=model_digest,
                pipeline_version=_pipeline_version(),
                started_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )
            output, report = deidentify(
                transcript,
                detect_fn=make_detector(local_model.structured),
                residual_fn=make_residual_detector(local_model.structured),
                low_confidence_threshold=settings.low_confidence_threshold,
                run_metadata=run_metadata,
            )
            save_outputs(output, report, output_dir)
        except (ValueError, RuntimeError, httpx.HTTPError, KeyError) as exc:
            failed += 1
            typer.echo(f"FAILED {path.name}: {exc}", err=True)
            continue

        if report.status == "needs_review":
            needs_review += 1
        typer.echo(
            f"{report.transcript_id}: status={report.status}, "
            f"{len(report.spans)} replacements, {len(report.review_items)} review items"
        )

    typer.echo(f"batch complete: {len(transcripts)} processed, {needs_review} need review, {failed} failed")
    if failed:
        raise typer.Exit(code=1)
    if needs_review:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
