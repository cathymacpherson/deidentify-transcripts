from __future__ import annotations

from pathlib import Path

import httpx
import typer

from .config import Settings
from .detect import make_detector
from .gate import make_residual_detector
from .io import load_transcript, save_outputs
from .model import LocalModel
from .pipeline import deidentify

app = typer.Typer(no_args_is_help=True)


@app.command()
def doctor() -> None:
    """Check local-only configuration, Ollama connectivity and the selected model."""
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
    typer.echo(f"OK: local endpoint {settings.base_url}; model {settings.model}")


@app.command()
def run(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    transcript_id: str | None = typer.Option(None, "--id", help="Override the transcript identifier"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir"),
) -> None:
    """De-identify a plain-text or JSON transcript using a downloaded local model."""
    try:
        settings = Settings.from_env()
        local_model = LocalModel(settings)
        transcript = load_transcript(input_path, transcript_id)
        if not transcript.turns:
            raise ValueError("the input transcript contains no non-empty turns")

        output, report = deidentify(
            transcript,
            detect_fn=make_detector(local_model.structured),
            residual_fn=make_residual_detector(local_model.structured),
            low_confidence_threshold=settings.low_confidence_threshold,
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


if __name__ == "__main__":
    app()

