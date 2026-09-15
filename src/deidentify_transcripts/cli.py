from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import httpx
import typer

from .config import Settings, env_template
from .detect import make_detector
from .gate import make_residual_detector
from .labelling import apply_second_opinion, label_transcript
from .review import render_report
from .inventory import (
    DEFAULT_ANOMALY_TOLERANCE,
    DEFAULT_THRESHOLD,
    audit_speaker_values,
    classify_speaker,
    discover_transcripts_recursive,
    display_speaker_value,
    is_multi_role,
    review_reason,
    tolerated_anomaly_note,
    LABELLED,
    PARTIAL,
    REVIEW,
    UNLABELLED,
    take_inventory,
)
from .baselines import BASELINES
from .io import load_transcript, save_outputs
from .metrics import Score, edit_actions, error_capture_curve, score_labels
from .splits import assign_transcripts, dominant_group_share, grouped_k_fold, load_mapping
from .supervised import train as train_labeller
from .model import LocalModel
from .pipeline import deidentify
from .schemas import RunMetadata

app = typer.Typer(no_args_is_help=True)


def _pipeline_version() -> str:
    try:
        return version("deidentify-transcripts")
    except PackageNotFoundError:
        return "unknown"


_TRANSCRIPT_SUFFIXES = (".txt", ".json", ".xlsx", ".xls")


def _discover_transcripts(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in _TRANSCRIPT_SUFFIXES)


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
    """De-identify a plain-text, JSON, or Excel transcript using the configured LLM endpoint."""
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
    """De-identify every plain-text, JSON, or Excel transcript in a directory."""
    transcripts = _discover_transcripts(input_dir)
    if not transcripts:
        typer.echo(
            f"FAILED: no {', '.join(_TRANSCRIPT_SUFFIXES)} transcripts found in {input_dir}",
            err=True,
        )
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


class _SecondOpinion:
    """Trained-model labels for a transcript, from a model that never saw its speaker.

    Training per speaker rather than once over everything keeps the comparison honest: a model
    that had already seen this transcript's clinician would agree with the LLM for the wrong
    reason, and the disagreement signal would be worth less than it appears.
    """

    def __init__(self, mapping_path: Path, all_paths: list[Path], labels: tuple[str, str]):
        from .splits import load_mapping, member_id_from_filename

        self.labels = labels
        self.mapping = load_mapping(mapping_path)
        self.member_id = member_id_from_filename
        self.by_path: dict[Path, str | None] = {}
        for path in all_paths:
            member = member_id_from_filename(path)
            self.by_path[path] = self.mapping.group_of.get(member) if member else None
        self.all_paths = all_paths
        self._cache: dict[str, object] = {}

    def _pairs_excluding(self, group: str):
        out = []
        for path in self.all_paths:
            if self.by_path.get(path) == group:
                continue
            try:
                turns = load_transcript(path).turns
            except (ValueError, KeyError, OSError):
                continue
            gold = [(t.speaker or "").strip() for t in turns]
            if any(g in self.labels for g in gold):
                out.append((turns, gold))
        return out

    def labels_for(self, path: Path, turns) -> list[str]:
        group = self.by_path.get(path)
        if group is None:
            return []
        if group not in self._cache:
            pairs = self._pairs_excluding(group)
            if not pairs:
                self._cache[group] = None
            else:
                self._cache[group] = train_labeller(
                    pairs, primary=self.labels[0], secondary=self.labels[1]
                )
        model = self._cache[group]
        return list(model.predict(turns).labels) if model is not None else []


@app.command("label")
def label(
    input_path: Path = typer.Argument(..., exists=True, readable=True,
                                      help="A transcript file, or a directory of them"),
    output_dir: Path = typer.Option(Path("output"), "--output-dir"),
    mapping_path: Path | None = typer.Option(
        None, "--mapping",
        help="Identity mapping. Adds the trained model as an independent second opinion.",
    ),
    reference_dir: Path = typer.Option(
        Path("data/labelled"), "--reference",
        help="Labelled transcripts used to train the second opinion",
    ),
    window: int = typer.Option(40, "--window", min=5),
    step: int = typer.Option(13, "--step", min=1),
    primary: str = typer.Option("C", "--primary"),
    secondary: str = typer.Option("T", "--secondary"),
) -> None:
    """Assign speaker labels and write a prioritised review report for each transcript.

    This CALLS THE CONFIGURED SERVER. For each input it writes the labelled transcript and a
    review report naming the turns most worth a human's attention.
    """
    labels = (primary, secondary)
    try:
        settings = Settings.from_env()
        model = LocalModel(settings)
    except (ValueError, httpx.HTTPError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    paths = discover_transcripts_recursive(input_path)
    if not paths:
        typer.echo(f"FAILED: no transcripts found at {input_path}", err=True)
        raise typer.Exit(code=1)

    second_opinion = None
    if mapping_path is not None:
        try:
            reference = discover_transcripts_recursive(reference_dir)
            second_opinion = _SecondOpinion(mapping_path, reference, labels)
        except (ValueError, OSError) as exc:
            typer.echo(f"WARNING: second opinion unavailable: {exc}", err=True)

    labelled_dir = output_dir / "labelled"
    review_dir = output_dir / "review"
    labelled_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for path in paths:
        try:
            transcript = load_transcript(path)
            turns = transcript.turns
            if not turns:
                raise ValueError("no turns")
            result = label_transcript(
                turns, model.structured, model.structured, size=window, step=step
            )
        except (ValueError, KeyError, RuntimeError, httpx.HTTPError, OSError) as exc:
            failed += 1
            typer.echo(f"FAILED {path.name}: {exc}", err=True)
            continue

        confidence = list(result.confidence)
        second_labels: list[str] = []
        if second_opinion is not None:
            second_labels = second_opinion.labels_for(path, turns)
            if second_labels:
                confidence = apply_second_opinion(result.votes, second_labels)

        # label_transcript already preserves manual labels; record where each came from.
        sources = [
            "manual" if (t.speaker or "").strip() in labels else "model" for t in turns
        ]
        for turn, assigned in zip(turns, result.labels):
            turn.speaker = assigned

        out = labelled_dir / f"{transcript.transcript_id}.json"
        out.write_text(
            json.dumps({
                "transcript_id": transcript.transcript_id,
                "turns": [
                    {
                        "turn_id": t.turn_id,
                        "speaker": t.speaker,
                        "speaker_confidence": round(c, 3),
                        "speaker_source": src,
                        "text": t.text,
                    }
                    for t, c, src in zip(turns, confidence, sources)
                ],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report_path = review_dir / f"{transcript.transcript_id}.review.md"
        report_path.write_text(
            render_report(
                transcript.transcript_id, turns, result.labels,
                result.votes, confidence, second_labels or None,
            ),
            encoding="utf-8",
        )
        flagged = sum(1 for c in confidence if c < 0.999)
        kept = f", {result.manual_count} manual kept" if result.manual_count else ""
        typer.echo(
            f"{transcript.transcript_id}: {len(turns)} turns{kept}, {flagged} flagged "
            f"({flagged/len(turns):.0%}), coverage {result.coverage:.0%} -> {report_path}"
        )

    if failed:
        raise typer.Exit(code=1)


@app.command("label-diagnose")
def label_diagnose(
    report: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
) -> None:
    """Summarise a label-report.csv: where the errors are and why they were not flagged.

    Reads the report locally and prints aggregate counts only - no transcript text, no filenames.
    Safe to share.
    """
    import csv as _csv
    from collections import Counter

    with report.open(encoding="utf-8", newline="") as handle:
        rows = list(_csv.DictReader(handle))
    if not rows:
        typer.echo("FAILED: report is empty", err=True)
        raise typer.Exit(code=1)

    scored = [r for r in rows if r["gold"] in ("C", "T")]
    errors = [r for r in scored if r["error"] == "WRONG"]
    anchored = [r for r in scored if r["anchor"]]
    typer.echo(
        f"{len(scored)} scored turns, {len(errors)} wrong ({len(errors)/len(scored):.1%}); "
        f"{len(anchored)} anchored ({len(anchored)/len(scored):.1%})"
    )

    anchor_err = [r for r in errors if r["anchor"]]
    typer.echo(
        f"\nerrors in anchored turns:   {len(anchor_err):>4} "
        f"({len(anchor_err)/max(len(anchored),1):.1%} of anchored turns)"
    )
    unanchored = len(scored) - len(anchored)
    typer.echo(
        f"errors in unanchored turns: {len(errors)-len(anchor_err):>4} "
        f"({(len(errors)-len(anchor_err))/max(unanchored,1):.1%} of unanchored turns)"
    )
    contradicted = [r for r in scored if r["anchor_contradicted"] == "yes"]
    typer.echo(
        f"anchors contradicted by windows: {len(contradicted)}; "
        f"of those, wrong: {sum(1 for r in contradicted if r['error'] == 'WRONG')}"
    )

    files = sorted({r["file"] for r in scored})
    if len(files) > 1:
        typer.echo("\nper-file error rate (is this uniform, or a few bad transcripts?):")
        typer.echo(f"  {'turns':>7} {'wrong':>7} {'rate':>7} {'anchored':>9} {'unclear':>8}")
        rates = []
        for name in files:
            rows_f = [r for r in scored if r["file"] == name]
            wrong = sum(1 for r in rows_f if r["error"] == "WRONG")
            anchored_f = sum(1 for r in rows_f if r["anchor"])
            unclear_f = sum(1 for r in rows_f if r["predicted"] == "unclear")
            rates.append(wrong / len(rows_f))
            typer.echo(
                f"  {len(rows_f):>7} {wrong:>7} {wrong/len(rows_f):>6.1%} "
                f"{anchored_f/len(rows_f):>8.0%} {unclear_f:>8}"
            )
        # Does a file's own flag rate predict how wrong it is? If so, transcripts can be
        # triaged before anyone looks at them, using only what is known without gold labels.
        from .triage import FileStats, spearman, triage_curve

        stats = []
        for name in files:
            rows_f = [r for r in scored if r["file"] == name]
            stats.append(FileStats(
                name=name,
                turns=len(rows_f),
                flagged=sum(1 for r in rows_f if float(r["confidence"]) < 0.999),
                errors=sum(1 for r in rows_f if r["error"] == "WRONG"),
            ))
        rho = spearman([s_.flag_rate for s_ in stats], [s_.error_rate for s_ in stats])
        curve = triage_curve(stats)
        # rho measures whether the ordering is right; lift measures whether acting on it pays.
        # A high rho with low lift means the ranking is correct but the files barely differ,
        # so routing on it saves little. Judge on lift.
        quarter = curve[max(0, len(curve) // 4 - 1)]
        lift = (
            quarter.share_of_errors / quarter.share_of_turns if quarter.share_of_turns else 0.0
        )
        typer.echo(
            f"\nflag rate vs error rate across files: rho = {rho:+.2f}, "
            f"lift at the worst quarter = {lift:.2f}x"
        )
        if lift >= 2.0:
            verdict = "worth acting on - the worst files carry far more than their share of error"
        elif lift >= 1.5:
            verdict = "marginal - a useful sort order, but not a basis for skipping files"
        else:
            verdict = (
                "not worth acting on - the worst-looking files carry barely more error than "
                "the rest, so per-file triage would save little"
            )
        typer.echo(f"  {verdict}")
        if len(stats) < 30:
            typer.echo(
                f"  (only {len(stats)} files - treat both figures as provisional)"
            )
        typer.echo("\nrouting the worst-looking files to full manual coding:")
        typer.echo(f"  {'files':>6} {'of files':>9} {'of turns':>9} {'of errors':>10} {'vs chance':>10}")
        for point in curve:
            if point.files_routed % max(1, len(curve) // 6) and point.files_routed != len(curve):
                continue
            lift = point.share_of_errors / point.share_of_turns if point.share_of_turns else 0
            typer.echo(
                f"  {point.files_routed:>6} {point.share_of_files:>8.0%} "
                f"{point.share_of_turns:>8.0%} {point.share_of_errors:>9.0%} {lift:>9.2f}x"
            )

        best, worst = min(rates), max(rates)
        typer.echo(
            f"\n  best file {best:.1%}, worst {worst:.1%} "
            f"({worst/max(best, 1e-9):.1f}x spread)"
        )
        half = sorted(rates, reverse=True)[: max(1, len(rates) // 2)]
        typer.echo(
            f"  the worst half of files carry "
            f"{sum(half)/max(sum(rates), 1e-9):.0%} of the total error"
        )

    typer.echo("\nconfidence distribution (the ranking signal):")
    buckets = Counter(r["confidence"] for r in scored)
    typer.echo(f"  {len(buckets)} distinct confidence value(s)")
    typer.echo(f"  {'confidence':>10} {'turns':>7} {'wrong':>7} {'error rate':>11}")
    for value, n in sorted(buckets.items(), key=lambda kv: -float(kv[0])):
        wrong = sum(1 for r in scored if r["confidence"] == value and r["error"] == "WRONG")
        typer.echo(f"  {value:>10} {n:>7} {wrong:>7} {wrong/n:>10.1%}")

    top = [r for r in scored if float(r["confidence"]) >= 0.999]
    top_wrong = sum(1 for r in top if r["error"] == "WRONG")
    typer.echo(
        f"\nCONFIDENTLY WRONG: {top_wrong} error(s) sit at maximum confidence "
        f"({top_wrong/max(len(errors),1):.1%} of all errors)."
    )
    typer.echo(
        "  These can never be flagged at any review budget - they are the ceiling on capture."
    )

    vote_shapes = Counter(len(r["votes"].split("|")) if r["votes"] else 0 for r in scored)
    typer.echo(f"\nviews per turn: {dict(sorted(vote_shapes.items()))}")

    if any(r.get("second_opinion") for r in scored):
        agree = [r for r in scored if r.get("second_opinion") == r["predicted"]]
        differ = [r for r in scored if r.get("second_opinion")
                  and r["second_opinion"] != r["predicted"]]
        typer.echo("\nsecond opinion (independent trained model):")
        for name, group in (("agrees", agree), ("disagrees", differ)):
            if group:
                wrong = sum(1 for r in group if r["error"] == "WRONG")
                typer.echo(
                    f"  {name:>9}: {len(group):>5} turns, {wrong:>4} wrong "
                    f"({wrong/len(group):.1%})"
                )
        caught = sum(1 for r in differ if r["error"] == "WRONG")
        typer.echo(
            f"  flagging every disagreement costs {len(differ)} turns "
            f"({len(differ)/len(scored):.0%}) and catches {caught}/{len(errors)} errors "
            f"({caught/max(len(errors),1):.0%})"
        )


@app.command("label-eval")
def label_eval(
    input_dir: Path = typer.Argument(
        Path("data/labelled"), exists=True, readable=True,
        help="Labelled transcripts to score the LLM passes against",
    ),
    limit: int = typer.Option(
        3, "--limit", min=1, help="How many transcripts to run. Start small: this calls the server."
    ),
    window: int = typer.Option(40, "--window", min=5, help="Turns per window"),
    step: int = typer.Option(13, "--step", min=1, help="Window step; smaller means more views"),
    primary: str = typer.Option("C", "--primary"),
    secondary: str = typer.Option("T", "--secondary"),
    output: Path = typer.Option(
        Path("label-report.csv"), "--output", "-o",
        help="Per-turn CSV: gold, predicted, confidence, votes. Contains transcript text.",
    ),
    mapping_path: Path | None = typer.Option(
        None, "--mapping",
        help="Identity mapping. Adds the trained model as an independent second opinion.",
    ),
) -> None:
    """Run the LLM anchor and window passes over labelled transcripts and score them.

    This CALLS THE CONFIGURED SERVER and sends transcript text to it. Start with --limit 1 to
    check timing and cost before running the corpus.
    """
    from .metrics import edit_actions, error_capture_curve

    labels = (primary, secondary)
    try:
        settings = Settings.from_env()
        model = LocalModel(settings)
    except (ValueError, httpx.HTTPError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    all_paths = discover_transcripts_recursive(input_dir)
    paths = all_paths[:limit]
    if not paths:
        typer.echo(f"FAILED: no transcripts found under {input_dir}", err=True)
        raise typer.Exit(code=1)

    if output.exists():
        typer.echo(f"NOTE: {output} exists and will be overwritten by this run.")

    second_opinion = None
    if mapping_path is not None:
        try:
            second_opinion = _SecondOpinion(mapping_path, all_paths, labels)
        except (ValueError, OSError) as exc:
            typer.echo(f"WARNING: second opinion unavailable: {exc}", err=True)

    gold: list[str] = []
    predicted: list[str] = []
    confidence: list[float] = []
    rows: list[list[str]] = []
    anchors_total = calls_total = unclear_total = no_second_opinion = 0
    started = datetime.now(timezone.utc)

    for path in paths:
        try:
            turns = load_transcript(path).turns
        except (ValueError, KeyError, OSError) as exc:
            typer.echo(f"  skipped a file: {exc}", err=True)
            continue
        if not any((t.speaker or "").strip() in labels for t in turns):
            continue
        try:
            result = label_transcript(
                turns, model.structured, model.structured, size=window, step=step,
                # The gold labels are in the speaker field; using them would be marking
                # its own homework.
                use_existing=False,
            )
        except (RuntimeError, httpx.HTTPError) as exc:
            typer.echo(f"FAILED during labelling: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        turn_confidence = list(result.confidence)
        second_labels: list[str] = []
        if second_opinion is not None:
            second_labels = second_opinion.labels_for(path, turns)
            if second_labels:
                turn_confidence = apply_second_opinion(result.votes, second_labels)
            else:
                # Silent degradation otherwise: this file keeps vote-only confidence while the
                # rest of the run has a second opinion, and the two are not comparable.
                no_second_opinion += 1
                typer.echo(
                    "  WARNING: no second opinion for this file - it matched no mapping row, "
                    "so its confidence is vote-agreement only."
                )

        gold.extend((t.speaker or "").strip() for t in turns)
        predicted.extend(result.labels)
        confidence.extend(turn_confidence)
        for i, (turn, vote) in enumerate(zip(turns, result.votes)):
            truth = (turn.speaker or "").strip()
            rows.append([
                path.name, str(turn.turn_id), truth, vote.winner,
                f"{turn_confidence[i]:.3f}",
                second_labels[i] if second_labels else "",
                vote.anchor or "",
                "yes" if vote.contradicts_anchor else "",
                "|".join(vote.votes),
                "WRONG" if truth in labels and truth != vote.winner else "",
                turn.text,
            ])
        anchors_total += result.anchor_count
        calls_total += result.window_count
        unclear_total += result.unclear_count
        warning = ""
        if result.coverage < 0.995:
            warning = (
                f"  <-- model returned only {result.coverage:.1%} of requested labels; "
                f"{result.unvoted_turns} turn(s) got no vote at all"
            )
        typer.echo(
            f"  {len(turns)} turns, {result.anchor_count} anchors, "
            f"{result.window_count} windows, {result.unclear_count} unclear, "
            f"coverage {result.coverage:.1%}, min views {result.min_views}{warning}"
        )

    if not gold:
        typer.echo("FAILED: nothing with usable labels was processed", err=True)
        raise typer.Exit(code=1)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    if no_second_opinion:
        typer.echo(
            f"\nWARNING: {no_second_opinion} of {len(paths)} file(s) had no second opinion. "
            "Their confidence is not comparable with the rest, and the pooled capture figure "
            "understates what the ensemble achieves."
        )
    result_score = score_labels(gold, predicted, labels=labels)
    typer.echo(
        f"\n{len(gold)} turns over {len(paths)} file(s) in {elapsed:.0f}s; "
        f"{anchors_total} anchors, {calls_total} window calls, {unclear_total} unclear"
    )
    typer.echo(
        f"  accuracy {result_score.accuracy:.3f}  macro-F1 {result_score.macro_f1:.3f}  "
        f"{primary}-F1 {result_score.per_class[primary].f1:.3f}  "
        f"{secondary}-F1 {result_score.per_class[secondary].f1:.3f}  "
        f"boundary {result_score.boundary_f1:.3f}"
    )
    if result_score.looks_flipped:
        typer.echo("  WARNING: roles appear inverted - the anchor pass failed on this sample.")

    curve = error_capture_curve(gold, predicted, confidence, labels=labels)
    if curve:
        actions = edit_actions(gold, predicted, labels=labels)
        typer.echo(
            f"\nreviewer view: {curve[0].errors_total} wrong turns in {actions} block(s)"
        )
        typer.echo(f"  {'flag':>6} {'turns':>8} {'capture':>9} {'left unflagged':>15}")
        for point in curve:
            typer.echo(
                f"  {point.budget:>5.0%} {point.reviewed:>8} "
                f"{point.error_capture:>8.1%} {point.errors_missed:>15}"
            )
    if rows:
        import csv as _csv

        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = _csv.writer(handle)
            writer.writerow([
                "file", "turn_id", "gold", "predicted", "confidence", "second_opinion",
                "anchor", "anchor_contradicted", "votes", "error", "text",
            ])
            writer.writerows(rows)
        typer.echo(f"\nper-turn report: {output}")
        typer.echo("  Contains transcript text - keep it out of version control.")

    per_transcript = elapsed / max(len(paths), 1)
    typer.echo(
        f"\n~{per_transcript:.0f}s per transcript at these settings."
    )


@app.command()
def evaluate(
    input_dir: Path = typer.Argument(
        Path("data/labelled"), exists=True, readable=True,
        help="Transcripts with manual speaker labels to score against",
    ),
    primary: str = typer.Option("C", "--primary", help="Label for the first role"),
    secondary: str = typer.Option("T", "--secondary", help="Label for the second role"),
    mapping_path: Path | None = typer.Option(
        None, "--mapping",
        help="Identity mapping CSV. Enables the trained model, scored on speaker-disjoint folds.",
    ),
    folds: int = typer.Option(5, "--folds", min=2, help="Number of cross-validation folds"),
    decode: str = typer.Option(
        "both", "--decode",
        help="Trained-model decoding: viterbi, marginal, or both for a side-by-side comparison",
    ),
) -> None:
    """Score the deterministic baselines against manually labelled transcripts.

    Establishes the bar any model-based approach has to clear. No model and no network: this runs
    entirely offline. Prints aggregate figures only - no filenames, no transcript text.
    """
    paths = discover_transcripts_recursive(input_dir)
    if not paths:
        typer.echo(f"FAILED: no transcripts found under {input_dir}", err=True)
        raise typer.Exit(code=1)

    labels = (primary, secondary)
    gold: list[str | None] = []
    predictions: dict[str, list[str]] = {name: [] for name in BASELINES}
    confidences: dict[str, list[float]] = {name: [] for name in BASELINES}
    scored_files = 0

    for path in paths:
        try:
            turns = load_transcript(path).turns
        except (ValueError, KeyError, OSError):
            continue
        if not any((t.speaker or "").strip() in labels for t in turns):
            continue  # nothing to score against in this file
        scored_files += 1
        gold.extend((t.speaker or "").strip() for t in turns)
        for name, fn in BASELINES.items():
            prediction = fn(turns)
            predictions[name].extend(prediction.labels)
            confidences[name].extend(prediction.confidence)

    if not scored_files:
        typer.echo(
            f"FAILED: none of the {len(paths)} transcript(s) carry {primary}/{secondary} labels",
            err=True,
        )
        raise typer.Exit(code=1)

    results: dict[str, Score] = {
        name: score_labels(gold, predictions[name], labels=labels) for name in BASELINES
    }
    first = next(iter(results.values()))
    typer.echo(
        f"{scored_files} labelled file(s); {first.scored_turns} turns scored, "
        f"{first.skipped_turns} skipped (no usable gold label)"
    )
    typer.echo(
        f"\n  {'baseline':<12} {'accuracy':>9} {'macro-F1':>9} "
        f"{primary + '-F1':>8} {secondary + '-F1':>8} {'boundary':>9}"
    )
    for name, result in results.items():
        typer.echo(
            f"  {name:<12} {result.accuracy:>9.3f} {result.macro_f1:>9.3f} "
            f"{result.per_class[primary].f1:>8.3f} {result.per_class[secondary].f1:>8.3f} "
            f"{result.boundary_f1:>9.3f}"
        )
        if result.looks_flipped:
            typer.echo(f"      ^ roles appear inverted (flipped macro-F1 {result.flipped_macro_f1:.3f})")

    best = max(results.items(), key=lambda kv: kv[1].macro_f1)
    typer.echo(
        f"\nBar to beat: macro-F1 {best[1].macro_f1:.3f} ({best[0]}). "
        f"Accuracy alone is misleading here - {results['majority'].accuracy:.1%} "
        "comes from labelling every turn the same."
    )

    if mapping_path is not None:
        _evaluate_trained(paths, mapping_path, folds, labels, decode)
    else:
        typer.echo(
            "\nThe trained model was not evaluated. It learns from the data, so it needs "
            "speaker-disjoint folds: pass --mapping to enable it."
        )

    curve = error_capture_curve(
        gold, predictions["rule"], confidences["rule"], labels=labels
    )
    if curve:
        actions = edit_actions(gold, predictions["rule"], labels=labels)
        total = curve[0].errors_total
        typer.echo(
            f"\nrule baseline, reviewer view: {total} wrong turns in "
            f"{actions} contiguous block(s) - roughly {actions} corrections, not {total}."
        )
        typer.echo(
            f"\n  {'flag':>6} {'turns':>8} {'errors caught':>14} {'capture':>9} "
            f"{'left unflagged':>15}"
        )
        for point in curve:
            typer.echo(
                f"  {point.budget:>5.0%} {point.reviewed:>8} "
                f"{point.errors_caught:>14} {point.error_capture:>8.1%} "
                f"{point.errors_missed:>15}"
            )
        typer.echo(
            "\n  'capture' is the share of the system's own mistakes that land in the flagged "
            "set.\n  Errors left unflagged are the ones the reviewer must find unaided - the "
            "number\n  that actually determines how hard this is to review."
        )


def _evaluate_trained(
    paths: list[Path], mapping_path: Path, k: int, labels: tuple[str, str],
    decode: str = "both",
) -> None:
    """Cross-validate the trained sequence labeller over speaker-disjoint folds."""
    primary, secondary = labels
    try:
        mapping = load_mapping(mapping_path)
        fold_list = grouped_k_fold(mapping, k=k)
    except (ValueError, OSError) as exc:
        typer.echo(f"\nFAILED to build folds: {exc}", err=True)
        return

    by_fold, unmatched = assign_transcripts(paths, fold_list)
    if unmatched:
        typer.echo(
            f"\nWARNING: {len(unmatched)} transcript(s) matched no mapping row and are "
            "excluded from the trained-model evaluation."
        )

    def load_pairs(paths_in: list[Path]):
        out = []
        for path in paths_in:
            try:
                turns = load_transcript(path).turns
            except (ValueError, KeyError, OSError):
                continue
            gold = [(t.speaker or "").strip() for t in turns]
            if any(g in labels for g in gold):
                out.append((turns, gold))
        return out

    modes = ["viterbi", "marginal"] if decode == "both" else [decode]
    typer.echo(f"\ntrained model, {len(fold_list)}-fold speaker-disjoint cross-validation:")
    header = f"  {'fold':>4} {'files':>6} {'turns':>8}"
    for mode in modes:
        header += f" {mode[:4] + ' F1':>9} {mode[:4] + ' bnd':>9}"
    typer.echo(header + f" {'largest grp':>12}")
    pooled_gold: list[str] = []
    pooled_pred: dict[str, list[str]] = {m: [] for m in modes}
    pooled_conf: dict[str, list[float]] = {m: [] for m in modes}
    per_fold: dict[str, list[float]] = {m: [] for m in modes}

    for fold in fold_list:
        test_pairs = load_pairs(by_fold[fold.index])
        train_pairs = load_pairs(
            [p for other in fold_list if other.index != fold.index for p in by_fold[other.index]]
        )
        if not test_pairs or not train_pairs:
            typer.echo(f"  {fold.index:>4} {'-':>6} {'-':>8}  (skipped: no data on one side)")
            continue
        labeller = train_labeller(train_pairs, primary=primary, secondary=secondary)

        fold_gold: list[str] = []
        fold_pred: dict[str, list[str]] = {m: [] for m in modes}
        for turns, gold in test_pairs:
            fold_gold.extend(gold)
            for mode in modes:
                prediction = labeller.predict(turns, decode=mode)
                fold_pred[mode].extend(prediction.labels)
                pooled_conf[mode].extend(prediction.confidence)
        pooled_gold.extend(fold_gold)
        line = f"  {fold.index:>4} {len(test_pairs):>6}"
        for i, mode in enumerate(modes):
            result = score_labels(fold_gold, fold_pred[mode], labels=labels)
            per_fold[mode].append(result.macro_f1)
            pooled_pred[mode].extend(fold_pred[mode])
            if i == 0:
                line += f" {result.scored_turns:>8}"
            line += f" {result.macro_f1:>9.3f} {result.boundary_f1:>9.3f}"
        typer.echo(line + f" {dominant_group_share(fold, mapping):>11.0%}")

    if not any(per_fold.values()):
        typer.echo("  no fold could be evaluated")
        return
    spread = 0.0
    for mode in modes:
        pooled = score_labels(pooled_gold, pooled_pred[mode], labels=labels)
        scores = per_fold[mode]
        spread = max(spread, max(scores) - min(scores))
        typer.echo(
            f"\n  {mode}: pooled macro-F1 {pooled.macro_f1:.3f}, boundary "
            f"{pooled.boundary_f1:.3f}; per-fold {min(scores):.3f}-{max(scores):.3f}"
        )
    if spread > 0.1:
        typer.echo(
            "  Per-fold spread is wide: performance depends materially on which speakers are "
            "held out, so the pooled figure alone overstates how settled this is."
        )

    # The reviewer view for the trained model, whose confidence comes from forward-backward
    # posteriors rather than a hand-made score - the signal actually worth measuring.
    best_mode = max(modes, key=lambda m: score_labels(
        pooled_gold, pooled_pred[m], labels=labels).macro_f1)
    curve = error_capture_curve(
        pooled_gold, pooled_pred[best_mode], pooled_conf[best_mode], labels=labels
    )
    if curve:
        actions = edit_actions(pooled_gold, pooled_pred[best_mode], labels=labels)
        typer.echo(
            f"\n  trained model ({best_mode}), reviewer view: "
            f"{curve[0].errors_total} wrong turns in {actions} block(s)"
        )
        typer.echo(f"  {'flag':>6} {'turns':>8} {'capture':>9} {'left unflagged':>15}")
        for point in curve:
            typer.echo(
                f"  {point.budget:>5.0%} {point.reviewed:>8} "
                f"{point.error_capture:>8.1%} {point.errors_missed:>15}"
            )


@app.command("speaker-audit")
def speaker_audit(
    targets: list[Path] = typer.Argument(
        None, exists=True, readable=True, help="Transcript files or directories (several allowed)"
    ),
    manifest: Path | None = typer.Option(
        None, "--manifest", help="Write per-file detail here (contains filenames)"
    ),
) -> None:
    """Survey every distinct speaker label used across a transcript tree.

    Read-only and recursive. Reports label values and counts; never prints transcript text or
    filenames. Use it to check how consistent the manual coding vocabulary actually is, and how
    often a single turn is marked as covering more than one speaker (e.g. "C/T").
    """
    targets = list(targets) if targets else [Path("data")]
    paths: list[Path] = []
    for one in targets:
        for found in discover_transcripts_recursive(one):
            if found not in paths:
                paths.append(found)
    if not paths:
        listed = ", ".join(str(t) for t in targets)
        typer.echo(f"FAILED: no transcripts found under {listed}", err=True)
        raise typer.Exit(code=1)

    audit = audit_speaker_values(paths)
    typer.echo(
        f"{audit.files_scanned} file(s), {audit.total_turns} turns"
        + (f", {audit.files_failed} unreadable" if audit.files_failed else "")
    )
    typer.echo(f"\n{len(audit.turns_by_value)} distinct speaker value(s):\n")
    typer.echo(f"  {'value':<20} {'turns':>7} {'%':>6} {'files':>6} {'mean words':>11}  kind")
    for value, turns in audit.turns_by_value.most_common():
        share = turns / audit.total_turns if audit.total_turns else 0
        kind = "MERGED" if is_multi_role(value) else classify_speaker(value)
        shown = display_speaker_value(value)
        typer.echo(
            f"  {shown:<20} {turns:>7} {share:>5.1%} {audit.files_by_value[value]:>6}"
            f" {audit.mean_words(value):>11.1f}  {kind}"
        )

    if audit.multi_role_values:
        typer.echo(
            f"\nMERGED turns: {audit.multi_role_turns} turn(s) across {audit.multi_role_files} "
            f"file(s) are marked as covering more than one speaker."
        )
        typer.echo(
            "  These cannot be given a single speaker label without splitting the turn first - "
            "see docs/SPEAKER_LABELLING.md."
        )
    else:
        typer.echo("\nNo merged-speaker turns found.")

    if manifest is not None:
        import csv as _csv

        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = _csv.writer(handle)
            writer.writerow(["file", "speaker_value", "turns", "kind"])
            for path in paths:
                try:
                    from .io import load_transcript as _load

                    counts: dict[str, int] = {}
                    for turn in _load(path).turns:
                        counts[turn.speaker.strip()] = counts.get(turn.speaker.strip(), 0) + 1
                except Exception:  # noqa: BLE001
                    writer.writerow([path.name, "", "", "UNREADABLE"])
                    continue
                for value, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                    kind = "MERGED" if is_multi_role(value) else classify_speaker(value)
                    writer.writerow([path.name, value, n, kind])
        typer.echo(f"\nmanifest: {manifest}")


@app.command()
def inventory(
    input_dir: Path = typer.Argument(Path("data"), exists=True, file_okay=False, readable=True),
    threshold: float = typer.Option(
        DEFAULT_THRESHOLD,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Fraction of turns needing a C/T label for a transcript to count as labelled",
    ),
    anomaly_tolerance: float = typer.Option(
        DEFAULT_ANOMALY_TOLERANCE,
        "--anomaly-tolerance",
        min=0.0,
        max=1.0,
        help="Proportion of odd turns a file may contain before it is sent to review/",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Print why each review/ file was held back. Includes FILENAMES and label values.",
    ),
    apply: bool = typer.Option(
        False, "--apply", help="Actually move files. Without this, nothing is written."
    ),
    manifest: Path | None = typer.Option(
        None, "--manifest", help="Write a per-file CSV manifest here (contains filenames)"
    ),
) -> None:
    """Sort transcripts into labelled/ and unlabelled/ by existing C/T speaker labels.

    A deterministic metadata check — no model, no network, no transcript text read or reported.
    Prints aggregate counts only. Dry-run unless --apply is given.
    """
    destinations = {
        LABELLED: input_dir / "labelled",
        PARTIAL: input_dir / "partial",
        UNLABELLED: input_dir / "unlabelled",
        REVIEW: input_dir / "review",
    }
    records = take_inventory(input_dir)
    if not records:
        typer.echo(f"FAILED: no transcripts found directly in {input_dir}", err=True)
        raise typer.Exit(code=1)

    buckets: dict[str, list] = {LABELLED: [], PARTIAL: [], UNLABELLED: [], REVIEW: []}
    for record in records:
        buckets[record.bucket(threshold, anomaly_tolerance)].append(record)

    typer.echo(f"{len(records)} transcripts scanned in {input_dir} (threshold {threshold:.0%})")
    for name in (LABELLED, PARTIAL, UNLABELLED, REVIEW):
        group = buckets[name]
        typer.echo(f"  {name:<10} {len(group):>4}")
    errors = [r for r in records if r.error is not None]
    unknown_vocab = [
        r for r in buckets[REVIEW] if r.error is None and r.unrecognised_rate > anomaly_tolerance
    ]
    merged_only = [
        r for r in buckets[REVIEW]
        if r.error is None and r.unrecognised_rate <= anomaly_tolerance and not r.role_labelled
    ]
    if unknown_vocab:
        typer.echo(f"  ...review, unrecognised vocabulary: {len(unknown_vocab)}")
    if merged_only:
        typer.echo(f"  ...review, merged-speaker marks only: {len(merged_only)}")
    if errors:
        typer.echo(f"  ...review, unreadable:                {len(errors)}")

    if explain and buckets[REVIEW]:
        typer.echo("\nwhy each review/ file was held back:")
        for record in buckets[REVIEW]:
            typer.echo(f"  {record.path.name}: {review_reason(record, threshold, anomaly_tolerance)}")
    elif buckets[REVIEW]:
        typer.echo("\nRe-run with --explain to see why each review/ file was held back.")

    if explain:
        noted = [
            (r, tolerated_anomaly_note(r))
            for r in records
            if r.bucket(threshold, anomaly_tolerance) != REVIEW and r.anomalous
        ]
        if noted:
            typer.echo("\nodd turns tolerated in files that were filed normally:")
            for record, note in noted:
                typer.echo(f"  {record.path.name}: {note}")

    # Anomalies tolerated at file level still need handling turn by turn.
    tolerated = [r for r in records if r.error is None and r.anomalous
                 and r.bucket(threshold, anomaly_tolerance) != REVIEW]
    if tolerated:
        merged_turns = sum(r.merged for r in tolerated)
        typo_turns = sum(r.other for r in tolerated)
        typer.echo(
            f"\n{len(tolerated)} file(s) filed normally despite containing odd turns "
            f"({typo_turns} mistyped label(s), {merged_turns} merged-speaker turn(s)). "
            "These turns go to the review queue individually; the files are still processable."
        )

    # A transcript can clear the threshold and still have gaps the labeller should fill.
    gappy = [r for r in buckets[LABELLED] if r.residual_unlabelled]
    if gappy:
        total_gaps = sum(r.residual_unlabelled for r in gappy)
        typer.echo(
            f"\n{len(gappy)} file(s) in labelled/ still contain unlabelled turns "
            f"({total_gaps} turns in total). The labeller fills gaps without touching "
            "existing labels, so these can be processed too."
        )

    unique_other: set[str] = set()
    for record in records:
        unique_other |= record.other_values
    if unique_other:
        typer.echo(
            f"\n{len(unique_other)} distinct out-of-vocabulary speaker value(s) seen. "
            "Use --manifest to see which files, then fix at source."
        )

    if manifest is not None:
        import csv as _csv

        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = _csv.writer(handle)
            writer.writerow(
                ["file", "bucket", "turns", "client", "therapist", "unlabelled",
                 "other", "merged", "odd_values", "completeness", "error"]
            )
            for record in records:
                writer.writerow([
                    record.path.name,
                    record.bucket(threshold, anomaly_tolerance),
                    record.total_turns,
                    record.client,
                    record.therapist,
                    record.unlabelled,
                    record.other,
                    record.merged,
                    "|".join(sorted(record.other_values | record.merged_values)),
                    f"{record.completeness:.4f}",
                    record.error or "",
                ])
        typer.echo(f"manifest: {manifest}")

    if not apply:
        typer.echo("\ndry run - nothing moved. Re-run with --apply to move the files.")
        return

    moved = 0
    for name, group in buckets.items():
        if not group:
            continue
        destinations[name].mkdir(parents=True, exist_ok=True)
        for record in group:
            target = destinations[name] / record.path.name
            if target.exists():
                typer.echo(f"FAILED: {target} already exists; nothing further moved.", err=True)
                raise typer.Exit(code=1)
            record.path.rename(target)
            moved += 1
    folders = ", ".join(f"{d.name}/" for d in sorted(destinations.values(), key=lambda p: p.name))
    typer.echo(f"\nmoved {moved} file(s) into {folders} under {input_dir}")


if __name__ == "__main__":
    app()
