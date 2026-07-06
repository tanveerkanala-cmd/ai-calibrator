"""`calibrate` — the command-line front end over the Calibration Core.

The CLI is a thin shell: every command maps to a pipeline stage on the Core, so
the same logic later powers the local API and desktop UI. Stages not yet built
print what they'll do and which milestone delivers them.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from pydantic import ValidationError

from .models import EngineBinding, Project, TaskType
from .fmt import pct, pct_delta
from .store import atomic_write_text, load_project, project_lock, save_project, write_project_gitignore

app = typer.Typer(
    add_completion=False,
    help="Turn your knowledge and standards into a tested, reliable AI.",
)


def _scorecard_or_exit(path: Path, rid: str):
    """Load a saved scorecard, or exit friendly — a scorecard.json can be
    corrupt/truncated/hand-edited, so never let a raw traceback escape."""
    from .drift import load_scorecard
    try:
        return load_scorecard(path, rid)
    except (OSError, ValueError, ValidationError) as exc:
        typer.secho(f"Could not read scorecard {rid!r}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def _load(path: Path) -> Project:
    try:
        return load_project(path)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        # Corrupt or incomplete project.yaml (hand-edited, partially written by an
        # old version, or truncated) — show a friendly message, not a traceback.
        typer.secho(
            f"The project at {path}/ is invalid or corrupted "
            f"({Path(path) / 'project.yaml'}):",
            fg=typer.colors.RED,
        )
        typer.secho(f"  {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name (also the folder name)."),
    goal: str = typer.Option(
        ..., "--goal", "-g", help="One sentence: what should this AI do?"
    ),
    task_type: TaskType = typer.Option(
        TaskType.ASSISTANT, "--task-type", "-t", help="The kind of task."
    ),
    path: Optional[Path] = typer.Option(
        None, "--path", help="Where to create it (default: ./<name>)."
    ),
) -> None:
    """Create a new calibration project."""
    if not name or not name.strip():
        typer.secho("Project name must not be empty.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if len(name.strip()) > 120:
        typer.secho("Project name too long (max 120 characters).", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if path is None:
        p = Path(name)
        if p.is_absolute() or len(p.parts) != 1 or p.parts[0] in ("..", "."):
            typer.secho(
                "Project name must be a simple folder name (no '/', '\\', or '..'). "
                "Use --path to create the project in a specific location.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
    target = path or Path(name)
    with project_lock(target):  # atomic against a concurrent `init` of the same path
        if (target / "project.yaml").exists():
            typer.secho(f"A project already exists at {target}/", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        project = Project(name=name, goal=goal, task_type=task_type)
        save_project(project, target)
        write_project_gitignore(target)
    typer.secho(f"✓ Created project '{name}' at {target}/", fg=typer.colors.GREEN)
    typer.echo(f"  goal: {goal}")
    e = project.engines
    typer.echo(f"  engines: {e.interviewer} (reasoning) · {e.judge} (judge) · {e.subject} (subject)")
    typer.echo("\nNext:  add materials, then `calibrate ingest` (M1).")


@app.command(name="import")
def import_(
    path: Path = typer.Argument(..., help="Project to create from the prompt."),
    prompt: Path = typer.Option(..., "--prompt", "-p", help="Path to the existing system prompt (a text file)."),
    goal: str = typer.Option(..., "--goal", "-g", help="One sentence: what should this AI do?"),
    task_type: TaskType = typer.Option(TaskType.ASSISTANT, "--task-type", "-t", help="The kind of task."),
    engine: Optional[str] = typer.Option(
        None, "--engine", help="Engine for extraction + the created project (model@provider). Default: the standard binding."
    ),
) -> None:
    """Reverse-calibrate: extract a tested behavior spec from an EXISTING system prompt. (M3+)"""
    from .engines import get_engine
    from .reverse import reverse_project

    if not prompt.exists() or not prompt.is_file():
        typer.secho(f"No prompt file at {prompt}.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    try:
        prompt_text = prompt.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        typer.secho(f"Could not read {prompt}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not prompt_text.strip():
        typer.secho("The prompt file is empty.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    name = path.resolve().name or "project"
    engine_spec = engine
    try:
        eng = get_engine(engine_spec or EngineBinding().compiler)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with project_lock(path):
        if (path / "project.yaml").exists():
            typer.secho(f"A project already exists at {path}/.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.echo(f"Reverse-calibrating {prompt} with {eng.name} …")
        try:
            project = reverse_project(name, goal, prompt_text, eng,
                                      task_type=task_type, engine_spec=engine_spec, project_dir=path)
        except Exception as exc:
            typer.secho(f"Import failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    spec = project.spec
    typer.secho(
        f"✓ Imported → {path}/  ({len(spec.standards)} standard(s), {len(spec.do_not)} never-rule(s), "
        f"{len(spec.eval_criteria)} criterion(s), {len(project.tests)} test(s) extracted).",
        fg=typer.colors.GREEN,
    )
    typer.echo("  original prompt saved as imported_prompt.txt")
    typer.echo("\nNext:  calibrate eval  (score it)  ·  calibrate coverage  ·  calibrate redteam")


@app.command()
def status(
    path: Path = typer.Argument(Path("."), help="Project directory."),
) -> None:
    """Show a project's progress through the pipeline."""
    project = _load(path)
    typer.secho(f"{project.name}", bold=True)
    typer.echo(f"  goal: {project.goal}")
    typer.echo(f"  task: {project.task_type.value}")

    stages = [
        ("materials ingested", bool(project.materials)),
        ("gaps identified", bool(project.gaps)),
        ("interview answered", any(i.answer for i in project.interview)),
        ("spec compiled", project.spec is not None),
        ("tests generated", bool(project.tests)),
    ]
    typer.echo("\n  progress:")
    for label, done in stages:
        mark = typer.style("✓", fg=typer.colors.GREEN) if done else "·"
        typer.echo(f"    {mark} {label}")


@app.command()
def engines(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    role: Optional[str] = typer.Argument(None, help="Role to rebind (e.g. subject, judge). Omit to just show bindings."),
    model: Optional[str] = typer.Argument(None, help="model@provider (e.g. gpt-4o-mini@openai, gemma4:e4b@ollama)."),
    all_roles: Optional[str] = typer.Option(None, "--all", help="Point EVERY role at this model@provider."),
) -> None:
    """Show — or set — which engine powers each role. (no engine)

    Examples:
      calibrate engines                                 # show current bindings
      calibrate engines . subject gpt-4o-mini@openai    # rebind one role
      calibrate engines . --all gemma4:e4b@ollama       # rebind every role
    """
    from .engines.base import validate_engine_spec
    from .models import EngineBinding

    valid_roles = list(EngineBinding.model_fields)

    def _check_spec(spec: str) -> str:
        try:
            return validate_engine_spec(spec)
        except ValueError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)

    if all_roles is not None and (role is not None or model is not None):
        typer.secho("Use EITHER `--all <model>` OR a `<role> <model>` pair — not both.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if all_roles is not None or (role is not None and model is not None):
        with project_lock(path):
            project = _load(path)
            if all_roles is not None:
                spec = _check_spec(all_roles)
                for r in valid_roles:
                    setattr(project.engines, r, spec)
                changed = f"all roles → {spec}"
            else:
                if role not in valid_roles:
                    typer.secho(f"Unknown role {role!r}. Valid: {', '.join(valid_roles)}.", fg=typer.colors.RED)
                    raise typer.Exit(code=1)
                spec = _check_spec(model)
                setattr(project.engines, role, spec)
                changed = f"{role} → {spec}"
            save_project(project, path)
        typer.secho(f"✓ Rebound {changed}", fg=typer.colors.GREEN)
    elif role is not None or model is not None:
        typer.secho("To set a binding, give BOTH a role and a model "
                    "(e.g. `calibrate engines . subject gpt-4o-mini@openai`), or use `--all`.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)
    else:
        project = _load(path)

    typer.secho("\nengine bindings (role → model@provider):", bold=True)
    for r, spec in project.engines.model_dump().items():
        typer.echo(f"  {r:<12} {spec}")


@app.command()
def auth() -> None:
    """Show how each engine provider signs in, and what looks configured."""
    from .auth import all_status

    typer.secho("engine sign-in status:", bold=True)
    for st in all_status():
        mark = typer.style("✓", fg=typer.colors.GREEN) if st.configured else "·"
        typer.echo(f"  {mark} {st.provider:<14} {st.detail}")
    typer.echo(
        "\nClaude:  browser login (no key) →  calibrate login claude"
        "\nOpenAI:  API key only →  set OPENAI_API_KEY (no third-party ChatGPT login)"
        "\nLocal:   Ollama needs no auth"
    )


@app.command()
def login(
    provider: str = typer.Argument(..., help="Which engine to sign in to: claude | openai"),
) -> None:
    """Sign in to a cloud engine (Claude: browser/OAuth; OpenAI: key guidance)."""
    p = provider.strip().lower()
    if p in ("claude", "anthropic"):
        from .auth import login_anthropic
        try:
            code = login_anthropic()
        except RuntimeError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)
        raise typer.Exit(code=code)
    if p in ("openai", "chatgpt", "gpt"):
        typer.echo(
            "OpenAI's API is key-based — there is no supported ChatGPT account "
            "login for third-party tools.\n"
            "Create a key at https://platform.openai.com/api-keys , then:\n"
            "  export OPENAI_API_KEY=sk-..."
        )
        raise typer.Exit(code=0)
    typer.secho(
        f"Unknown provider {provider!r}. Use 'claude' or 'openai'.",
        fg=typer.colors.RED,
    )
    raise typer.Exit(code=1)


@app.command()
def ingest(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    source: Optional[Path] = typer.Option(
        None, "--source", help="Materials dir (default: <project>/materials)."
    ),
    no_index: bool = typer.Option(
        False, "--no-index", help="Skip building the retrieval (vector) index."
    ),
) -> None:
    """Parse materials, extract the gap list, and build the retrieval index. (M1)"""
    from .engines import get_engine
    from .ingest import ingest_project

    # Hold the project lock across load→mutate→save so a concurrent calibrate
    # process can't lose this run's results.
    with project_lock(path):
        project = _load(path)
        src = source or (Path(path) / "materials")
        if not src.exists() or not any(src.iterdir()):
            typer.secho(
                f"No materials found in {src}/. Add files there, then re-run.",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)

        try:
            engine = get_engine(project.engines.extractor)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)

        typer.echo(f"Ingesting {src}/ using {engine.name} …")
        try:
            result = ingest_project(
                project, src, engine, project_dir=path, build_index=not no_index
            )
        except Exception as exc:  # network / auth / parse errors → friendly message
            typer.secho(f"Ingest failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        save_project(project, path)

    typer.secho(
        f"✓ Ingested {result.materials} file(s), {result.chunks} chunk(s), "
        f"{result.facts} fact(s).",
        fg=typer.colors.GREEN,
    )
    if result.indexed is None:
        typer.echo("  retrieval index: skipped (install the `rag` extra to enable)")
    else:
        typer.echo(f"  retrieval index: {result.indexed} chunk(s) embedded")

    typer.secho(f"\n{result.gaps} gap(s) to resolve in the interview:", bold=True)
    for g in project.gaps:
        typer.echo(f"  · {g.dimension}")
    typer.echo("\nNext:  calibrate interview  (M2)")


@app.command()
def interview(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    accept_drafts: bool = typer.Option(
        False, "--accept-drafts",
        help="Accept every drafted answer without prompting (non-interactive).",
    ),
    regenerate: bool = typer.Option(
        False, "--regenerate", help="Re-generate the questions from the current gaps.",
    ),
) -> None:
    """Ask adaptive, gap-driven questions (propose-and-ratify) and store answers. (M2)"""
    from .engines import get_engine
    from .interview import generate_questions

    with project_lock(path):
        project = _load(path)
        if not project.gaps:
            typer.secho("No gaps yet — run `calibrate ingest` first.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

        if regenerate or not project.interview:
            try:
                engine = get_engine(project.engines.interviewer)
            except (RuntimeError, ValueError, NotImplementedError) as exc:
                typer.secho(str(exc), fg=typer.colors.RED)
                raise typer.Exit(code=1)
            typer.echo(f"Generating questions with {engine.name} …")
            try:
                project.interview = generate_questions(project, engine)
            except Exception as exc:
                typer.secho(f"Question generation failed: {exc}", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            save_project(project, path)
        questions = list(project.interview)  # snapshot; lock released before prompting

    pending = [it for it in questions if not it.answer]
    if not pending:
        typer.secho("All questions answered. Next:  calibrate compile  (M3)",
                    fg=typer.colors.GREEN)
        raise typer.Exit(code=0)

    # Gather answers WITHOUT holding the lock — the interactive prompts can take
    # minutes, and holding the project lock across them would block every other
    # command on this project. Collect by question id, then apply atomically.
    mode = "auto-accepting drafts" if accept_drafts else "Enter = accept the draft"
    typer.secho(f"{len(pending)} question(s) to answer ({mode}):\n", bold=True)
    answers: dict[str, str] = {}
    for item in pending:
        if accept_drafts:
            answers[item.id] = item.draft_answer or ""
        else:
            typer.secho(f"[{item.dimension}] {item.question}", bold=True)
            if item.rationale:
                typer.echo(f"  why: {item.rationale}")
            typer.echo(f"  draft: {item.draft_answer}")
            resp = typer.prompt("  your answer (Enter to accept draft)",
                                default="", show_default=False)
            answers[item.id] = resp.strip() or (item.draft_answer or "")
            typer.echo("")

    # Apply under the lock against a FRESH load, so a concurrent edit isn't
    # clobbered by our stale in-memory copy (store.py load→mutate→save contract).
    with project_lock(path):
        project = _load(path)
        by_id = {it.id: it for it in project.interview}
        for qid, ans in answers.items():
            if qid in by_id:
                by_id[qid].answer = ans
        save_project(project, path)
        answered = sum(1 for it in project.interview if it.answer)
        total = len(project.interview)

    typer.secho(f"✓ {answered}/{total} answered.", fg=typer.colors.GREEN)
    typer.echo("Next:  calibrate compile  (M3)")


@app.command()
def compile(path: Path = typer.Argument(Path("."), help="Project directory.")) -> None:
    """Synthesize the behavior spec + system prompt + RAG + rubric + tests. (M3)"""
    from .compile import compile_project
    from .engines import get_engine

    with project_lock(path):
        project = _load(path)
        if not any(it.answer for it in project.interview):
            typer.secho(
                "No interview answers yet — run `calibrate interview` first.",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=1)

        try:
            engine = get_engine(project.engines.compiler)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)

        typer.echo(f"Compiling the spec + artifacts with {engine.name} …")
        try:
            result = compile_project(project, engine, project_dir=path)
        except Exception as exc:
            typer.secho(f"Compile failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        save_project(project, path)

    typer.secho(
        f"✓ Spec compiled: {result.standards} standard(s), {result.edge_cases} "
        f"edge case(s), {result.criteria} eval criterion(s), {result.tests} test(s).",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"  bundle → {result.build_dir}/")
    for f in result.files:
        typer.echo(f"    {f}")
    typer.echo("\nNext:  calibrate eval  (M4)")


@app.command(name="eval")
def eval_(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    refine: bool = typer.Option(
        False, "--refine", help="Loop: diagnose failures, refine the spec, re-run.",
    ),
    rounds: int = typer.Option(3, "--rounds", help="Max refine rounds."),
    threshold: float = typer.Option(0.8, "--threshold", help="Target pass rate (0-1)."),
    judge_passes: int = typer.Option(
        1, "--judge-passes", help="Grade each criterion N times and majority-vote (self-consistency)."
    ),
) -> None:
    """Run tests, grade against the rubric, score, and (optionally) refine. (M4)"""
    from .engine_log import wrap_engine
    from .engines import get_engine
    from .eval import low_confidence_results, next_run_id, run_eval, save_scorecard

    if not (1 <= rounds <= 100):  # bounds match the API (EvalBody), always validated
        typer.secho("--rounds must be between 1 and 100.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        typer.secho("--threshold must be a number between 0 and 1.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not (1 <= judge_passes <= 9):
        typer.secho("--judge-passes must be between 1 and 9.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with project_lock(path):
        project = _load(path)
        if project.spec is None or not project.tests:
            typer.secho("Nothing to evaluate — run `calibrate compile` first.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

        try:
            log_on = project.log_interactions
            subject = get_engine(project.engines.subject)
            judge = wrap_engine(get_engine(project.engines.judge), "judge", path, enabled=log_on)
            refiner = (wrap_engine(get_engine(project.engines.compiler), "compiler", path, enabled=log_on)
                       if refine else None)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)

        typer.echo(
            f"Evaluating {len(project.tests)} test(s): subject={subject.name}, judge={judge.name}"
            + (f", refiner={refiner.name}" if refiner else "") + " …"
        )
        try:
            if refine:
                from .compile import write_build_bundle
                from .pipeline import calibrate_loop
                cards = calibrate_loop(
                    project, subject, judge, refiner,
                    threshold=threshold, max_rounds=rounds, judge_passes=judge_passes, project_dir=path,
                )
                save_project(project, path)  # refined standards persist
                write_build_bundle(project.spec, project.tests, path)  # refresh build/ to match
            else:
                card = run_eval(project, subject, judge, run_id=next_run_id(path), judge_passes=judge_passes)
                save_scorecard(path, card)
                cards = [card]
        except Exception as exc:
            typer.secho(f"Eval failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    for i, card in enumerate(cards, 1):
        graded = [r for r in card.results if r.criteria]
        passed = sum(1 for r in graded if r.passed)
        typer.echo(f"  round {i} [{card.run_id}]: {pct(card.pass_rate)} ({passed}/{len(graded)} graded)")

    final = cards[-1]
    ok = final.pass_rate >= threshold
    typer.secho(
        f"\nFinal pass rate: {pct(final.pass_rate)}   (weighted score: {pct(final.weighted_score)})",
        fg=typer.colors.GREEN if ok else typer.colors.YELLOW,
    )
    # Triage order: tests whose HIGH-weight criteria failed come first.
    from .models import Weight

    def _worst(r):  # highest weight among this test's failed criteria
        return max(((c.weight or Weight.MEDIUM).numeric for c in r.criteria if not c.passed), default=0)

    for r in sorted([r for r in final.results if not r.passed], key=_worst, reverse=True)[:10]:
        why = "; ".join(
            f"[{(c.weight or Weight.MEDIUM).value}] " + (c.rationale or c.criterion_id)
            for c in r.criteria if not c.passed
        ) or "no criteria"
        typer.echo(f"  · {r.test_id}: {why}")

    if judge_passes > 1:
        low = low_confidence_results(final)
        if low:
            typer.secho(f"\n⚠ {len(low)} verdict(s) the judge was split on — worth a human check:",
                        fg=typer.colors.YELLOW)
            for tid, c in low[:10]:
                typer.echo(f"  · {tid} / {c.criterion_id}: {pct(c.confidence)} agreement → {'pass' if c.passed else 'fail'}")

    typer.echo(f"\nScorecards saved under {Path(path)}/evals/.")
    if ok:
        typer.secho("Threshold met. Next:  calibrate export  (M5)", fg=typer.colors.GREEN)
    elif not refine:
        typer.echo("Below threshold — try:  calibrate eval --refine")


@app.command()
def ci(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    threshold: float = typer.Option(0.8, "--threshold", help="Min pass rate for the eval stage (0-1)."),
    tolerance: float = typer.Option(0.0, "--tolerance", help="Allowed pass-rate drop before drift fails."),
    judge_passes: int = typer.Option(1, "--judge-passes", help="Judge self-consistency passes (1-9)."),
    baseline: Optional[str] = typer.Option(None, "--baseline", help="Run id to drift against (default: previous run)."),
    as_json: bool = typer.Option(False, "--json", help="Print a machine-readable JSON result."),
) -> None:
    """The whole gate in one command: lint -> eval -> drift -> snapshot.

    Exit codes: 0 = gate passed, 1 = couldn't gate (spec/engine problems), 2 = the AI failed the gate.
    """
    import json as _json

    from .ci import ci_dict, run_ci
    from .engine_log import wrap_engine
    from .engines import get_engine

    if not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        typer.secho("--threshold must be a number between 0 and 1.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not math.isfinite(tolerance) or tolerance < 0:
        typer.secho("--tolerance must be a number >= 0.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not (1 <= judge_passes <= 9):
        typer.secho("--judge-passes must be between 1 and 9.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with project_lock(path):
        project = _load(path)
        if project.spec is None or not project.tests:
            typer.secho("Nothing to gate — run `calibrate compile` (or `import`) first.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        # Factories: engines are acquired only if lint passes — a lint-broken spec
        # shouldn't demand credentials, and an engine problem shouldn't mask lint.
        log_on = project.log_interactions
        subject = lambda: get_engine(project.engines.subject)  # noqa: E731
        judge = lambda: wrap_engine(get_engine(project.engines.judge), "judge", path, enabled=log_on)  # noqa: E731
        try:
            result = run_ci(project, subject, judge, project_dir=path, threshold=threshold,
                            tolerance=tolerance, judge_passes=judge_passes, baseline=baseline)
        except Exception as exc:
            typer.secho(f"CI gate could not run: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    if as_json:
        typer.echo(_json.dumps(ci_dict(result), indent=2))
    else:
        marks = {"pass": ("✓", typer.colors.GREEN), "fail": ("✗", typer.colors.RED),
                 "skip": ("-", typer.colors.YELLOW)}
        for s in result.stages:
            mark, color = marks[s.status]
            typer.secho(f" {mark} {s.name:<9}{s.detail}", fg=color)
        typer.secho(f"\nCI gate: {'PASS' if result.ok else 'FAIL'}",
                    fg=typer.colors.GREEN if result.ok else typer.colors.RED, bold=True)

    if not result.ok:
        lint_failed = any(s.name == "lint" and s.status == "fail" for s in result.stages)
        raise typer.Exit(code=1 if lint_failed else 2)


@app.command()
def run(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (localhost by default)."),
    port: int = typer.Option(8600, "--port", help="Port."),
    guard: bool = typer.Option(False, "--guard", help="Re-check every live answer against the spec's deterministic checks."),
    force: bool = typer.Option(False, "--force", help="Serve even if the last `ci` gate FAILED."),
) -> None:
    """Serve the calibrated AI itself — an OpenAI-compatible endpoint that won't boot on a red gate.

    Point any OpenAI-protocol client at http://HOST:PORT/v1 (model name = project name).
    """
    from .ci import certification_status

    project = _load(path)
    if project.spec is None:
        typer.secho("Nothing to serve — run `calibrate compile` (or `import`) first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    # The boot gate: an AI that can't prove it follows its rules shouldn't serve.
    status, detail = certification_status(project, path)
    if status == "pass":
        typer.secho(f"✓ Certified: {detail}", fg=typer.colors.GREEN)
    elif status == "fail" and not force:
        typer.secho(f"✗ REFUSING TO SERVE — {detail}", fg=typer.colors.RED, bold=True)
        typer.echo("  (--force to serve anyway, clearly at your own risk)")
        raise typer.Exit(code=2)
    else:
        color = typer.colors.RED if status == "fail" else typer.colors.YELLOW
        typer.secho(f"⚠ UNCERTIFIED ({status}): {detail}", fg=color, bold=True)

    try:
        from .runtime import create_ai_app
        application = create_ai_app(path, guard=guard)
        import uvicorn
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except ImportError:
        typer.secho("Serving needs the `api` extra:  pip install -e '.[api]'", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if host not in ("127.0.0.1", "localhost", "::1"):
        typer.secho(f"⚠  Binding to {host} exposes the (unauthenticated) AI beyond localhost.",
                    fg=typer.colors.YELLOW)
    typer.echo(f"Serving '{project.name}' (subject: {project.engines.subject}"
               + (", guard ON" if guard else "") + f") at http://{host}:{port}/v1")
    import json as _json
    import shlex
    payload = _json.dumps({"model": project.name,
                           "messages": [{"role": "user", "content": "hello"}]})
    typer.echo(f'  try:  curl -s http://{host}:{port}/v1/chat/completions '
               f'-H "Content-Type: application/json" -d {shlex.quote(payload)}')
    uvicorn.run(application, host=host, port=port, log_level="warning")


@app.command()
def absorb(path: Path = typer.Argument(Path("."), help="Project directory.")) -> None:
    """Close the flywheel: fold live feedback (from `calibrate run`) into examples + pinned tests. (no engine)"""
    from .compile import write_build_bundle
    from .flywheel import absorb_feedback

    with project_lock(path):
        project = _load(path)
        result = absorb_feedback(project, path)
        if result.ups + result.downs + result.skipped == 0:
            typer.secho("No live feedback to absorb yet.", fg=typer.colors.YELLOW)
            typer.echo("  `calibrate run` records it: POST /v1/feedback "
                       '{"completion_id": "...", "verdict": "down", "correction": "..."}')
            raise typer.Exit(code=0)
        save_project(project, path)
        if project.spec is not None and project.tests:
            write_build_bundle(project.spec, project.tests, path)

    typer.secho(f"✓ Absorbed {result.ups + result.downs} feedback record(s): "
                f"{result.ups} up / {result.downs} down.", fg=typer.colors.GREEN)
    typer.echo(f"  examples added: {result.examples_added}   pinned tests added: {result.tests_added}"
               + (f" ({', '.join(result.test_ids)})" if result.test_ids else "")
               + (f"   skipped: {result.skipped}" if result.skipped else ""))
    if result.tests_added or result.examples_added:
        typer.secho("The AI just learned from real use — its certification is now stale.",
                    fg=typer.colors.YELLOW)
        typer.echo("Run `calibrate ci` to re-certify against the suite that now includes it.")


@app.command(name="add-check")
def add_check(
    path: Path = typer.Argument(..., help="Project directory."),
    criterion: str = typer.Argument(..., help="Eval-criterion id to attach the check to."),
    kind: str = typer.Argument(..., help="contains | not_contains | regex | max_chars | min_chars | non_empty"),
    value: str = typer.Argument("", help="The term / pattern / number (unused for non_empty)."),
) -> None:
    """Attach a deterministic check to a criterion — graded exactly by code, not the judge. (no engine)"""
    from pydantic import ValidationError

    from .compile import write_build_bundle
    from .models import Check

    with project_lock(path):
        project = _load(path)
        if project.spec is None:
            typer.secho("Nothing to check — run `calibrate compile` (or `import`) first.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        crit = next((c for c in project.spec.eval_criteria if c.id == criterion), None)
        if crit is None:
            ids = ", ".join(c.id for c in project.spec.eval_criteria) or "(none)"
            typer.secho(f"No criterion {criterion!r}. Ids: {ids}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        try:
            crit.check = Check(kind=kind, value=value)
        except ValidationError:
            typer.secho("kind must be one of: contains, not_contains, regex, max_chars, min_chars, non_empty.",
                        fg=typer.colors.RED)
            raise typer.Exit(code=1)
        save_project(project, path)
        if project.tests:
            write_build_bundle(project.spec, project.tests, path)
    typer.secho(f"✓ Criterion {criterion!r} is now graded deterministically: {kind} {value!r}.", fg=typer.colors.GREEN)
    typer.echo("  it will be checked exactly (no judge) on the next `calibrate eval`.")


@app.command(name="judge-check")
def judge_check(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    sample: int = typer.Option(10, "--sample", help="How many of the judge's verdicts to review."),
) -> None:
    """Calibrate the judge: confirm a sample of its verdicts and measure its agreement with you. (no engine)"""
    from .drift import load_scorecard
    from .eval import latest_run_id
    from .judge_check import gradings, judge_agreement, save_labels

    if sample < 1:
        typer.secho("--sample must be >= 1.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    _load(path)
    rid = latest_run_id(path)
    if not rid:
        typer.secho("No scorecard yet — run `calibrate eval` first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    card = load_scorecard(path, rid)
    items = gradings(card)
    if not items:
        typer.secho("No graded verdicts in the latest scorecard.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    items = items[:sample]

    typer.echo(f"Reviewing {len(items)} of the judge's verdicts from {rid} — confirm or correct each:")
    labels = []
    for it in items:
        typer.secho(f"\n[{it['criterion_id']}]  judge said {'PASS' if it['judge_passed'] else 'FAIL'}", bold=True)
        if it["rationale"]:
            typer.echo(f"  judge's reason: {it['rationale']}")
        typer.echo(f"  output: {it['output'][:200]}")
        agree = typer.confirm("  Do you agree with the judge?", default=True)
        human_passed = it["judge_passed"] if agree else (not it["judge_passed"])
        labels.append({"test_id": it["test_id"], "criterion_id": it["criterion_id"], "passed": human_passed})

    save_labels(path, rid, labels)  # ground truth is an asset: feeds train-engine
    ag = judge_agreement(card, labels)
    rate = ag.agreement_rate
    typer.secho(f"\nJudge agreement with you: {pct(rate)} ({ag.agreed}/{ag.total})",
                fg=typer.colors.GREEN if rate >= 0.8 else typer.colors.YELLOW, bold=True)
    for cid in ag.unreliable_criteria():
        a, t = ag.by_criterion[cid]
        typer.secho(f"  ⚠ {cid}: judge agreed only {a}/{t} — make this criterion more objective.",
                    fg=typer.colors.YELLOW)
    if rate < 0.8:
        typer.echo("\nTrust this scorecard less; reword the flagged criteria, or grade with `eval --judge-passes 3`.")
    else:
        typer.secho("\nThe judge tracks your judgment well — the scorecard is trustworthy.", fg=typer.colors.GREEN)
    typer.echo(f"Labels saved to evals/{rid}/human-labels.json — `calibrate train-engine judge` "
               "uses them as ground truth.")


@app.command()
def lint(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    deep: bool = typer.Option(False, "--deep", help="Also detect self-contradictions (uses an engine)."),
) -> None:
    """Lint the behavior spec for quality issues before you eval. Exits 1 on errors. (no engine unless --deep)"""
    from .lint import lint_spec, lint_unknown_fields

    project = _load(path)
    if project.spec is None:
        typer.secho("Nothing to lint — run `calibrate compile` (or `import`) first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    report = lint_spec(project.spec, project.tests)
    report.issues.extend(lint_unknown_fields(project))
    if deep:
        from .engines import get_engine
        from .lint import lint_contradictions
        try:
            eng = get_engine(project.engines.compiler)
            typer.echo("Checking for self-contradictions …")
            report.issues.extend(lint_contradictions(project.spec, eng))
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)
        except Exception as exc:  # contradiction check is best-effort; don't fail the lint
            typer.secho(f"  (contradiction check skipped: {exc})", fg=typer.colors.YELLOW)

    colors = {"error": typer.colors.RED, "warn": typer.colors.YELLOW, "info": typer.colors.BLUE}
    for i in report.issues:
        typer.secho(f"  [{i.severity:<5}] {i.code}: {i.message}", fg=colors.get(i.severity))
    if not report.issues:
        typer.secho("✓ No lint issues — the spec looks well-formed.", fg=typer.colors.GREEN)
    else:
        n_info = len(report.issues) - len(report.errors) - len(report.warnings)
        typer.secho(f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s), {n_info} info.",
                    fg=typer.colors.RED if report.errors else typer.colors.YELLOW)
    raise typer.Exit(code=1 if report.errors else 0)


@app.command()
def coverage(path: Path = typer.Argument(Path("."), help="Project directory.")) -> None:
    """Behavioral coverage: which spec criteria have targeted tests. (no engine, instant)"""
    from .coverage import analyze_coverage

    project = _load(path)
    if project.spec is None:
        typer.secho("Nothing to analyze — run `calibrate compile` first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    report = analyze_coverage(project.spec, project.tests)
    typer.secho(
        f"Behavioral coverage: {pct(report.coverage_rate)} "
        f"({len(report.covered_criteria)}/{report.total_criteria} criteria targeted)",
        bold=True,
    )
    for c in report.criteria:
        mark = typer.style("✓", fg=typer.colors.GREEN) if c.covered else typer.style("·", fg=typer.colors.YELLOW)
        targeted = ", ".join(c.targeted_by) if c.targeted_by else "— no targeted test"
        typer.echo(f"  {mark} [{c.weight:<6}] {c.id}: {targeted}")
    if report.broad_tests:
        typer.echo(f"\n  broad grade-all tests (weak coverage): {', '.join(report.broad_tests)}")
    for w in report.warnings:
        typer.secho(f"  ⚠ {w}", fg=typer.colors.YELLOW)
    if not report.warnings and report.coverage_rate == 1.0:
        typer.secho("\n✓ Every criterion has a targeted test.", fg=typer.colors.GREEN)


@app.command()
def redteam(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    max_probes: int = typer.Option(12, "--max-probes", help="Max adversarial probes to generate."),
    add_tests: bool = typer.Option(
        False, "--add-tests", help="Promote confirmed violations into the test suite as regressions.",
    ),
) -> None:
    """Adversarially probe the configured AI to break its own rules. (M4+)"""
    from .compile import write_build_bundle
    from .engines import get_engine
    from .redteam import promote_to_tests, run_redteam

    if not (1 <= max_probes <= 50):
        typer.secho("--max-probes must be between 1 and 50.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with project_lock(path):
        project = _load(path)
        if project.spec is None:
            typer.secho("Nothing to red-team — run `calibrate compile` first.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        try:
            generator = get_engine(project.engines.compiler)
            subject = get_engine(project.engines.subject)
            judge = get_engine(project.engines.judge)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)

        typer.echo(f"Red-teaming: subject={subject.name}, judge={judge.name} (≤{max_probes} probes) …")
        try:
            report = run_redteam(project, generator, subject, judge, project_dir=path, max_probes=max_probes)
        except Exception as exc:
            typer.secho(f"Red-team failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        added = 0
        if add_tests and report.violations:
            added = promote_to_tests(project, report)
            save_project(project, path)
            write_build_bundle(project.spec, project.tests, path)

    color = typer.colors.GREEN if not report.violations else typer.colors.RED
    typer.secho(
        f"\nHeld {pct(report.hold_rate)} — {len(report.violations)}/{report.probes} probe(s) caused a violation.",
        fg=color, bold=True,
    )
    for r in report.violations:
        typer.secho(f"  ✗ [{r.severity}] {r.target}", fg=typer.colors.RED)
        typer.echo(f"     probe ({r.tactic}): {r.input[:100]}")
        if r.rationale:
            typer.echo(f"     why: {r.rationale}")
    if add_tests and added:
        typer.secho(f"\n+ Added {added} regression test(s). Run `calibrate eval --refine` to fix them.",
                    fg=typer.colors.GREEN)
    typer.echo(f"\nSaved → {Path(path)}/evals/{report.run_id}/redteam.json")


@app.command()
def rightsize(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    models: Optional[str] = typer.Option(
        None, "--models", help="Comma-separated model@provider candidates (default: the Claude tier ladder)."
    ),
    threshold: float = typer.Option(0.8, "--threshold", help="Pass-rate bar (0-1)."),
) -> None:
    """Find the cheapest model that still meets your pass bar — runs your tests across models. (M4+)"""
    from .engines import get_engine
    from .rightsize import DEFAULT_LADDER
    from .rightsize import rightsize as run_rightsize

    if not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        typer.secho("--threshold must be a number between 0 and 1.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    project = _load(path)
    if project.spec is None or not project.tests:
        typer.secho("Nothing to rightsize — run `calibrate compile` first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    specs = [s.strip() for s in models.split(",") if s.strip()] if models else list(DEFAULT_LADDER)
    try:
        judge = get_engine(project.engines.judge)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Rightsizing across {len(specs)} model(s), judge={judge.name} (this runs your tests N× — may take a while) …")
    report = run_rightsize(project, specs, judge, get_engine, threshold=threshold, project_dir=path)

    typer.secho(f"\n  {'model':<28}{'pass':>6}  {'$ in/out':>10}  note", bold=True)
    for r in report.results:
        if r.error:
            typer.secho(f"  {r.spec:<28}{'—':>6}  {'—':>10}  error: {r.error[:48]}", fg=typer.colors.YELLOW)
            continue
        price = f"{r.in_price}/{r.out_price}" if r.in_price is not None else "unknown"
        meets = r.pass_rate >= threshold
        note = "✓ meets bar" if meets else "below bar"
        typer.secho(f"  {r.spec:<28}{pct(r.pass_rate):>5}  {price:>10}  {note}",
                    fg=typer.colors.GREEN if meets else None)

    rec = report.recommended
    if rec:
        typer.secho(f"\n→ Recommended: {rec.spec}  ({pct(rec.pass_rate)}, cheapest that meets {threshold:.0%})",
                    fg=typer.colors.GREEN)
        typer.echo(f"  to adopt: point engines.subject at {rec.spec} in project.yaml")
    else:
        typer.secho(f"\nNo candidate met the {threshold:.0%} bar — try `calibrate eval --refine` or lower --threshold.",
                    fg=typer.colors.YELLOW)
    typer.echo(f"\nSaved → {Path(path)}/evals/rightsize.json")


@app.command()
def diff(
    before: Path = typer.Argument(..., help="Baseline project."),
    after: Path = typer.Argument(..., help="Project to compare against the baseline."),
) -> None:
    """Show how the behavior spec changed between two projects. (no engine)"""
    from .specdiff import diff_specs

    pa, pb = _load(before), _load(after)
    if pa.spec is None or pb.spec is None:
        typer.secho("Both projects need a compiled spec (run `calibrate compile` or `import`).", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    d = diff_specs(pa.spec, pb.spec)
    if not d.changed:
        typer.secho("No behavior change between the two specs.", fg=typer.colors.GREEN)
        return

    def _section(title: str, added: list[str], removed: list[str]) -> None:
        if not added and not removed:
            return
        typer.secho(f"\n{title}:", bold=True)
        for x in added:
            typer.secho(f"  + {x}", fg=typer.colors.GREEN)
        for x in removed:
            typer.secho(f"  - {x}", fg=typer.colors.RED)

    _section("Standards", d.standards_added, d.standards_removed)
    _section("Never-rules", d.do_not_added, d.do_not_removed)
    _section("Edge cases", d.edge_cases_added, d.edge_cases_removed)
    if d.criteria_added or d.criteria_removed or d.criteria_changed:
        typer.secho("\nCriteria:", bold=True)
        for x in d.criteria_added:
            typer.secho(f"  + {x}", fg=typer.colors.GREEN)
        for x in d.criteria_removed:
            typer.secho(f"  - {x}", fg=typer.colors.RED)
        for x in d.criteria_changed:
            typer.secho(f"  ~ {x} (description/weight changed)", fg=typer.colors.YELLOW)


@app.command()
def drift(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    baseline: Optional[str] = typer.Option(
        None, "--baseline", help="Baseline run id (default: the latest saved scorecard)."
    ),
    tolerance: float = typer.Option(
        0.0, "--tolerance", help="Allowed pass-rate drop before flagging drift (0-1)."
    ),
) -> None:
    """Re-run the suite and flag behavior drift vs a baseline. Exits 2 on regression (CI-friendly). (M4+)"""
    from .drift import load_scorecard, run_drift
    from .engines import get_engine
    from .eval import latest_run_id

    if not math.isfinite(tolerance) or not (0.0 <= tolerance <= 1.0):
        typer.secho("--tolerance must be a number between 0 and 1.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with project_lock(path):
        project = _load(path)
        if project.spec is None or not project.tests:
            typer.secho("Nothing to check — run `calibrate compile` first.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        base_id = baseline or latest_run_id(path)
        if not base_id:
            typer.secho("No baseline scorecard yet — run `calibrate eval` first to set one.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        try:
            base_card = load_scorecard(path, base_id)
        except FileNotFoundError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)
        try:
            subject = get_engine(project.engines.subject)
            judge = get_engine(project.engines.judge)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.echo(f"Drift check vs baseline {base_id}: subject={subject.name}, judge={judge.name} …")
        try:
            report, _ = run_drift(project, subject, judge, baseline=base_card, project_dir=path, tolerance=tolerance)
        except Exception as exc:
            typer.secho(f"Drift check failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    typer.secho(
        f"\nbaseline {report.baseline_run}: {pct(report.baseline_rate)}  →  "
        f"{report.candidate_run}: {pct(report.candidate_rate)}  (Δ {pct_delta(report.delta)})",
        bold=True,
    )
    if report.regressed_tests:
        typer.secho(f"  ✗ {len(report.regressed_tests)} regressed (pass→fail): "
                    f"{', '.join(report.regressed_tests[:10])}", fg=typer.colors.RED)
    if report.fixed_tests:
        typer.secho(f"  ✓ {len(report.fixed_tests)} improved (fail→pass): "
                    f"{', '.join(report.fixed_tests[:10])}", fg=typer.colors.GREEN)
    if report.regressed:
        typer.secho("\n⚠ DRIFT DETECTED — behavior regressed beyond tolerance.", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    typer.secho("\n✓ No drift — behavior held within tolerance.", fg=typer.colors.GREEN)


@app.command()
def snapshot(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    check: bool = typer.Option(False, "--check", help="Compare the latest outputs to the pinned golden; exit 2 if changed."),
) -> None:
    """Pin or check golden outputs — catch output changes the pass/fail rubric misses. (no engine)"""
    from .eval import latest_run_id
    from .snapshot import compare, load_golden, outputs_of, save_golden

    _load(path)  # validate project
    rid = latest_run_id(path)
    if not rid:
        typer.secho("No scorecard yet — run `calibrate eval` first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    latest = outputs_of(_scorecard_or_exit(path, rid))

    if not check:
        save_golden(path, latest)
        typer.secho(f"✓ Pinned {len(latest)} golden output(s) from {rid} → golden.json.", fg=typer.colors.GREEN)
        return

    golden = load_golden(path)
    if golden is None:
        typer.secho("No golden yet — run `calibrate snapshot` (without --check) to pin one.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    d = compare(golden, latest)
    if not (d.changed or d.added or d.removed):
        typer.secho("✓ Outputs match the golden.", fg=typer.colors.GREEN)
        return
    for t in d.changed:
        typer.secho(f"  ~ {t}: output changed", fg=typer.colors.YELLOW)
    for t in d.removed:
        typer.secho(f"  - {t}: test missing from the latest run", fg=typer.colors.RED)
    for t in d.added:
        typer.echo(f"  + {t}: new test (not in golden)")
    if d.drifted:
        typer.secho(f"\n⚠ {len(d.changed)} output(s) changed vs golden.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=2)


@app.command()
def report(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    html: bool = typer.Option(False, "--html", help="Also write a shareable single-file HTML certificate."),
    badge: bool = typer.Option(False, "--badge", help="Also write badge.json (shields.io endpoint format)."),
) -> None:
    """Generate a shareable calibration report (the AI's 'nutrition label'). (no engine)"""
    from .coverage import analyze_coverage
    from .drift import load_scorecard
    from .eval import latest_run_id
    from .report import calibration_confidence, render_report, save_report

    project = _load(path)
    if project.spec is None:
        typer.secho("Nothing to report — run `calibrate compile` first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    cov = analyze_coverage(project.spec, project.tests)
    latest = None
    rid = latest_run_id(path)
    if rid:
        try:
            latest = load_scorecard(path, rid)
        except (FileNotFoundError, ValueError):
            latest = None

    markdown = render_report(project, cov, latest)
    out = save_report(path, markdown)
    conf = calibration_confidence(cov.coverage_rate, latest.pass_rate if latest else 0.0, latest is not None)
    typer.secho(f"Calibration Confidence: {pct(conf)}", bold=True)
    typer.echo(f"  coverage {pct(cov.coverage_rate)}"
               + (f" × pass rate {pct(latest.pass_rate)}" if latest else "  (no eval yet — run `calibrate eval`)"))
    typer.secho(f"✓ Report → {out}", fg=typer.colors.GREEN)
    if html:
        from .report import render_html_report, save_html_report
        out_html = save_html_report(path, render_html_report(project, cov, latest, path))
        typer.secho(f"✓ Certificate → {out_html}", fg=typer.colors.GREEN)
    if badge:
        from .report import badge_dict, save_badge
        b = badge_dict(project, path)
        out_badge = save_badge(path, b)
        typer.secho(f"✓ Badge → {out_badge}   [{b['label']}: {b['message']} · {b['color']}]", fg=typer.colors.GREEN)
        typer.echo("  README embed: https://img.shields.io/endpoint?url=<public URL of badge.json>")


@app.command()
def teach(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    n: int = typer.Option(5, "--n", help="How many sample outputs to judge."),
) -> None:
    """Calibrate by example: approve/reject sample outputs; the tool infers your standards. (M3+)"""
    from .compile import write_build_bundle
    from .engines import get_engine
    from .teach import Judged, apply_learned, infer_standards, propose_candidates

    if not (1 <= n <= 20):
        typer.secho("--n must be between 1 and 20.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with project_lock(path):
        project = _load(path)
        try:
            generator = get_engine(project.engines.compiler)
            subject = get_engine(project.engines.subject)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)

        typer.echo(f"Generating {n} sample output(s) to judge (subject={subject.name}) …")
        try:
            candidates = propose_candidates(project, generator, subject, n=n)
        except Exception as exc:
            typer.secho(f"Could not generate candidates: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if not candidates:
            typer.secho("No candidates to judge.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

        judged: list[Judged] = []
        for c in candidates:
            typer.secho(f"\nINPUT:  {c.input}", bold=True)
            typer.echo(f"OUTPUT: {c.output}")
            approved = typer.confirm("  Approve this output?", default=True)
            reason = typer.prompt("  Why? (optional, Enter to skip)", default="", show_default=False).strip() or None
            judged.append(Judged(input=c.input, output=c.output, approved=approved, reason=reason))

        typer.echo("\nInferring your standards from these judgments …")
        try:
            learned = infer_standards(project.goal, judged, generator)
        except Exception as exc:
            typer.secho(f"Inference failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        result = apply_learned(project, judged, learned)
        save_project(project, path)
        if project.tests:  # refresh the build bundle if one exists
            write_build_bundle(project.spec, project.tests, path)

    typer.secho(
        f"\n✓ Learned {result.standards_added} standard(s) + {result.do_not_added} never-rule(s) "
        f"from {len(judged)} judgment(s); recorded {result.examples_recorded} example(s).",
        fg=typer.colors.GREEN,
    )
    for s in result.standards:
        typer.echo(f"  + standard: {s}")
    for s in result.do_not:
        typer.echo(f"  + never:    {s}")
    typer.echo("\nNext:  calibrate compile  (regenerate tests/rubric)  or  calibrate eval")


@app.command()
def merge(
    out: Path = typer.Argument(..., help="Path of the merged project to create."),
    sources: list[Path] = typer.Option([], "--from", help="A stakeholder's project dir (repeat for each)."),
    goal: Optional[str] = typer.Option(None, "--goal", help="Goal for the merged AI (default: the first source's)."),
    report_only: bool = typer.Option(False, "--report-only", help="Detect + print conflicts; don't create the merged project."),
) -> None:
    """Merge multiple stakeholders' calibrated projects into one, reconciling conflicts. (org use)"""
    import yaml as _yaml

    from .engines import get_engine
    from .stakeholders import build_merged_spec, conflict_dict, detect_conflicts, gather

    if len(sources) < 2:
        typer.secho("Need at least two --from projects to merge.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    named: dict = {}
    first = None
    for src in sources:
        proj = _load(src)
        if proj.spec is None:
            typer.secho(f"{src}/ has no spec — run `calibrate compile` there first.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if proj.name in named:
            typer.secho(f"Duplicate stakeholder name {proj.name!r} across sources — rename one project.",
                        fg=typer.colors.RED)
            raise typer.Exit(code=1)
        named[proj.name] = proj.spec
        first = first or proj

    try:
        engine = get_engine(first.engines.compiler)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    statements = gather(named)
    typer.echo(f"Analyzing {len(statements)} rule(s) from {len(named)} stakeholder(s) for conflicts …")
    try:
        conflicts = detect_conflicts(statements, engine)
    except Exception as exc:
        typer.secho(f"Conflict detection failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    drops: set[int] = set()
    additions: list[str] = []
    audit: list[dict] = []
    if conflicts:
        typer.secho(f"\n{len(conflicts)} conflict(s) found:", bold=True)
    else:
        typer.secho("\nNo conflicts found — merging cleanly.", fg=typer.colors.GREEN)

    for c in conflicts:
        typer.secho(f"\n[{c.id}] ({c.severity})", fg=typer.colors.RED, bold=True)
        typer.echo(f"  A [{c.a.stakeholder}]: {c.a.text}")
        typer.echo(f"  B [{c.b.stakeholder}]: {c.b.text}")
        typer.echo(f"  why: {c.explanation}")
        if report_only:
            continue
        choice = typer.prompt("  keep (a)/(b) or (m)erge", default="a").strip().lower()[:1]
        if choice == "b":
            drops.add(c.a.idx)
            ruling = f"keep B [{c.b.stakeholder}]"
        elif choice == "m":
            merged_text = typer.prompt("  merged rule").strip()
            if merged_text:
                drops.update({c.a.idx, c.b.idx})
                additions.append(merged_text)
                ruling = f"merge → {merged_text}"
            else:
                # empty merged rule → not a merge; keep A rather than record a
                # phantom "merge → " ruling that added nothing.
                drops.add(c.b.idx)
                ruling = f"keep A [{c.a.stakeholder}] (empty merge)"
        else:
            drops.add(c.b.idx)
            ruling = f"keep A [{c.a.stakeholder}]"
        rationale = typer.prompt("  rationale (optional)", default="", show_default=False).strip() or None
        audit.append({"conflict": conflict_dict(c), "ruling": ruling, "rationale": rationale})

    if report_only:
        typer.echo("\n(report only — no project created. Re-run without --report-only to reconcile.)")
        raise typer.Exit(code=0)

    goal_final = goal or first.goal
    spec = build_merged_spec(named, goal=goal_final, task_type=first.task_type, drops=drops, additions=additions)
    merged = Project(name=out.name, goal=goal_final, task_type=first.task_type, spec=spec)
    with project_lock(out):
        if (out / "project.yaml").exists():
            typer.secho(f"A project already exists at {out}/.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        save_project(merged, out)
        atomic_write_text(out / "reconciliation.yaml",
                          _yaml.safe_dump({"stakeholders": list(named), "conflicts": audit}, sort_keys=False))
    typer.secho(
        f"\n✓ Merged {len(named)} stakeholder(s) → {out}/  "
        f"({len(spec.standards)} standard(s), {len(spec.do_not)} never-rule(s); {len(conflicts)} conflict(s) reconciled).",
        fg=typer.colors.GREEN,
    )
    typer.echo("  reconciliation audit → reconciliation.yaml")
    typer.echo("\nNext:  calibrate compile  (regenerate tests/rubric for the merged spec)")


@app.command(name="log")
def log_cmd(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    enable: Optional[bool] = typer.Option(
        None, "--on/--off", help="Turn engine-decision logging on or off (omit to just show status)."
    ),
) -> None:
    """Toggle local logging of engine decisions — the data the Engine-Trainer learns from."""
    with project_lock(path):
        project = _load(path)
        if enable is not None:
            project.log_interactions = enable
            save_project(project, path)
    state = "ON" if project.log_interactions else "OFF"
    typer.secho(f"engine logging: {state}", fg=typer.colors.GREEN if project.log_interactions else None, bold=True)
    if project.log_interactions:
        typer.echo(f"  decisions append to {Path(path)}/logs/<role>.jsonl on each `calibrate eval`.")
        typer.echo("  localize a role later with:  calibrate train-engine judge")
    else:
        typer.echo("  turn on with:  calibrate log --on   (stays local; off by default for privacy)")


@app.command(name="train-engine")
def train_engine_cmd(
    role: str = typer.Argument(..., help="Tool role to localize: judge | compiler | extractor | interviewer | predictor"),
    path: Path = typer.Argument(Path("."), help="Project directory."),
    base: Optional[str] = typer.Option(None, "--base", help="Open base model to fine-tune."),
    prove: bool = typer.Option(False, "--prove", help="Replay logged inputs through a candidate engine and measure agreement."),
    candidate: Optional[str] = typer.Option(None, "--candidate", help="Candidate engine spec to prove (model@provider)."),
    threshold: float = typer.Option(0.9, "--threshold", help="Min agreement to trust the local engine (0-1)."),
) -> None:
    """Localize a cloud role onto your own model from logged decisions — the autonomy loop. (v1)"""
    from .train_engine import TRAINABLE_ROLES, export_engine_bundle, prove_engine, read_log

    role = role.strip().lower()
    if role not in TRAINABLE_ROLES:
        typer.secho(f"role must be one of: {', '.join(sorted(TRAINABLE_ROLES))}.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    _load(path)  # validate the project exists / is loadable

    if prove:
        if not candidate:
            typer.secho("--prove needs --candidate <model@provider>.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
            typer.secho("--threshold must be a number between 0 and 1.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        from .engines import get_engine
        try:
            cand = get_engine(candidate)
        except (RuntimeError, ValueError, NotImplementedError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.echo(f"Proving {candidate} against logged {role} decisions …")
        try:
            proof = prove_engine(path, role, cand, threshold=threshold)
        except Exception as exc:
            typer.secho(f"Prove failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if proof.samples == 0:
            typer.secho(f"No logged {role} decisions. Run `calibrate log --on`, then `calibrate eval`, then retry.",
                        fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        typer.secho(
            f"agreement: {pct(proof.agreement)} over {proof.samples} sample(s) (threshold {threshold:.0%})",
            fg=typer.colors.GREEN if proof.passes else typer.colors.YELLOW, bold=True,
        )
        if proof.passes:
            typer.secho(f"✓ The local engine reproduces the cloud {role} — safe to set engines.{role} = {candidate}.",
                        fg=typer.colors.GREEN)
        else:
            typer.secho("✗ Not yet — keep the cloud engine, or train on more logged data.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0 if proof.passes else 1)

    if not read_log(path, role):
        typer.secho(
            f"No logged {role} decisions yet. Turn on logging (`calibrate log --on`), run `calibrate eval`, then retry.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)
    try:
        result = export_engine_bundle(path, role, base_model=base)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho(
        f"✓ Engine-training bundle → {result.bundle_dir}/  "
        f"({result.examples} example(s) on {result.base_model})",
        fg=typer.colors.GREEN,
    )
    for f in result.files:
        typer.echo(f"    {f}")
    typer.echo(
        f"\nTrain on a GPU (see README), serve it, then prove it matches:\n"
        f"  calibrate train-engine {role} --prove --candidate <your-model@ollama>"
    )


@app.command()
def export(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    name: Optional[str] = typer.Option(
        None, "--name", help="Bundle / model name (default: from the project name)."
    ),
) -> None:
    """Package the calibrated config into a runnable bundle (Ollama Modelfile + more). (M5)"""
    from .export import export_bundle

    project = _load(path)
    if project.spec is None:
        typer.secho("Nothing to export — run `calibrate compile` first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    try:
        result = export_bundle(project, project_dir=path, name=name)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho(f"✓ Exported calibrated AI → {result.bundle_dir}/", fg=typer.colors.GREEN)
    for f in result.files:
        typer.echo(f"    {f}")
    typer.echo(
        f"\nRun it locally with Ollama:\n"
        f"  ollama pull {result.base_model}\n"
        f"  ollama create {result.name} -f {result.bundle_dir}/Modelfile\n"
        f"  ollama run {result.name}\n"
        f"…or programmatically:  python {result.bundle_dir}/run.py \"your question\""
    )


@app.command(name="examples-to-tests")
def examples_to_tests(path: Path = typer.Argument(Path("."), help="Project directory.")) -> None:
    """Turn the spec's good/bad examples into regression tests (§9 golden anchors). (no engine)"""
    from .compile import tests_from_examples, write_build_bundle

    with project_lock(path):
        project = _load(path)
        if project.spec is None:
            typer.secho("Nothing to convert — run `calibrate compile` (or `import`) first.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        new = tests_from_examples(project.spec, project.tests)
        if not new:
            typer.secho("No new example-derived tests to add.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)
        project.tests.extend(new)
        save_project(project, path)
        write_build_bundle(project.spec, project.tests, path)
    typer.secho(f"✓ Added {len(new)} regression test(s) from the spec's examples.", fg=typer.colors.GREEN)
    typer.echo("Run `calibrate eval` to include them.")


@app.command(name="export-evals")
def export_evals(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    fmt: str = typer.Option("promptfoo", "--format", help="Eval harness format (currently: promptfoo)."),
) -> None:
    """Export the test suite + rubric to an external eval harness (promptfoo). (no engine)"""
    project = _load(path)
    if project.spec is None or not project.tests:
        typer.secho("Nothing to export — run `calibrate compile` (or `import`) first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    if fmt != "promptfoo":
        typer.secho(f"Unsupported format {fmt!r} (currently only: promptfoo).", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    from .interop import export_promptfoo

    out = export_promptfoo(project, project_dir=path)
    typer.secho(f"✓ Wrote {out}", fg=typer.colors.GREEN)
    typer.echo("Run it with:  promptfoo eval -c promptfooconfig.yaml")
    typer.echo("  (edit the `providers:` line to your model, and set the grader's API key)")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (localhost by default)."),
    port: int = typer.Option(8765, "--port", help="Port."),
    projects: Optional[Path] = typer.Option(
        None, "--projects", help="Projects root dir (default: ~/.ai-calibrator/projects)."
    ),
) -> None:
    """Run the local API + web UI; open the printed URL in your browser. (M6)"""
    if not (0 <= port <= 65535):
        typer.secho(
            f"--port must be between 0 and 65535 (got {port}).", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)
    try:
        import uvicorn
        from .api import create_app, default_projects_root
    except (ImportError, RuntimeError):
        typer.secho("The API needs the `api` extra:  pip install -e '.[api]'", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    root = projects or default_projects_root()
    is_local = host in ("127.0.0.1", "localhost", "::1")
    if not is_local:
        typer.secho(
            f"⚠  Binding to {host} exposes the (unauthenticated) API beyond localhost.\n"
            "   The Host/Origin guard still applies — but there is NO authentication, so only do\n"
            "   this on a trusted network. Bind a specific address (e.g. --host 192.168.1.50),\n"
            "   not 0.0.0.0, so the guard can match the host you actually connect to.",
            fg=typer.colors.RED,
        )
    application = create_app(root, allowed_hosts=None if is_local else [host])
    typer.secho(f"AI Calibrator → http://{host}:{port}", fg=typer.colors.GREEN)
    typer.echo(f"  projects in {root}")
    uvicorn.run(application, host=host, port=port, log_level="warning")


@app.command()
def finetune(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    base: Optional[str] = typer.Option(None, "--base", help="Open base model to fine-tune."),
    gate: bool = typer.Option(False, "--gate", help="Compare two eval scorecards instead of building."),
    baseline: Optional[str] = typer.Option(None, "--baseline", help="Baseline run id (with --gate)."),
    candidate: Optional[str] = typer.Option(None, "--candidate", help="Candidate run id (with --gate)."),
) -> None:
    """Advanced tier: build a fine-tuning dataset + recipe, or run the prove-it gate. (v1)"""

    project = _load(path)

    if gate:
        if not (baseline and candidate):
            typer.secho("--gate needs --baseline <run-id> and --candidate <run-id>.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        from .finetune import beats_baseline

        base_card, cand_card = _scorecard_or_exit(path, baseline), _scorecard_or_exit(path, candidate)
        win = beats_baseline(base_card, cand_card)
        typer.echo(f"baseline [{baseline}]: {pct(base_card.pass_rate)}    candidate [{candidate}]: {pct(cand_card.pass_rate)}")
        if win:
            typer.secho("✓ ACCEPT — the fine-tune beats the configured baseline. Keep it.", fg=typer.colors.GREEN)
        else:
            typer.secho("✗ REJECT — it doesn't beat the baseline. Stay on configuration.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0 if win else 1)

    if project.spec is None:
        typer.secho("Nothing to fine-tune — run `calibrate compile` first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    from .finetune import export_finetune
    try:
        result = export_finetune(project, project_dir=path, base_model=base)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if result.examples == 0:
        typer.secho(
            "⚠ No training examples yet. The Advanced tier needs human-authored / "
            "corrected examples — add examples to the spec (or capture eval "
            "corrections) before fine-tuning will help.",
            fg=typer.colors.YELLOW,
        )
    typer.secho(
        f"✓ Fine-tuning bundle → {result.bundle_dir}/  "
        f"({result.examples} example(s), {result.method} on {result.base_model})",
        fg=typer.colors.GREEN,
    )
    for f in result.files:
        typer.echo(f"    {f}")
    typer.echo(
        "\nNext: train on a GPU (see finetune/README.md), then prove it wins:\n"
        "  calibrate finetune --gate --baseline <run> --candidate <run>"
    )


def main() -> None:
    # A limited terminal encoding (ascii / cp1252 console) must degrade glyphs
    # (✓ ⚠ →) to '?', never crash — Rich's --help rendering raised a raw
    # UnicodeEncodeError otherwise. (audit: locale robustness)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError):  # non-reconfigurable stream (tests, pipes)
            pass
    app()


if __name__ == "__main__":
    main()
