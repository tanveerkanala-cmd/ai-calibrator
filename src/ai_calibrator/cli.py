"""`calibrate` — the command-line front end over the Calibration Core.

The CLI is a thin shell: every command maps to a pipeline stage on the Core, so
the same logic also powers the local API and the web UI.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from pydantic import ValidationError

from . import __version__
from .models import EngineBinding, Project, TaskType
from .fmt import pct, pct_delta
from .store import atomic_write_text, load_project, project_lock, save_project, write_project_gitignore

app = typer.Typer(
    add_completion=False,
    help="Turn your knowledge and standards into a tested, reliable AI.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"calibrate {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,  # print and exit before any subcommand is resolved
        help="Show the installed version and exit.",
    ),
) -> None:
    """Turn your knowledge and standards into a tested, reliable AI."""


def _validate_port(port: int) -> None:
    """Reject an out-of-range port before uvicorn does (with a raw error)."""
    if not (1 <= port <= 65535):
        typer.secho(f"--port must be between 1 and 65535 (got {port}).", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def _scorecard_or_exit(path: Path, rid: str):
    """Load a saved scorecard, or exit friendly — a scorecard.json can be
    corrupt/truncated/hand-edited, so never let a raw traceback escape."""
    from .drift import load_scorecard
    try:
        return load_scorecard(path, rid)
    except (OSError, ValueError, ValidationError) as exc:
        typer.secho(f"Could not read scorecard {rid!r}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def _lock_wait_notice() -> None:
    """Printed once when a project lock is contended, so a wait isn't a silent hang."""
    typer.secho("  · project is busy with another calibrate operation — waiting for the lock…",
                fg=typer.colors.YELLOW)


def _resolve_project(path: Path, projects: Optional[Path]) -> Path:
    """Resolve the project location. With ``--projects <root>`` the argument is a
    project NAME under that root (the same store `calibrate serve` / the web UI
    use), so a project created there is reachable by name, not just full path."""
    if projects is not None:
        return Path(projects) / path
    return path


def _cleanup_empty_project_dir(path: Path, we_created: bool) -> None:
    """Remove a directory this command created for a project that never got built.

    ``import`` and ``merge`` legitimately create their destination before the work
    that can fail, so a failure leaves a directory holding only a ``.lock`` — the
    exact litter ``_require_project`` prevents everywhere else. Only ever removes a
    directory WE created, and only when nothing but the lock is in it."""
    if not we_created:
        return
    try:
        d = Path(path)
        if not d.is_dir():
            return
        leftovers = [f for f in d.iterdir() if f.name != ".lock"]
        if leftovers:
            return  # something real is in there — never touch it
        (d / ".lock").unlink(missing_ok=True)
        d.rmdir()
    except OSError:
        pass  # cleanup is best-effort; never mask the original failure


def _require_project(path: Path, on_error=None) -> None:
    """Exit friendly if there's no project here — WITHOUT creating anything.

    ``project_lock`` mkdirs the directory and drops a ``.lock`` file, so calling
    it for a typo'd/nonexistent project name litters a junk directory. Call this
    before the lock so a missing project is reported without side effects.

    ``on_error(reason)`` lets a machine-readable caller (``ci --json``) render the
    failure in its own format instead of a coloured sentence."""
    if not (Path(path) / "project.yaml").exists():
        msg = (f"No calibrator project at {Path(path)} (missing project.yaml). "
               "Run `calibrate init` first.")
        if on_error is not None:
            raise on_error(msg)
        typer.secho(msg, fg=typer.colors.RED)
        raise typer.Exit(code=1)


def _warn_unknown_keys(project: Project) -> None:
    """Surface a typo'd top-level key (kept, but inert) — e.g. a misspelled
    `engines:` silently leaves every role on the cloud default, which quietly
    breaks the privacy of local-only mode. Warn so the typo doesn't hide."""
    extra = getattr(project, "__pydantic_extra__", None) or {}
    if not extra:
        return
    import difflib
    known = list(Project.model_fields)
    for key in extra:
        near = difflib.get_close_matches(str(key), known, n=1, cutoff=0.7)
        hint = f" — did you mean '{near[0]}'?" if near else ""
        typer.secho(f"⚠ Unknown key {key!r} in project.yaml{hint} It is ignored; "
                    "any real setting it was meant to be is using its default.",
                    fg=typer.colors.YELLOW)


def _load(path: Path, on_error=None) -> Project:
    try:
        project = load_project(path)
        _warn_unknown_keys(project)
        return project
    except FileNotFoundError as exc:
        if on_error is not None:
            raise on_error(str(exc))
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except (OSError, yaml.YAMLError, ValidationError, ValueError) as exc:
        # Corrupt, incomplete, or UNREADABLE project.yaml (hand-edited, partially
        # written by an old version, truncated, no read permission, or a directory
        # where the file should be) — a friendly message, never a traceback. OSError
        # belongs here for the same reason _scorecard_or_exit and report.py catch it.
        msg = (f"The project at {path}/ could not be read "
               f"({Path(path) / 'project.yaml'}): {exc}")
        if on_error is not None:
            raise on_error(msg)
        typer.secho(
            f"The project at {path}/ is invalid or corrupted, or could not be read "
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
    if not goal or not goal.strip():
        # The goal seeds gap analysis, the interview, and the spec — an empty one
        # gives the whole pipeline nothing to work with.
        typer.secho("--goal must not be empty: one sentence on what this AI should do.",
                    fg=typer.colors.RED)
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
    # Validate the name BEFORE any filesystem op. The model catches names the CLI
    # pre-checks miss (reserved device names like CON, trailing dot/space, control
    # chars); on Windows those also make the lock's mkdir(target) fail at the OS
    # level, so this must run before project_lock — surface a friendly message,
    # not a raw pydantic traceback or an OSError.
    try:
        project = Project(name=name, goal=goal, task_type=task_type)
    except ValidationError as exc:
        msg = exc.errors()[0].get("msg", "invalid project name") if exc.errors() else "invalid project name"
        typer.secho(f"Invalid project name: {msg}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if target.exists() and not target.is_dir():
        # A plain file sits where the project dir would go — project_lock's mkdir
        # would raise a raw FileExistsError. Fail friendly instead.
        typer.secho(f"A file named {target} already exists here — pick another project "
                    "name, or use --path for a different location.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    with project_lock(target, on_wait=_lock_wait_notice):  # atomic against a concurrent `init` of the same path
        if (target / "project.yaml").exists():
            typer.secho(f"A project already exists at {target}/", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        save_project(project, target)
        write_project_gitignore(target)
    typer.secho(f"✓ Created project '{name}' at {target}/", fg=typer.colors.GREEN)
    typer.echo(f"  goal: {goal}")
    e = project.engines
    typer.echo(f"  engines: {e.interviewer} (reasoning) · {e.judge} (judge) · {e.subject} (subject)")
    typer.echo("\nNext:  add materials, then `calibrate ingest`.")


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
    """Reverse-calibrate: extract a tested behavior spec from an EXISTING system prompt."""
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

    # project_lock() mkdirs the destination, which raises a raw FileExistsError
    # when the path is an existing FILE. `init` guards this; import must too.
    if path.exists() and not path.is_dir():
        typer.secho(f"A file named {path} already exists — pick another destination.",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    name = path.resolve().name or "project"
    engine_spec = engine
    try:
        eng = get_engine(engine_spec or EngineBinding().compiler)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Remember whether we are the ones creating the directory: if the engine call
    # fails we must not leave a junk project dir holding only a .lock behind (the
    # litter _require_project exists to prevent), but we must never delete a
    # directory the user already had.
    _we_created = not path.exists()
    try:
        with project_lock(path, on_wait=_lock_wait_notice):
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
    except typer.Exit:
        _cleanup_empty_project_dir(path, _we_created)
        raise

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
    path: Path = typer.Argument(Path("."), help="Project directory (or name, with --projects)."),
    projects: Optional[Path] = typer.Option(
        None, "--projects", help="Treat the argument as a project NAME under this root (serve/UI store)."
    ),
) -> None:
    """Show a project's progress through the pipeline."""
    path = _resolve_project(path, projects)
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
        _require_project(path)  # no junk .lock dir for a typo'd name
        with project_lock(path, on_wait=_lock_wait_notice):
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
        f"Unknown provider {provider!r}. Use 'claude' (aka 'anthropic') or 'openai'.",
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
    """Parse materials, extract the gap list, and build the retrieval index."""
    from .engines import get_engine
    from .ingest import MAX_EXTRACT_CHARS, ingest_project

    # Hold the project lock across load→mutate→save so a concurrent calibrate
    # process can't lose this run's results.
    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
        project = _load(path)
        src = source or (Path(path) / "materials")
        if src.exists() and not src.is_dir():  # --source pointed at a file, not a dir
            typer.secho(f"--source {src} must be a directory of materials, not a file.",
                        fg=typer.colors.RED)
            raise typer.Exit(code=1)
        # A --source that isn't there is a typo (or the wrong working directory) —
        # not the deliberate "I deleted every material" that an existing-but-empty
        # folder means. Treating the two alike wiped the facts, gaps and index of a
        # healthy project and reported success. The default materials/ can't be
        # missing: save_project recreates it on every save.
        if source is not None and not src.exists():
            typer.secho(f"--source {src} does not exist — check the path (and your working "
                        "directory). Nothing was changed.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if not src.exists() or not any(src.iterdir()):
            if not project.materials:
                typer.secho(
                    f"No materials found in {src}/. Add files there, then re-run.",
                    fg=typer.colors.YELLOW,
                )
                raise typer.Exit(code=1)
            # An empty folder on a project that HAS ingested materials is a
            # deliberate "remove everything". Refusing here left the facts, gaps and
            # retrieval index built from the deleted files in place, still feeding
            # every graded and served prompt.
            typer.secho(
                f"{src}/ is empty — clearing the {len(project.materials)} previously "
                "ingested file(s) and their extracted facts"
                + ("." if no_index else ", and dropping the retrieval index."),
                fg=typer.colors.YELLOW,
            )

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
    if result.analyzed < result.materials:
        typer.secho(
            f"  ⚠ only {result.analyzed} of {result.materials} file(s) fit the "
            f"{MAX_EXTRACT_CHARS:,}-character analysis window — the facts and the gap "
            "list below come from those files alone. Move the rest into their own "
            "project (or trim them) if they need to inform the interview.",
            fg=typer.colors.YELLOW,
        )
    if result.skipped:
        typer.secho(f"  ⚠ skipped {len(result.skipped)} file(s) that couldn't be parsed:",
                    fg=typer.colors.YELLOW)
        for rel, reason in result.skipped:
            typer.echo(f"    · {rel} — {reason}")
    if result.indexed is not None and not result.chunks:
        # Reporting "0 chunk(s) embedded" for a drop reads as a build that found
        # nothing, when what happened is that an index was deleted.
        typer.echo("  retrieval index: dropped (no material text left to index)")
    elif result.indexed is not None:
        typer.echo(f"  retrieval index: {result.indexed} chunk(s) embedded")
    elif no_index:
        typer.echo("  retrieval index: skipped (--no-index)")
        # Skipping the build is not the same as retrieving nothing: an index from
        # an earlier ingest is still on disk and still consulted, so it can serve
        # text from files this ingest just replaced or removed.
        if (Path(path) / "knowledge.lancedb").exists():
            typer.secho(
                "  ⚠ the index from an earlier ingest is still in place and still feeds every "
                "eval and run — it may hold text from files you just changed or deleted. "
                "Re-run without --no-index to rebuild it from the current materials.",
                fg=typer.colors.YELLOW,
            )
    else:
        from . import rag
        if not rag.index_available():
            typer.secho(
                "  ⚠ retrieval index NOT built — the `rag` extra isn't installed, so your\n"
                "    exported AI will answer from the system prompt ONLY and will NOT be able\n"
                "    to use these documents. To fix, in your ai-calibrator clone run\n"
                "    pip install -e '.[rag]'  then re-run  `calibrate ingest`.",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.echo("  retrieval index: not built (no indexable chunks)")

    typer.secho(f"\n{result.gaps} gap(s) to resolve in the interview:", bold=True)
    for g in project.gaps:
        typer.echo(f"  · {g.dimension}")
    typer.echo("\nNext:  calibrate interview")


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
    """Ask adaptive, gap-driven questions (propose-and-ratify) and store answers."""
    from .engines import get_engine
    from .interview import generate_questions

    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
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
            typer.echo(f"Drafting one question per gap with {engine.name} "
                       f"({len(project.gaps)} gap(s)) …")

            # Persist after each gap so an engine timeout mid-interview keeps the
            # questions already drafted instead of discarding the whole run.
            drafted = 0

            def _progress(items, done, total):
                nonlocal drafted
                drafted = done  # gaps reached, not items held: a snapshot also carries the answers
                project.interview = list(items)
                save_project(project, path)
                typer.echo(f"  · drafted {done}/{total} gap(s) …")

            try:
                project.interview = generate_questions(project, engine, on_progress=_progress)
            except Exception as exc:
                typer.secho(f"Question generation stopped after {drafted} "
                            f"of {len(project.gaps)} gap(s): {exc}", fg=typer.colors.RED)
                typer.echo("  Progress was saved and your answers were kept — re-run "
                           "`calibrate interview --regenerate` to finish.")
                raise typer.Exit(code=1)
            save_project(project, path)

            from .interview import uncovered_gaps
            missing = uncovered_gaps(project, project.interview)
            if missing:
                typer.secho(
                    f"⚠ Drafted {len(project.interview)} question(s) covering "
                    f"{len(project.gaps) - len(missing)} of {len(project.gaps)} gap(s). "
                    f"Uncovered: {', '.join(missing)}.\n"
                    "  Re-run `calibrate interview --regenerate` to retry the missed gaps.",
                    fg=typer.colors.YELLOW,
                )
        questions = list(project.interview)  # snapshot; lock released before prompting

    pending = [it for it in questions if not it.answer]
    if not pending:
        typer.secho("All questions answered. Next:  calibrate compile",
                    fg=typer.colors.GREEN)
        raise typer.Exit(code=0)

    # Gather answers WITHOUT holding the lock — the interactive prompts can take
    # minutes, and holding the project lock across them would block every other
    # command on this project. Collect by question id, then apply atomically.
    mode = "auto-accepting drafts" if accept_drafts else "Enter = accept the draft"
    typer.secho(f"{len(pending)} question(s) to answer ({mode}):\n", bold=True)
    # Prompt once per unique id — a hand-edited project.yaml can have duplicate
    # interview ids, and prompting for each would collect two answers under one
    # dict key (losing the first). The apply step fans the answer to every match.
    seen_ids: set[str] = set()
    pending = [it for it in pending if not (it.id in seen_ids or seen_ids.add(it.id))]
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
    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
        project = _load(path)
        # Apply to EVERY item whose id matches — a dict-by-id would drop all but
        # the last of any duplicate-id items (possible via a hand-edited
        # project.yaml), silently discarding a collected answer.
        for it in project.interview:
            if it.id in answers:
                it.answer = answers[it.id]
        save_project(project, path)
        answered = sum(1 for it in project.interview if it.answer)
        total = len(project.interview)

    typer.secho(f"✓ {answered}/{total} answered.", fg=typer.colors.GREEN)
    typer.echo("Next:  calibrate compile")


@app.command()
def compile(path: Path = typer.Argument(Path("."), help="Project directory.")) -> None:
    """Synthesize the behavior spec + system prompt + RAG + rubric + tests."""
    from .compile import compile_project
    from .engines import get_engine

    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
        project = _load(path)
        # A merged project carries a spec but no interview, and compile_project
        # already preserves an existing spec when there is nothing to synthesize
        # from. Gating on the interview alone made `merge` a dead end: the very
        # next step it tells you to run always refused.
        if not any(it.answer for it in project.interview) and project.spec is None:
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
        from .engines.base import EngineOutputError
        try:
            result = compile_project(project, engine, project_dir=path)
        except EngineOutputError as exc:
            # The compiler model produced unreadable output. If synthesis had
            # already succeeded (it's the test-gen call that failed), the spec is
            # captured on the project — save it so a retry resumes from tests.
            raw = getattr(exc, "raw", "")
            if raw:
                err_path = atomic_write_text(Path(path) / "build" / "compile-error.txt", raw)
            if project.spec is not None:
                save_project(project, path)
            typer.secho(f"Compile failed: {exc}", fg=typer.colors.RED)
            if raw:
                typer.echo(f"  The model's raw output was saved to {err_path} for inspection.")
            if project.spec is not None:
                typer.echo("  The synthesized spec was saved — re-run `calibrate compile` to retry test generation.")
            raise typer.Exit(code=1)
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
    typer.echo("\nNext:  calibrate eval")


def _retrieval_off_reason(project_dir: Path) -> str:
    """Why retrieval is OFF for this project, or '' when it actually works.

    Two failures with two different fixes: the `rag` extra isn't installed at
    all, and an index that exists but cannot be queried (embedder model absent,
    corrupt table, version skew). Reporting the first as the second sends the
    owner hunting for a broken index they never built."""
    from . import rag
    if not rag.index_available():
        return "the `rag` extra is not installed — pip install -e '.[rag]' in your clone"
    why = rag.probe(project_dir)
    if not why:
        return ""
    if why == "no index":
        return "your documents are NOT in play"
    return f"index present but unusable — {why}"


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
    max_tests: Optional[int] = typer.Option(
        None, "--max-tests", help="Grade only the first N tests (a smoke check on a slow model)."
    ),
) -> None:
    """Run tests, grade against the rubric, score, and (optionally) refine."""
    from .engine_log import wrap_engine
    from .engines import get_engine
    from .eval import EvalInterrupted, low_confidence_results, next_run_id, run_eval, save_scorecard

    if not (1 <= rounds <= 100):  # bounds match the API (EvalBody), always validated
        typer.secho("--rounds must be between 1 and 100.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        typer.secho("--threshold must be a number between 0 and 1.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not (1 <= judge_passes <= 9):
        typer.secho("--judge-passes must be between 1 and 9.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if max_tests is not None and max_tests < 1:
        typer.secho("--max-tests must be >= 1.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if max_tests is not None and refine:
        typer.secho("--max-tests can't be combined with --refine (the loop needs the full suite).",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
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

        n_tests = min(len(project.tests), max_tests) if max_tests else len(project.tests)
        typer.echo(
            f"Evaluating {n_tests} test(s): subject={subject.name}, judge={judge.name}"
            + (f", refiner={refiner.name}" if refiner else "")
            + (f"  (--max-tests {max_tests})" if max_tests else "") + " …"
        )
        # If the project has documents but no usable retrieval index, we're grading
        # a prompt-only bot — say so, so a high pass rate isn't read as "it can use
        # my docs" when it can't (the exported bot would answer blind). A project
        # with no materials has nothing to retrieve, so it says nothing at all.
        # The probe is for real: an index directory that exists proves nothing.
        if project.materials:
            detail = _retrieval_off_reason(path)
            if detail:
                typer.secho(f"  retrieval: OFF — grading a prompt-only bot ({detail}).",
                            fg=typer.colors.YELLOW)

        def _progress(done, total, test_id):
            typer.echo(f"  · [{done}/{total}] {test_id} — subject + judge …")

        try:
            if refine:
                from .compile import write_build_bundle
                from .pipeline import calibrate_loop

                def _persist_spec(proj):
                    # Checkpoint each round's refinement BEFORE the next round is
                    # graded: scorecards are saved as they are earned, so saving the
                    # spec only after the loop would let an interruption leave runs
                    # on disk that no recorded spec ever produced.
                    save_project(proj, path)
                    write_build_bundle(proj.spec, proj.tests, path)

                cards = calibrate_loop(
                    project, subject, judge, refiner,
                    threshold=threshold, max_rounds=rounds, judge_passes=judge_passes, project_dir=path,
                    on_spec_change=_persist_spec,
                )
                save_project(project, path)  # refined standards persist
                write_build_bundle(project.spec, project.tests, path)  # refresh build/ to match
            else:
                try:
                    card = run_eval(project, subject, judge, run_id=next_run_id(path),
                                    judge_passes=judge_passes, project_dir=path,
                                    max_tests=max_tests, on_progress=_progress)
                except EvalInterrupted as interrupted:
                    # Ctrl-C: save what completed so the run isn't wasted.
                    save_scorecard(path, interrupted.partial)
                    done = len(interrupted.partial.results)
                    typer.secho(f"\n⚠ Interrupted — saved a PARTIAL scorecard "
                                f"[{interrupted.partial.run_id}] with {done} graded test(s).",
                                fg=typer.colors.YELLOW)
                    raise typer.Exit(code=130)
                save_scorecard(path, card)
                cards = [card]
        except (EvalInterrupted, typer.Exit):
            raise  # a clean interrupt/exit must not be reframed as "Eval failed"
        except Exception as exc:
            typer.secho(f"Eval failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    for i, card in enumerate(cards, 1):
        graded = [r for r in card.results if r.criteria]
        passed = sum(1 for r in graded if r.passed)
        ungraded = len(card.results) - len(graded)
        typer.echo(f"  round {i} [{card.run_id}]: {pct(card.pass_rate)} ({passed}/{len(graded)} passed"
                   + (f", {ungraded} of {len(card.results)} not graded)" if ungraded else ")"))

    final = cards[-1]
    ok = final.pass_rate >= threshold
    typer.secho(
        f"\nFinal pass rate: {pct(final.pass_rate)}   (weighted score: {pct(final.weighted_score)})",
        fg=typer.colors.GREEN if ok else typer.colors.YELLOW,
    )
    # An ungraded test is excluded from the rate above (it was never actually
    # graded, so counting it as a failure would be a lie) — but so is reporting
    # 100% on a suite where a third of the tests expect criteria the spec doesn't
    # have. Name the shortfall and where the fix is.
    skipped = [r.test_id for r in final.results if not r.criteria]
    if skipped:
        typer.secho(
            f"  ⚠ {len(skipped)} of {len(final.results)} test(s) were NOT graded — their "
            f"`expects` name criterion id(s) the spec doesn't define "
            f"({', '.join(skipped[:5])}{', …' if len(skipped) > 5 else ''}). The rate above "
            "covers the graded tests only; run `calibrate lint` to fix them.",
            fg=typer.colors.YELLOW,
        )
    # Triage order: tests whose HIGH-weight criteria failed come first.
    from .models import Weight

    def _worst(r):  # highest weight among this test's failed criteria
        return max(((c.weight or Weight.MEDIUM).numeric for c in r.criteria if not c.passed), default=0)

    for r in sorted([r for r in final.results if not r.passed], key=_worst, reverse=True)[:10]:
        # Uniform format: each failed criterion as `id [weight]: reason`.
        why = "; ".join(
            f"{c.criterion_id} [{(c.weight or Weight.MEDIUM).value}]: "
            f"{c.rationale or 'no rationale'}"
            for c in r.criteria if not c.passed
        ) or "no criteria graded"
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
        typer.secho("Threshold met. Next:  calibrate export", fg=typer.colors.GREEN)
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

    def _fail(reason: str, human: str | None = None, code: int = 1):
        """Every --json exit path emits JSON. A pipeline that pipes `ci --json`
        into a parser must get a structured reason, not a coloured sentence."""
        if as_json:
            typer.echo(_json.dumps({"ok": False, "gate": "error", "reason": reason}))
        else:
            typer.secho(human or reason, fg=typer.colors.RED)
        return typer.Exit(code=code)

    if not math.isfinite(threshold) or not (0.0 <= threshold <= 1.0):
        raise _fail("--threshold must be a number between 0 and 1.")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise _fail("--tolerance must be a number >= 0.")
    if not (1 <= judge_passes <= 9):
        raise _fail("--judge-passes must be between 1 and 9.")

    _require_project(path, on_error=_fail)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
        project = _load(path, on_error=_fail)
        if project.spec is None or not project.tests:
            reason = "nothing to gate — run `calibrate compile` (or `import`) first"
            raise _fail(reason, f"Nothing to gate — {reason}.")
        # Factories: engines are acquired only if lint passes — a lint-broken spec
        # shouldn't demand credentials, and an engine problem shouldn't mask lint.
        log_on = project.log_interactions
        subject = lambda: get_engine(project.engines.subject)  # noqa: E731
        judge = lambda: wrap_engine(get_engine(project.engines.judge), "judge", path, enabled=log_on)  # noqa: E731
        try:
            result = run_ci(project, subject, judge, project_dir=path, threshold=threshold,
                            tolerance=tolerance, judge_passes=judge_passes, baseline=baseline)
        except Exception as exc:
            raise _fail(f"CI gate could not run: {exc}")

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
    path: Path = typer.Argument(Path("."), help="Project directory (or name, with --projects)."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (localhost by default)."),
    port: int = typer.Option(8600, "--port", help="Port."),
    guard: bool = typer.Option(False, "--guard", help="Re-check every live answer against the spec's deterministic checks."),
    force: bool = typer.Option(False, "--force", help="Serve even if the last `ci` gate FAILED."),
    projects: Optional[Path] = typer.Option(
        None, "--projects", help="Treat the argument as a project NAME under this root (serve/UI store)."
    ),
) -> None:
    """Serve the calibrated AI itself — an OpenAI-compatible endpoint that won't boot on a red gate.

    Point any OpenAI-protocol client at http://HOST:PORT/v1 (model name = project name).
    """
    from .ci import certification_status

    _validate_port(port)
    path = _resolve_project(path, projects)
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

    is_local = host in ("127.0.0.1", "localhost", "::1")
    try:
        from .runtime import create_ai_app
        # Non-local bind: widen the Host allowlist to that ONE host, so the
        # anti-rebinding/CSRF guard keeps protecting it (mirrors `serve`).
        application = create_ai_app(path, guard=guard, allowed_hosts=None if is_local else [host])
        import uvicorn
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except ImportError:
        typer.secho("Serving needs the `api` extra:  pip install -e '.[api]'  (in your ai-calibrator clone)",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not is_local:
        typer.secho(f"⚠  Binding to {host} exposes the (unauthenticated) AI beyond localhost.",
                    fg=typer.colors.YELLOW)
    # Compares the --host flag; nothing is bound here. (bandit reads the literal
    # as a bind-all — it is the opposite: this branch exists to WARN about one.)
    if host in ("0.0.0.0", "::"):  # nosec B104
        # A wildcard bind allowlists the literal wildcard, but clients send the
        # machine's real address in Host — so every request would 400. Say so.
        typer.secho(f"⚠  --host {host} listens on every interface, but the Host guard only "
                    f"allows the literal '{host}', so real clients will get "
                    "400 'host not allowed'.", fg=typer.colors.YELLOW, bold=True)
        typer.echo("   Bind the address clients will actually use, e.g. "
                   "`--host 192.168.1.50`, or keep it on 127.0.0.1 and put a proxy in front.")
    # --guard only enforces criteria that carry a deterministic check, and only
    # `calibrate add-check` creates one. Requesting it on a project without any
    # would otherwise print "guard ON" over an endpoint checking nothing.
    guard_armed = guard and bool(
        project.spec is not None and [c for c in project.spec.eval_criteria if c.check is not None]
    )
    if guard and not guard_armed:
        typer.secho("⚠ --guard requested, but no criterion has a deterministic check — "
                    "nothing will be enforced on live answers.", fg=typer.colors.YELLOW, bold=True)
        typer.echo("  Add one with `calibrate add-check <project> <criterion> <kind> <value>`.")
    typer.echo(f"Serving '{project.name}' (subject: {project.engines.subject}"
               + (", guard ON" if guard_armed else "") + f") at http://{host}:{port}/v1")
    import json as _json
    import shlex
    payload = _json.dumps({"model": project.name,
                           "messages": [{"role": "user", "content": "hello"}]})
    typer.echo(f'  try:  curl -s http://{host}:{port}/v1/chat/completions '
               f'-H "Content-Type: application/json" -d {shlex.quote(payload)}')
    try:
        uvicorn.run(application, host=host, port=port, log_level="warning")
    except OSError as exc:  # e.g. EADDRINUSE — friendly, not a raw traceback
        typer.secho(f"Could not bind {host}:{port} — {exc}. "
                    "The port may be in use; retry with a different --port.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def absorb(path: Path = typer.Argument(Path("."), help="Project directory.")) -> None:
    """Close the flywheel: fold live feedback (from `calibrate run`) into examples + pinned tests. (no engine)"""
    from .compile import write_build_bundle
    from .flywheel import absorb_feedback

    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
        project = _load(path)
        # The save runs as absorb's commit step, while the records are still in
        # the inbox: a save that fails leaves them there to absorb again.
        result = absorb_feedback(project, path, commit=lambda: save_project(project, path))
        if result.ups + result.downs + result.skipped == 0:
            typer.secho("No live feedback to absorb yet.", fg=typer.colors.YELLOW)
            typer.echo("  `calibrate run` records it: POST /v1/feedback "
                       '{"completion_id": "...", "verdict": "down", "correction": "..."}')
            raise typer.Exit(code=0)
        if project.spec is not None and project.tests:
            write_build_bundle(project.spec, project.tests, path)

    typer.secho(f"✓ Absorbed {result.ups + result.downs} feedback record(s): "
                f"{result.ups} up / {result.downs} down.", fg=typer.colors.GREEN)
    typer.echo(f"  examples added: {result.examples_added}   pinned tests added: {result.tests_added}"
               + (f" ({', '.join(result.test_ids)})" if result.test_ids else "")
               + (f"   skipped: {result.skipped}" if result.skipped else ""))
    if result.tests_added:
        # Only a new TEST moves config_hash; an examples-only absorb leaves the
        # fingerprint (and therefore the gate) exactly where it was.
        typer.secho("The AI just learned from real use — its certification is now stale.",
                    fg=typer.colors.YELLOW)
        typer.echo("Run `calibrate ci` to re-certify against the suite that now includes it.")
    elif result.examples_added:
        typer.echo("  Examples only — the certification fingerprint is unchanged, so the gate "
                   "still reflects what it certified. Re-run `calibrate ci` when you want the "
                   "new material graded.")


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

    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
        project = _load(path)
        if project.spec is None:
            typer.secho("Nothing to check — run `calibrate compile` (or `import`) first.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        crit = next((c for c in project.spec.eval_criteria if c.id == criterion), None)
        if crit is None:
            ids = ", ".join(c.id for c in project.spec.eval_criteria) or "(none)"
            typer.secho(f"No criterion {criterion!r}. Ids: {ids}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if kind in ("max_chars", "min_chars"):
            try:
                n = int(value.strip())
                if n < 0:
                    raise ValueError
            except ValueError:
                typer.secho(f"{kind} needs a non-negative integer, got {value!r}.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
        if kind in ("contains", "not_contains", "regex") and not value.strip():
            # An empty needle is inside every string, so `contains` with the value
            # omitted is a check that can never fail — a criterion that reports
            # PASS on output nothing actually graded.
            typer.secho(f"{kind} needs a value to check for — e.g. "
                        f"`calibrate add-check <project> {criterion} {kind} '30-day'`.",
                        fg=typer.colors.RED)
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
    card = _scorecard_or_exit(path, rid)
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
    if ag.unmatched:
        typer.secho(f"  ⚠ {ag.unmatched} label(s) had no matching judge verdict in {rid} and are "
                    "not part of that rate.", fg=typer.colors.YELLOW)
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
    """Adversarially probe the configured AI to break its own rules."""
    from .compile import write_build_bundle
    from .engines import get_engine
    from .redteam import promote_to_tests, run_redteam

    if not (1 <= max_probes <= 50):
        typer.secho("--max-probes must be between 1 and 50.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
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

    if not report.probes:
        # No probes ran, so nothing was attacked — never report that as holding.
        typer.secho(
            "\n⚠ No probes were generated — nothing was attacked, so this is not a pass.",
            fg=typer.colors.YELLOW, bold=True,
        )
        typer.echo("  The spec needs concrete rules to attack (standards, never-rules, edge cases,")
        typer.echo("  or a refusal policy), and the generator must return usable probes. Re-run after")
        typer.echo("  adding rules, or try again if the generator returned nothing usable.")
    else:
        ungraded = len(report.ungraded)
        if report.violations:
            color = typer.colors.RED
        elif ungraded:
            color = typer.colors.YELLOW  # unjudged probes are not a clean hold
        else:
            color = typer.colors.GREEN
        typer.secho(
            f"\nHeld {pct(report.hold_rate)} — {len(report.violations)}/{report.probes} probe(s) caused a violation.",
            fg=color, bold=True,
        )
        if ungraded:
            # A probe the judge could not grade never withstood anything, so it is
            # left out of the hold rate rather than quietly counted as a pass.
            typer.secho(
                f"  ⚠ {ungraded}/{report.probes} probe(s) could not be judged and are not counted "
                "as held. Re-run, or grade with a stronger judge engine.",
                fg=typer.colors.YELLOW,
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
    """Find the cheapest model that still meets your pass bar — runs your tests across models."""
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
        price = ("local" if r.local else
                 f"{r.in_price}/{r.out_price}" if r.in_price is not None else "unknown")
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

    if d.fields_changed:
        typer.secho("\nBehavior fields:", bold=True)
        for name, before_v, after_v in d.fields_changed:
            typer.secho(f"  ~ {name}", fg=typer.colors.YELLOW)
            typer.secho(f"      - {before_v if before_v is not None else '(unset)'}", fg=typer.colors.RED)
            typer.secho(f"      + {after_v if after_v is not None else '(unset)'}", fg=typer.colors.GREEN)

    _section("Standards", d.standards_added, d.standards_removed)
    _section("Never-rules", d.do_not_added, d.do_not_removed)
    _section("Edge cases", d.edge_cases_added, d.edge_cases_removed)
    # Knowledge sources count as a behavior change (the system prompt gains or
    # loses its grounding paragraph), so a knowledge-only diff must print
    # something rather than an empty report under a "changed" verdict.
    _section("Knowledge sources", d.knowledge_added, d.knowledge_removed)
    if d.criteria_added or d.criteria_removed or d.criteria_changed:
        typer.secho("\nCriteria:", bold=True)
        for x in d.criteria_added:
            typer.secho(f"  + {x}", fg=typer.colors.GREEN)
        for x in d.criteria_removed:
            typer.secho(f"  - {x}", fg=typer.colors.RED)
        for x in d.criteria_changed:
            typer.secho(f"  ~ {x} (description/weight/check changed)", fg=typer.colors.YELLOW)


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
    """Re-run the suite and flag behavior drift vs a baseline. Exits 2 on regression (CI-friendly)."""
    from .drift import run_drift
    from .engines import get_engine
    from .eval import latest_run_id

    if not math.isfinite(tolerance) or not (0.0 <= tolerance <= 1.0):
        typer.secho("--tolerance must be a number between 0 and 1.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
        project = _load(path)
        if project.spec is None or not project.tests:
            typer.secho("Nothing to check — run `calibrate compile` first.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        # Default to the latest FULL run: a --max-tests smoke run as the baseline
        # compares two different test sets, hiding every regression it never ran.
        base_id = baseline or latest_run_id(path, full_only=True)
        if not base_id:
            typer.secho("No full baseline scorecard yet — run `calibrate eval` first to set one.",
                        fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
        base_card = _scorecard_or_exit(path, base_id)  # corrupt/missing baseline → friendly exit
        if base_card.partial:  # only reachable via an explicitly pinned --baseline
            typer.secho(f"Baseline {base_id} is a PARTIAL run (interrupted, or --max-tests) — "
                        "it covers only some tests, so a comparison against it is not meaningful. "
                        "Pin a full run with --baseline, or run `calibrate eval`.",
                        fg=typer.colors.RED)
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
    # Resolve the newest run either way, so a corrupt scorecard still surfaces
    # honestly ("Could not read scorecard") instead of being silently skipped.
    rid = latest_run_id(path)
    if not rid:
        typer.secho("No scorecard yet — run `calibrate eval` first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    card = _scorecard_or_exit(path, rid)
    latest = outputs_of(card)

    if not check:
        # Never PIN from a partial run: the golden is the most reference-y artifact
        # there is, and a --max-tests or interrupted run would overwrite a complete
        # golden with a strict subset, silently narrowing every future --check.
        if card.partial:
            typer.secho(f"{rid} is a PARTIAL run (interrupted, or --max-tests) — pinning it would "
                        "replace the golden with a subset of the suite. Run a full "
                        "`calibrate eval` first.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
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
    # Headline the latest FULL run: a `--max-tests` smoke run summarizes nothing,
    # and a certificate is a claim about the whole suite.
    rid = latest_run_id(path, full_only=True)
    if rid:
        try:
            latest = load_scorecard(path, rid)
        except (OSError, ValueError, ValidationError):  # incl. PermissionError on read
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
    """Calibrate by example: approve/reject sample outputs; the tool infers your standards."""
    from .compile import write_build_bundle
    from .engines import get_engine
    from .teach import Judged, apply_learned, infer_standards, propose_candidates

    if not (1 <= n <= 20):
        typer.secho("--n must be between 1 and 20.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Generate + judge + infer WITHOUT the lock — the interactive approve/reject
    # loop can take minutes, and holding the project lock across it would block
    # every other command on this project. Apply atomically at the end (like
    # `interview`), against a FRESH load so a concurrent edit isn't clobbered.
    project = _load(path)
    try:
        generator = get_engine(project.engines.compiler)
        subject = get_engine(project.engines.subject)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Generating {n} sample output(s) to judge (subject={subject.name}) …")
    try:
        candidates = propose_candidates(project, generator, subject, n=n, project_dir=path)
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

    # Persist the human's judgments BEFORE the inference call. They are the
    # expensive part — minutes of a person's attention — and they do not depend on
    # `learned` at all (apply_learned records them either way). An engine failure
    # here used to discard the entire session.
    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
        project = _load(path)
        saved = apply_learned(project, judged, None)
        save_project(project, path)
        if project.tests:
            write_build_bundle(project.spec, project.tests, path)

    typer.echo("\nInferring your standards from these judgments …")
    try:
        learned = infer_standards(project.goal, judged, generator)
    except Exception as exc:
        typer.secho(f"Inference failed: {exc}", fg=typer.colors.RED)
        typer.secho(f"  Your {len(judged)} judgment(s) were saved as examples — nothing was lost. "
                    "Re-run `calibrate teach` when the engine is available to infer standards.",
                    fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    with project_lock(path, on_wait=_lock_wait_notice):  # short critical section: reload → apply → save
        project = _load(path)
        # No judgments here: phase 1 already recorded every one of them as an
        # example. Passing them again appends a SECOND copy of each, doubling the
        # examples every session and training each pair at double weight in the
        # fine-tune dataset.
        result = apply_learned(project, [], learned)
        save_project(project, path)
        if project.tests:  # refresh the build bundle if one exists
            write_build_bundle(project.spec, project.tests, path)
    result.examples_recorded = saved.examples_recorded  # what phase 1 actually wrote

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


def _merged_name(out: Path) -> str:
    """Name the merged project after its destination directory.

    `.` and `..` are this CLI's own idiom for "here", and `Path.name` is empty or
    unusable for both — so resolve first, exactly as `import` does."""
    return out.resolve().name or "project"


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
    from .stakeholders import (build_merged_spec, conflict_dict, detect_conflicts, gather,
                               scalar_conflicts)

    if len(sources) < 2:
        typer.secho("Need at least two --from projects to merge.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    # Validate the destination up front: project_lock() mkdirs it, which raises a
    # raw FileExistsError when the path is an existing FILE — and it would do so
    # AFTER the interactive reconciliation loop, discarding every ruling typed.
    # --report-only writes nothing, so the destination is none of its business:
    # checking it there makes a read-only conflict report impossible the moment
    # the merged project exists, which is exactly when you want to re-read it.
    if not report_only:
        if out.exists() and not out.is_dir():
            typer.secho(f"A file named {out} already exists — pick another destination.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if (out / "project.yaml").exists():
            typer.secho(f"A project already exists at {out}/.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        # The merged project is NAMED after the destination, and that name is
        # validated when the Project is built — after the loop. Check it here.
        from .models import validate_project_name
        try:
            validate_project_name(_merged_name(out))
        except ValueError as exc:
            typer.secho(f"Can't name the merged project after {out} — {exc}.", fg=typer.colors.RED)
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

    # Scalar behavior fields (voice, format, refusal policy) never reach the
    # engine's conflict detector — it only ever sees standards and never-rules —
    # so a disagreement there would otherwise be resolved silently and reported as
    # "no conflicts". Surface it, and record it in the audit file.
    scalars = scalar_conflicts(named)

    drops: set[int] = set()
    additions: list[str] = []
    audit: list[dict] = []
    if conflicts or scalars:
        typer.secho(f"\n{len(conflicts) + len(scalars)} conflict(s) found:", bold=True)
    else:
        typer.secho("\nNo conflicts found — merging cleanly.", fg=typer.colors.GREEN)

    for field, vals in scalars:
        typer.secho(f"\n[{field}] stakeholders disagree", fg=typer.colors.RED, bold=True)
        for n, v in vals:
            typer.echo(f"  [{n}]: {v}")
        typer.secho(f"  → keeping {vals[0][0]}'s value (first by stakeholder name); "
                    f"edit the merged spec to change it.", fg=typer.colors.YELLOW)

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
    # Read the resolution back off the merged spec instead of naming the first
    # stakeholder with a non-empty value: persona is resolved as one object, so
    # that guess would record a per-field ruling the merge never made. A field
    # whose winning value belongs to nobody in the list resolves to (none).
    resolved = {"persona.voice": spec.persona.voice, "persona.reading_level": spec.persona.reading_level,
                "format": spec.format, "refusal_policy": spec.refusal_policy}
    scalar_audit = [
        {"field": field, "values": [{"stakeholder": n, "value": v} for n, v in vals],
         "resolved_to": {"stakeholder": next((n for n, v in vals if v == resolved.get(field)), None),
                         "value": resolved.get(field)}}
        for field, vals in scalars
    ]
    merged = Project(name=_merged_name(out), goal=goal_final, task_type=first.task_type, spec=spec)
    _we_created = not out.exists()
    try:
        with project_lock(out):
            if (out / "project.yaml").exists():
                typer.secho(f"A project already exists at {out}/.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            save_project(merged, out)
            atomic_write_text(out / "reconciliation.yaml",
                              _yaml.safe_dump({"stakeholders": list(named), "conflicts": audit,
                                               "field_conflicts": scalar_audit},
                                              sort_keys=False, allow_unicode=True))
    except typer.Exit:
        _cleanup_empty_project_dir(out, _we_created)
        raise
    typer.secho(
        f"\n✓ Merged {len(named)} stakeholder(s) → {out}/  "
        f"({len(spec.standards)} standard(s), {len(spec.do_not)} never-rule(s); "
        f"{len(conflicts)} rule conflict(s) reconciled, {len(scalars)} field conflict(s) resolved).",
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
    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
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
    """Localize a cloud role onto your own model from logged decisions — the autonomy loop. (Advanced tier)"""
    from .train_engine import LOGGED_ROLES, TRAINABLE_ROLES, export_engine_bundle, prove_engine, read_log

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
            hint = ("Run `calibrate log --on`, then `calibrate eval`, then retry."
                    if role in LOGGED_ROLES else
                    "Only judge and compiler decisions are logged today, so there is nothing to replay.")
            typer.secho(f"No logged {role} decisions. {hint}", fg=typer.colors.YELLOW)
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
        if role in LOGGED_ROLES:
            typer.secho(
                f"No logged {role} decisions yet. Turn on logging (`calibrate log --on`), run `calibrate eval`, then retry.",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.secho(
                f"Nothing records the {role} role yet — only the judge (`calibrate eval`, `calibrate ci`) and the "
                "compiler (`calibrate eval --refine`) are logged, so there is no data to train on.",
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
    """Package the calibrated config into a runnable bundle (Ollama Modelfile + more)."""
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
    """Turn the spec's good/bad examples into regression tests — golden anchors. (no engine)"""
    from .compile import tests_from_examples, write_build_bundle

    _require_project(path)  # no junk .lock dir for a typo'd name
    with project_lock(path, on_wait=_lock_wait_notice):
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
    """Run the local API + web UI; open the printed URL in your browser."""
    _validate_port(port)  # 1..65535 (port 0 would print a wrong URL for a random bind)
    try:
        import uvicorn
        from .api import create_app, default_projects_root
    except (ImportError, RuntimeError):
        typer.secho("The API needs the `api` extra:  pip install -e '.[api]'  (in your ai-calibrator clone)",
                    fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Resolve to an absolute path so the startup banner is unambiguous from any
    # working directory (a bare relative "projects" reads as a mystery); the app
    # resolves it the same way for /api/health.
    root = (projects or default_projects_root()).resolve()
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
    try:
        uvicorn.run(application, host=host, port=port, log_level="warning")
    except OSError as exc:  # e.g. EADDRINUSE — friendly, not a raw traceback
        typer.secho(f"Could not bind {host}:{port} — {exc}. "
                    "The port may be in use; retry with a different --port.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def finetune(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    base: Optional[str] = typer.Option(None, "--base", help="Open base model to fine-tune."),
    gate: bool = typer.Option(False, "--gate", help="Compare two eval scorecards instead of building."),
    baseline: Optional[str] = typer.Option(None, "--baseline", help="Baseline run id (with --gate)."),
    candidate: Optional[str] = typer.Option(None, "--candidate", help="Candidate run id (with --gate)."),
) -> None:
    """Advanced tier: build a fine-tuning dataset + recipe, or run the prove-it gate."""

    project = _load(path)

    if gate:
        if not (baseline and candidate):
            typer.secho("--gate needs --baseline <run-id> and --candidate <run-id>.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        from .finetune import beats_baseline, held_out_rate, training_overlap

        base_card, cand_card = _scorecard_or_exit(path, baseline), _scorecard_or_exit(path, candidate)
        typer.echo(f"baseline [{baseline}]: {pct(base_card.pass_rate)}    "
                   f"candidate [{candidate}]: {pct(cand_card.pass_rate)}")

        # The dataset is built from spec.examples, and examples-to-tests / absorb
        # turn those same examples into ex_*/fb_* tests — so the headline rate can
        # include prompts the model trained on, which a memorizing fine-tune passes
        # by construction. Decide the gate on the HELD-OUT tests only.
        overlap = training_overlap(project, cand_card)
        graded = [r for r in cand_card.results if r.criteria]
        if overlap:
            excl = set(overlap)
            cand_held, n_held = held_out_rate(cand_card, excl)
            base_held, n_base = held_out_rate(base_card, excl)
            typer.secho(f"⚠ {len(overlap)} of {len(graded)} graded test(s) use an input that is "
                        "also a TRAINING prompt "
                        f"({', '.join(overlap[:5])}{', …' if len(overlap) > 5 else ''}).",
                        fg=typer.colors.YELLOW, bold=True)
            if n_held == 0:
                typer.secho("✗ CANNOT JUDGE — every graded test is a training prompt, so this "
                            "comparison cannot distinguish learning from memorization.",
                            fg=typer.colors.RED, bold=True)
                typer.echo("  Add tests whose inputs are NOT in the dataset "
                           "(`calibrate redteam --add-tests`, or write them by hand), re-run both "
                           "evals, and gate again.")
                raise typer.Exit(code=2)
            # held_out_rate reports 0.0 for "nothing to score" — reading that as a
            # measured baseline rate hands the candidate an automatic win off a
            # comparison that never happened (a baseline run that predates these
            # tests never graded one of them).
            if n_base == 0:
                typer.secho("✗ CANNOT JUDGE — the baseline run graded none of the held-out "
                            "test(s), so there is nothing to compare the candidate against.",
                            fg=typer.colors.RED, bold=True)
                typer.echo(f"  Re-run the baseline eval ({baseline}) on the CURRENT test suite, "
                           "then gate again.")
                raise typer.Exit(code=2)
            if n_base != n_held:
                typer.secho(f"⚠ The two runs graded different held-out tests ({n_base} in the "
                            f"baseline, {n_held} in the candidate) — the rates below aren't over "
                            "the same tests. Re-run the baseline eval on the CURRENT suite.",
                            fg=typer.colors.YELLOW)
            typer.echo(f"  gating on the {n_held} held-out test(s): "
                       f"baseline {pct(base_held)} → candidate {pct(cand_held)}")
            win = cand_held > base_held
        else:
            win = beats_baseline(base_card, cand_card)

        def _prov(card):
            bits = [f"subject={card.subject}"] if card.subject else []
            if card.judge:
                bits.append(f"judge={card.judge}")
            return "  (" + ", ".join(bits) + ")" if bits else ""
        typer.echo(f"  baseline {_prov(base_card).strip()}")
        typer.echo(f"  candidate {_prov(cand_card).strip()}")
        # A comparison is only meaningful if the SAME judge graded both runs.
        if base_card.judge and cand_card.judge and base_card.judge != cand_card.judge:
            typer.secho(f"⚠ Different judges graded these runs ({base_card.judge} vs "
                        f"{cand_card.judge}) — the comparison isn't apples-to-apples.",
                        fg=typer.colors.YELLOW)
        if base_card.partial or cand_card.partial:
            typer.secho("⚠ One of these scorecards is PARTIAL (an interrupted or --max-tests "
                        "run) — re-run a full eval before trusting the gate.", fg=typer.colors.YELLOW)
        if win:
            typer.secho("✓ ACCEPT — the fine-tune beats the configured baseline. Keep it.", fg=typer.colors.GREEN)
        else:
            typer.secho("✗ REJECT — it doesn't beat the baseline. Stay on configuration.", fg=typer.colors.YELLOW)
        # 0 = accept, 2 = a clean REJECT (distinct from 1 = an error such as a
        # missing/unreadable scorecard, which exits 1 via _scorecard_or_exit).
        raise typer.Exit(code=0 if win else 2)

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
    if result.excluded_engine:
        # Say why they were left out: a model trained on its own synthesized
        # answers learns nothing new, so those rows are not training targets.
        typer.secho(
            f"  {result.excluded_engine} compiler-written example(s) excluded — a model "
            "trained on its own output learns nothing. Import your own examples "
            "(`calibrate examples --import`) or capture corrections with `calibrate teach`.",
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
        "\nNext: run it in one step —  calibrate train  (installs prereqs + trains),\n"
        "or train manually on a GPU (see finetune/README.md). Then prove it wins:\n"
        "  calibrate finetune --gate --baseline <run> --candidate <run>"
    )


def _version_tuple(v: str) -> tuple[int, int, int]:
    """Leading (major, minor, patch) of a version string, for a floor comparison.

    Deliberately tolerant — '2.2.0+cu121' and '1.0.0rc1' have to compare, and a
    component it cannot read counts as 0 rather than derailing an install."""
    parts = []
    for chunk in (v.split(".") + ["0", "0", "0"])[:3]:
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits or 0))
    return (parts[0], parts[1], parts[2])


def _dep_satisfied(module: str, requirement: str) -> bool:
    """True when ``module`` is importable AND meets the >= floor in ``requirement``.

    find_spec only answers "is it importable", so on its own an already-present
    but too-old transformers or trl is neither reported nor upgraded, and the
    generated trainer then fails on an argument that release doesn't have. A
    version that can't be read counts as satisfied: trusting what is installed
    beats forcing a reinstall on a guess."""
    import importlib.metadata
    import importlib.util
    if importlib.util.find_spec(module) is None:
        return False
    floor = requirement.partition(">=")[2].strip()
    if not floor:
        return True
    try:
        have = importlib.metadata.version(module)
    except Exception:
        return True
    return _version_tuple(have) >= _version_tuple(floor)


@app.command()
def train(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    base: Optional[str] = typer.Option(None, "--base", help="Open base model (e.g. Qwen/Qwen2.5-3B-Instruct)."),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Training epochs (default: 5 for <50 examples, else 3)."),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Cap total training steps (a fast smoke run)."),
    qlora: bool = typer.Option(False, "--qlora", help="Load the base in 4-bit (CUDA + bitsandbytes only) — fits a smaller GPU."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Install missing training dependencies without prompting."),
) -> None:
    """Fine-tune your calibrated AI locally, in ONE step: build the bundle, install
    the prereqs (with your OK), and run training. (Advanced tier)

    Detects your hardware (CUDA / Apple-Silicon MPS / CPU), installs the training
    stack if it's missing, downloads the base model, and trains a LoRA adapter.
    Then prove it actually helps with `calibrate finetune --gate` — never rely on a
    fine-tune that doesn't beat your prompt+RAG baseline."""
    import importlib.util
    import os
    import subprocess

    project = _load(path)
    if project.spec is None:
        typer.secho("Nothing to fine-tune — run `calibrate compile` first.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    # 1. build / refresh the bundle (dataset + recipe + train.py)
    if epochs is not None and epochs < 1:
        typer.secho("--epochs must be >= 1.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if max_steps is not None and max_steps < 1:
        typer.secho("--max-steps must be >= 1.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    from .finetune import export_finetune
    try:
        result = export_finetune(project, project_dir=path, base_model=base,
                                 epochs=epochs, max_steps=max_steps)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if result.examples == 0:
        typer.secho("No training examples — add examples (or capture eval corrections) first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if result.examples < 20:
        typer.secho(f"⚠ Only {result.examples} example(s). A fine-tune usually needs 50+ to beat the "
                    "prompt+RAG baseline — the gate may well REJECT this. Training anyway.",
                    fg=typer.colors.YELLOW)
    typer.echo(f"Bundle: {result.bundle_dir}/  ({result.examples} example(s), {result.method} on {result.base_model})")
    # Estimate the work up front so a long run isn't a surprise. batch=1,
    # grad-accum=8 → one optimizer step per 8 examples per epoch.
    import yaml as _yaml
    recipe = _yaml.safe_load((Path(result.bundle_dir) / "recipe.yaml").read_text(encoding="utf-8"))
    if recipe.get("max_steps", -1) and recipe["max_steps"] > 0:
        typer.echo(f"Plan: up to {recipe['max_steps']} step(s) (--max-steps cap).")
    else:
        est = max(1, (result.examples + 7) // 8) * int(recipe.get("epochs", 3))
        typer.echo(f"Plan: ~{est} optimizer step(s) over {recipe.get('epochs')} epoch(s).")

    # 2. ensure the training stack (offered, not forced)
    # Install the REQUIREMENT STRINGS the `train` extra declares, not bare module
    # names: find_spec only answers "is it importable", so an already-present but
    # too-old transformers/trl/peft was neither detected nor upgraded — and the
    # generated trainer then fails on an argument that release doesn't have.
    # _dep_satisfied compares the INSTALLED version against each floor, so an
    # outdated module is upgraded and not just an absent one.
    _TRAIN_REQS = {
        "torch": "torch>=2.2", "transformers": "transformers>=4.46", "trl": "trl>=1.0",
        "peft": "peft>=0.11", "datasets": "datasets>=2.19", "accelerate": "accelerate>=0.30",
    }
    stale = [m for m, req in _TRAIN_REQS.items() if not _dep_satisfied(m, req)]
    need = [_TRAIN_REQS[m] for m in stale]
    if qlora and importlib.util.find_spec("bitsandbytes") is None:
        need.append("bitsandbytes")
    if need:
        typer.secho(f"\nTraining needs: {', '.join(stale) or 'bitsandbytes'}  "
                    "(torch is a large download — several GB).", fg=typer.colors.YELLOW)
        typer.echo("  Installing with the version floors the bundle's train.py requires.")
        typer.echo("  (or install once yourself, in your ai-calibrator clone:  pip install -e '.[train]')")
        if not yes and not typer.confirm("  Install them now?", default=True):
            raise typer.Exit(code=1)
        if subprocess.run([sys.executable, "-m", "pip", "install", *need]).returncode != 0:
            typer.secho("Dependency install failed — install manually and retry.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    # 3. hardware
    import torch
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    if qlora and device != "cuda":
        typer.secho("--qlora needs a CUDA GPU (bitsandbytes) — training in full precision instead.",
                    fg=typer.colors.YELLOW)
        qlora = False
    typer.secho(f"\nTraining on {device.upper()} — first run downloads the base model …", fg=typer.colors.CYAN)

    # 4. run the tool's generated, device-aware train.py
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "TOKENIZERS_PARALLELISM": "false"}
    if qlora:
        env["QLORA"] = "1"
    if subprocess.run([sys.executable, "train.py"], cwd=result.bundle_dir, env=env).returncode != 0:
        typer.secho("\nTraining failed (see the output above). Common fixes: fewer GB → smaller --base "
                    "or --qlora (CUDA); on Apple Silicon a 7B needs ~24GB+ unified memory.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    adapter = Path(result.bundle_dir) / "adapter"
    typer.secho(f"\n✓ Trained a LoRA adapter → {adapter}/", fg=typer.colors.GREEN)
    typer.echo(
        "\nNext — PROVE it helps before relying on it (a fine-tune isn't automatically better):\n"
        "  1. serve the fine-tuned model (merge the adapter + `ollama create`, or an endpoint)\n"
        "  2. point the project's `subject` engine at it and run `calibrate eval`\n"
        "  3. calibrate finetune --gate --baseline <pre-ft-run> --candidate <new-run>\n"
        "     — keep it ONLY if the gate says it beats your baseline."
    )


@app.command()
def examples(
    path: Path = typer.Argument(Path("."), help="Project directory."),
    import_file: Optional[Path] = typer.Option(
        None, "--import", help="Bulk-import input/output pairs from a .csv / .jsonl / .json / .yaml file."),
    dedup: bool = typer.Option(False, "--dedup", help="Remove duplicate examples (same input)."),
) -> None:
    """Review, import, and dedup training examples — the fuel for fine-tuning. (no engine)

    Most owners already HAVE examples (past replies, an FAQ, a spreadsheet). Import
    them in one shot: `calibrate examples --import my-qa.csv`. Column/key names are
    matched flexibly (input/question/prompt…, good_output/output/answer…)."""
    from .examples_io import dedup_examples, examples_status, load_examples_report, merge_examples

    if _load(path).spec is None:
        # Examples attach to the behavior spec, so the Guided loop must run first.
        typer.secho(
            "No spec yet — examples attach to the behavior spec, so build it first:\n"
            "  calibrate ingest  →  calibrate interview  →  calibrate compile\n"
            "then re-run this import.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    if import_file is not None:
        try:
            report = load_examples_report(import_file)
        except ValueError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=1)
        _require_project(path)  # no junk .lock dir for a typo'd name
        with project_lock(path, on_wait=_lock_wait_notice):                 # load→mutate→save under the lock (fresh reload)
            project = _load(path)
            added, skipped = merge_examples(project.spec, report.examples)
            save_project(project, path)
        msg = f"✓ Imported {added} example(s) from {import_file.name}"
        typer.secho(msg + (f" ({skipped} duplicate(s) skipped)." if skipped else "."), fg=typer.colors.GREEN)
        if report.without_output:
            typer.secho(f"  · {report.without_output} imported without an output "
                        "(input only — add answers before fine-tuning).", fg=typer.colors.CYAN)
        if report.skipped:
            typer.secho(f"  ⚠ skipped {len(report.skipped)} malformed row(s):", fg=typer.colors.YELLOW)
            for s in report.skipped[:10]:
                typer.echo(f"    · {s}")
            if len(report.skipped) > 10:
                typer.echo(f"    · … and {len(report.skipped) - 10} more")

    if dedup:
        _require_project(path)  # no junk .lock dir for a typo'd name
        with project_lock(path, on_wait=_lock_wait_notice):
            project = _load(path)
            removed = dedup_examples(project.spec)
            save_project(project, path)
        typer.secho(f"✓ Removed {removed} duplicate example(s)." if removed else "No duplicates to remove.",
                    fg=typer.colors.GREEN)

    project = _load(path)
    st = examples_status(project.spec)
    dupe_note = f" ({st['duplicates']} duplicate(s) — `--dedup` to clean)" if st["duplicates"] else ""
    typer.secho(f"\n{st['unique_inputs']} unique training example(s){dupe_note}; "
                f"{st['with_output']} with an output.", bold=True)
    if st["enough_to_finetune"]:
        typer.secho(f"✓ Enough to try a fine-tune (recommended ≥{st['recommended']}). Next:  calibrate train",
                    fg=typer.colors.GREEN)
    else:
        typer.secho(f"→ {st['short_by']} more for a solid fine-tune (recommended ≥{st['recommended']}). "
                    "Add via --import <file>, `calibrate teach`, or captured eval corrections.",
                    fg=typer.colors.CYAN)
    for ex in project.spec.examples[:6]:
        out = (ex.good_output or "")[:60]
        typer.echo(f"  • {ex.input[:60]}" + (f"  →  {out}" if out else "  (no output yet)"))
    if len(project.spec.examples) > 6:
        typer.echo(f"  … and {len(project.spec.examples) - 6} more")


def main() -> None:
    # A limited terminal encoding (ascii / cp1252 console) must degrade glyphs
    # (✓ ⚠ →) to '?', never crash — Rich's --help rendering raised a raw
    # UnicodeEncodeError otherwise.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError):  # non-reconfigurable stream (tests, pipes)
            pass
    app()


if __name__ == "__main__":
    main()
