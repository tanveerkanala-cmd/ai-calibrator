"""M6 — Local HTTP API over the Calibration Core (localhost only).

A thin FastAPI layer: every endpoint maps to a core pipeline call, so the same
logic that powers the CLI also drives the web/desktop UI. Engine-dependent
endpoints build the project's configured engine; an upstream engine failure
surfaces as 502/504 (see ``_engine_http_error``), a bad request as 400.

Run with `calibrate serve`. Needs the `api` extra:  pip install -e '.[api]'
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import yaml

try:
    from fastapi import Depends, FastAPI, HTTPException, UploadFile
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field, ValidationError
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise RuntimeError(
        "The API needs the `api` extra:  pip install -e '.[api]'  (in your ai-calibrator clone)"
    ) from exc

from .auth import all_status
from .models import EngineBinding, Project, TaskType
from .store import load_project, project_lock, save_project, write_project_gitignore
from .webguard import MAX_BODY_BYTES, install_guard

WEB_DIR = Path(__file__).parent / "web"


def default_projects_root() -> Path:
    base = os.getenv("CALIBRATOR_HOME") or (Path.home() / ".ai-calibrator")
    return Path(base) / "projects"


class CreateProjectBody(BaseModel):
    name: str
    goal: str
    task_type: TaskType = TaskType.ASSISTANT


class AnswersBody(BaseModel):
    answers: dict[str, str]


class EvalBody(BaseModel):
    refine: bool = False
    # Validated at the boundary → a bad value is a clean 422, not a 500 deep in
    # the loop. allow_inf_nan rejects NaN/Inf (against which the threshold check
    # would silently misbehave).
    rounds: int = Field(default=3, ge=1, le=100)
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, allow_inf_nan=False)
    judge_passes: int = Field(default=1, ge=1, le=9)  # self-consistency: majority-vote over N judge calls


class RedTeamBody(BaseModel):
    max_probes: int = Field(default=12, ge=1, le=50)
    add_tests: bool = False


class RightsizeBody(BaseModel):
    models: list[str] = Field(default_factory=list)  # empty → default Claude ladder
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, allow_inf_nan=False)


class DriftBody(BaseModel):
    baseline: str | None = None  # None → latest saved scorecard
    tolerance: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)


class TeachDraftBody(BaseModel):
    n: int = Field(default=5, ge=1, le=20)


class JudgedItem(BaseModel):
    input: str
    output: str
    approved: bool
    reason: str | None = None


class TeachLearnBody(BaseModel):
    judgments: list[JudgedItem]


class LogBody(BaseModel):
    enabled: bool


class ExamplesBody(BaseModel):
    examples: list[dict] = Field(default_factory=list)   # [{input, good_output?, ...}] — flexible keys
    dedup: bool = True


class MergeDetectBody(BaseModel):
    sources: list[str]


class MergeApplyBody(BaseModel):
    out: str
    sources: list[str]
    goal: str | None = None
    drops: list[int] = Field(default_factory=list)
    additions: list[str] = Field(default_factory=list)


class ImportBody(BaseModel):
    name: str
    goal: str
    prompt: str
    task_type: TaskType = TaskType.ASSISTANT
    engine: str | None = None


class DiffBody(BaseModel):
    before: str
    after: str


class JudgeLabelsBody(BaseModel):
    labels: list[dict] = Field(default_factory=list)  # [{test_id, criterion_id, passed}]
    run_id: str | None = None  # None → latest


class CiBody(BaseModel):
    threshold: float = Field(0.8, ge=0.0, le=1.0)
    tolerance: float = Field(0.0, ge=0.0, le=1.0)  # le bounds it AND rejects inf (a rate delta is 0..1)
    judge_passes: int = Field(1, ge=1, le=9)
    baseline: str | None = None


class EnginesBody(BaseModel):
    role: str | None = None        # set one role, or...
    model: str | None = None       # ...its model@provider
    all: str | None = None         # ...set every role to this model@provider


class TryBody(BaseModel):
    message: str = Field(..., min_length=1, max_length=100_000)


class FeedbackBody(BaseModel):
    turns: list[str] = Field(..., min_length=1)
    output: str = Field(..., min_length=1)
    verdict: str  # "up" | "down" (validated in the endpoint for a friendly 400)
    correction: str | None = None
    reason: str | None = None


def _engine_factory():
    """Dependency: returns the engine builder (overridable in tests)."""
    from .engines import get_engine
    return get_engine


def _engine_http_error(exc: Exception) -> "HTTPException":
    """Map an engine/provider failure to the right HTTP status class.

    An upstream engine timeout is a gateway timeout (504) and any other engine /
    provider failure is a bad gateway (502) — NOT a client 400, so an API client
    branching on 4xx-vs-5xx retries/blames correctly. Anything that isn't an
    engine failure stays a 400 (a genuinely bad request)."""
    from .engines.base import EngineError, EngineTimeout
    if isinstance(exc, EngineTimeout):
        return HTTPException(504, str(exc))
    if isinstance(exc, EngineError):
        return HTTPException(502, str(exc))
    return HTTPException(400, str(exc))


# The material-upload ceiling IS the request-body ceiling: an upload is a request
# body, and one number is easier to reason about than two that must agree.
MAX_UPLOAD_BYTES = MAX_BODY_BYTES

# How long a request waits for a contended project lock before returning 423.
# Long enough that quick concurrent writes serialize cleanly; short enough that a
# request behind a minutes-long engine op fails fast instead of hanging.
_LOCK_WAIT_SECONDS = 10.0


def create_app(projects_root: Path | None = None, allowed_hosts: list[str] | None = None) -> "FastAPI":
    # Resolve to an absolute path so the banner / health endpoint are unambiguous
    # from any working directory (a bare relative "projects" reads as a mystery).
    root = (Path(projects_root) if projects_root else default_projects_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    from . import __version__
    app = FastAPI(title="AI Calibrator", version=__version__)

    # Always enforced — never fully disabled. Installs the body cap as well as
    # the Host/Origin guard, and is shared with `calibrate run` (runtime.py) so
    # neither server can end up with one and not the other. See webguard.py.
    install_guard(app, allowed_hosts, max_body_bytes=MAX_UPLOAD_BYTES)

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request, exc: RequestValidationError):
        # A body with NaN/Infinity is REJECTED by our `allow_inf_nan=False`
        # fields (intended: a clean 422). But the default handler echoes the raw
        # input in the error detail, and serializing NaN then raises inside
        # Starlette (json with allow_nan=False) → an unhandled 500. Sanitize
        # non-finite floats so the rejection surfaces as the 422 it should be.
        def _san(o):
            if isinstance(o, float) and not math.isfinite(o):
                return str(o)  # "nan" / "inf" / "-inf" — JSON-safe
            if isinstance(o, dict):
                return {k: _san(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_san(v) for v in o]
            return o
        return JSONResponse(status_code=422, content={"detail": _san(jsonable_encoder(exc.errors()))})

    def _safe(name: str) -> str:
        # Reject an invalid name rather than silently rewriting it — otherwise a
        # client that POSTs "../evil" gets back a project called "evil" and every
        # follow-up call keyed on its submitted name 404s. Same rules as the CLI.
        from .models import validate_project_name
        try:
            return validate_project_name(name)
        except ValueError as exc:
            raise HTTPException(400, f"invalid project name: {exc}")

    def _dir(name: str) -> Path:
        return root / _safe(name)

    def _load(name: str) -> Project:
        try:
            return load_project(_dir(name))
        except FileNotFoundError:
            raise HTTPException(404, f"no project {name!r}")
        except (yaml.YAMLError, ValidationError, ValueError):
            # Corrupt/invalid project.yaml on disk → a clean 400, never a 500.
            raise HTTPException(400, f"project {name!r} is invalid or corrupted")

    @contextmanager
    def _held(d: Path, name: str):
        """Hold a project's exclusive lock, or give up with 423.

        Poll the lock briefly rather than blocking indefinitely: quick concurrent
        writes (answers, engine rebinds) serialize within the window and don't
        lose updates, but a request stuck behind a multi-minute engine op fails
        fast with 423 instead of hanging the connection — and, with it, one of
        the threadpool slots the whole server shares.

        Takes the directory rather than the routing key so the routes that must
        run BEFORE a project exists (create, import, merge) get the same bounded
        wait as the rest; ``_locked`` layers the 404 on top for the routes that
        require an existing project."""
        import time

        from .locking import LockBusy
        lock = project_lock(d, blocking=False)
        deadline = time.monotonic() + _LOCK_WAIT_SECONDS
        while True:
            try:
                lock.acquire()
                break
            except LockBusy:
                if time.monotonic() >= deadline:
                    raise HTTPException(
                        423, f"an operation is already in progress on project {name!r} — retry shortly")
                time.sleep(0.05)
        try:
            yield d
        finally:
            lock.release()

    @contextmanager
    def _locked(name: str):
        """Serialize a project's read-modify-write across concurrent requests.

        Yields the project directory with its exclusive lock held, so the
        enclosed ``load → mutate → save`` can't interleave with another request
        on the same project (which would lose updates or collide on run ids).
        404s before locking, so it never creates a stray dir for a missing
        project."""
        d = _dir(name)
        if not (d / "project.yaml").exists():
            raise HTTPException(404, f"no project {name!r}")
        with _held(d, name):
            yield d

    def _state(p: Project, key: str | None = None) -> dict:
        """Project state for the web UI.

        ``name`` is the ROUTING key — the directory component every route resolves
        through — not the project's stored ``name`` field. The two diverge whenever
        a project folder is copied or renamed, and the UI builds every subsequent
        request URL from this value: publishing the stored name there makes the UI
        show one project and mutate another. ``display_name`` keeps the stored one
        for headings."""
        return {
            "name": key if key is not None else p.name,
            "display_name": p.name,
            "goal": p.goal,
            "task_type": p.task_type.value,
            "materials": [m.model_dump(mode="json") for m in p.materials],
            "gaps": [g.model_dump(mode="json") for g in p.gaps],
            "interview": [it.model_dump(mode="json") for it in p.interview],
            "has_spec": p.spec is not None,
            "tests": len(p.tests),
            "engines": p.engines.model_dump(),
            "log_interactions": p.log_interactions,
        }

    @app.get("/api/health")
    def health():
        return {"ok": True, "projects_root": str(root)}  # absolute (root is resolved)

    @app.get("/api/auth")
    def auth():
        return [s.__dict__ for s in all_status()]

    @app.get("/api/projects")
    def list_projects():
        if not root.exists():
            return []
        return sorted(d.name for d in root.iterdir() if (d / "project.yaml").exists())

    @app.post("/api/projects")
    def create_project(body: CreateProjectBody):
        name = _safe(body.name)  # canonical: stored name == directory name == routing key
        d = root / name
        project = Project(name=name, goal=body.goal, task_type=body.task_type)
        # Answer the permanent condition before waiting on the lock: a name that
        # already exists will still exist in ten seconds, so making the caller
        # wait out the lock window to be told "retry shortly" is both slow and
        # wrong. The in-lock re-check below is what makes the create atomic.
        if (d / "project.yaml").exists():
            raise HTTPException(409, "project already exists")
        # Atomic create: hold the project lock across the exists-check + write so
        # two concurrent POSTs for the same name can't both pass the check (one
        # wins with 200, the other deterministically gets 409 — never a partial
        # write or a 500).
        with _held(d, name):
            if (d / "project.yaml").exists():
                raise HTTPException(409, "project already exists")
            save_project(project, d)
            write_project_gitignore(d)
        return _state(project, name)

    @app.get("/api/projects/{name}")
    def get_project(name: str):
        return _state(_load(name), name)

    @app.delete("/api/projects/{name}")
    def delete_project(name: str):
        """Remove a project the web UI / API created (a mistyped or throwaway one)
        — the delete half of the CRUD surface. Serialized via the project lock so
        it can't race an in-flight operation on the same project."""
        import shutil

        d = _dir(name)
        if not (d / "project.yaml").exists():
            raise HTTPException(404, f"no project {name!r}")
        # The lock file lives INSIDE the tree being deleted, and the two platforms
        # fail that differently — so each gets the approach that is correct for it.
        #
        # POSIX: unlink succeeds even on an open file, so deleting in place
        # destroys the lock that is providing the mutual exclusion. A waiter then
        # creates a fresh `.lock` at the same path and acquires it while this
        # delete is still running, and two holders proceed at once. Renaming the
        # tree aside takes the lock file with it, so this request keeps holding
        # the same inode and a waiter correctly finds no project.
        #
        # Windows: that same rename is refused (the open handle blocks renaming
        # the directory), and it is refused for the reason that makes in-place
        # deletion safe there — the handle also blocks unlinking `.lock`, so the
        # exclusion survives the rmtree. Delete in place, then clean up the
        # leftover lock file once the handle is closed.
        if os.name == "nt":  # pragma: no cover - exercised on the Windows CI leg
            from .store import LOCK_FILE
            with _held(d, name):
                shutil.rmtree(d, ignore_errors=True)
            # Retry ONLY when that stale lock file (or nothing) is all that is
            # left: releasing the lock lets a concurrent create rebuild a project
            # under this name, and a blind second rmtree would destroy it.
            try:
                leftovers = {p.name for p in d.iterdir()}
            except OSError:
                leftovers = set()
            if d.exists() and leftovers <= {LOCK_FILE}:
                shutil.rmtree(d, ignore_errors=True)
            remains = d
        else:
            stash = d.parent / f".{d.name}.deleting-{os.getpid()}-{threading.get_ident()}"
            with _held(d, name):
                try:
                    os.replace(d, stash)
                except OSError as exc:
                    raise HTTPException(409, f"could not delete {name!r}: {exc}. Check permissions "
                                             "or another process holding its files open.")
            shutil.rmtree(stash, ignore_errors=True)
            remains = stash
        # ignore_errors swallows every OSError, so verify rather than assume: a
        # delete that removed nothing (or half the tree) must not report success
        # while the project's uploaded documents are still on disk.
        if remains.exists():
            raise HTTPException(409, f"could not fully delete {name!r} — files remain at {remains}. "
                                     "Check permissions or another process holding them open.")
        return {"deleted": name}

    @app.delete("/api/projects/{name}/materials/{filename}")
    def delete_material(name: str, filename: str):
        """Remove one uploaded material file. The filename is basename-only, so it
        can't traverse outside the project's materials/ directory."""
        d = _dir(name)
        if not (d / "project.yaml").exists():
            raise HTTPException(404, f"no project {name!r}")
        target = d / "materials" / Path(filename).name
        if not target.is_file():
            raise HTTPException(404, f"no material {filename!r}")
        with _held(d, name):
            target.unlink(missing_ok=True)
        return {"deleted": filename}

    @app.post("/api/projects/{name}/materials")
    async def upload_material(name: str, file: UploadFile):
        d = _dir(name)
        if not (d / "project.yaml").exists():
            raise HTTPException(404, f"no project {name!r}")
        mats = d / "materials"
        mats.mkdir(exist_ok=True)
        # `.name` already defeats traversal, but it leaves three names that are not
        # files: "" (from "."), ".." (the project dir), and anything past the
        # filesystem's component limit. Each one reaches os.replace below and comes
        # back as a 500 + traceback; every other bad input on this API is a clean 4xx.
        base = Path(file.filename or "upload.txt").name
        if base in ("", ".", "..") or len(base.encode("utf-8")) > 200:
            raise HTTPException(400, "invalid filename — give the file a plain name under 200 bytes")
        target = mats / base
        # Stream to a unique temp file, then atomically rename into place. A
        # disconnect mid-upload leaves no partial file, and two concurrent uploads
        # of the same name can't interleave bytes — the last complete one wins
        # cleanly instead of corrupting the target. The size cap is _BodyLimit's:
        # by the time this runs the body has already been received in full.
        fd, tmp_name = tempfile.mkstemp(dir=str(mats), prefix=".upload-", suffix=".tmp")
        tmp = Path(tmp_name)
        ours = True  # we own the temp file until the rename takes it
        try:
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            try:
                os.replace(tmp, target)
            except OSError as exc:  # name the filesystem still rejects → 4xx, not a 500
                raise HTTPException(400, f"could not store {base!r}: {exc.strerror or exc}") from exc
            ours = False
        finally:
            if ours:
                tmp.unlink(missing_ok=True)
        return {"uploaded": target.name}

    @app.post("/api/projects/{name}/ingest")
    def ingest(name: str, make_engine=Depends(_engine_factory)):
        from .ingest import ingest_project
        with _locked(name) as d:
            project = _load(name)
            materials = d / "materials"
            # An empty folder on a project that HAS ingested materials is a
            # deliberate "remove everything": let it through so the facts, gaps and
            # retrieval index built from the deleted files are cleared too. Only a
            # project that never had any still gets the "add some first" message.
            if (not materials.exists() or not any(materials.iterdir())) and not project.materials:
                raise HTTPException(400, f"No materials to ingest — POST files to /api/projects/{name}/materials first.")
            try:
                engine = make_engine(project.engines.extractor)
                result = ingest_project(project, materials, engine, project_dir=d)
            except HTTPException:
                raise
            except Exception as exc:
                raise _engine_http_error(exc)
            save_project(project, d)
        return {
            "materials": result.materials, "gaps": result.gaps,
            "facts": result.facts, "indexed": result.indexed,
            # How many parsed files actually fit the extractor's context window;
            # fewer than `materials` means the gap list saw only those.
            "analyzed": result.analyzed,
            "skipped": [{"path": rel, "reason": reason} for rel, reason in result.skipped],
            "state": _state(project, name),
        }

    @app.post("/api/projects/{name}/interview")
    def interview(name: str, make_engine=Depends(_engine_factory)):
        from .interview import generate_questions, uncovered_gaps
        with _locked(name) as d:
            project = _load(name)
            if not project.gaps:
                raise HTTPException(400, f"No gaps yet — POST /api/projects/{name}/ingest first.")
            try:
                engine = make_engine(project.engines.interviewer)

                # Persist after each gap so a timeout keeps partial progress.
                def _progress(items, done, total):
                    project.interview = list(items)
                    save_project(project, d)

                project.interview = generate_questions(project, engine, on_progress=_progress)
            except HTTPException:
                raise
            except Exception as exc:
                raise _engine_http_error(exc)
            save_project(project, d)
        return {**_state(project, name), "uncovered_gaps": uncovered_gaps(project, project.interview)}

    @app.post("/api/projects/{name}/answers")
    def submit_answers(name: str, body: AnswersBody):
        with _locked(name) as d:
            project = _load(name)
            # Apply to EVERY item whose id matches, exactly as the CLI does — a
            # dict-by-id keeps only the last of any duplicate-id items (possible
            # via a hand-edited project.yaml), silently dropping an answer the
            # owner gave.
            applied = 0
            for it in project.interview:
                if it.id in body.answers:
                    it.answer = body.answers[it.id]
                    applied += 1
            save_project(project, d)
        return {"applied": applied, "state": _state(project, name)}

    @app.post("/api/projects/{name}/examples")
    def add_examples(name: str, body: ExamplesBody):
        """Bulk-add training examples (the fuel for the Advanced tier). Same
        flexible shapes as `calibrate examples --import` — but from the request
        body, never a server file path (no path-traversal surface)."""
        from .examples_io import _row_to_example, examples_status, merge_examples
        new = [ex for row in body.examples if isinstance(row, dict) and (ex := _row_to_example(row))]
        with _locked(name) as d:
            project = _load(name)
            if project.spec is None:
                raise HTTPException(400, f"No spec yet — POST /api/projects/{name}/compile first (examples attach to the spec).")
            spec = project.spec
            added, skipped = merge_examples(spec, new, dedup=body.dedup)
            save_project(project, d)
        return {"added": added, "skipped": skipped, **examples_status(spec)}

    @app.post("/api/projects/{name}/compile")
    def compile_(name: str, make_engine=Depends(_engine_factory)):
        from .compile import compile_project
        with _locked(name) as d:
            project = _load(name)
            # A merged project has a spec but no interview; compile preserves an
            # existing spec, so gating on the interview alone made merge a dead end.
            if not any(it.answer for it in project.interview) and project.spec is None:
                raise HTTPException(400, f"No interview answers yet — POST /api/projects/{name}/interview, then submit answers.")
            try:
                engine = make_engine(project.engines.compiler)
                result = compile_project(project, engine, project_dir=d)
            except HTTPException:
                raise
            except Exception as exc:
                raise _engine_http_error(exc)
            save_project(project, d)
        return {"criteria": result.criteria, "tests": result.tests,
                "files": result.files, "state": _state(project, name)}

    @app.post("/api/projects/{name}/eval")
    def evaluate(name: str, body: EvalBody, make_engine=Depends(_engine_factory)):
        from .engine_log import wrap_engine
        from .eval import next_run_id, run_eval, save_scorecard
        with _locked(name) as d:
            project = _load(name)
            if project.spec is None or not project.tests:
                raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
            log_on = project.log_interactions
            try:
                subject = make_engine(project.engines.subject)
                judge = wrap_engine(make_engine(project.engines.judge), "judge", d, enabled=log_on)
                if body.refine:
                    from .compile import write_build_bundle
                    from .pipeline import calibrate_loop
                    refiner = wrap_engine(make_engine(project.engines.compiler), "compiler", d, enabled=log_on)

                    def _persist_spec(proj):
                        # Checkpoint each round's refinement BEFORE the next round is
                        # graded: scorecards are saved as they are earned, so saving the
                        # spec only after the loop would let a failed or abandoned
                        # request leave runs on disk that no recorded spec produced.
                        save_project(proj, d)
                        write_build_bundle(proj.spec, proj.tests, d)

                    cards = calibrate_loop(project, subject, judge, refiner,
                                           threshold=body.threshold, max_rounds=body.rounds,
                                           judge_passes=body.judge_passes, project_dir=d,
                                           on_spec_change=_persist_spec)
                    save_project(project, d)
                    write_build_bundle(project.spec, project.tests, d)  # refresh build/ to match
                else:
                    card = run_eval(project, subject, judge, run_id=next_run_id(d),
                                    judge_passes=body.judge_passes, project_dir=d)
                    save_scorecard(d, card)
                    cards = [card]
            except HTTPException:
                raise
            except Exception as exc:
                raise _engine_http_error(exc)
        return {"rounds": [
            {"run_id": c.run_id, "pass_rate": c.pass_rate, "weighted_score": c.weighted_score,
             "results": [r.model_dump(mode="json") for r in c.results]}
            for c in cards
        ]}

    @app.post("/api/projects/{name}/ci")
    def ci_(name: str, body: CiBody, make_engine=Depends(_engine_factory)):
        from .ci import ci_dict, run_ci
        from .engine_log import wrap_engine
        with _locked(name) as d:
            project = _load(name)
            if project.spec is None or not project.tests:
                raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
            try:
                # factories: engines resolve only after the lint stage passes
                subject = lambda: make_engine(project.engines.subject)  # noqa: E731
                judge = lambda: wrap_engine(make_engine(project.engines.judge), "judge", d,  # noqa: E731
                                            enabled=project.log_interactions)
                result = run_ci(project, subject, judge, project_dir=d, threshold=body.threshold,
                                tolerance=body.tolerance, judge_passes=body.judge_passes,
                                baseline=body.baseline)
            except HTTPException:
                raise
            except Exception as exc:
                raise _engine_http_error(exc)
        return ci_dict(result)

    @app.post("/api/projects/{name}/export")
    def export(name: str):
        from .export import export_bundle
        project = _load(name)
        if project.spec is None:
            raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
        try:
            result = export_bundle(project, project_dir=_dir(name))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"bundle_dir": result.bundle_dir, "name": result.name,
                "base_model": result.base_model, "files": result.files}

    @app.post("/api/projects/{name}/rightsize")
    def rightsize_(name: str, body: RightsizeBody, make_engine=Depends(_engine_factory)):
        from .rightsize import DEFAULT_LADDER, rightsize, rightsize_dict
        with _locked(name) as d:
            project = _load(name)
            if project.spec is None or not project.tests:
                raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
            specs = body.models or list(DEFAULT_LADDER)
            try:
                judge = make_engine(project.engines.judge)
                report = rightsize(project, specs, judge, make_engine,
                                   threshold=body.threshold, project_dir=d)
            except HTTPException:
                raise
            except Exception as exc:
                raise _engine_http_error(exc)
        return rightsize_dict(report)

    @app.post("/api/projects/{name}/try")
    def try_(name: str, body: TryBody, make_engine=Depends(_engine_factory)):
        """One exchange with the calibrated AI (subject engine + compiled prompt) —
        the workbench's 'Try & flag' box. Same encoding as the runtime/eval."""
        from . import rag
        from .compile import render_system_prompt
        from .eval import conversation_prompt
        project = _load(name)
        if project.spec is None:
            raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
        try:
            subject = make_engine(project.engines.subject)
            # Augment with retrieved knowledge when indexed, so the workbench "try"
            # shows the AI as DEPLOYED (matching `run`), not a prompt-only variant.
            eff_system = rag.augment_system(render_system_prompt(project.spec), _dir(name), body.message)
            output = str(subject.complete(conversation_prompt([], body.message),
                                          system=eff_system) or "").strip()
        except HTTPException:
            raise
        except Exception as exc:
            raise _engine_http_error(exc)
        return {"turns": [body.message], "output": output}

    @app.post("/api/projects/{name}/feedback")
    def feedback_(name: str, body: FeedbackBody):
        """Record thumbs-up/down from the workbench (same inbox the runtime feeds)."""
        from datetime import datetime, timezone

        from .flywheel import append_feedback, read_feedback
        from .runtime import MAX_CHAT_CHARS
        _load(name)
        if body.verdict not in ("up", "down"):
            raise HTTPException(400, "verdict must be 'up' or 'down'")
        turns = [t for t in body.turns if isinstance(t, str) and t.strip()]
        if not turns or not body.output.strip():
            raise HTTPException(400, "turns and output must be non-empty")
        # Same cap as the runtime's /v1/feedback, which feeds this same inbox: an
        # absorbed record becomes a permanent test input sent to BOTH the subject
        # and the judge on every future eval, so an unbounded payload here is a
        # one-request, permanent cost amplifier.
        # Every field that lands in the record, `reason` included: an absorbed
        # record becomes a permanent test input, so anything uncounted here is an
        # uncapped permanent cost.
        if (sum(len(t) for t in turns) + len(body.output) + len(body.correction or "")
                + len(body.reason or "")) > MAX_CHAT_CHARS:
            raise HTTPException(400, f"feedback too large (>{MAX_CHAT_CHARS} characters)")
        d = _dir(name)
        from .locking import LockBusy
        try:
            # Bounded, like every other mutating route: the lock is held across
            # whole engine runs, and a waiting request holds a threadpool slot.
            append_feedback(d, {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "turns": turns, "output": body.output, "verdict": body.verdict,
                "correction": body.correction, "reason": body.reason,
            }, wait_seconds=_LOCK_WAIT_SECONDS)
        except LockBusy:
            raise HTTPException(
                423, f"an operation is already in progress on project {name!r} — retry shortly")
        return {"recorded": True, "pending": len(read_feedback(d))}

    @app.get("/api/projects/{name}/feedback")
    def feedback_pending_(name: str):
        from .flywheel import read_feedback
        _load(name)
        records = read_feedback(_dir(name))
        return {"pending": len(records), "records": records}

    @app.post("/api/projects/{name}/absorb")
    def absorb_(name: str):
        from .compile import write_build_bundle
        from .flywheel import absorb_dict, absorb_feedback
        with _locked(name) as d:
            project = _load(name)
            # Save as absorb's commit step, while the records are still in the
            # inbox: a save that fails leaves them there to absorb again.
            result = absorb_feedback(project, d, commit=lambda: save_project(project, d))
            if project.spec is not None and project.tests:
                write_build_bundle(project.spec, project.tests, d)
        out = absorb_dict(result)
        out["state"] = _state(project, name)
        return out

    @app.post("/api/projects/{name}/examples-to-tests")
    def examples_to_tests_(name: str):
        from .compile import tests_from_examples, write_build_bundle
        with _locked(name) as d:
            project = _load(name)
            if project.spec is None:
                raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
            new = tests_from_examples(project.spec, project.tests)
            project.tests.extend(new)
            save_project(project, d)
            write_build_bundle(project.spec, project.tests, d)
        return {"added": len(new), "state": _state(project, name)}

    @app.get("/api/projects/{name}/promptfoo")
    def promptfoo_(name: str):
        from .interop import to_promptfoo
        project = _load(name)
        if project.spec is None or not project.tests:
            raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
        return {"config": to_promptfoo(project)}

    @app.get("/api/projects/{name}/judge-check")
    def judge_check_sample_(name: str):
        from .drift import load_scorecard
        from .eval import latest_run_id
        from .judge_check import gradings
        _load(name)
        d = _dir(name)
        rid = latest_run_id(d)
        if not rid:
            raise HTTPException(400, f"no scorecard — POST /api/projects/{name}/eval first")
        try:
            card = load_scorecard(d, rid)
        except (FileNotFoundError, ValueError, ValidationError) as exc:
            raise HTTPException(409, f"scorecard {rid!r} is unreadable: {exc}")
        return {"run_id": rid, "gradings": gradings(card)}

    @app.post("/api/projects/{name}/judge-check")
    def judge_check_score_(name: str, body: JudgeLabelsBody):
        from .drift import load_scorecard
        from .eval import latest_run_id
        from .judge_check import agreement_dict, judge_agreement, save_labels
        _load(name)
        d = _dir(name)
        rid = body.run_id or latest_run_id(d)
        if not rid:
            raise HTTPException(400, f"no scorecard — POST /api/projects/{name}/eval first")
        try:
            card = load_scorecard(d, rid)
            save_labels(d, rid, body.labels)  # persist: feeds train-engine as ground truth
        except (FileNotFoundError, ValueError, ValidationError) as exc:
            raise HTTPException(400, str(exc))
        out = agreement_dict(judge_agreement(card, body.labels))
        out["labels_saved"] = f"evals/{rid}/human-labels.json"
        return out

    @app.post("/api/projects/{name}/snapshot")
    def snapshot_pin_(name: str):
        from .drift import load_scorecard
        from .eval import latest_run_id
        from .snapshot import outputs_of, save_golden
        _load(name)
        d = _dir(name)
        rid = latest_run_id(d)
        if not rid:
            raise HTTPException(400, f"no scorecard — POST /api/projects/{name}/eval first")
        try:
            card = load_scorecard(d, rid)
        except (FileNotFoundError, ValueError, ValidationError) as exc:
            raise HTTPException(409, f"scorecard {rid!r} is unreadable: {exc}")
        # Never PIN from a partial run: the golden is the most reference-y artifact
        # there is, and an interrupted or truncated run would replace a complete
        # golden with a strict subset, silently narrowing every future check.
        if card.partial:
            raise HTTPException(409, f"{rid} is a PARTIAL run — pinning it would replace the golden "
                                     "with a subset of the suite. Run a full eval first.")
        outs = outputs_of(card)
        save_golden(d, outs)
        return {"pinned": len(outs), "run_id": rid}

    @app.get("/api/projects/{name}/snapshot")
    def snapshot_check_(name: str):
        from .drift import load_scorecard
        from .eval import latest_run_id
        from .snapshot import compare, load_golden, outputs_of, snapshot_dict
        _load(name)
        d = _dir(name)
        golden = load_golden(d)
        if golden is None:
            raise HTTPException(400, "no golden — POST /snapshot to pin one first")
        rid = latest_run_id(d)
        if not rid:
            raise HTTPException(400, f"no scorecard — POST /api/projects/{name}/eval first")
        try:
            card = load_scorecard(d, rid)
        except (FileNotFoundError, ValueError, ValidationError) as exc:
            raise HTTPException(409, f"scorecard {rid!r} is unreadable: {exc}")
        return snapshot_dict(compare(golden, outputs_of(card)))

    @app.get("/api/projects/{name}/lint")
    def lint_(name: str):
        from .lint import lint_dict, lint_spec, lint_unknown_fields
        project = _load(name)
        if project.spec is None:
            raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
        report = lint_spec(project.spec, project.tests)
        report.issues.extend(lint_unknown_fields(project))
        return lint_dict(report)

    @app.get("/api/projects/{name}/coverage")
    def coverage_(name: str):
        from .coverage import analyze_coverage, coverage_dict
        project = _load(name)
        if project.spec is None:
            raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
        return coverage_dict(analyze_coverage(project.spec, project.tests))

    @app.put("/api/projects/{name}/engines")
    def set_engines(name: str, body: EnginesBody):
        """Rebind role→model@provider. Body: {all} or {role, model}."""
        from .engines.base import validate_engine_spec
        from .models import EngineBinding
        if body.all is not None and (body.role or body.model):
            raise HTTPException(400, "use either `all` or a `role`+`model` pair, not both")
        with _locked(name) as d:
            project = _load(name)
            try:
                if body.all is not None:
                    spec = validate_engine_spec(body.all)
                    for r in EngineBinding.model_fields:
                        setattr(project.engines, r, spec)
                elif body.role and body.model:
                    if body.role not in EngineBinding.model_fields:
                        raise ValueError(f"unknown role {body.role!r}")
                    setattr(project.engines, body.role, validate_engine_spec(body.model))
                else:
                    raise ValueError("provide `all`, or both `role` and `model`")
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            save_project(project, d)
        return {"engines": project.engines.model_dump(), "state": _state(project, name)}

    @app.get("/api/projects/{name}/badge")
    def badge_(name: str):
        """shields.io endpoint JSON — embed the live badge with
        https://img.shields.io/endpoint?url=<this URL> (host must be reachable by shields)."""
        from .report import badge_dict
        project = _load(name)
        return badge_dict(project, _dir(name))

    @app.get("/api/projects/{name}/certification")
    def certification_(name: str):
        from .ci import certification_status, latest_gate
        project = _load(name)
        d = _dir(name)
        status, detail = certification_status(project, d)
        return {"status": status, "detail": detail, "gate": latest_gate(d)}

    @app.get("/api/projects/{name}/report")
    def report_(name: str):
        from .coverage import analyze_coverage
        from .drift import load_scorecard
        from .eval import latest_run_id
        from .report import render_report, report_dict
        project = _load(name)
        if project.spec is None:
            raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
        cov = analyze_coverage(project.spec, project.tests)
        latest = None
        rid = latest_run_id(_dir(name), full_only=True)  # a partial run summarizes nothing
        if rid:
            try:
                latest = load_scorecard(_dir(name), rid)
            except (FileNotFoundError, ValueError):
                latest = None
        return {**report_dict(project, cov, latest), "markdown": render_report(project, cov, latest)}

    @app.post("/api/projects/{name}/drift")
    def drift_(name: str, body: DriftBody, make_engine=Depends(_engine_factory)):
        from .drift import drift_dict, load_scorecard, run_drift
        from .eval import latest_run_id
        with _locked(name) as d:
            project = _load(name)
            if project.spec is None or not project.tests:
                raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
            # Must be a FULL run: comparing against a --max-tests / interrupted
            # smoke run measures two different test sets, so every regression on a
            # test the baseline never ran reads as "no regressions".
            base_id = body.baseline or latest_run_id(d, full_only=True)
            if not base_id:
                raise HTTPException(400, f"no full baseline scorecard — POST /api/projects/{name}/eval first")
            try:
                base_card = load_scorecard(d, base_id)
                if base_card.partial:
                    raise HTTPException(400, f"baseline {base_id} is a PARTIAL run (interrupted, or "
                                             "--max-tests) — not comparable; run a full eval first")
                subject = make_engine(project.engines.subject)
                judge = make_engine(project.engines.judge)
                report, _ = run_drift(project, subject, judge, baseline=base_card,
                                      project_dir=d, tolerance=body.tolerance)
            except HTTPException:
                raise
            except FileNotFoundError as exc:
                raise HTTPException(400, str(exc))
            except Exception as exc:
                raise _engine_http_error(exc)
        return drift_dict(report)

    @app.post("/api/projects/{name}/redteam")
    def redteam_(name: str, body: RedTeamBody, make_engine=Depends(_engine_factory)):
        from .redteam import promote_to_tests, redteam_dict, run_redteam
        added = 0
        with _locked(name) as d:
            project = _load(name)
            if project.spec is None:
                raise HTTPException(400, f"Nothing here yet — compile this project first (POST /api/projects/{name}/compile, or /import).")
            try:
                generator = make_engine(project.engines.compiler)
                subject = make_engine(project.engines.subject)
                judge = make_engine(project.engines.judge)
                report = run_redteam(project, generator, subject, judge,
                                     project_dir=d, max_probes=body.max_probes)
                if body.add_tests and report.violations:
                    from .compile import write_build_bundle
                    added = promote_to_tests(project, report)
                    save_project(project, d)
                    write_build_bundle(project.spec, project.tests, d)
            except HTTPException:
                raise
            except Exception as exc:
                raise _engine_http_error(exc)
        return {**redteam_dict(report), "tests_added": added}

    @app.post("/api/projects/{name}/teach/draft")
    def teach_draft(name: str, body: TeachDraftBody, make_engine=Depends(_engine_factory)):
        from .teach import propose_candidates
        project = _load(name)  # read-only: produce candidates to judge
        try:
            generator = make_engine(project.engines.compiler)
            subject = make_engine(project.engines.subject)
            candidates = propose_candidates(project, generator, subject, n=body.n, project_dir=_dir(name))
        except HTTPException:
            raise
        except Exception as exc:
            raise _engine_http_error(exc)
        return {"candidates": [{"id": c.id, "input": c.input, "output": c.output} for c in candidates]}

    @app.post("/api/projects/{name}/teach/learn")
    def teach_learn(name: str, body: TeachLearnBody, make_engine=Depends(_engine_factory)):
        from .teach import Judged, apply_learned, infer_standards
        if not body.judgments:
            raise HTTPException(400, "no judgments")
        with _locked(name) as d:
            project = _load(name)
            judged = [Judged(input=j.input, output=j.output, approved=j.approved, reason=j.reason)
                      for j in body.judgments]
            # Persist the human's judgments BEFORE the inference call, exactly as the
            # CLI does. They are the expensive part — a person's attention — and they
            # do not depend on `learned`, so an engine failure below must not discard
            # them. The second apply_learned folds in the standards ONLY (empty
            # judgment list), so nothing is recorded twice.
            apply_learned(project, judged, None)
            save_project(project, d)
            try:
                generator = make_engine(project.engines.compiler)
                learned = infer_standards(project.goal, judged, generator)
                result = apply_learned(project, [], learned)
                save_project(project, d)
                if project.spec is not None and project.tests:
                    from .compile import write_build_bundle
                    write_build_bundle(project.spec, project.tests, d)
            except HTTPException:
                raise
            except Exception as exc:
                raise _engine_http_error(exc)
        return {"standards_added": result.standards_added, "do_not_added": result.do_not_added,
                "standards": result.standards, "do_not": result.do_not, "state": _state(project, name)}

    def _merge_sources(sources: list[str]):
        """Load + validate >=2 distinct stakeholder projects with specs."""
        if len(sources) < 2:
            raise HTTPException(400, "need at least two sources")
        named: dict = {}
        first = None
        for s in sources:
            proj = _load(s)  # 404 if missing
            if proj.spec is None:
                raise HTTPException(400, f"{s!r} has no spec — compile it first")
            if proj.name in named:
                raise HTTPException(400, f"duplicate stakeholder name {proj.name!r}")
            named[proj.name] = proj.spec
            first = first or proj
        return named, first

    @app.post("/api/import")
    def import_prompt(body: ImportBody, make_engine=Depends(_engine_factory)):
        from .reverse import reverse_project
        if not body.prompt.strip():
            raise HTTPException(400, "prompt is empty")
        name = _safe(body.name)
        d = root / name
        if (d / "project.yaml").exists():   # permanent — don't wait out the lock for it
            raise HTTPException(409, "project already exists")
        with _held(d, name):
            if (d / "project.yaml").exists():
                raise HTTPException(409, "project already exists")
            try:
                eng = make_engine(body.engine or EngineBinding().compiler)
                project = reverse_project(name, body.goal, body.prompt, eng,
                                          task_type=body.task_type, engine_spec=body.engine, project_dir=d)
            except HTTPException:
                raise
            except Exception as exc:
                raise _engine_http_error(exc)
        return _state(project, name)

    @app.post("/api/diff")
    def diff_(body: DiffBody):
        from .specdiff import diff_dict, diff_specs
        pa, pb = _load(body.before), _load(body.after)
        if pa.spec is None or pb.spec is None:
            raise HTTPException(400, "both projects must have a compiled spec")
        return diff_dict(diff_specs(pa.spec, pb.spec))

    @app.post("/api/merge/detect")
    def merge_detect(body: MergeDetectBody, make_engine=Depends(_engine_factory)):
        from .stakeholders import (build_merged_spec, conflict_dict, detect_conflicts, gather,
                                   scalar_conflicts)
        named, first = _merge_sources(body.sources)
        statements = gather(named)
        try:
            engine = make_engine(first.engines.compiler)
            conflicts = detect_conflicts(statements, engine)
        except HTTPException:
            raise
        except Exception as exc:
            raise _engine_http_error(exc)
        # Read each resolution back off the spec the merge produces. persona is
        # resolved as ONE object (the first stakeholder by name with any persona
        # field), so naming the first stakeholder with a non-empty value would
        # promise a per-field ruling /api/merge/apply never makes.
        preview = build_merged_spec(named, goal=first.goal, task_type=first.task_type)
        resolved = {"persona.voice": preview.persona.voice,
                    "persona.reading_level": preview.persona.reading_level,
                    "format": preview.format, "refusal_policy": preview.refusal_policy}
        return {
            "stakeholders": list(named),
            "statements": [{"idx": s.idx, "text": s.text, "kind": s.kind, "stakeholder": s.stakeholder}
                           for s in statements],
            "conflicts": [conflict_dict(c) for c in conflicts],
            # Scalar behavior fields the engine's detector never sees (it is handed
            # standards and never-rules only). Reported so a client can show a
            # refusal-policy disagreement instead of it being resolved silently.
            "field_conflicts": [
                {"field": field, "values": [{"stakeholder": n, "value": v} for n, v in vals],
                 "resolved_to": {"stakeholder": next((n for n, v in vals if v == resolved.get(field)), None),
                                 "value": resolved.get(field)}}
                for field, vals in scalar_conflicts(named)
            ],
        }

    @app.post("/api/merge/apply")
    def merge_apply(body: MergeApplyBody):
        from .stakeholders import merged_project
        named, first = _merge_sources(body.sources)
        out_name = _safe(body.out)
        out_dir = root / out_name
        goal = body.goal or first.goal
        if (out_dir / "project.yaml").exists():   # permanent — don't wait out the lock
            raise HTTPException(409, "merged project already exists")
        with _held(out_dir, out_name):
            if (out_dir / "project.yaml").exists():
                raise HTTPException(409, "merged project already exists")
            proj = merged_project(out_name, named, goal=goal, task_type=first.task_type,
                                  drops=set(body.drops), additions=body.additions)
            save_project(proj, out_dir)
        return _state(proj, out_name)

    @app.post("/api/projects/{name}/log")
    def set_log(name: str, body: LogBody):
        with _locked(name) as d:
            project = _load(name)
            project.log_interactions = body.enabled
            save_project(project, d)
        return {"log_interactions": project.log_interactions}

    @app.post("/api/projects/{name}/train-engine/{role}")
    def train_engine_(name: str, role: str):
        from .train_engine import LOGGED_ROLES, TRAINABLE_ROLES, export_engine_bundle, read_log
        if role not in TRAINABLE_ROLES:
            raise HTTPException(400, f"role must be one of: {', '.join(sorted(TRAINABLE_ROLES))}")
        d = _dir(name)
        if not (d / "project.yaml").exists():
            raise HTTPException(404, f"no project {name!r}")
        if not read_log(d, role):
            if role in LOGGED_ROLES:
                raise HTTPException(400, f"no logged {role} decisions — enable logging and run eval first")
            raise HTTPException(400, f"nothing records the {role} role yet — only judge and compiler "
                                     "decisions are logged, so there is no data to train on")
        try:
            result = export_engine_bundle(d, role)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"role": result.role, "examples": result.examples,
                "base_model": result.base_model, "bundle_dir": result.bundle_dir, "files": result.files}

    # Static web UI last, so /api/* routes win.
    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app
