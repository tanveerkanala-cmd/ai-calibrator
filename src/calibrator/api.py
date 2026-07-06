"""M6 — Local HTTP API over the Calibration Core (localhost only).

A thin FastAPI layer: every endpoint maps to a core pipeline call, so the same
logic that powers the CLI also drives the web/desktop UI. Engine-dependent
endpoints build the project's configured engine and surface failures as HTTP 400.

Run with `calibrate serve`. Needs the `api` extra:  pip install -e '.[api]'
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import yaml

try:
    from fastapi import Depends, FastAPI, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field, ValidationError
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise RuntimeError("The API needs the `api` extra:  pip install -e '.[api]'") from exc

from .auth import all_status
from .models import EngineBinding, Project, TaskType
from .store import load_project, project_lock, save_project, write_project_gitignore

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
    tolerance: float = Field(0.0, ge=0.0)
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


MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB cap on material uploads
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "testserver"}


def create_app(projects_root: Path | None = None, allowed_hosts: list[str] | None = None) -> "FastAPI":
    root = Path(projects_root) if projects_root else default_projects_root()
    root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="AI Calibrator")

    allowed = set(_LOOPBACK_HOSTS) | {h.lower() for h in (allowed_hosts or [])}

    @app.middleware("http")
    async def _guard(request, call_next):
        # Always enforced — never fully disabled. The Host allowlist blocks
        # DNS-rebinding; the Origin check blocks cross-origin CSRF on mutating
        # requests. To expose beyond localhost, bind a specific reachable address
        # (serve adds it to the allowlist) so BOTH checks still protect you.
        # Fail closed: an absent/unparseable Host is rejected.
        raw = request.headers.get("host") or ""
        host = (raw.split("]")[0].lstrip("[") if raw.startswith("[") else raw.split(":")[0]).lower().rstrip(".")
        if host not in allowed:
            return JSONResponse(status_code=400, content={"detail": "host not allowed"})
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and (urlsplit(origin).hostname or "").lower().rstrip(".") not in allowed:
                return JSONResponse(status_code=403, content={"detail": "cross-origin request blocked"})
        return await call_next(request)

    def _safe(name: str) -> str:
        s = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
        if not s:
            raise HTTPException(400, "invalid project name")
        if len(s) > 120:  # names become directory names — filesystems cap ~255 bytes
            raise HTTPException(400, "project name too long (max 120 characters)")
        return s

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
        with project_lock(d):
            yield d

    def _state(p: Project) -> dict:
        return {
            "name": p.name,
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
        return {"ok": True, "projects_root": str(root)}

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
        # Atomic create: hold the project lock across the exists-check + write so
        # two concurrent POSTs for the same name can't both pass the check (one
        # wins with 200, the other deterministically gets 409 — never a partial
        # write or a 500).
        with project_lock(d):
            if (d / "project.yaml").exists():
                raise HTTPException(409, "project already exists")
            save_project(project, d)
            write_project_gitignore(d)
        return _state(project)

    @app.get("/api/projects/{name}")
    def get_project(name: str):
        return _state(_load(name))

    @app.post("/api/projects/{name}/materials")
    async def upload_material(name: str, file: UploadFile):
        d = _dir(name)
        if not (d / "project.yaml").exists():
            raise HTTPException(404, f"no project {name!r}")
        mats = d / "materials"
        mats.mkdir(exist_ok=True)
        target = mats / Path(file.filename or "upload.txt").name
        # Stream to a unique temp file, then atomically rename into place. A
        # cap-exceed / disconnect mid-upload leaves no partial file, and two
        # concurrent uploads of the same name can't interleave bytes — the last
        # complete one wins cleanly instead of corrupting the target.
        fd, tmp_name = tempfile.mkstemp(dir=str(mats), prefix=".upload-", suffix=".tmp")
        tmp: Path | None = Path(tmp_name)
        size = 0
        try:
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "file too large (max 25 MB)")
                    out.write(chunk)
            os.replace(tmp, target)
            tmp = None
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        return {"uploaded": target.name}

    @app.post("/api/projects/{name}/ingest")
    def ingest(name: str, make_engine=Depends(_engine_factory)):
        from .ingest import ingest_project
        with _locked(name) as d:
            project = _load(name)
            try:
                engine = make_engine(project.engines.extractor)
                result = ingest_project(project, d / "materials", engine, project_dir=d)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, str(exc))
            save_project(project, d)
        return {
            "materials": result.materials, "gaps": result.gaps,
            "facts": result.facts, "indexed": result.indexed, "state": _state(project),
        }

    @app.post("/api/projects/{name}/interview")
    def interview(name: str, make_engine=Depends(_engine_factory)):
        from .interview import generate_questions
        with _locked(name) as d:
            project = _load(name)
            if not project.gaps:
                raise HTTPException(400, "No gaps yet — run `calibrate ingest` first.")
            try:
                engine = make_engine(project.engines.interviewer)
                project.interview = generate_questions(project, engine)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, str(exc))
            save_project(project, d)
        return _state(project)

    @app.post("/api/projects/{name}/answers")
    def submit_answers(name: str, body: AnswersBody):
        with _locked(name) as d:
            project = _load(name)
            by_id = {it.id: it for it in project.interview}
            applied = 0
            for qid, ans in body.answers.items():
                if qid in by_id:
                    by_id[qid].answer = ans
                    applied += 1
            save_project(project, d)
        return {"applied": applied, "state": _state(project)}

    @app.post("/api/projects/{name}/compile")
    def compile_(name: str, make_engine=Depends(_engine_factory)):
        from .compile import compile_project
        with _locked(name) as d:
            project = _load(name)
            if not any(it.answer for it in project.interview):
                raise HTTPException(400, "No interview answers yet — run `calibrate interview` first.")
            try:
                engine = make_engine(project.engines.compiler)
                result = compile_project(project, engine, project_dir=d)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, str(exc))
            save_project(project, d)
        return {"criteria": result.criteria, "tests": result.tests,
                "files": result.files, "state": _state(project)}

    @app.post("/api/projects/{name}/eval")
    def evaluate(name: str, body: EvalBody, make_engine=Depends(_engine_factory)):
        from .engine_log import wrap_engine
        from .eval import next_run_id, run_eval, save_scorecard
        with _locked(name) as d:
            project = _load(name)
            if project.spec is None or not project.tests:
                raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
            log_on = project.log_interactions
            try:
                subject = make_engine(project.engines.subject)
                judge = wrap_engine(make_engine(project.engines.judge), "judge", d, enabled=log_on)
                if body.refine:
                    from .pipeline import calibrate_loop
                    refiner = wrap_engine(make_engine(project.engines.compiler), "compiler", d, enabled=log_on)
                    cards = calibrate_loop(project, subject, judge, refiner,
                                           threshold=body.threshold, max_rounds=body.rounds,
                                           judge_passes=body.judge_passes, project_dir=d)
                    save_project(project, d)
                    from .compile import write_build_bundle
                    write_build_bundle(project.spec, project.tests, d)  # refresh build/ to match
                else:
                    card = run_eval(project, subject, judge, run_id=next_run_id(d),
                                    judge_passes=body.judge_passes)
                    save_scorecard(d, card)
                    cards = [card]
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, str(exc))
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
                raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
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
                raise HTTPException(400, str(exc))
        return ci_dict(result)

    @app.post("/api/projects/{name}/export")
    def export(name: str):
        from .export import export_bundle
        project = _load(name)
        if project.spec is None:
            raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
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
                raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
            specs = body.models or list(DEFAULT_LADDER)
            try:
                judge = make_engine(project.engines.judge)
                report = rightsize(project, specs, judge, make_engine,
                                   threshold=body.threshold, project_dir=d)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, str(exc))
        return rightsize_dict(report)

    @app.post("/api/projects/{name}/try")
    def try_(name: str, body: TryBody, make_engine=Depends(_engine_factory)):
        """One exchange with the calibrated AI (subject engine + compiled prompt) —
        the workbench's 'Try & flag' box. Same encoding as the runtime/eval."""
        from .compile import render_system_prompt
        from .eval import conversation_prompt
        project = _load(name)
        if project.spec is None:
            raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
        try:
            subject = make_engine(project.engines.subject)
            output = str(subject.complete(conversation_prompt([], body.message),
                                          system=render_system_prompt(project.spec)) or "").strip()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, str(exc))
        return {"turns": [body.message], "output": output}

    @app.post("/api/projects/{name}/feedback")
    def feedback_(name: str, body: FeedbackBody):
        """Record thumbs-up/down from the workbench (same inbox the runtime feeds)."""
        from datetime import datetime, timezone

        from .flywheel import append_feedback, read_feedback
        _load(name)
        if body.verdict not in ("up", "down"):
            raise HTTPException(400, "verdict must be 'up' or 'down'")
        turns = [t for t in body.turns if isinstance(t, str) and t.strip()]
        if not turns or not body.output.strip():
            raise HTTPException(400, "turns and output must be non-empty")
        d = _dir(name)
        append_feedback(d, {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "turns": turns, "output": body.output, "verdict": body.verdict,
            "correction": body.correction, "reason": body.reason,
        })
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
            result = absorb_feedback(project, d)
            save_project(project, d)
            if project.spec is not None and project.tests:
                write_build_bundle(project.spec, project.tests, d)
        out = absorb_dict(result)
        out["state"] = _state(project)
        return out

    @app.post("/api/projects/{name}/examples-to-tests")
    def examples_to_tests_(name: str):
        from .compile import tests_from_examples, write_build_bundle
        with _locked(name) as d:
            project = _load(name)
            if project.spec is None:
                raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
            new = tests_from_examples(project.spec, project.tests)
            project.tests.extend(new)
            save_project(project, d)
            write_build_bundle(project.spec, project.tests, d)
        return {"added": len(new), "state": _state(project)}

    @app.get("/api/projects/{name}/promptfoo")
    def promptfoo_(name: str):
        from .interop import to_promptfoo
        project = _load(name)
        if project.spec is None or not project.tests:
            raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
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
            raise HTTPException(400, "no scorecard — run eval first")
        return {"run_id": rid, "gradings": gradings(load_scorecard(d, rid))}

    @app.post("/api/projects/{name}/judge-check")
    def judge_check_score_(name: str, body: JudgeLabelsBody):
        from .drift import load_scorecard
        from .eval import latest_run_id
        from .judge_check import agreement_dict, judge_agreement, save_labels
        _load(name)
        d = _dir(name)
        rid = body.run_id or latest_run_id(d)
        if not rid:
            raise HTTPException(400, "no scorecard — run eval first")
        try:
            card = load_scorecard(d, rid)
            save_labels(d, rid, body.labels)  # persist: feeds train-engine as ground truth
        except (FileNotFoundError, ValueError) as exc:
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
            raise HTTPException(400, "no scorecard — run eval first")
        outs = outputs_of(load_scorecard(d, rid))
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
            raise HTTPException(400, "no scorecard — run eval first")
        return snapshot_dict(compare(golden, outputs_of(load_scorecard(d, rid))))

    @app.get("/api/projects/{name}/lint")
    def lint_(name: str):
        from .lint import lint_dict, lint_spec, lint_unknown_fields
        project = _load(name)
        if project.spec is None:
            raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
        report = lint_spec(project.spec, project.tests)
        report.issues.extend(lint_unknown_fields(project))
        return lint_dict(report)

    @app.get("/api/projects/{name}/coverage")
    def coverage_(name: str):
        from .coverage import analyze_coverage, coverage_dict
        project = _load(name)
        if project.spec is None:
            raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
        return coverage_dict(analyze_coverage(project.spec, project.tests))

    @app.put("/api/projects/{name}/engines")
    def set_engines(name: str, body: EnginesBody):
        """Rebind role→model@provider. Body: {all} or {role, model}."""
        from .engines.base import validate_engine_spec
        from .models import EngineBinding
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
        return {"engines": project.engines.model_dump(), "state": _state(project)}

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
            raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
        cov = analyze_coverage(project.spec, project.tests)
        latest = None
        rid = latest_run_id(_dir(name))
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
                raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
            base_id = body.baseline or latest_run_id(d)
            if not base_id:
                raise HTTPException(400, "no baseline scorecard — run eval first")
            try:
                base_card = load_scorecard(d, base_id)
                subject = make_engine(project.engines.subject)
                judge = make_engine(project.engines.judge)
                report, _ = run_drift(project, subject, judge, baseline=base_card,
                                      project_dir=d, tolerance=body.tolerance)
            except HTTPException:
                raise
            except FileNotFoundError as exc:
                raise HTTPException(400, str(exc))
            except Exception as exc:
                raise HTTPException(400, str(exc))
        return drift_dict(report)

    @app.post("/api/projects/{name}/redteam")
    def redteam_(name: str, body: RedTeamBody, make_engine=Depends(_engine_factory)):
        from .redteam import promote_to_tests, redteam_dict, run_redteam
        added = 0
        with _locked(name) as d:
            project = _load(name)
            if project.spec is None:
                raise HTTPException(400, "Nothing here yet — run `calibrate compile` (or `import`) first.")
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
                raise HTTPException(400, str(exc))
        return {**redteam_dict(report), "tests_added": added}

    @app.post("/api/projects/{name}/teach/draft")
    def teach_draft(name: str, body: TeachDraftBody, make_engine=Depends(_engine_factory)):
        from .teach import propose_candidates
        project = _load(name)  # read-only: produce candidates to judge
        try:
            generator = make_engine(project.engines.compiler)
            subject = make_engine(project.engines.subject)
            candidates = propose_candidates(project, generator, subject, n=body.n)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, str(exc))
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
            try:
                generator = make_engine(project.engines.compiler)
                learned = infer_standards(project.goal, judged, generator)
                result = apply_learned(project, judged, learned)
                save_project(project, d)
                if project.tests:
                    from .compile import write_build_bundle
                    write_build_bundle(project.spec, project.tests, d)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, str(exc))
        return {"standards_added": result.standards_added, "do_not_added": result.do_not_added,
                "standards": result.standards, "do_not": result.do_not, "state": _state(project)}

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
        with project_lock(d):
            if (d / "project.yaml").exists():
                raise HTTPException(409, "project already exists")
            try:
                eng = make_engine(body.engine or EngineBinding().compiler)
                project = reverse_project(name, body.goal, body.prompt, eng,
                                          task_type=body.task_type, engine_spec=body.engine, project_dir=d)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(400, str(exc))
        return _state(project)

    @app.post("/api/diff")
    def diff_(body: DiffBody):
        from .specdiff import diff_dict, diff_specs
        pa, pb = _load(body.before), _load(body.after)
        if pa.spec is None or pb.spec is None:
            raise HTTPException(400, "both projects must have a compiled spec")
        return diff_dict(diff_specs(pa.spec, pb.spec))

    @app.post("/api/merge/detect")
    def merge_detect(body: MergeDetectBody, make_engine=Depends(_engine_factory)):
        from .stakeholders import conflict_dict, detect_conflicts, gather
        named, first = _merge_sources(body.sources)
        statements = gather(named)
        try:
            engine = make_engine(first.engines.compiler)
            conflicts = detect_conflicts(statements, engine)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, str(exc))
        return {
            "stakeholders": list(named),
            "statements": [{"idx": s.idx, "text": s.text, "kind": s.kind, "stakeholder": s.stakeholder}
                           for s in statements],
            "conflicts": [conflict_dict(c) for c in conflicts],
        }

    @app.post("/api/merge/apply")
    def merge_apply(body: MergeApplyBody):
        from .stakeholders import merged_project
        named, first = _merge_sources(body.sources)
        out_name = _safe(body.out)
        out_dir = root / out_name
        goal = body.goal or first.goal
        with project_lock(out_dir):
            if (out_dir / "project.yaml").exists():
                raise HTTPException(409, "merged project already exists")
            proj = merged_project(out_name, named, goal=goal, task_type=first.task_type,
                                  drops=set(body.drops), additions=body.additions)
            save_project(proj, out_dir)
        return _state(proj)

    @app.post("/api/projects/{name}/log")
    def set_log(name: str, body: LogBody):
        with _locked(name) as d:
            project = _load(name)
            project.log_interactions = body.enabled
            save_project(project, d)
        return {"log_interactions": project.log_interactions}

    @app.post("/api/projects/{name}/train-engine/{role}")
    def train_engine_(name: str, role: str):
        from .train_engine import TRAINABLE_ROLES, export_engine_bundle, read_log
        if role not in TRAINABLE_ROLES:
            raise HTTPException(400, f"role must be one of: {', '.join(sorted(TRAINABLE_ROLES))}")
        d = _dir(name)
        if not (d / "project.yaml").exists():
            raise HTTPException(404, f"no project {name!r}")
        if not read_log(d, role):
            raise HTTPException(400, f"no logged {role} decisions — enable logging and run eval first")
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
