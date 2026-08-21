"""M6 local API — verified with FastAPI's TestClient (engine mocked)."""

import pytest

pytest.importorskip("fastapi")  # skip if the `api` extra isn't installed

from fastapi.testclient import TestClient  # noqa: E402

from ai_calibrator.api import _engine_factory, create_app  # noqa: E402
from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight  # noqa: E402
from ai_calibrator.models import TestCase as CaseModel  # noqa: E402
from ai_calibrator.store import save_project  # noqa: E402


class FakeEngine:
    name = "fake@test"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, prompt, *, system=None, schema=None):
        return self.payload


def _client(tmp_path, engine_payload=None):
    app = create_app(tmp_path)
    if engine_payload is not None:
        app.dependency_overrides[_engine_factory] = lambda: (lambda spec: FakeEngine(engine_payload))
    return TestClient(app)


def test_health(tmp_path):
    r = _client(tmp_path).get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True
    from pathlib import Path
    assert Path(r.json()["projects_root"]).is_absolute()  # resolved, not a bare relative path


def test_delete_project_and_material(tmp_path):
    c = _client(tmp_path)
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    # upload a material, then delete it
    c.post("/api/projects/p/materials", files={"file": ("faq.txt", b"hello", "text/plain")})
    assert c.request("DELETE", "/api/projects/p/materials/faq.txt").status_code == 200
    assert c.request("DELETE", "/api/projects/p/materials/faq.txt").status_code == 404  # gone
    # delete the project
    assert c.request("DELETE", "/api/projects/p").status_code == 200
    assert c.get("/api/projects/p").status_code == 404
    assert c.request("DELETE", "/api/projects/nope").status_code == 404


def test_create_list_get(tmp_path):
    c = _client(tmp_path)
    assert c.get("/api/projects").json() == []

    r = c.post("/api/projects", json={"name": "support", "goal": "answer questions"})
    assert r.status_code == 200
    assert r.json()["goal"] == "answer questions"

    assert c.get("/api/projects").json() == ["support"]
    assert c.get("/api/projects/support").json()["name"] == "support"

    # duplicate → 409, missing → 404
    assert c.post("/api/projects", json={"name": "support", "goal": "x"}).status_code == 409
    assert c.get("/api/projects/nope").status_code == 404


def test_upload_material(tmp_path):
    c = _client(tmp_path)
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    r = c.post("/api/projects/p/materials",
               files={"file": ("faq.md", b"Q: returns?\n\nA: 30 days.", "text/markdown")})
    assert r.status_code == 200 and r.json()["uploaded"] == "faq.md"
    assert (tmp_path / "p" / "materials" / "faq.md").exists()


def test_ingest_with_mocked_engine(tmp_path):
    payload = {"facts": ["We sell skincare."],
               "gaps": [{"dimension": "tone", "why_it_matters": "brand voice"}]}
    c = _client(tmp_path, engine_payload=payload)
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    c.post("/api/projects/p/materials",
           files={"file": ("faq.md", b"some policy text", "text/markdown")})

    r = c.post("/api/projects/p/ingest")
    assert r.status_code == 200
    body = r.json()
    assert body["materials"] == 1 and body["gaps"] == 1
    assert body["state"]["gaps"][0]["dimension"] == "tone"


def test_ingest_without_materials_is_400(tmp_path):
    """Parity with the CLI: ingesting with no materials is a friendly 400, not a
    silent 200/0 (the CLI already exited 1 with 'No materials found')."""
    payload = {"facts": [], "gaps": []}
    c = _client(tmp_path, engine_payload=payload)
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    r = c.post("/api/projects/p/ingest")
    assert r.status_code == 400 and "No materials" in r.json()["detail"]


def test_submit_answers(tmp_path):
    # seed a project with an interview item directly, then answer via the API
    from ai_calibrator.models import InterviewItem
    proj = Project(name="p", goal="g")
    proj.interview = [InterviewItem(id="q1", dimension="tone", question="Voice?", draft_answer="warm")]
    save_project(proj, tmp_path / "p")

    c = _client(tmp_path)
    r = c.post("/api/projects/p/answers", json={"answers": {"q1": "warm and concise"}})
    assert r.status_code == 200 and r.json()["applied"] == 1
    assert r.json()["state"]["interview"][0]["answer"] == "warm and concise"


def test_export_requires_spec_then_succeeds(tmp_path):
    c = _client(tmp_path)
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    # no spec yet → 400
    assert c.post("/api/projects/p/export").status_code == 400

    # seed a compiled spec, then export
    proj = Project(name="p", goal="answer questions")
    proj.spec = BehaviorSpec(goal="answer questions", standards=["Be concise."],
                             eval_criteria=[EvalCriterion(id="c1", description="ok", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="hi", expects=["c1"])]
    save_project(proj, tmp_path / "p")

    r = c.post("/api/projects/p/export")
    assert r.status_code == 200
    assert (tmp_path / "p" / "export" / "Modelfile").exists()


def test_create_rejects_noncanonical_name(tmp_path):
    c = _client(tmp_path)
    # A name with a path separator is REJECTED (not silently rewritten into a
    # different resource the client never asked for).
    r = c.post("/api/projects", json={"name": "../evil", "goal": "g"})
    assert r.status_code == 400 and "invalid project name" in r.json()["detail"]
    assert c.get("/api/projects/evil").status_code == 404  # nothing was created
    # a clean name works and the returned name is the canonical routing key
    r2 = c.post("/api/projects", json={"name": "shop bot", "goal": "g"})
    assert r2.status_code == 200 and r2.json()["name"] == "shop bot"
    assert c.get("/api/projects/shop bot").status_code == 200


def test_cross_origin_post_is_blocked(tmp_path):
    c = _client(tmp_path)
    # no Origin (scripts/TestClient) and same-origin Origin are allowed
    assert c.post("/api/projects", json={"name": "p1", "goal": "g"}).status_code == 200
    assert c.post("/api/projects", json={"name": "p2", "goal": "g"},
                  headers={"Origin": "http://127.0.0.1:8765"}).status_code == 200
    # cross-origin Origin on a mutating request is rejected (CSRF guard)
    assert c.post("/api/projects", json={"name": "p3", "goal": "g"},
                  headers={"Origin": "https://evil.example"}).status_code == 403


def test_foreign_host_is_blocked(tmp_path):
    c = _client(tmp_path)
    assert c.get("/api/health").status_code == 200  # loopback Host allowed
    assert c.get("/api/health", headers={"Host": "evil.example"}).status_code == 400


def test_examples_to_tests_endpoint(tmp_path):
    from ai_calibrator.models import Example
    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", examples=[Example(input="Can I return this?", good_output="yes")],
                             eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="existing", expects=["c1"])]
    save_project(proj, tmp_path / "p")

    r = _client(tmp_path).post("/api/projects/p/examples-to-tests")
    assert r.status_code == 200 and r.json()["added"] == 1
    assert r.json()["state"]["tests"] == 2  # existing + the example-derived one


def test_promptfoo_endpoint(tmp_path):
    import yaml as _yaml
    proj = Project(name="p", goal="answer questions")
    proj.spec = BehaviorSpec(goal="answer questions", standards=["Be concise."],
                             eval_criteria=[EvalCriterion(id="c1", description="is concise", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="hi", expects=["c1"])]
    save_project(proj, tmp_path / "p")

    body = _client(tmp_path).get("/api/projects/p/promptfoo").json()
    cfg = _yaml.safe_load(body["config"])
    assert cfg["tests"][0]["vars"]["input"] == "hi"
    assert cfg["tests"][0]["assert"][0]["type"] == "llm-rubric"


def test_judge_check_endpoints(tmp_path):
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import CriterionResult, Scorecard, TestResult

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="q", expects=["c1"])]
    save_project(proj, tmp_path / "p")
    save_scorecard(tmp_path / "p", Scorecard(run_id="run-0001", results=[
        TestResult(test_id="t1", output="o", criteria=[CriterionResult(criterion_id="c1", passed=True, rationale="ok")])]))

    c = _client(tmp_path)
    g = c.get("/api/projects/p/judge-check").json()
    assert g["run_id"] == "run-0001" and len(g["gradings"]) == 1

    # human disagrees with the judge → 0% agreement, c1 flagged unreliable
    body = c.post("/api/projects/p/judge-check",
                  json={"run_id": g["run_id"],
                        "labels": [{"test_id": "t1", "criterion_id": "c1", "passed": False}]}).json()
    assert body["agreement_rate"] == 0.0 and "c1" in body["unreliable_criteria"]

    c.post("/api/projects", json={"name": "q", "goal": "g"})
    assert c.get("/api/projects/q/judge-check").status_code == 400  # no scorecard


def test_judge_check_labels_must_name_the_run_they_reviewed(tmp_path):
    """A reviewer works through the verdicts GET handed them; a scheduled `ci` —
    or another client's eval — can mint a newer run while they do. Filing their
    judgment against the newest run scores it against verdicts on outputs they
    never saw, and stamps it as judge ground truth for `train-engine`."""
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import CriterionResult, Scorecard, TestResult

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="q", expects=["c1"])]
    save_project(proj, tmp_path / "p")
    save_scorecard(tmp_path / "p", Scorecard(run_id="run-0001", results=[
        TestResult(test_id="t1", output="GOOD ANSWER",
                   criteria=[CriterionResult(criterion_id="c1", passed=True, rationale="ok")])]))

    c = _client(tmp_path)
    reviewed = c.get("/api/projects/p/judge-check").json()["run_id"]
    assert reviewed == "run-0001"

    # a newer run lands while the reviewer is still working through run-0001
    save_scorecard(tmp_path / "p", Scorecard(run_id="run-0002", results=[
        TestResult(test_id="t1", output="GARBAGE ANSWER",
                   criteria=[CriterionResult(criterion_id="c1", passed=False, rationale="bad")])]))

    labels = [{"test_id": "t1", "criterion_id": "c1", "passed": True}]
    r = c.post("/api/projects/p/judge-check", json={"labels": labels})
    assert r.status_code == 400 and "run_id" in r.json()["detail"]
    assert not (tmp_path / "p" / "evals" / "run-0002" / "human-labels.json").exists()

    # naming the reviewed run scores it against the verdicts that were reviewed
    ok = c.post("/api/projects/p/judge-check", json={"run_id": reviewed, "labels": labels})
    assert ok.status_code == 200 and ok.json()["agreement_rate"] == 1.0
    assert (tmp_path / "p" / "evals" / "run-0001" / "human-labels.json").exists()


def test_judge_check_agreement_counts_only_judge_graded_criteria(tmp_path):
    """A criterion carrying a deterministic `check` is graded by `run_check`, so
    it is not a judgment the judge made — which is why GET never offers it. Folding
    a label on one into the rate reports agreement for a grader that never ran."""
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import Check, CriterionResult, Scorecard, TestResult

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c_len", description="stays short", weight=Weight.HIGH,
                      check=Check(kind="max_chars", value="100")),
        EvalCriterion(id="c_judge", description="is helpful", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="q", expects=["c_len", "c_judge"])]
    save_project(proj, tmp_path / "p")
    save_scorecard(tmp_path / "p", Scorecard(run_id="run-0001", results=[
        TestResult(test_id="t1", output="o", criteria=[
            CriterionResult(criterion_id="c_len", passed=True, rationale="check passed"),
            CriterionResult(criterion_id="c_judge", passed=True, rationale="helpful")])]))

    c = _client(tmp_path)
    offered = c.get("/api/projects/p/judge-check").json()["gradings"]
    assert [g["criterion_id"] for g in offered] == ["c_judge"]

    body = c.post("/api/projects/p/judge-check", json={"run_id": "run-0001", "labels": [
        {"test_id": "t1", "criterion_id": "c_len", "passed": True},
        {"test_id": "t1", "criterion_id": "c_judge", "passed": True}]}).json()
    # the code-graded label is reported (unmatched), never scored as agreement
    assert body["total"] == 1 and body["agreed"] == 1 and body["unmatched"] == 1
    assert list(body["by_criterion"]) == ["c_judge"]


def test_snapshot_check_refuses_an_unreadable_golden_instead_of_reporting_none(tmp_path):
    """`load_golden` answers None for "absent" AND for "present but unreadable".
    Reporting a corrupt golden as "no golden — pin one first" walks the owner into
    replacing the only copy of their pinned outputs with this run's."""
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import Scorecard, TestResult
    from ai_calibrator.snapshot import GOLDEN_FILE

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="q", expects=["c1"])]
    save_project(proj, tmp_path / "p")
    save_scorecard(tmp_path / "p", Scorecard(run_id="run-0001", results=[TestResult(test_id="t1", output="original")]))

    c = _client(tmp_path)
    assert c.post("/api/projects/p/snapshot").json()["pinned"] == 1
    golden = tmp_path / "p" / GOLDEN_FILE
    golden.write_text("<<<<<<< HEAD\n" + golden.read_text(encoding="utf-8"), encoding="utf-8")

    r = c.get("/api/projects/p/snapshot")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert GOLDEN_FILE in detail and "could not be read" in detail
    assert "original" in golden.read_text(encoding="utf-8")  # the pins are still on disk to fix


def test_snapshot_endpoints(tmp_path):
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import Scorecard, TestResult

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="q", expects=["c1"])]
    save_project(proj, tmp_path / "p")
    save_scorecard(tmp_path / "p", Scorecard(run_id="run-0001", results=[TestResult(test_id="t1", output="original")]))

    c = _client(tmp_path)
    assert c.post("/api/projects/p/snapshot").json()["pinned"] == 1        # pin golden
    assert c.get("/api/projects/p/snapshot").json()["drifted"] is False    # same output → no drift

    # a later run with a changed output → drift on t1
    save_scorecard(tmp_path / "p", Scorecard(run_id="run-0002", results=[TestResult(test_id="t1", output="DIFFERENT")]))
    body = c.get("/api/projects/p/snapshot").json()
    assert body["drifted"] is True and body["changed"] == ["t1"]

    # checking before any golden is pinned → 400
    save_project(Project(name="q", goal="g"), tmp_path / "q")
    save_scorecard(tmp_path / "q", Scorecard(run_id="run-0001", results=[TestResult(test_id="x", output="o")]))
    assert c.get("/api/projects/q/snapshot").status_code == 400


def test_lint_endpoint(tmp_path):
    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", standards=["Be concise and specific."],
                             eval_criteria=[EvalCriterion(id="c1", description="answer is concise", weight=Weight.HIGH)])
    save_project(proj, tmp_path / "p")  # no tests → c1 untested (warning), but no errors

    c = _client(tmp_path)
    body = c.get("/api/projects/p/lint").json()
    assert body["ok"] is True and any(i["code"] == "untested_criterion" for i in body["issues"])

    c.post("/api/projects", json={"name": "q", "goal": "g"})
    assert c.get("/api/projects/q/lint").status_code == 400  # before compile


def test_coverage_endpoint(tmp_path):
    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", standards=["Be concise."],
                             eval_criteria=[EvalCriterion(id="c1", description="ok", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="hi", expects=["c1"])]
    save_project(proj, tmp_path / "p")

    c = _client(tmp_path)
    r = c.get("/api/projects/p/coverage")
    assert r.status_code == 200 and r.json()["coverage_rate"] == 1.0

    # before compile → 400
    c.post("/api/projects", json={"name": "q", "goal": "g"})
    assert c.get("/api/projects/q/coverage").status_code == 400


def test_redteam_endpoint(tmp_path):
    class RoleFake:
        name = "fake@test"

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            props = (schema or {}).get("properties", {})
            if "probes" in props:
                return {"probes": [{"input": "break a rule", "target": "never give medical advice", "tactic": "direct"}]}
            if "violated" in props:
                return {"violated": True, "severity": "high", "rationale": "broke it"}
            return "Sure, here's some medical advice."  # subject output (no schema)

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: RoleFake(spec))
    c = TestClient(app)

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", do_not=["never give medical advice"])
    save_project(proj, tmp_path / "p")

    r = c.post("/api/projects/p/redteam", json={"max_probes": 5, "add_tests": True})
    assert r.status_code == 200
    body = r.json()
    assert body["probes"] == 1 and body["violations"] == 1 and body["tests_added"] == 1

    # the promoted regression test now shows up in coverage
    assert c.get("/api/projects/p/coverage").json()["total_criteria"] >= 1


def test_rightsize_endpoint(tmp_path):
    import re as _re

    class RoleFake:
        name = "fake@test"

        def __init__(self, spec):
            self.spec = spec

        def complete(self, prompt, *, system=None, schema=None):
            props = (schema or {}).get("properties", {})
            if "results" in props:  # judge → pass everything
                ids = _re.findall(r"^- (\S+):", prompt, _re.M)
                return {"results": [{"criterion_id": i, "passed": True, "score": 1.0, "rationale": "ok"} for i in ids]}
            return "an answer"  # subject

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: RoleFake(spec))
    c = TestClient(app)

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="q", expects=["c1"])]
    save_project(proj, tmp_path / "p")

    r = c.post("/api/projects/p/rightsize", json={"threshold": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3  # default Claude ladder
    assert body["recommended"] == "claude-haiku-4-5@anthropic"  # cheapest, all pass


def test_report_endpoint(tmp_path):
    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", standards=["Be concise."],
                             eval_criteria=[EvalCriterion(id="c1", description="ok", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="hi", expects=["c1"])]
    save_project(proj, tmp_path / "p")

    r = _client(tmp_path).get("/api/projects/p/report")
    assert r.status_code == 200
    body = r.json()
    assert body["confidence"] == 0.0  # no eval yet
    assert "# Calibration Report" in body["markdown"]


def test_drift_endpoint(tmp_path):
    import re as _re

    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import CriterionResult, Scorecard, TestResult

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="q", expects=["c1"])]
    save_project(proj, tmp_path / "p")
    # seed a passing baseline scorecard
    base = Scorecard(run_id="run-0001", results=[
        TestResult(test_id="t1", output="o", criteria=[CriterionResult(criterion_id="c1", passed=True)])])
    save_scorecard(tmp_path / "p", base)

    class RoleFake:
        name = "fake@test"

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            if "results" in (schema or {}).get("properties", {}):  # judge: fail (no GOOD marker)
                ids = _re.findall(r"^- (\S+):", prompt, _re.M)
                return {"results": [{"criterion_id": i, "passed": False, "score": 0.0, "rationale": "x"} for i in ids]}
            return "BAD answer"  # subject regresses vs the passing baseline

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: RoleFake(spec))
    r = TestClient(app).post("/api/projects/p/drift", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["regressed"] is True and body["regressed_tests"] == ["t1"]


def test_teach_endpoints(tmp_path):
    class RoleFake:
        name = "fake@test"

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            props = (schema or {}).get("properties", {})
            if "inputs" in props:
                return {"inputs": ["q1?", "q2?"]}
            if "standards" in props:
                return {"standards": ["Be concise."], "do_not": ["No jargon."]}
            return "a candidate answer"  # subject

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: RoleFake(spec))
    c = TestClient(app)
    c.post("/api/projects", json={"name": "p", "goal": "g"})  # no spec yet

    draft = c.post("/api/projects/p/teach/draft", json={"n": 2})
    assert draft.status_code == 200
    cands = draft.json()["candidates"]
    assert len(cands) == 2 and cands[0]["output"] == "a candidate answer"

    judgments = [
        {"input": cands[0]["input"], "output": cands[0]["output"], "approved": True, "reason": "good"},
        {"input": cands[1]["input"], "output": cands[1]["output"], "approved": False, "reason": "bad"},
    ]
    learn = c.post("/api/projects/p/teach/learn", json={"judgments": judgments})
    assert learn.status_code == 200
    body = learn.json()
    assert body["standards_added"] == 1 and body["do_not_added"] == 1
    assert c.get("/api/projects/p").json()["has_spec"] is True  # spec bootstrapped from judgments


def test_log_toggle_eval_logging_and_train_engine(tmp_path):
    import re as _re

    class RoleFake:
        name = "fake@test"

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            if "results" in (schema or {}).get("properties", {}):  # judge
                ids = _re.findall(r"^- (\S+):", prompt, _re.M)
                return {"results": [{"criterion_id": i, "passed": True, "score": 1.0, "rationale": "ok"} for i in ids]}
            return "an answer"  # subject

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: RoleFake(spec))
    c = TestClient(app)

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="q", expects=["c1"])]
    save_project(proj, tmp_path / "p")

    judge_log = tmp_path / "p" / "logs" / "judge.jsonl"

    # off by default → eval logs nothing
    c.post("/api/projects/p/eval", json={})
    assert not judge_log.exists()

    # turn logging on → state reflects it → eval now logs the judge's decisions
    assert c.post("/api/projects/p/log", json={"enabled": True}).json()["log_interactions"] is True
    assert c.get("/api/projects/p").json()["log_interactions"] is True
    c.post("/api/projects/p/eval", json={})
    assert judge_log.exists()

    # train-engine assembles a bundle from the logged decisions
    r = c.post("/api/projects/p/train-engine/judge")
    assert r.status_code == 200 and r.json()["examples"] >= 1
    assert c.post("/api/projects/p/train-engine/bogus").status_code == 400  # unknown role


def test_corrupt_project_yaml_returns_400_not_500(tmp_path):
    # A malformed or schema-invalid project.yaml on disk must be a clean 400, never a 500.
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "project.yaml").write_text("{ not: valid: yaml: [")
    (tmp_path / "incomplete").mkdir()
    (tmp_path / "incomplete" / "project.yaml").write_text("name: x\n")  # missing required goal

    c = _client(tmp_path)
    assert c.get("/api/projects/broken").status_code == 400
    assert c.get("/api/projects/incomplete").status_code == 400


def test_import_endpoint(tmp_path):
    class RoleFake:
        name = "fake@test"

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            props = (schema or {}).get("properties", {})
            if "tests" in props:
                return {"tests": [{"id": "t1", "input": "q", "expects": ["clarity"], "notes": ""}]}
            return {"persona": {"voice": "concise"}, "standards": ["Be clear."], "do_not": [], "edge_cases": [],
                    "format": "", "refusal_policy": "",
                    "eval_criteria": [{"id": "clarity", "description": "d", "weight": "high"}], "examples": []}

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: RoleFake(spec))
    c = TestClient(app)

    r = c.post("/api/import", json={"name": "imported", "goal": "answer questions",
                                    "prompt": "You are concise. Always be clear."})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "imported" and body["has_spec"] is True and body["tests"] >= 1
    # it's a real project now — coverage works on the extracted spec/tests
    assert c.get("/api/projects/imported/coverage").json()["total_criteria"] >= 1
    # duplicate → 409, empty prompt → 400
    assert c.post("/api/import", json={"name": "imported", "goal": "g", "prompt": "x"}).status_code == 409
    assert c.post("/api/import", json={"name": "blank", "goal": "g", "prompt": "   "}).status_code == 400


def test_diff_endpoint(tmp_path):
    a = Project(name="a", goal="g")
    a.spec = BehaviorSpec(goal="g", standards=["keep", "drop"],
                          eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    b = Project(name="b", goal="g")
    b.spec = BehaviorSpec(goal="g", standards=["keep", "new"],
                          eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    save_project(a, tmp_path / "a")
    save_project(b, tmp_path / "b")

    c = _client(tmp_path)
    r = c.post("/api/diff", json={"before": "a", "after": "b"})
    assert r.status_code == 200
    body = r.json()
    assert body["changed"] is True
    assert body["standards"]["added"] == ["new"] and body["standards"]["removed"] == ["drop"]


def test_merge_endpoints(tmp_path):
    legal = Project(name="legal", goal="org goal")
    legal.spec = BehaviorSpec(goal="org goal", standards=["always add a disclaimer"])
    sales = Project(name="sales", goal="sales goal")
    sales.spec = BehaviorSpec(goal="sales goal", standards=["never add a disclaimer"])
    save_project(legal, tmp_path / "legal")
    save_project(sales, tmp_path / "sales")

    class ConflictFake:
        name = "fake@test"

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            return {"conflicts": [{"a": 1, "b": 2, "explanation": "contradict", "severity": "high"}]}

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: ConflictFake(spec))
    c = TestClient(app)

    det = c.post("/api/merge/detect", json={"sources": ["legal", "sales"]})
    assert det.status_code == 200
    conf = det.json()["conflicts"][0]
    assert conf["a"]["stakeholder"] == "legal" and conf["b"]["stakeholder"] == "sales"

    # resolve "keep A" → drop B's statement, build the merged project
    ap = c.post("/api/merge/apply", json={"out": "org", "sources": ["legal", "sales"], "drops": [conf["b"]["idx"]]})
    assert ap.status_code == 200 and ap.json()["name"] == "org" and ap.json()["has_spec"] is True

    merged = c.get("/api/projects/org").json()
    assert merged["has_spec"] is True
    assert c.post("/api/merge/detect", json={"sources": ["legal"]}).status_code == 400  # need >= 2


def test_csrf_guard_stays_on_when_host_is_widened(tmp_path):
    # Exposing to a specific remote host must NOT disable the cross-origin guard.
    app = create_app(tmp_path, allowed_hosts=["192.168.1.50"])
    client = TestClient(app, base_url="http://192.168.1.50")  # Host = allowed remote
    # same-origin from the allowed host works
    assert client.post("/api/projects", json={"name": "a", "goal": "g"},
                       headers={"Origin": "http://192.168.1.50:8765"}).status_code == 200
    # cross-origin is still blocked even though the host was widened
    assert client.post("/api/projects", json={"name": "b", "goal": "g"},
                       headers={"Origin": "https://evil.example"}).status_code == 403


def test_ci_endpoint(tmp_path):
    import re as _re

    class RoleFake:
        name = "fake@test"

        def __init__(self, spec):
            self.spec = spec

        def complete(self, prompt, *, system=None, schema=None):
            props = (schema or {}).get("properties", {})
            if "results" in props:  # judge → pass everything
                ids = _re.findall(r"^- (\S+):", prompt, _re.M)
                return {"results": [{"criterion_id": i, "passed": True, "score": 1.0, "rationale": "ok"} for i in ids]}
            return "an answer"  # subject

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: RoleFake(spec))
    c = TestClient(app)

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", standards=["Always answer with the documented policy."],
                             refusal_policy="decline medical questions",
                             eval_criteria=[EvalCriterion(id="c1", description="matches the documented policy",
                                                          weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="q", expects=["c1"])]
    save_project(proj, tmp_path / "p")

    r = c.post("/api/projects/p/ci", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["run_id"] == "run-0001"
    assert {s["name"]: s["status"] for s in body["stages"]} == {
        "lint": "pass", "eval": "pass", "drift": "skip", "snapshot": "skip"}

    # second run: drift now has a baseline
    r2 = c.post("/api/projects/p/ci", json={"threshold": 0.5}).json()
    assert {s["name"]: s["status"] for s in r2["stages"]}["drift"] == "pass"

    # validation: bad threshold → 422 from pydantic
    assert c.post("/api/projects/p/ci", json={"threshold": 7}).status_code == 422

    # labels persist via judge-check POST
    r3 = c.post("/api/projects/p/judge-check",
                json={"run_id": r2["run_id"],
                      "labels": [{"test_id": "t1", "criterion_id": "c1", "passed": False}]})
    assert r3.status_code == 200 and r3.json()["labels_saved"].endswith("human-labels.json")
    assert (tmp_path / "p" / "evals" / r2["run_id"] / "human-labels.json").exists()


def test_badge_and_certification_endpoints(tmp_path):
    c = _client(tmp_path)
    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    save_project(proj, tmp_path / "p")

    b = c.get("/api/projects/p/badge")
    assert b.status_code == 200
    assert b.json() == {"schemaVersion": 1, "label": "calibrated",
                        "message": "uncalibrated", "color": "lightgrey"}
    cert = c.get("/api/projects/p/certification").json()
    assert cert["status"] == "none" and cert["gate"] is None
    assert c.get("/api/projects/nope/badge").status_code == 404


def test_absorb_endpoint(tmp_path):
    from ai_calibrator.flywheel import append_feedback

    c = _client(tmp_path)
    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    save_project(proj, tmp_path / "p")
    append_feedback(tmp_path / "p", {"turns": ["q"], "output": "a", "verdict": "down"})

    r = c.post("/api/projects/p/absorb")
    assert r.status_code == 200
    body = r.json()
    assert body["downs"] == 1 and body["tests_added"] == 1 and body["state"]["tests"] == 1

    # idempotent: nothing left to absorb
    assert c.post("/api/projects/p/absorb").json()["tests_added"] == 0


def test_try_and_feedback_endpoints(tmp_path):
    """The workbench flywheel: try → thumbs → pending → absorb."""

    class Subject:
        name = "fake@test"

        def __init__(self, spec):
            self.spec = spec

        def complete(self, prompt, *, system=None, schema=None):
            assert prompt == "User: how long?\nAssistant:"       # runtime/eval encoding
            assert system and "30-day" in system                 # compiled spec is the system prompt
            return "30 days."

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: Subject(spec))
    c = TestClient(app)

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", standards=["Always cite the 30-day window."],
                             eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    save_project(proj, tmp_path / "p")

    r = c.post("/api/projects/p/try", json={"message": "how long?"})
    assert r.status_code == 200 and r.json() == {"turns": ["how long?"], "output": "30 days."}

    fb = c.post("/api/projects/p/feedback", json={
        "turns": ["how long?"], "output": "30 days.", "verdict": "down",
        "correction": "30 days, unworn only.", "reason": "missing condition"})
    assert fb.status_code == 200 and fb.json() == {"recorded": True, "pending": 1}
    assert c.get("/api/projects/p/feedback").json()["pending"] == 1

    # validation is friendly
    assert c.post("/api/projects/p/feedback", json={"turns": ["q"], "output": "a", "verdict": "meh"}).status_code == 400
    assert c.post("/api/projects/p/feedback", json={"turns": ["  "], "output": "a", "verdict": "up"}).status_code == 400
    assert c.post("/api/projects/p/try", json={"message": ""}).status_code == 422

    # absorb drains the same inbox
    r2 = c.post("/api/projects/p/absorb").json()
    assert r2["downs"] == 1 and r2["tests_added"] == 1
    assert c.get("/api/projects/p/feedback").json()["pending"] == 0


def test_overlong_project_name_is_400_not_500(tmp_path):
    """A >255-char name passed _safe() and blew up in mkdir (OSError→500)."""
    c = _client(tmp_path)
    r = c.post("/api/projects", json={"name": "a" * 1000, "goal": "g"})
    assert r.status_code == 400 and "too long" in r.json()["detail"]
    # and every routed endpoint rejects it the same way (name is the routing key)
    assert c.get(f"/api/projects/{'a' * 1000}").status_code == 400


def test_set_engines_endpoint(tmp_path):
    from ai_calibrator.store import load_project
    c = _client(tmp_path)
    c.post("/api/projects", json={"name": "p", "goal": "g"})

    r = c.put("/api/projects/p/engines", json={"all": "gemma4:e4b@ollama"})
    assert r.status_code == 200
    assert all(v == "gemma4:e4b@ollama" for v in r.json()["engines"].values())

    r2 = c.put("/api/projects/p/engines", json={"role": "judge", "model": "gpt-4o-mini@openai"})
    assert r2.status_code == 200 and r2.json()["engines"]["judge"] == "gpt-4o-mini@openai"
    assert load_project(tmp_path / "p").engines.judge == "gpt-4o-mini@openai"

    # validation
    assert c.put("/api/projects/p/engines", json={"role": "judge", "model": "x@bogus"}).status_code == 400
    assert c.put("/api/projects/p/engines", json={"role": "wizard", "model": "x@openai"}).status_code == 400
    assert c.put("/api/projects/p/engines", json={}).status_code == 400


def test_set_engines_rejects_all_plus_role(tmp_path):
    c = _client(tmp_path)
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    r = c.put("/api/projects/p/engines",
              json={"all": "gemma4:e4b@ollama", "role": "subject", "model": "gpt-4o@openai"})
    assert r.status_code == 400 and "not both" in r.json()["detail"]


def test_ingest_with_materials_succeeds(tmp_path):
    """Complement to the no-materials 400: with a file present, ingest runs."""
    payload = {"facts": ["x"], "gaps": [{"dimension": "tone", "why_it_matters": "w"}]}
    c = _client(tmp_path, engine_payload=payload)
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    c.post("/api/projects/p/materials", files={"file": ("f.md", b"policy text", "text/markdown")})
    assert c.post("/api/projects/p/ingest").status_code == 200


def test_api_bulk_add_examples(tmp_path):
    import pytest
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_calibrator.api import create_app
    from ai_calibrator.models import BehaviorSpec, Project
    from ai_calibrator.store import save_project
    p = Project(name="ex", goal="g"); p.spec = BehaviorSpec(goal="g")
    save_project(p, tmp_path / "ex")
    c = TestClient(create_app(tmp_path))
    r = c.post("/api/projects/ex/examples", json={"examples": [
        {"question": "a", "answer": "A"}, {"input": "b", "output": "B"}, {"input": "a", "output": "dup"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["added"] == 2 and body["skipped"] == 1 and body["unique_inputs"] == 2


def test_nan_body_is_422_not_500(tmp_path):
    """A NaN/Infinity float in the body is rejected by allow_inf_nan=False; the
    handler must return a clean 422, not an unhandled 500 (the default handler
    500s trying to serialize the NaN it echoes back)."""
    c = _client(tmp_path)
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    hdr = {"content-type": "application/json"}
    for url, raw in [
        ("/api/projects/p/eval", '{"threshold": NaN}'),
        ("/api/projects/p/eval", '{"threshold": Infinity}'),
        ("/api/projects/p/ci", '{"tolerance": NaN}'),
        ("/api/projects/p/drift", '{"tolerance": -Infinity}'),
    ]:
        r = c.post(url, content=raw, headers=hdr)
        assert r.status_code == 422, (url, raw, r.status_code)
    # a finite out-of-range value is still a clean 422
    assert c.post("/api/projects/p/eval", json={"threshold": 5}).status_code == 422


def test_upload_with_an_unusable_filename_is_a_400_not_a_500(tmp_path):
    """`.name` defeats traversal but leaves "" (from "."), ".." and over-long
    names, which reach os.replace and escape as a 500 + traceback. Every other
    bad input on this API is a clean 4xx."""
    c = _client(tmp_path)
    c.post("/api/projects", json={"name": "p", "goal": "g"})

    for bad in (".", "..", "x" * 300):
        r = c.post("/api/projects/p/materials", files={"file": (bad, b"hello", "text/plain")})
        assert r.status_code == 400, (bad, r.status_code)
        assert "invalid filename" in r.json()["detail"]

    # No temp files left behind, and nothing written outside materials/.
    mats = tmp_path / "p" / "materials"
    assert [f.name for f in mats.iterdir() if f.is_file()] == []


def test_upload_still_accepts_a_traversal_style_name_as_a_plain_file(tmp_path):
    c = _client(tmp_path)
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    r = c.post("/api/projects/p/materials",
               files={"file": ("../../evil.txt", b"hello", "text/plain")})
    assert r.status_code == 200 and r.json()["uploaded"] == "evil.txt"
    assert (tmp_path / "p" / "materials" / "evil.txt").exists()
    assert not (tmp_path / "evil.txt").exists()




def test_merge_detect_reports_the_field_resolution_that_ships(tmp_path):
    """detect's `resolved_to` must name what apply actually ships. persona is
    resolved PER FIELD (each field independently, first stakeholder by name that
    supplies one), so no stakeholder's voice or reading level is lost to another's
    empty one."""
    from ai_calibrator.models import Persona
    from ai_calibrator.store import load_project

    for nm, persona in (("alpha", Persona(reading_level="grade 5")),
                        ("beta", Persona(voice="terse")),
                        ("gamma", Persona(voice="chatty"))):
        p = Project(name=nm, goal="g")
        p.spec = BehaviorSpec(goal="g", persona=persona)
        save_project(p, tmp_path / nm)

    c = _client(tmp_path, engine_payload={"conflicts": []})
    sources = ["alpha", "beta", "gamma"]
    det = c.post("/api/merge/detect", json={"sources": sources})
    assert det.status_code == 200
    voice = next(f for f in det.json()["field_conflicts"] if f["field"] == "persona.voice")

    assert c.post("/api/merge/apply", json={"out": "org", "sources": sources}).status_code == 200
    shipped = load_project(tmp_path / "org").spec.persona.voice
    # alpha has no voice, so the first name that does (beta) supplies it.
    assert shipped == "terse"
    assert voice["resolved_to"]["value"] == shipped     # detect promised what apply did


def test_api_refine_persists_each_round_before_the_next_one_is_graded(tmp_path):
    # A refine round that dies upstream must not leave a scorecard on disk that no
    # recorded spec produced: the standards the earlier round added are saved as
    # they are earned, exactly as the CLI's `eval --refine` does.
    import re as _re

    from ai_calibrator.engines.base import EngineError
    from ai_calibrator.store import load_project

    class FlakyRoleFake:
        name = "fake@test"
        subject_calls = []

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            props = (schema or {}).get("properties", {})
            if "new_standards" in props:  # refiner
                return {"new_standards": ["Cite the policy section you relied on."]}
            if "results" in props:  # judge → fail everything, so round 1 refines
                ids = _re.findall(r"^- (\S+):", prompt, _re.M)
                return {"results": [{"criterion_id": i, "passed": False, "score": 0.0,
                                     "rationale": "no"} for i in ids]}
            self.subject_calls.append(1)  # subject
            if len(self.subject_calls) > 1:  # the provider dies during round 2
                raise EngineError("upstream is down")
            return "an answer"

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: FlakyRoleFake(spec))
    c = TestClient(app)

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="q", expects=["c1"])]
    save_project(proj, tmp_path / "p")

    r = c.post("/api/projects/p/eval", json={"refine": True, "rounds": 3})
    assert r.status_code == 502  # upstream failure, surfaced honestly
    assert (tmp_path / "p" / "evals" / "run-0001" / "scorecard.json").exists()
    assert "Cite the policy section you relied on." in load_project(tmp_path / "p").spec.standards


def test_delete_moves_the_tree_aside_before_removing_it(tmp_path, monkeypatch):
    """POSIX: deleting IN PLACE destroys the lock file providing the mutual
    exclusion — unlink succeeds on an open file there, so a waiter creates a
    fresh `.lock` at the same path and acquires it while this delete is still
    running. Renaming takes the lock file with the tree, so the name is free the
    instant the rename lands and a project recreated under it is never caught by
    the removal."""
    import os
    import shutil
    from pathlib import Path

    from ai_calibrator.store import load_project

    if os.name == "nt":
        pytest.skip("platform-specific: Windows cannot rename a directory holding an open handle")

    c = _client(tmp_path)
    assert c.post("/api/projects", json={"name": "p", "goal": "old"}).status_code == 200

    real_rmtree = shutil.rmtree
    removed = []

    def watching_rmtree(path, *a, **kw):
        removed.append(Path(path))
        # A create wins the name the instant the rename frees it — the window the
        # in-place delete could not survive.
        save_project(Project(name="p", goal="brand new"), tmp_path / "p")
        real_rmtree(path, *a, **kw)

    monkeypatch.setattr(shutil, "rmtree", watching_rmtree)
    r = c.request("DELETE", "/api/projects/p")

    assert r.status_code == 200, r.text
    assert removed and removed[0] != tmp_path / "p"          # never the live path
    assert load_project(tmp_path / "p").goal == "brand new"  # the new project is untouched


def test_delete_retry_cannot_destroy_a_project_recreated_in_the_window(tmp_path, monkeypatch):
    """Windows: the tree is deleted in place, and the open lock handle blocks
    unlinking `.lock` — so the directory survives the first pass and a second one
    runs after the lock is released. A create can win the name in between, and
    that retry must leave the new project alone."""
    import os
    import shutil
    from pathlib import Path

    from ai_calibrator.store import load_project

    if os.name != "nt":
        pytest.skip("platform-specific: POSIX renames the tree aside, so there is no second pass")

    c = _client(tmp_path)
    assert c.post("/api/projects", json={"name": "p", "goal": "old"}).status_code == 200

    real_rmtree = shutil.rmtree
    seen = []

    def fake_rmtree(path, *a, **kw):
        p = Path(path)
        seen.append(p)
        if len(seen) > 1:
            real_rmtree(p, *a, **kw)
            return
        for child in p.iterdir():
            if child.name == ".lock":
                continue  # an open handle: Windows refuses to unlink it
            real_rmtree(child) if child.is_dir() else child.unlink()
        save_project(Project(name="p", goal="brand new"), p)  # a concurrent create

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)
    r = c.request("DELETE", "/api/projects/p")

    assert load_project(tmp_path / "p").goal == "brand new"  # survived the retry
    assert r.status_code == 409  # files remain, so the delete never claims success


def test_delete_reports_failure_when_the_tree_cannot_be_moved(tmp_path, monkeypatch):
    """A delete that removed nothing must not claim success."""
    import os

    if os.name == "nt":
        pytest.skip("platform-specific: Windows deletes in place, so os.replace is not on that path")
    c = _client(tmp_path)
    assert c.post("/api/projects", json={"name": "p", "goal": "g"}).status_code == 200

    def refuse(src, dst):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(os, "replace", refuse)
    r = c.request("DELETE", "/api/projects/p")

    assert r.status_code == 409 and "could not delete" in r.text
    assert (tmp_path / "p" / "project.yaml").exists()   # still there, as reported


def test_api_snapshot_refuses_to_pin_a_partial_run(tmp_path):
    from ai_calibrator.eval import save_scorecard
    from ai_calibrator.models import Scorecard, TestResult

    proj = Project(name="p", goal="g")
    proj.spec = BehaviorSpec(goal="g", eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    proj.tests = [CaseModel(id="t1", input="q", expects=["c1"]), CaseModel(id="t2", input="q2", expects=["c1"])]
    save_project(proj, tmp_path / "p")
    save_scorecard(tmp_path / "p", Scorecard(run_id="run-0001", results=[
        TestResult(test_id="t1", output="one"), TestResult(test_id="t2", output="two")]))

    c = _client(tmp_path)
    assert c.post("/api/projects/p/snapshot").json()["pinned"] == 2

    # An interrupted run graded half the suite; pinning it would shrink the golden.
    save_scorecard(tmp_path / "p", Scorecard(run_id="run-0002", partial=True,
                                             results=[TestResult(test_id="t1", output="one")]))
    r = c.post("/api/projects/p/snapshot")
    assert r.status_code == 409 and "PARTIAL" in r.json()["detail"]

    # The golden still covers both — read it directly rather than inferring it
    # from a check against the partial run. CHECKING against a partial run is
    # refused for the same reason PINNING is: t2 is ungraded, not missing, and
    # reporting it as `removed` is drift the model never caused.
    import json as _json
    assert set(_json.loads((tmp_path / "p" / "golden.json").read_text())) == {"t1", "t2"}
    r = c.get("/api/projects/p/snapshot")
    assert r.status_code == 409 and "PARTIAL" in r.json()["detail"]


def test_merge_detect_reports_the_persona_apply_really_keeps(tmp_path):
    from ai_calibrator.models import Persona
    from ai_calibrator.store import load_project

    class NoConflictFake:
        name = "fake@test"

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            return {"conflicts": []}

    for nm, persona in [("alpha", Persona(reading_level="grade 3")),
                        ("beta", Persona(voice="pirate", reading_level="grade 9")),
                        ("gamma", Persona(voice="formal", reading_level="grade 12"))]:
        p = Project(name=nm, goal=f"{nm} goal")
        p.spec = BehaviorSpec(goal=f"{nm} goal", persona=persona, standards=[f"{nm} rule"])
        save_project(p, tmp_path / nm)

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: NoConflictFake(spec))
    c = TestClient(app)

    det = c.post("/api/merge/detect", json={"sources": ["alpha", "beta", "gamma"]})
    assert det.status_code == 200
    reported = {f["field"]: f["resolved_to"] for f in det.json()["field_conflicts"]}

    assert c.post("/api/merge/apply", json={"out": "org", "sources": ["alpha", "beta", "gamma"]}).status_code == 200
    kept = load_project(tmp_path / "org").spec.persona

    # Per-field: reading_level from alpha (first by name with one), voice from
    # beta (alpha supplies none). Neither stakeholder's value is lost to the
    # other's empty field, and detect promises exactly that.
    assert kept.voice == "pirate" and kept.reading_level == "grade 3"
    assert reported["persona.voice"] == {"stakeholder": "beta", "value": "pirate"}
    assert reported["persona.reading_level"] == {"stakeholder": "alpha", "value": "grade 3"}




def test_ingest_after_deleting_every_material_clears_the_corpus(tmp_path):
    """Parity with the CLI: an empty materials folder on a project that HAS been
    ingested is a legitimate 'remove everything', not a 400. Refusing it left the
    facts, gaps and retrieval index of deleted files in every served prompt."""
    from ai_calibrator.models import Gap, Material

    proj = Project(name="p", goal="g")
    proj.materials = [Material(path="faq.md", kind="md", summary="old policy")]
    proj.facts = ["Returns are accepted for 30 days."]
    proj.gaps = [Gap(dimension="tone")]
    save_project(proj, tmp_path / "p")
    (tmp_path / "p" / "knowledge.lancedb").mkdir()

    c = _client(tmp_path, engine_payload={"facts": [], "gaps": []})
    r = c.post("/api/projects/p/ingest")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["materials"] == 0 and body["facts"] == 0 and body["gaps"] == 0
    assert body["state"]["materials"] == [] and body["state"]["gaps"] == []
    assert not (tmp_path / "p" / "knowledge.lancedb").exists()


def test_ingest_reports_how_many_files_were_analyzed(tmp_path, monkeypatch):
    """The web UI can only warn about a truncated corpus if the route says so."""
    import ai_calibrator.ingest as ing

    monkeypatch.setattr(ing, "MAX_EXTRACT_CHARS", 60)
    c = _client(tmp_path, engine_payload={"facts": [], "gaps": []})
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    c.post("/api/projects/p/materials", files={"file": ("a.md", b"a" * 400, "text/markdown")})
    c.post("/api/projects/p/materials", files={"file": ("b.md", b"b" * 400, "text/markdown")})

    body = c.post("/api/projects/p/ingest").json()
    assert body["materials"] == 2
    # Neither file FIT the window — the first was truncated at the cap and the
    # second never reached the extractor at all. Counting a partially-read file
    # as analyzed is what let a single oversized material report "1 of 1
    # analyzed" and suppress the warning entirely.
    assert body["analyzed"] == 0




def test_import_endpoint_writes_the_project_gitignore(tmp_path):
    """The API import path must ignore logs/, evals/ and .env exactly as `init`
    does — an imported project is the one most likely to end up in git."""
    class RoleFake:
        name = "fake@test"

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            props = (schema or {}).get("properties", {})
            if "tests" in props:
                return {"tests": [{"id": "t1", "input": "q", "expects": ["clarity"], "notes": ""}]}
            return {"persona": {"voice": "concise"}, "standards": ["Be clear."], "do_not": [], "edge_cases": [],
                    "format": "", "refusal_policy": "",
                    "eval_criteria": [{"id": "clarity", "description": "d", "weight": "high"}], "examples": []}

    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: RoleFake(spec))
    c = TestClient(app)
    assert c.post("/api/import", json={"name": "imp", "goal": "g", "prompt": "Be clear."}).status_code == 200
    body = (tmp_path / "imp" / ".gitignore").read_text(encoding="utf-8")
    assert "logs/" in body and "evals/" in body and ".env" in body


def test_train_engine_route_names_the_roles_that_are_logged(tmp_path):
    """Only the judge and compiler are ever wrapped by the logging engine."""
    save_project(Project(name="p", goal="g"), tmp_path / "p")
    r = _client(tmp_path).post("/api/projects/p/train-engine/extractor")
    assert r.status_code == 400
    assert "nothing records the extractor role" in r.json()["detail"]


def test_teach_learn_keeps_judgments_when_inference_fails(tmp_path):
    """Parity with the CLI: the judgments are a person's attention and do not
    depend on the inferred standards, so an engine failure must not discard them."""
    from ai_calibrator.store import load_project

    from ai_calibrator.engines.base import EngineError

    class BoomInfer:
        name = "fake@test"

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            raise EngineError("upstream is down")

    c = _client(tmp_path)
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: BoomInfer(spec))
    c2 = TestClient(app, raise_server_exceptions=False)

    r = c2.post("/api/projects/p/teach/learn", json={"judgments": [
        {"input": "q1", "output": "a1", "approved": True, "reason": None},
        {"input": "q2", "output": "a2", "approved": False, "reason": "too vague"},
    ]})
    assert r.status_code in (502, 504)                      # the failure is reported
    spec = load_project(tmp_path / "p").spec
    assert spec is not None
    assert [e.input for e in spec.examples] == ["q1", "q2"]  # and nothing was lost


def test_teach_learn_records_each_judgment_once(tmp_path):
    """The checkpoint must not double-record — the defect the CLI had."""
    from ai_calibrator.store import load_project

    class Infer:
        name = "fake@test"

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            return {"standards": ["S1"], "do_not": []}

    c = _client(tmp_path)
    c.post("/api/projects", json={"name": "p", "goal": "g"})
    app = create_app(tmp_path)
    app.dependency_overrides[_engine_factory] = lambda: (lambda spec: Infer(spec))
    r = TestClient(app).post("/api/projects/p/teach/learn", json={"judgments": [
        {"input": "q1", "output": "a1", "approved": True, "reason": None}]})
    assert r.status_code == 200

    spec = load_project(tmp_path / "p").spec
    assert len(spec.examples) == 1
    assert spec.standards == ["S1"]


def test_merge_apply_writes_the_protective_gitignore(tmp_path):
    """Parity with the CLI's merge and with POST /api/projects."""
    from ai_calibrator.models import BehaviorSpec, Project
    from ai_calibrator.store import save_project

    class NoConflict:
        name = "fake@test"

        def __init__(self, spec):
            pass

        def complete(self, prompt, *, system=None, schema=None):
            return {"conflicts": []}

    for nm in ("alpha", "beta"):
        p = Project(name=nm, goal="g")
        p.spec = BehaviorSpec(goal="g", standards=[f"{nm} rule"])
        save_project(p, tmp_path / nm)

    app_ = create_app(tmp_path)
    app_.dependency_overrides[_engine_factory] = lambda: (lambda spec: NoConflict(spec))
    r = TestClient(app_).post("/api/merge/apply",
                              json={"out": "org", "sources": ["alpha", "beta"]})
    assert r.status_code == 200, r.text
    gitignore = tmp_path / "org" / ".gitignore"
    assert gitignore.is_file(), "merged project has no .gitignore"
    assert "logs/" in gitignore.read_text(encoding="utf-8")


def test_answers_are_not_applied_to_a_question_that_changed(tmp_path):
    """The API twin of the CLI's misattachment guard. A client that echoes what
    it asked gets its answer dropped rather than recorded against a question the
    user never saw; a client that echoes nothing keeps the old behavior."""
    import pytest
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_calibrator.api import create_app
    from ai_calibrator.models import InterviewItem, Project
    from ai_calibrator.store import load_project, save_project

    p = Project(name="p", goal="g")
    p.interview = [InterviewItem(id="q1", dimension="refund", question="What is the refund window?")]
    save_project(p, tmp_path / "p")
    c = TestClient(create_app(tmp_path))

    r = c.post("/api/projects/p/answers",
               json={"answers": {"q1": "30 days"}, "asked": {"q1": "What tone?"}})

    assert r.status_code == 200
    assert r.json()["applied"] == 0 and r.json()["misattached"] == 1
    assert load_project(tmp_path / "p").interview[0].answer is None

    r2 = c.post("/api/projects/p/answers",
                json={"answers": {"q1": "30 days"}, "asked": {"q1": "What is the refund window?"}})
    assert r2.json()["applied"] == 1
    assert load_project(tmp_path / "p").interview[0].answer == "30 days"


def test_merge_apply_refuses_drop_indices_from_a_stale_detect(tmp_path):
    """`drops` are positions in the set /merge/detect gathered. A source edited
    between the two calls renumbers that set, so the same indices delete
    different rules — silently, with a 200 — and the merged AI ships the
    contradiction the owner thought they had resolved."""
    import pytest
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_calibrator.api import create_app
    from ai_calibrator.models import BehaviorSpec, Project
    from ai_calibrator.store import load_project, save_project

    for name, standards in (("legal", ["L-s1", "Always add a disclaimer"]),
                            ("sales", ["S-s1", "Never add disclaimers"])):
        p = Project(name=name, goal="g")
        p.spec = BehaviorSpec(goal="g", standards=standards)
        save_project(p, tmp_path / name)

    c = TestClient(create_app(tmp_path))
    fingerprint = "0" * 16          # what a stale client would echo back

    # A source changes between detect and apply.
    legal = load_project(tmp_path / "legal")
    legal.spec.standards.insert(0, "L-NEW from teach")
    save_project(legal, tmp_path / "legal")

    r = c.post("/api/merge/apply", json={"out": "merged", "sources": ["legal", "sales"],
                                         "drops": [4], "statements_fingerprint": fingerprint})

    assert r.status_code == 409
    assert "no longer name the same rules" in r.json()["detail"]
    assert not (tmp_path / "merged").exists()
