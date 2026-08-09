"""`calibrate ci` — the composed lint → eval → drift → snapshot gate."""

import re

from ai_calibrator.ci import ci_dict, run_ci
from ai_calibrator.eval import run_eval, save_scorecard
from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
from ai_calibrator.models import TestCase as CaseModel
from ai_calibrator.snapshot import save_golden


class Judge:
    """Passes iff the output contains GOOD."""
    name = "judge@test"

    def complete(self, prompt, *, system=None, schema=None):
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        good = "GOOD" in prompt
        return {"results": [{"criterion_id": i, "passed": good, "score": 1.0 if good else 0.0,
                             "rationale": "r"} for i in ids]}


class Subject:
    name = "subject@test"

    def __init__(self, text):
        self.text = text

    def complete(self, prompt, *, system=None, schema=None):
        return self.text


def _project():
    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", standards=["Always answer with the documented policy."],
                          refusal_policy="decline medical questions",
                          eval_criteria=[EvalCriterion(id="c1", description="answer matches the documented policy",
                                                       weight=Weight.HIGH)])
    p.tests = [CaseModel(id="t1", input="a question", expects=["c1"])]
    return p


def _statuses(result):
    return {s.name: s.status for s in result.stages}


def test_ci_all_green_first_run(tmp_path):
    r = run_ci(_project(), Subject("GOOD answer"), Judge(), project_dir=tmp_path)
    assert _statuses(r) == {"lint": "pass", "eval": "pass",
                            "drift": "skip",       # no baseline yet
                            "snapshot": "skip"}    # no golden pinned
    assert r.ok and r.run_id == "run-0001" and r.pass_rate == 1.0
    d = ci_dict(r)
    assert d["ok"] is True and len(d["stages"]) == 4


def test_ci_lint_error_stops_before_any_engine_acquisition(tmp_path):
    """Lint failure must exit before engines are even ACQUIRED — a broken spec
    shouldn't demand credentials, and an engine problem shouldn't mask lint."""
    def exploding_factory():
        raise AssertionError("engine factory must not be called when lint fails")

    p = _project()
    p.spec.eval_criteria = []  # no criteria → lint error
    r = run_ci(p, exploding_factory, exploding_factory, project_dir=tmp_path)
    assert _statuses(r) == {"lint": "fail", "eval": "skip", "drift": "skip", "snapshot": "skip"}
    assert not r.ok and r.run_id is None


def test_ci_accepts_engine_factories(tmp_path):
    """Factories resolve lazily after lint passes (the CLI/API path)."""
    calls = []

    def subject_factory():
        calls.append("subject")
        return Subject("GOOD answer")

    def judge_factory():
        calls.append("judge")
        return Judge()

    r = run_ci(_project(), subject_factory, judge_factory, project_dir=tmp_path)
    assert r.ok and calls == ["subject", "judge"]


def test_drift_stage_does_not_report_pass_when_the_suite_was_recompiled(tmp_path):
    """The documented loop is compile -> eval -> answer more questions -> compile
    -> ci, and `compile` re-mints t1..tN positionally. The baseline then graded a
    different question under the same id, so there is nothing to compare — and
    "no regressions" would certify a comparison that never happened."""
    p = _project()
    save_scorecard(tmp_path, run_eval(p, Subject("GOOD baseline"), Judge(), run_id="run-0001"))

    # `compile` re-mints t1 onto a different question, exactly as it does in the
    # real workflow; the id is unchanged.
    p.tests = [CaseModel(id="t1", input="a COMPLETELY different question", expects=["c1"])]

    r = run_ci(p, Subject("BAD now"), Judge(), project_dir=tmp_path, threshold=0.0)
    drift = next(s for s in r.stages if s.name == "drift")
    assert drift.status == "skip"                      # not "pass"
    assert "not comparable" in drift.detail and "changed content" in drift.detail
    assert "no regressions" not in drift.detail


def test_drift_stage_still_compares_the_subset_that_did_not_change(tmp_path):
    """The skip is keyed on "nothing left to compare", NOT on "no test flipped".

    One re-minted probe with the rest still holding is the ordinary state after
    answering another interview question. Skipping there would mean the drift
    stage never passes again until someone re-baselines, which is how a gate
    teaches people to ignore it."""
    p = _project()
    p.tests = [CaseModel(id="t1", input="a question", expects=["c1"]),
               CaseModel(id="t2", input="a second question", expects=["c1"])]
    save_scorecard(tmp_path, run_eval(p, Subject("GOOD baseline"), Judge(), run_id="run-0001"))

    # `compile` re-mints only t1; t2 still asks what it asked.
    p.tests = [CaseModel(id="t1", input="a DIFFERENT question", expects=["c1"]),
               CaseModel(id="t2", input="a second question", expects=["c1"])]

    r = run_ci(p, Subject("GOOD still"), Judge(), project_dir=tmp_path, threshold=0.0)
    drift = next(s for s in r.stages if s.name == "drift")
    assert drift.status == "pass"                       # it DID compare something
    assert "1 shared test(s)" in drift.detail           # and says how much
    assert "1 not comparable" in drift.detail           # and what it left out


def test_drift_stage_still_compares_when_the_suite_is_unchanged(tmp_path):
    """The fix must not make every ordinary run incomparable: an unchanged suite
    re-evaluated still drifts against its baseline the way it always did."""
    p = _project()
    save_scorecard(tmp_path, run_eval(p, Subject("GOOD baseline"), Judge(), run_id="run-0001"))
    r = run_ci(p, Subject("BAD now"), Judge(), project_dir=tmp_path, threshold=0.0)
    drift = next(s for s in r.stages if s.name == "drift")
    assert drift.status == "fail" and "t1" in drift.detail


def test_ci_eval_below_threshold_fails_but_later_stages_still_run(tmp_path):
    p = _project()
    save_scorecard(tmp_path, run_eval(p, Subject("GOOD baseline"), Judge(), run_id="run-0001"))
    r = run_ci(p, Subject("BAD now"), Judge(), project_dir=tmp_path, threshold=0.8)
    st = _statuses(r)
    assert st["eval"] == "fail"
    assert st["drift"] == "fail"          # t1 flipped pass→fail vs run-0001
    assert not r.ok


def test_ci_drift_regression_fails_even_when_eval_passes_threshold(tmp_path):
    p = _project()
    p.tests.append(CaseModel(id="t2", input="another", expects=["c1"]))

    class SplitSubject:  # t1 GOOD, t2 BAD → 50% pass
        name = "s@test"

        def complete(self, prompt, *, system=None, schema=None):
            return "GOOD" if "a question" in prompt else "BAD"

    save_scorecard(tmp_path, run_eval(p, Subject("GOOD everywhere"), Judge(), run_id="run-0001"))
    r = run_ci(p, SplitSubject(), Judge(), project_dir=tmp_path, threshold=0.5)
    st = _statuses(r)
    assert st["eval"] == "pass"           # 50% >= threshold 0.5
    assert st["drift"] == "fail"          # but t2 regressed vs baseline
    assert "t2" in next(s.detail for s in r.stages if s.name == "drift")


def test_ci_snapshot_drift_fails_and_match_passes(tmp_path):
    p = _project()
    save_golden(tmp_path, {"t1": "GOOD answer"})
    ok = run_ci(p, Subject("GOOD answer"), Judge(), project_dir=tmp_path)
    assert _statuses(ok)["snapshot"] == "pass"

    changed = run_ci(p, Subject("GOOD but reworded"), Judge(), project_dir=tmp_path)
    st = changed.stages[-1]
    assert st.status == "fail" and "t1" in st.detail
    assert not changed.ok


def test_ci_explicit_baseline(tmp_path):
    p = _project()
    save_scorecard(tmp_path, run_eval(p, Subject("BAD old"), Judge(), run_id="run-0001"))   # failing baseline
    save_scorecard(tmp_path, run_eval(p, Subject("GOOD mid"), Judge(), run_id="run-0002"))
    # default baseline would be run-0002; pin run-0001 → t1 goes fail→pass = no regression
    r = run_ci(p, Subject("GOOD now"), Judge(), project_dir=tmp_path, baseline="run-0001")
    drift = next(s for s in r.stages if s.name == "drift")
    assert drift.status == "pass" and "run-0001" in drift.detail and "fixed" in drift.detail

    # a bogus explicit baseline must fail loudly, not silently skip
    r2 = run_ci(p, Subject("GOOD"), Judge(), project_dir=tmp_path, baseline="nope")
    assert next(s for s in r2.stages if s.name == "drift").status == "fail"


def test_gate_record_persisted_and_certification_status(tmp_path):
    """ci persists its verdict (pass AND fail); `run`'s boot gate reads it."""
    from ai_calibrator.ci import certification_status, config_hash, latest_gate

    p = _project()
    assert certification_status(p, tmp_path)[0] == "none"        # never gated

    r = run_ci(p, Subject("GOOD"), Judge(), project_dir=tmp_path)
    assert r.ok
    gate = latest_gate(tmp_path)
    assert gate["ok"] is True and gate["run_id"] == "run-0001"
    assert gate["config_hash"] == config_hash(p) and gate["finished_at"]
    assert certification_status(p, tmp_path)[0] == "pass"

    # spec change → the old certification is STALE, not still-green
    p.spec.standards.append("Always answer in French.")
    status, detail = certification_status(p, tmp_path)
    assert status == "stale" and "re-run" in detail

    # failing gate is recorded too — a red gate is a fact, not a secret
    run_ci(p, Subject("BAD"), Judge(), project_dir=tmp_path)
    status, detail = certification_status(p, tmp_path)
    assert status == "fail" and "eval" in detail


def test_gate_record_survives_lint_fail(tmp_path):
    from ai_calibrator.ci import latest_gate

    p = _project()
    p.spec.eval_criteria = []
    run_ci(p, Subject("x"), Judge(), project_dir=tmp_path)
    gate = latest_gate(tmp_path)
    assert gate["ok"] is False and gate["run_id"] is None


def test_gate_file_does_not_confuse_run_listing(tmp_path):
    """evals/last-gate.json (a FILE) must not break latest_run_id's dir scan."""
    from ai_calibrator.eval import latest_run_id

    p = _project()
    run_ci(p, Subject("GOOD"), Judge(), project_dir=tmp_path)
    assert latest_run_id(tmp_path) == "run-0001"


def test_certification_stales_when_tests_or_checks_change(tmp_path):
    """config_hash covers the grading contract + suite, not just the prompt."""
    from ai_calibrator.ci import certification_status
    from ai_calibrator.models import Check

    p = _project()
    run_ci(p, Subject("GOOD"), Judge(), project_dir=tmp_path)
    assert certification_status(p, tmp_path)[0] == "pass"

    p.tests.append(CaseModel(id="fb_1", input="new pinned test"))     # new test → stale
    assert certification_status(p, tmp_path)[0] == "stale"
    p.tests.pop()
    assert certification_status(p, tmp_path)[0] == "pass"

    p.spec.eval_criteria[0].check = Check(kind="contains", value="x")  # grading change → stale
    assert certification_status(p, tmp_path)[0] == "stale"


def test_criteria_reordering_does_not_stale_certification(tmp_path):
    """Criteria were once hashed in list order (tests are sorted) — reordering
    YAML entries spuriously staled the certification."""
    from ai_calibrator.ci import certification_status, config_hash

    p = _project()
    p.spec.eval_criteria.append(EvalCriterion(id="c2", description="another thing", weight=Weight.LOW))
    p.tests[0].expects = []
    run_ci(p, Subject("GOOD"), Judge(), project_dir=tmp_path)
    assert certification_status(p, tmp_path)[0] == "pass"

    p.spec.eval_criteria.reverse()                       # pure reorder → still certified
    assert certification_status(p, tmp_path)[0] == "pass"

    p.spec.eval_criteria[0].description = "changed"      # content change → stale
    assert certification_status(p, tmp_path)[0] == "stale"
    assert config_hash(p) != ""


def test_config_hash_ignores_list_reordering(tmp_path):
    """Reordering standards/do_not/edge_cases (or criteria) in the YAML must NOT
    stale a certification — only real content changes do."""
    from ai_calibrator.ci import config_hash
    from ai_calibrator.models import EdgeCase

    p = _project()
    p.spec.standards = ["Always cite the policy number.", "Be warm and concise."]
    p.spec.do_not = ["Never promise exceptions.", "Never invent a policy."]
    p.spec.edge_cases = [EdgeCase(situation="used item", ruling="30 days if defective"),
                         EdgeCase(situation="gift", ruling="store credit")]
    p.spec.eval_criteria.append(EvalCriterion(id="c2", description="d2", weight=Weight.LOW))
    base = config_hash(p)

    p.spec.standards.reverse()
    p.spec.do_not.reverse()
    p.spec.edge_cases.reverse()
    p.spec.eval_criteria.reverse()
    assert config_hash(p) == base                       # pure reorder → same fingerprint

    p.spec.standards[0] = "Always cite the policy number AND the fee."
    assert config_hash(p) != base                       # real edit → different
