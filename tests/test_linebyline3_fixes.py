"""Regression tests for the 3rd line-by-line audit fixes."""

import pytest

from ai_calibrator.models import BehaviorSpec, Persona


def test_render_system_prompt_no_double_space_when_voice_empty():
    from ai_calibrator.compile import render_system_prompt
    spec = BehaviorSpec(goal="g", persona=Persona(voice="", reading_level="grade 8"))
    out = render_system_prompt(spec)
    assert "VOICE:  " not in out                       # no double space
    assert "VOICE: (reading level: grade 8)" in out


def test_load_golden_non_dict_is_unpinned(tmp_path):
    from ai_calibrator.snapshot import GOLDEN_FILE, load_golden
    (tmp_path / GOLDEN_FILE).write_text("[1, 2, 3]")   # valid JSON, but not a dict
    assert load_golden(tmp_path) is None               # treated as no/corrupt golden, not {}


def test_refine_spec_dedups_by_trimmed_content(tmp_path):
    from ai_calibrator.models import CriterionResult, Project, Scorecard, TestResult
    from ai_calibrator.pipeline import refine_spec

    class Gen:
        name = "g@t"
        def complete(self, prompt, *, system=None, schema=None):
            return {"new_standards": ["  be concise  ", "be concise", "  ", "cite sources"]}

    p = Project(name="p", goal="g")
    p.spec = BehaviorSpec(goal="g", standards=["cite sources"])   # already present
    card = Scorecard(run_id="r", results=[TestResult(test_id="t1", output="x", criteria=[
        CriterionResult(criterion_id="c1", passed=False, score=0.0, rationale="bad")])])
    fresh = refine_spec(p, card, Gen())
    assert fresh == ["be concise"]                     # trimmed, de-duped, existing excluded, blank dropped


def test_agreement_dict_no_divzero_on_empty_criterion(tmp_path):
    from ai_calibrator.judge_check import JudgeAgreement, agreement_dict
    ag = JudgeAgreement(total=0, agreed=0, by_criterion={"c1": (0, 0)})
    out = agreement_dict(ag)                            # must not raise ZeroDivisionError
    assert out["by_criterion"]["c1"]["rate"] == 0.0


def test_validate_port_helper():
    import typer
    from ai_calibrator.cli import _validate_port
    for ok in (1, 80, 8600, 65535):
        _validate_port(ok)                             # no raise
    for bad in (0, -1, 65536, 100000):
        with pytest.raises(typer.Exit):
            _validate_port(bad)


def test_ci_body_rejects_infinite_tolerance():
    from pydantic import ValidationError

    from ai_calibrator.api import CiBody
    with pytest.raises(ValidationError):
        CiBody(tolerance=float("inf"))
    with pytest.raises(ValidationError):
        CiBody(tolerance=2.0)                          # > 1.0 rejected
    assert CiBody(tolerance=0.05).tolerance == 0.05    # valid still works
