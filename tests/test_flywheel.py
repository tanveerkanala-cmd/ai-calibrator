"""The flywheel — live feedback → spec examples + pinned regression tests."""

import json

from ai_calibrator.flywheel import absorb_feedback, append_feedback, read_feedback
from ai_calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
from ai_calibrator.models import TestCase as CaseModel


def _project():
    p = Project(name="p", goal="answer return questions")
    p.spec = BehaviorSpec(goal="g", eval_criteria=[
        EvalCriterion(id="c1", description="on-policy", weight=Weight.HIGH)])
    p.tests = [CaseModel(id="t1", input="existing test", expects=["c1"])]
    return p


def test_absorb_down_with_correction_pins_example_and_test(tmp_path):
    p = _project()
    append_feedback(tmp_path, {"turns": ["Can I return after 40 days?"],
                               "output": "Sure, any time!", "verdict": "down",
                               "correction": "No — the window is 30 days.",
                               "reason": "invented policy"})
    r = absorb_feedback(p, tmp_path)
    assert (r.downs, r.examples_added, r.tests_added) == (1, 1, 1)

    ex = p.spec.examples[-1]
    assert ex.bad_output == "Sure, any time!" and ex.good_output == "No — the window is 30 days."
    assert ex.why == "invented policy"

    t = p.tests[-1]
    assert t.id == "fb_1" and t.input == "Can I return after 40 days?"
    assert t.expects == [] and "live feedback (down)" in t.notes


def test_absorb_up_and_multi_turn_keeps_follow_ups(tmp_path):
    p = _project()
    append_feedback(tmp_path, {"turns": ["hi", "and after 40 days?"],
                               "output": "The window is 30 days.", "verdict": "up"})
    r = absorb_feedback(p, tmp_path)
    assert r.ups == 1
    assert p.spec.examples[-1].good_output == "The window is 30 days."
    assert p.spec.examples[-1].input == "and after 40 days?"   # example = the turn that produced it
    t = p.tests[-1]
    assert t.input == "hi" and t.follow_ups == ["and after 40 days?"]  # test replays the whole chat


def test_absorb_is_idempotent_and_archives(tmp_path):
    p = _project()
    append_feedback(tmp_path, {"turns": ["q"], "output": "a", "verdict": "down"})
    first = absorb_feedback(p, tmp_path)
    assert first.tests_added == 1
    assert read_feedback(tmp_path) == []                                  # inbox emptied
    assert "q" in (tmp_path / "logs" / "feedback-absorbed.jsonl").read_text(encoding="utf-8")  # audit trail

    # same feedback arriving again → recognized as duplicate, nothing re-added
    append_feedback(tmp_path, {"turns": ["q"], "output": "a", "verdict": "down"})
    again = absorb_feedback(p, tmp_path)
    assert again.downs == 1 and again.examples_added == 0 and again.tests_added == 0


def test_absorb_skips_malformed_and_survives_junk_file(tmp_path):
    p = _project()
    d = tmp_path / "logs"
    d.mkdir()
    (d / "feedback.jsonl").write_text(
        'not json\n'
        '{"turns": [], "output": "a", "verdict": "down"}\n'          # no turns
        '{"turns": ["q"], "output": "", "verdict": "down"}\n'        # no output
        '{"turns": ["q"], "output": "a", "verdict": "meh"}\n'        # bad verdict
        '{"turns": ["ok"], "output": "fine", "verdict": "up"}\n')    # valid
    r = absorb_feedback(p, tmp_path)
    assert r.skipped == 3 and r.ups == 1 and r.tests_added == 1


def test_absorb_bootstraps_spec_and_fb_ids_dont_collide(tmp_path):
    p = Project(name="p", goal="g")          # no spec at all
    p.tests = [CaseModel(id="fb_1", input="taken")]
    append_feedback(tmp_path, {"turns": ["q"], "output": "a", "verdict": "down"})
    r = absorb_feedback(p, tmp_path)
    assert p.spec is not None and r.test_ids == ["fb_2"]   # skipped the taken id


def test_absorbing_feedback_makes_certification_stale(tmp_path):
    """The whole point: learning from live use un-certifies until re-proven."""
    from ai_calibrator.ci import config_hash

    p = _project()
    before = config_hash(p)
    append_feedback(tmp_path, {"turns": ["q"], "output": "a", "verdict": "down"})
    absorb_feedback(p, tmp_path)
    assert config_hash(p) != before


def test_feedback_round_trip_unicode(tmp_path):
    append_feedback(tmp_path, {"turns": ["¿devolución? 🙂"], "output": "sí — 30 días", "verdict": "up"})
    rec = read_feedback(tmp_path)[0]
    assert rec["turns"] == ["¿devolución? 🙂"] and rec["output"] == "sí — 30 días"
    raw = (tmp_path / "logs" / "feedback.jsonl").read_text(encoding="utf-8")
    assert "🙂" in raw  # ensure_ascii=False — the file stays human-readable
    assert json.loads(raw)  # and valid JSON


def test_concurrent_appends_never_lost_during_absorb(tmp_path):
    """A record appended during absorb's read→truncate window was silently
    destroyed. append_feedback now serializes on the project lock (which absorb
    callers hold) — conservation must be exact."""
    import threading

    from ai_calibrator.store import project_lock

    N = 150
    done = threading.Event()

    def writer():
        for i in range(N):
            append_feedback(tmp_path, {"turns": [f"q{i}"], "output": f"a{i}", "verdict": "up"})
        done.set()

    t = threading.Thread(target=writer)
    absorbed_total = 0
    p = _project()
    t.start()
    while not done.is_set():  # absorb aggressively while the writer hammers
        with project_lock(tmp_path):
            r = absorb_feedback(p, tmp_path)
        absorbed_total += r.ups + r.downs + r.skipped
    t.join()
    with project_lock(tmp_path):
        r = absorb_feedback(p, tmp_path)  # sweep the tail
    absorbed_total += r.ups + r.downs + r.skipped

    assert absorbed_total == N                              # nothing destroyed
    assert read_feedback(tmp_path) == []                    # inbox drained
    archived = (tmp_path / "logs" / "feedback-absorbed.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(archived) == N                               # every record archived exactly once
