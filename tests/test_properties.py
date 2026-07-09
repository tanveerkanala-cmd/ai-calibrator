"""Property-based tests (Hypothesis) — generative adversarial inputs against the
invariant-critical functions. Complements the example-based suite: Hypothesis
explores inputs a human wouldn't type and shrinks any failure to a minimal case.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

# a broad text strategy: unicode incl. control chars, surrogates excluded
TEXT = st.text(st.characters(codec="utf-8"), max_size=400)
ANY = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True) | TEXT,
    lambda c: st.lists(c, max_size=5) | st.dictionaries(TEXT, c, max_size=5),
    max_leaves=20,
)


# --- coerce: must NEVER crash, and honor its contracts on ANY input ----------
@given(ANY)
def test_coerce_never_crashes(x):
    from calibrator.coerce import as_list, as_opt_str, as_str, is_str
    assert isinstance(as_list(x), list)
    assert isinstance(as_str(x), str)
    r = as_opt_str(x)
    assert r is None or (isinstance(r, str) and r.strip())   # opt: None or non-blank
    assert isinstance(is_str(x), bool)


@given(TEXT)
def test_safe_token_output_is_safe_or_rejects(s):
    from calibrator.coerce import safe_token
    try:
        out = safe_token(s, "field")
    except ValueError:
        return                                    # rejecting bad input is fine
    # if accepted, the token must be filesystem/injection safe
    assert ".." not in out
    assert not out.startswith("/") and not out.endswith("/")
    assert out == out.strip() and out != ""


# --- parsing: loads_tolerant only ever raises ValueError; chunk_text is lossless-ish
@given(TEXT)
def test_loads_tolerant_only_raises_valueerror(s):
    from calibrator.engines.base import loads_tolerant
    try:
        loads_tolerant(s)
    except ValueError:
        pass                                       # the only allowed failure
    # any OTHER exception type escaping is a bug (test fails on raise)


@given(TEXT, st.integers(min_value=1, max_value=500))
def test_chunk_text_covers_input_without_loss(text, size):
    from calibrator.parsing import chunk_text
    chunks = chunk_text(text, size=size)
    assert isinstance(chunks, list)
    # every chunk is bounded, and concatenation preserves the non-space content
    for c in chunks:
        assert isinstance(c, str)
    joined = "".join(chunks)
    assert "".join(text.split()) == "".join(joined.split()) or not text.strip()


# --- checks: run_check must NEVER crash and always returns (bool, str) --------
@given(
    st.sampled_from(["contains", "not_contains", "regex", "max_chars", "min_chars", "non_empty", "bogus"]),
    TEXT, TEXT,
)
@settings(deadline=None)   # regex path has a timeout
def test_run_check_never_crashes(kind, value, output):
    from calibrator.checks import run_check
    from calibrator.models import Check
    try:
        chk = Check(kind=kind, value=value)
    except Exception:
        return                                     # invalid kind rejected at construction
    passed, why = run_check(chk, output)
    assert isinstance(passed, bool) and isinstance(why, str)


# --- scoring: pass_rate / weighted_score always in [0,1] (or 0 for empty) -----
@given(st.lists(st.lists(st.tuples(st.booleans(), st.sampled_from(["high", "medium", "low"])), max_size=6), max_size=8))
def test_scorecard_metrics_in_unit_interval(spec):
    from calibrator.models import CriterionResult, Scorecard, TestResult, Weight
    results = []
    for i, crits in enumerate(spec):
        crs = [CriterionResult(criterion_id=f"c{j}", passed=p, score=1.0 if p else 0.0,
                               weight=Weight(w)) for j, (p, w) in enumerate(crits)]
        results.append(TestResult(test_id=f"t{i}", output="x", criteria=crs))
    card = Scorecard(run_id="r", results=results)
    assert 0.0 <= card.pass_rate <= 1.0
    assert 0.0 <= card.weighted_score <= 1.0


# --- fmt: honest percentages — never "100%" unless exactly 1.0 ----------------
@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_pct_never_falsely_100(x):
    from calibrator.fmt import pct
    s = pct(x)
    assert isinstance(s, str)
    if s == "100%":
        assert x == 1.0                            # only a true 1.0 may print 100%
    if s == "0%":
        assert x == 0.0                            # only a true 0.0 may print 0%


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1.0, max_value=1.0))
def test_pct_delta_is_a_string(x):
    from calibrator.fmt import pct_delta
    assert isinstance(pct_delta(x), str)


# --- config_hash: deterministic + order-independent over spec list fields -----
@given(
    st.lists(st.text(st.characters(min_codepoint=32, codec="utf-8"), min_size=1, max_size=30), max_size=6, unique=True),
    st.lists(st.text(st.characters(min_codepoint=32, codec="utf-8"), min_size=1, max_size=30), max_size=6, unique=True),
)
def test_config_hash_is_order_independent(standards, do_not):
    from calibrator.ci import config_hash
    from calibrator.models import BehaviorSpec, Project

    def make(std, dn):
        p = Project(name="p", goal="g")
        p.spec = BehaviorSpec(goal="g", standards=list(std), do_not=list(dn))
        return p

    h1 = config_hash(make(standards, do_not))
    h2 = config_hash(make(list(reversed(standards)), list(reversed(do_not))))
    assert h1 == h2                                # reordering rules must not change the cert
    assert config_hash(make(standards, do_not)) == h1   # deterministic


# --- model round-trip: dump → yaml → load → dump must be STABLE (no silent loss)
_SAFE = st.text(st.characters(min_codepoint=32, codec="utf-8"), max_size=60)


@given(
    st.lists(_SAFE, max_size=5),
    st.lists(_SAFE, max_size=5),
    _SAFE,
    st.dictionaries(st.text(st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=8), _SAFE, max_size=3),
)
def test_project_yaml_roundtrip_is_idempotent(standards, do_not, goal, extra):
    import yaml

    from calibrator.models import Project
    reserved = set(Project.model_fields)
    extras = {k: v for k, v in extra.items() if k not in reserved}
    # Build via model_validate — the REAL path unknown fields take (loading a
    # project.yaml), so they pass through validation/normalization like the product.
    p = Project.model_validate({
        "name": "p", "goal": goal or "g",
        "spec": {"goal": goal or "g", "standards": standards, "do_not": do_not},
        **extras,
    })
    once = yaml.safe_dump(p.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    reloaded = Project.model_validate(yaml.safe_load(once))
    twice = yaml.safe_dump(reloaded.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    assert once == twice                           # a second round-trip is a fixed point


# --- config_hash: a CONTENT change must change the hash (no collision) --------
@given(st.lists(_SAFE.filter(lambda s: s.strip()), min_size=1, max_size=5, unique=True), _SAFE.filter(lambda s: s.strip()))
def test_config_hash_detects_a_new_standard(standards, extra):
    from calibrator.ci import config_hash
    from calibrator.models import BehaviorSpec, Project

    def make(std):
        p = Project(name="p", goal="g")
        p.spec = BehaviorSpec(goal="g", standards=list(std))
        return p

    base = make(standards)
    if extra in standards:
        return                                     # not actually a new standard
    assert config_hash(base) != config_hash(make(standards + [extra]))


# --- weighted_score is monotonic: flipping fail→pass never lowers it ----------
@given(st.lists(st.tuples(st.booleans(), st.sampled_from(["high", "medium", "low"])), min_size=1, max_size=8),
       st.integers(min_value=0, max_value=7))
def test_weighted_score_monotonic_on_flip(crits, idx):
    from calibrator.models import CriterionResult, Scorecard, TestResult, Weight

    def card_from(cs):
        crs = [CriterionResult(criterion_id=f"c{j}", passed=p, score=1.0 if p else 0.0, weight=Weight(w))
               for j, (p, w) in enumerate(cs)]
        return Scorecard(run_id="r", results=[TestResult(test_id="t", output="x", criteria=crs)])

    before = card_from(crits).weighted_score
    flipped = list(crits)
    i = idx % len(flipped)
    flipped[i] = (True, flipped[i][1])             # force this criterion to pass
    after = card_from(flipped).weighted_score
    assert after >= before - 1e-9                  # passing more never lowers the weighted score


# --- loads_tolerant preserves any valid JSON value -----------------------------
@given(st.recursive(st.none() | st.booleans() | st.integers() | _SAFE,
                    lambda c: st.lists(c, max_size=4) | st.dictionaries(_SAFE, c, max_size=4), max_leaves=15))
def test_loads_tolerant_preserves_valid_json(obj):
    import json

    from calibrator.engines.base import loads_tolerant
    assert loads_tolerant(json.dumps(obj)) == obj


# --- pct_delta: a real nonzero delta must never render as a "zero" string ------
@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1.0, max_value=1.0))
def test_pct_delta_never_masks_a_real_change(x):
    from calibrator.fmt import pct_delta
    s = pct_delta(x)
    if x != 0.0:
        assert s not in ("0%", "+0%", "-0%", "±0%", "0.0%", "+0.0%")   # must not read as "no change"


# --- compile: synthesize_spec/spec_from_dict must COERCE any engine dict, never crash
@given(ANY)
@settings(max_examples=200)
def test_spec_from_dict_never_crashes_on_arbitrary_engine_output(out):
    from calibrator.compile import spec_from_dict
    from calibrator.models import TaskType
    if not isinstance(out, dict):
        return
    try:
        spec = spec_from_dict(out, goal="g", task_type=TaskType.ASSISTANT)
    except (ValueError, TypeError):
        return                                      # a clean typed error is acceptable
    # if it built a spec, every field must be well-typed (no raw junk leaked in)
    assert all(isinstance(s, str) for s in spec.standards)
    assert all(isinstance(s, str) for s in spec.do_not)
    assert all(isinstance(c.id, str) and isinstance(c.description, str) for c in spec.eval_criteria)


# --- engine spec parsing: any string parses cleanly or raises ValueError, never else
@given(TEXT)
def test_parse_engine_spec_never_crashes(s):
    from calibrator.engines.base import parse_engine_spec, validate_engine_spec
    try:
        model, provider = parse_engine_spec(s)
        assert isinstance(model, str) and isinstance(provider, str)
    except ValueError:
        pass                                        # parse only ever raises ValueError
    try:
        assert validate_engine_spec(s) == s         # returns the spec, or...
    except ValueError:
        pass                                        # ...raises ValueError — never any other type


# --- runtime: encode_messages must handle any message list gracefully ---------
@given(st.lists(st.dictionaries(
    st.sampled_from(["role", "content", "name"]),
    st.none() | TEXT | st.lists(st.dictionaries(st.sampled_from(["type", "text"]), TEXT, max_size=3), max_size=3),
    max_size=3), max_size=6))
@settings(max_examples=200)
def test_encode_messages_never_crashes(messages):
    from calibrator.runtime import encode_messages
    try:
        out = encode_messages(messages)
    except (ValueError, KeyError, TypeError):
        return                                      # rejecting malformed input is fine
    assert isinstance(out, str)


# --- tests_from_examples: ids are unique and generation is deterministic -------
@given(st.lists(st.tuples(_SAFE.filter(lambda s: s.strip()), _SAFE), max_size=8))
def test_tests_from_examples_unique_ids_deterministic(pairs):
    from calibrator.compile import tests_from_examples
    from calibrator.models import BehaviorSpec, Example
    spec = BehaviorSpec(goal="g", examples=[Example(input=i, good_output=o) for i, o in pairs])
    t1 = tests_from_examples(spec)
    t2 = tests_from_examples(spec)
    ids = [t.id for t in t1]
    assert len(ids) == len(set(ids))                # no id collisions
    assert [t.id for t in t1] == [t.id for t in t2] # deterministic


# --- Scorecard JSON round-trip is idempotent too ------------------------------
@given(st.lists(st.lists(st.tuples(st.booleans(), st.sampled_from(["high", "medium", "low"]), _SAFE), max_size=4), max_size=5))
def test_scorecard_json_roundtrip(spec):
    import json

    from calibrator.models import CriterionResult, Scorecard, TestResult, Weight
    results = []
    for i, crits in enumerate(spec):
        crs = [CriterionResult(criterion_id=f"c{j}", passed=p, score=1.0 if p else 0.0,
                               weight=Weight(w), rationale=r) for j, (p, w, r) in enumerate(crits)]
        results.append(TestResult(test_id=f"t{i}", output="o", criteria=crs))
    card = Scorecard(run_id="run-0001", results=results)
    once = json.dumps(card.model_dump(mode="json"))
    twice = json.dumps(Scorecard.model_validate(json.loads(once)).model_dump(mode="json"))
    assert once == twice
