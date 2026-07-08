"""RAG indexing — guarded (the `rag` extra is heavy; skip when absent).

Closes a real coverage gap: build_index had NO tests because the rag extra was
never installed in this session's CI. These run wherever lancedb +
sentence-transformers are present."""

import pytest

lancedb = pytest.importorskip("lancedb")
pytest.importorskip("sentence_transformers")

from calibrator import rag  # noqa: E402


def test_index_available_true_when_extra_present():
    assert rag.index_available() is True


def test_build_index_embeds_and_is_queryable(tmp_path):
    records = [
        {"id": "c1", "text": "Returns accepted within 30 days with a receipt.", "source": "p.md"},
        {"id": "c2", "text": "Refunds go to the original payment method.", "source": "p.md"},
        {"id": "c3", "text": "Final-sale items cannot be returned.", "source": "p.md"},
    ]
    n = rag.build_index(tmp_path, records)
    assert n == 3
    assert (tmp_path / "knowledge.lancedb").exists()

    # the index is genuinely queryable — semantic retrieval works
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(rag.EMBED_MODEL)
    db = lancedb.connect(str(tmp_path / "knowledge.lancedb"))
    tbl = db.open_table(rag.TABLE)
    qv = model.encode(["how do I get my money back?"]).tolist()[0]
    hits = tbl.search(qv).limit(1).to_list()
    assert hits[0]["id"] == "c2"          # the refund chunk is nearest


def test_build_index_empty_records(tmp_path):
    assert rag.build_index(tmp_path, []) == 0     # no crash, no table


def test_ingest_reports_indexed_count(tmp_path):
    """The full ingest wiring: with the extra present, build_index runs (not skipped)."""
    from calibrator.ingest import ingest_project
    from calibrator.models import Project

    class FakeExtractor:
        name = "fake@test"
        def complete(self, prompt, *, system=None, schema=None):
            return {"facts": ["Returns within 30 days."], "gaps": []}

    (tmp_path / "materials").mkdir()
    (tmp_path / "materials" / "p.md").write_text("Returns are accepted within 30 days with a receipt.")
    result = ingest_project(Project(name="p", goal="g"), tmp_path / "materials",
                            FakeExtractor(), project_dir=tmp_path)
    assert result.indexed == result.chunks and result.indexed >= 1   # built, not skipped (None)


# --- retrieval wired into eval + runtime (the "test what you serve" fix) ------

def _indexed_project(tmp_path):
    from calibrator.models import BehaviorSpec, EvalCriterion, Project, Weight
    from calibrator.models import TestCase as Case
    rag.build_index(tmp_path, [
        {"id": "c1", "text": "The return window is exactly 30 days from delivery.", "source": "p.md"},
        {"id": "c2", "text": "Final-sale items cannot be returned.", "source": "p.md"},
    ])
    p = Project(name="p", goal="answer return questions")
    p.spec = BehaviorSpec(goal="g", knowledge_sources=["p.md"],
                          eval_criteria=[EvalCriterion(id="c1", description="d", weight=Weight.HIGH)])
    p.tests = [Case(id="t1", input="how long do I have to return?", expects=["c1"])]
    return p


class SpySubject:
    name = "spy@test"
    def __init__(self): self.systems = []
    def complete(self, prompt, *, system=None, schema=None):
        self.systems.append(system)
        return "answer"


class PassJudge:
    name = "j@test"
    def complete(self, prompt, *, system=None, schema=None):
        import re
        ids = re.findall(r"^- (\S+):", prompt, re.M)
        return {"results": [{"criterion_id": i, "passed": True, "score": 1.0, "rationale": "ok"} for i in ids]}


def test_run_eval_injects_retrieved_knowledge(tmp_path):
    from calibrator.eval import run_eval
    p = _indexed_project(tmp_path)
    subj = SpySubject()
    run_eval(p, subj, PassJudge(), run_id="r", project_dir=tmp_path)   # RAG on
    assert "RELEVANT KNOWLEDGE" in subj.systems[0]
    assert "30 days from delivery" in subj.systems[0]                  # the retrieved chunk


def test_run_eval_no_retrieval_without_project_dir(tmp_path):
    from calibrator.eval import run_eval
    p = _indexed_project(tmp_path)
    subj = SpySubject()
    run_eval(p, subj, PassJudge(), run_id="r")                        # project_dir omitted
    assert "RELEVANT KNOWLEDGE" not in (subj.systems[0] or "")         # unchanged behavior


def test_runtime_and_eval_inject_identically(tmp_path):
    """What you test IS what you serve: the augmented system is byte-identical."""
    import pytest as _pytest
    _pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from calibrator import rag
    from calibrator.compile import render_system_prompt
    from calibrator.runtime import create_ai_app
    from calibrator.store import save_project

    p = _indexed_project(tmp_path)
    save_project(p, tmp_path)
    base_system = render_system_prompt(p.spec)
    query = "how long do I have to return?"
    eval_system = rag.augment_system(base_system, tmp_path, query)

    spy = SpySubject()
    c = TestClient(create_ai_app(tmp_path, engine=spy))
    c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": query}]})
    assert spy.systems[0] == eval_system            # runtime injects exactly what eval graded
    assert "30 days from delivery" in spy.systems[0]
