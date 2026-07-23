"""M1 ingest logic, verified with a mocked engine (no network / SDK needed)."""

from ai_calibrator.ingest import GAP_SCHEMA, extract_gaps, ingest_project
from ai_calibrator.models import Project, TaskType


class FakeEngine:
    """Records calls and returns a canned structured payload."""

    name = "fake@test"

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete(self, prompt, *, system=None, schema=None):
        self.calls.append({"prompt": prompt, "system": system, "schema": schema})
        return self.payload


def test_extract_gaps_parses_engine_output():
    engine = FakeEngine(
        {
            "facts": ["We sell skincare products."],
            "gaps": [
                {"dimension": "refusal policy", "why_it_matters": "no medical claims"},
                {"dimension": "tone", "why_it_matters": "brand voice"},
            ],
        }
    )
    facts, gaps = extract_gaps("answer questions", "support_assistant",
                               [("faq.md", "Q/A text")], engine)

    assert facts == ["We sell skincare products."]
    assert [g.dimension for g in gaps] == ["refusal policy", "tone"]
    assert gaps[0].why_it_matters == "no medical claims"
    # structured output was requested with our schema
    assert engine.calls[0]["schema"] is GAP_SCHEMA
    assert engine.calls[0]["system"] is not None


def test_extract_gaps_drops_json_shard_gibberish():
    # A small local model sometimes leaks raw JSON / prompt scaffolding as a
    # "gap" — those must never reach the user's gap list or get persisted.
    engine = FakeEngine({
        "facts": ["We ship internationally.", '```jsonall-important-fields: [fact, gap]'],
        "gaps": [
            {"dimension": "tone", "why_it_matters": "brand voice"},
            {"dimension": '```jsonall-important-fields: [fact, gap] --- {', "why_it_matters": "x"},
            {"dimension": 'gap_content": "detailed definition"},', "why_it_matters": "y"},
        ],
    })
    facts, gaps = extract_gaps("g", "assistant", [("f.md", "t")], engine)
    assert [g.dimension for g in gaps] == ["tone"]          # only the clean gap survives
    assert facts == ["We ship internationally."]            # shard fact dropped too


def test_ingest_project_populates_materials_and_gaps(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "faq.md").write_text("Q: returns?\n\nA: within 30 days.")

    project = Project(name="t", goal="answer product questions",
                      task_type=TaskType.SUPPORT_ASSISTANT)
    engine = FakeEngine({"facts": [], "gaps": [{"dimension": "tone", "why_it_matters": "x"}]})

    result = ingest_project(project, materials, engine,
                            project_dir=tmp_path, build_index=False)

    assert result.materials == 1
    assert result.gaps == 1
    assert project.materials[0].path == "faq.md"
    assert project.materials[0].kind == "md"
    assert project.materials[0].summary  # non-empty excerpt
    assert project.gaps[0].dimension == "tone"


def test_ingest_empty_dir_skips_engine(tmp_path):
    materials = tmp_path / "materials"
    materials.mkdir()
    project = Project(name="t", goal="g")
    engine = FakeEngine({"facts": [], "gaps": []})

    result = ingest_project(project, materials, engine, build_index=False)

    assert result.materials == 0 and result.gaps == 0
    assert engine.calls == []  # no engine call on an empty corpus


def test_gap_schema_is_strict_compatible():
    # every object disallows extra props and lists all properties as required
    assert GAP_SCHEMA["additionalProperties"] is False
    assert set(GAP_SCHEMA["required"]) == set(GAP_SCHEMA["properties"])
    item = GAP_SCHEMA["properties"]["gaps"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])


def test_ingest_isolates_per_file_failures(tmp_path, monkeypatch):
    """One unparseable file must NOT abort the whole batch — the good files are
    ingested and the bad one is reported in `skipped` (finding: non-isolated
    ingest). Dependency-free: the parser is stubbed to raise on one file, so the
    test exercises the isolation logic without needing the optional docs extra."""
    import ai_calibrator.ingest as ing
    from ai_calibrator.ingest import parse_materials

    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "good1.md").write_text("first policy")
    (materials / "good2.md").write_text("second policy")
    (materials / "broken.bin").write_text("triggers a parse failure")

    real = ing.read_document

    def flaky(path):
        if path.name == "broken.bin":
            raise ValueError("simulated parse failure")
        return real(path)

    monkeypatch.setattr(ing, "read_document", flaky)

    docs, skipped = parse_materials(materials)

    names = sorted(p.name for p, _ in docs)
    assert names == ["good1.md", "good2.md"]                 # good files survived
    assert [rel for rel, _ in skipped] == ["broken.bin"]     # bad one named, not fatal
    assert "simulated parse failure" in skipped[0][1]


def test_ingest_project_surfaces_skipped(tmp_path, monkeypatch):
    """ingest_project carries the skip list through to IngestResult."""
    import ai_calibrator.ingest as ing

    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "ok.md").write_text("real content")
    (materials / "bad.bin").write_text("boom")

    real = ing.read_document

    def flaky(path):
        if path.name == "bad.bin":
            raise ValueError("bad")
        return real(path)

    monkeypatch.setattr(ing, "read_document", flaky)

    proj = Project(name="p", goal="g")
    eng = FakeEngine({"facts": [], "gaps": []})
    result = ingest_project(proj, materials, eng, build_index=False)

    assert result.materials == 1
    assert [rel for rel, _ in result.skipped] == ["bad.bin"]
