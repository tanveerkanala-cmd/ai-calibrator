"""M1 ingest logic, verified with a mocked engine (no network / SDK needed)."""

from calibrator.ingest import GAP_SCHEMA, extract_gaps, ingest_project
from calibrator.models import Project, TaskType


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
