

def test_a_binary_field_does_not_make_a_project_unsaveable(tmp_path):
    """`extra="allow"` carries unknown fields through load->save. A YAML
    `!!binary` one loaded happily and then raised inside model_dump on the next
    save, so the project became permanently unsaveable and every mutating
    command ended in a traceback naming nothing."""
    from ai_calibrator.store import load_project, save_project

    p = tmp_path / "p"
    p.mkdir()
    (p / "project.yaml").write_text(
        'name: p\ngoal: g\nblob: !!binary "/////w=="\n', encoding="utf-8")

    project = load_project(p)
    save_project(project, p)          # used to raise UnicodeDecodeError

    assert load_project(p).goal == "g"


def test_two_turn_splits_of_the_same_text_do_not_collide():
    """A turn may contain a NUL — an engine's JSON can carry \\u0000 and it
    round-trips through the project file. With a bare NUL join, "a\\x00b" and
    ("a", "b") hashed identically, so a recompile that split one turn into two
    let an old scorecard's verdicts be credited to a question never asked."""
    from ai_calibrator.models import content_hash

    assert content_hash("a\x00b") != content_hash("a", "b")


def test_a_project_name_is_capped_in_bytes_not_characters():
    """ext4/xfs/btrfs cap a path component at 255 BYTES, so 86 CJK characters is
    already over while passing a 120-character test — and the failure then
    arrives as an ENAMETOOLONG from mkdir, which this check exists to prevent."""
    import pytest

    from ai_calibrator.models import validate_project_name

    with pytest.raises(ValueError):
        validate_project_name("中" * 86)
    assert validate_project_name("ok-name") == "ok-name"
