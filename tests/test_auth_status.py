"""Auth status must not green-check an obvious placeholder key (C5)."""

from ai_calibrator.auth import anthropic_status, openai_status


def test_placeholder_anthropic_key_not_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")   # the docs placeholder
    st = anthropic_status()
    assert st.configured is False and "placeholder" in st.detail


def test_angle_placeholder_not_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "<your-key>")
    assert openai_status().configured is False


def test_real_looking_key_is_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-abcdefgh12345678ZZ")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefgh12345678")
    assert anthropic_status().configured is True
    assert openai_status().configured is True
