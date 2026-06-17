"""Tests for the grounded Gemini match brief (network mocked; key-gated)."""
from src.ai import match_brief as mb


def test_noop_without_key(monkeypatch):
    monkeypatch.setattr(mb, "api_key", lambda: None)
    assert mb.is_available() is False
    assert mb.brief({"Match": "A vs B"}) is None


def test_render_facts_lists_only_provided():
    txt = mb._render({"Match": "A vs B", "Model": "A 60%", "Empty": ""})
    assert "Match: A vs B" in txt and "Model: A 60%" in txt
    assert "Empty" not in txt                       # blank facts dropped


def test_brief_parses_gemini_shape(monkeypatch):
    monkeypatch.setattr(mb, "api_key", lambda: "k")

    class _R:
        status_code = 200

        @staticmethod
        def json():
            return {"candidates": [{"content": {"parts": [{"text": "A lean preview."}]}}]}

    captured = {}

    def _post(url, params=None, json=None, timeout=None):
        captured["prompt"] = json["contents"][0]["parts"][0]["text"]
        return _R()
    monkeypatch.setattr(mb.requests, "post", _post)
    out = mb.brief({"Match": "A vs B", "Model": "A 60%"})
    assert out == "A lean preview."
    # the grounding instruction + the facts are in the prompt
    assert "Do NOT invent" in captured["prompt"] and "A 60%" in captured["prompt"]


def test_brief_none_on_error(monkeypatch):
    monkeypatch.setattr(mb, "api_key", lambda: "k")

    class _R:
        status_code = 500
        @staticmethod
        def json():
            return {}
    monkeypatch.setattr(mb.requests, "post", lambda *a, **k: _R())
    assert mb.brief({"Match": "A vs B"}) is None
    assert mb.brief({}) is None                      # empty facts
