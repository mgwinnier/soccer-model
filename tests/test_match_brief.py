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


def test_brief_parses_text_and_sources(monkeypatch):
    monkeypatch.setattr(mb, "api_key", lambda: "k")

    class _R:
        status_code = 200

        @staticmethod
        def json():
            return {"candidates": [{
                "content": {"parts": [{"text": "A real, cited preview."}]},
                "groundingMetadata": {"groundingChunks": [
                    {"web": {"uri": "https://bbc.com/x", "title": "BBC"}},
                    {"web": {"uri": "https://bbc.com/x", "title": "BBC dup"}},  # deduped
                ]}}]}

    captured = {}

    def _post(url, params=None, json=None, timeout=None):
        captured["body"] = json
        return _R()
    monkeypatch.setattr(mb.requests, "post", _post)
    out = mb.brief({"Match": "A vs B", "Model": "A 60%"})
    assert out["text"] == "A real, cited preview."
    assert out["sources"] == [{"title": "BBC", "uri": "https://bbc.com/x"}]   # deduped to 1
    # search grounding requested + the card data is in the prompt
    assert captured["body"]["tools"] == [{"google_search": {}}]
    assert "A 60%" in captured["body"]["contents"][0]["parts"][0]["text"]


def test_brief_falls_back_without_search_tool(monkeypatch):
    monkeypatch.setattr(mb, "api_key", lambda: "k")
    calls = {"n": 0}

    class _R:
        def __init__(self, code):
            self.status_code = code
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "plain"}]}}]}

    def _post(url, params=None, json=None, timeout=None):
        calls["n"] += 1
        # first call (with search tool) 400s -> retried without tools -> 200
        return _R(400) if "tools" in json else _R(200)
    monkeypatch.setattr(mb.requests, "post", _post)
    out = mb.brief({"Match": "A vs B"})
    assert out["text"] == "plain" and calls["n"] == 2


def test_brief_surfaces_error(monkeypatch):
    monkeypatch.setattr(mb, "api_key", lambda: "k")
    monkeypatch.setattr(mb.requests, "post", lambda *a, **k: type(
        "R", (), {"status_code": 500, "text": "",
                  "json": staticmethod(lambda: {"error": {"message": "boom"}})})())
    out = mb.brief({"Match": "A vs B"})
    assert out and "500" in out["error"] and "boom" in out["error"]   # error surfaced for the UI
    assert mb.brief({}) is None                                       # empty facts -> None
