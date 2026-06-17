"""Tests for the betting-tuned, structured Gemini brief (network mocked; key-gated)."""
import json

from src.ai import match_brief as mb


def _resp(payload: dict, code: int = 200):
    class _R:
        status_code = code

        @staticmethod
        def json():
            return payload

        text = ""
    return _R()


def test_noop_without_key(monkeypatch):
    monkeypatch.setattr(mb, "api_key", lambda: None)
    assert mb.is_available() is False
    assert mb.brief({"Match": "A vs B"}) is None


def test_render_facts_lists_only_provided():
    txt = mb._render({"Match": "A vs B", "Model": "A 60%", "Empty": ""})
    assert "Match: A vs B" in txt and "Model: A 60%" in txt
    assert "Empty" not in txt                       # blank facts dropped


def test_extract_json_from_fenced_text():
    raw = ('Here is the read.\n```json\n{"summary": "s", "angles": [], '
           '"confidence": "high", "watch": "w"}\n```\nthanks')
    obj = mb._extract_json(raw)
    assert obj["summary"] == "s" and obj["confidence"] == "high"


def test_clean_angles_filters_offschema():
    angles = mb._clean_angles([
        {"market": "Total Goals", "lean": "Over 2.5", "read": "support", "why": "wind low, sourced"},
        {"market": "Cards", "lean": "x", "read": "support", "why": "off-vocab market dropped"},
        {"market": "BTTS", "lean": "Yes", "read": "wild", "why": "bad read coerced to neutral"},
        {"market": "Match Result", "lean": "England", "read": "undercut", "why": ""},  # no why → drop
    ])
    markets = [a["market"] for a in angles]
    assert markets == ["Total Goals", "BTTS"]
    assert angles[1]["read"] == "neutral"           # invalid read coerced


def test_brief_parses_structured_and_sources(monkeypatch):
    monkeypatch.setattr(mb, "api_key", lambda: "k")
    payload = {"candidates": [{
        "content": {"parts": [{"text": json.dumps({
            "summary": "England missing Saka; leans under.",
            "angles": [{"market": "Total Goals", "lean": "Under 2.5", "read": "support",
                        "why": "Saka out per BBC; attack weakened"}],
            "confidence": "medium", "watch": "if Saka is named to start"})}]},
        "groundingMetadata": {"groundingChunks": [
            {"web": {"uri": "https://bbc.com/x", "title": "BBC"}},
            {"web": {"uri": "https://bbc.com/x", "title": "BBC dup"}},  # deduped
        ]}}]}
    captured = {}

    def _post(url, params=None, json=None, timeout=None):
        captured["body"] = json
        return _resp(payload)
    monkeypatch.setattr(mb.requests, "post", _post)

    out = mb.brief({"Match": "Eng vs Cro", "Flagged bets": "Total Goals: Under 2.5 ...",
                    "Variance": "upset risk 30%"})
    assert out["summary"].startswith("England missing Saka")
    assert out["text"] == out["summary"]            # back-compat alias
    assert out["angles"][0]["market"] == "Total Goals" and out["angles"][0]["read"] == "support"
    assert out["confidence"] == "medium" and out["watch"]
    assert out["sources"] == [{"title": "BBC", "uri": "https://bbc.com/x"}]   # deduped to 1
    # search grounding requested + the betting facts reached the prompt
    assert captured["body"]["tools"] == [{"google_search": {}}]
    prompt = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "Flagged bets" in prompt and "Variance" in prompt


def test_brief_degrades_nonjson_to_summary(monkeypatch):
    monkeypatch.setattr(mb, "api_key", lambda: "k")
    payload = {"candidates": [{"content": {"parts": [{"text": "Just a plain sentence, no JSON."}]}}]}
    monkeypatch.setattr(mb.requests, "post", lambda *a, **k: _resp(payload))
    out = mb.brief({"Match": "A vs B"})
    assert out["summary"] == "Just a plain sentence, no JSON." and out["angles"] == []


def test_brief_falls_back_without_search_tool(monkeypatch):
    monkeypatch.setattr(mb, "api_key", lambda: "k")
    calls = {"n": 0}
    good = {"candidates": [{"content": {"parts": [{"text": '{"summary":"plain","angles":[]}'}]}}]}

    def _post(url, params=None, json=None, timeout=None):
        calls["n"] += 1
        # first call (with search tool) 400s -> retried without tools -> 200
        return _resp(good) if "tools" not in json else _resp({}, 400)
    monkeypatch.setattr(mb.requests, "post", _post)
    out = mb.brief({"Match": "A vs B"})
    assert out["summary"] == "plain" and calls["n"] == 2


def test_brief_surfaces_error(monkeypatch):
    monkeypatch.setattr(mb, "api_key", lambda: "k")
    monkeypatch.setattr(mb.requests, "post", lambda *a, **k: type(
        "R", (), {"status_code": 500, "text": "",
                  "json": staticmethod(lambda: {"error": {"message": "boom"}})})())
    out = mb.brief({"Match": "A vs B"})
    assert out and "500" in out["error"] and "boom" in out["error"]   # error surfaced for the UI
    assert mb.brief({}) is None                                       # empty facts -> None
