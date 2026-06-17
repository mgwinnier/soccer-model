"""Grounded AI match brief via Google Gemini (key-gated, never fabricates).

Turns the structured facts we ALREADY have (model probs, expected goals, xG, confirmed XIs +
formations, player market values, key absences, value gap, form, the best bet) into a 2-3 sentence
preview. The prompt is strict: **use only the provided facts; invent nothing** — no history, no
head-to-head trends, no reasons for absences, no stats that aren't given. So it's pure synthesis of
real data, consistent with the project's no-fabrication rule.

Graceful no-op without ``GEMINI_API_KEY`` (env / .env / Streamlit ``st.secrets``).
"""
from __future__ import annotations

import os

import requests

from ..config import load_secrets

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_SYSTEM = (
    "You are a sharp football betting analyst writing a punchy ~3-sentence pre-match brief. The "
    "card already shows the model's numbers — your job is to ADD real context with Google Search, "
    "in this order:\n"
    "1) TEAM NEWS FIRST — who is actually in/out and the real, sourced reason (late injuries, "
    "suspensions, rotation, a returning starter), plus the confirmed XI / formation angle.\n"
    "2) THEN THE BETTING TAKEAWAY — tie it to the model's lean and its best bet at the offered "
    "price; say whether the news strengthens or undercuts that bet.\n"
    "STRICT RULES: state ONLY what the card data or a search result supports; cite sources. If you "
    "cannot verify something — especially WHY a player is missing — do not say it; never guess or "
    "invent history, trends, or stats. Frame the model as 'the model thinks/leans', never a "
    "guarantee. Be specific and punchy: ~3 sentences, no hedging fluff, no preamble, no bullets."
)


def api_key() -> str | None:
    load_secrets()
    return os.environ.get("GEMINI_API_KEY") or None


def is_available() -> bool:
    return bool(api_key())


def _model() -> str:
    return os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash"


def _render(facts: dict) -> str:
    lines = []
    for k, v in facts.items():
        if v:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _call(key: str, prompt: str, search: bool, timeout: float):
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 320, "topP": 0.9}}
    if search:
        body["tools"] = [{"google_search": {}}]      # live grounding (real, citable)
    return requests.post(_ENDPOINT.format(model=_model()), params={"key": key},
                         json=body, timeout=timeout)


def _parse(resp) -> dict | None:
    cand = (resp.json().get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in ((cand.get("content") or {}).get("parts") or []))
    text = text.strip()
    if not text:
        return None
    sources = []
    for c in ((cand.get("groundingMetadata") or {}).get("groundingChunks") or []):
        w = c.get("web") or {}
        if w.get("uri"):
            sources.append({"title": w.get("title") or w["uri"], "uri": w["uri"]})
    # dedupe by uri, keep order
    seen, uniq = set(), []
    for s in sources:
        if s["uri"] not in seen:
            seen.add(s["uri"])
            uniq.append(s)
    return {"text": text, "sources": uniq[:5]}


def _err(resp) -> str:
    try:
        e = resp.json().get("error") or {}
        return e.get("message") or resp.text[:160]
    except Exception:  # noqa: BLE001
        return (getattr(resp, "text", "") or "")[:160]


def brief(facts: dict, timeout: float = 25.0) -> dict | None:
    """A context-adding brief grounded in Google Search.

    Returns ``{"text", "sources":[{title,uri}]}`` on success, ``{"error": ...}`` on an API failure
    (so the UI can show why), or ``None`` only when there's no key / no facts. Falls back to
    no-search synthesis on ANY error from the grounded call (e.g. search tool/billing not enabled)."""
    key = api_key()
    if not key or not facts:
        return None
    prompt = f"{_SYSTEM}\n\nCARD DATA:\n{_render(facts)}\n\nWrite the brief now:"
    try:
        r = _call(key, prompt, search=True, timeout=timeout)
        if r.status_code == 200:
            parsed = _parse(r)
            if parsed:
                return parsed
        # grounded call failed or returned no text -> retry plain synthesis
        r2 = _call(key, prompt, search=False, timeout=timeout)
        if r2.status_code == 200:
            parsed = _parse(r2)
            if parsed:
                return parsed
            return {"error": "model returned no text (possibly safety-filtered)"}
        return {"error": f"HTTP {r2.status_code}: {_err(r2)}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:140]}"}
