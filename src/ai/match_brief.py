"""Grounded AI match brief via Google Gemini (key-gated, never fabricates).

Turns the structured facts we ALREADY have (model probs, expected goals, confirmed XIs + formations,
player market values, key absences, the variance signals, and **every bet the model flagged**) into a
**betting-tuned, structured** read. The job is two-sided and strict:

1. Use **Google Search** to surface the line-movers a pure DC+Elo model is blind to — late team news,
   weather, dead-rubber / qualification motivation, manager rotation intent, referee card/pen tendencies,
   tactical/formation shifts.
2. For **each bet the model already flagged**, judge whether that sourced news *supports* or *undercuts*
   it — never invent a new edge or a number; only contextualise the model's existing lean.

Output is JSON (parsed leniently from the grounded text — Gemini can't combine the search tool with a
strict response schema), returning ``{"summary","angles":[...],"confidence","watch","sources":[...]}``.
``text`` is kept as an alias of ``summary`` for the existing renderer / back-compat.

Graceful no-op without ``GEMINI_API_KEY`` (env / .env / Streamlit ``st.secrets``).
"""
from __future__ import annotations

import json
import os
import re

import requests

from ..config import load_secrets

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# The dashboard's market labels — angles must use these exactly so the UI can place each chip.
MARKETS = ("Match Result", "Total Goals", "Spread", "BTTS")
READS = ("support", "undercut", "neutral")

_SYSTEM = (
    "You are a sharp football betting analyst. The card already has the model's calibrated numbers and "
    "the exact bets it flagged — your job is to ADD the real, sourced context a pure statistical model "
    "CANNOT see, then judge each flagged bet against it.\n"
    "USE GOOGLE SEARCH to find the line-movers, in this priority:\n"
    "  • confirmed/expected XI and LATE team news (injuries, suspensions, a returning starter);\n"
    "  • weather — heat, wind, rain (pushes totals down / unders);\n"
    "  • motivation — dead rubber, already-through, must-win, qualification math (affects result, totals, cards);\n"
    "  • manager rotation intent (a rested/weakened XI changes strength);\n"
    "  • referee tendencies — cards/penalties per game (cards, BTTS, totals);\n"
    "  • late tactical/formation shifts that change the matchup.\n"
    "THEN, for each bet under 'Flagged bets', decide whether the news STRENGTHENS it (read='support'), "
    "WEAKENS it (read='undercut'), or is unclear/irrelevant (read='neutral'). Undercut/fade calls matter "
    "as much as confirmations — do not just agree with the model.\n"
    "STRICT RULES: state ONLY what a search result or the card data supports; every 'why' must be "
    "grounded — if you cannot verify it (especially WHY a player is missing), use read='neutral' or omit "
    "the angle. Never invent history, trends, stats, a new bet, or a probability. Frame the model as "
    "'the model leans', never a guarantee.\n"
    "Return ONLY a JSON object, no prose, no code fences, with this exact shape:\n"
    '{"summary": "<=2 sentences: the confirmed team news + your overall read", '
    '"angles": [{"market": "<one of: Match Result | Total Goals | Spread | BTTS>", '
    '"lean": "<the flagged selection, e.g. Over 2.5 or England>", '
    '"read": "support|undercut|neutral", "why": "<short, grounded reason>"}], '
    '"confidence": "low|medium|high", '
    '"watch": "<one line: what would flip this, e.g. if the XI confirms Saka starts>"}'
)


def api_key() -> str | None:
    load_secrets()
    k = os.environ.get("GEMINI_API_KEY")
    if not k:                                   # robust to st.secrets/import-timing on the cloud
        try:
            import streamlit as _st
            k = _st.secrets.get("GEMINI_API_KEY")
        except Exception:  # noqa: BLE001
            k = None
    return k or None


def is_available() -> bool:
    return bool(api_key())


def _model() -> str:
    return os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"


def _render(facts: dict) -> str:
    lines = []
    for k, v in facts.items():
        if v:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _call(key: str, prompt: str, search: bool, timeout: float):
    body = {"contents": [{"parts": [{"text": prompt}]}],
            # ~900 tokens: enough for a summary + several JSON angles. 2.5+ "thinking" tokens
            # otherwise eat the output budget and truncate — turn it off for this short task.
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 900, "topP": 0.9,
                                 "thinkingConfig": {"thinkingBudget": 0}}}
    if search:
        # NOTE: Gemini does NOT allow combining google_search with responseSchema/JSON mime — so we
        # keep grounding and ask for JSON in the prompt, then parse it out of the text.
        body["tools"] = [{"google_search": {}}]
    return requests.post(_ENDPOINT.format(model=_model()), params={"key": key},
                         json=body, timeout=timeout)


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of model text: strip ```json fences, find the first balanced {...}."""
    if not text:
        return None
    t = text.strip()
    # strip code fences if present
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    # find the first balanced object
    start = t.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = t[start:i + 1]
                try:
                    obj = json.loads(blob)
                    return obj if isinstance(obj, dict) else None
                except Exception:  # noqa: BLE001
                    return None
    return None


def _clean_angles(raw) -> list[dict]:
    """Keep only well-formed, in-vocabulary angles (drops anything off-schema — honest by construction)."""
    out = []
    if not isinstance(raw, list):
        return out
    for a in raw:
        if not isinstance(a, dict):
            continue
        market = str(a.get("market") or "").strip()
        if market not in MARKETS:
            continue
        read = str(a.get("read") or "neutral").strip().lower()
        if read not in READS:
            read = "neutral"
        why = str(a.get("why") or "").strip()
        if not why:                              # a "why"-less angle is unsourced → drop it
            continue
        out.append({"market": market, "lean": str(a.get("lean") or "").strip(),
                    "read": read, "why": why})
    return out


def _sources(cand: dict) -> list[dict]:
    sources = []
    for c in ((cand.get("groundingMetadata") or {}).get("groundingChunks") or []):
        w = c.get("web") or {}
        if w.get("uri"):
            sources.append({"title": w.get("title") or w["uri"], "uri": w["uri"]})
    seen, uniq = set(), []
    for s in sources:
        if s["uri"] not in seen:
            seen.add(s["uri"])
            uniq.append(s)
    return uniq[:5]


def _parse(resp) -> dict | None:
    """Parse a Gemini response into the structured brief. Degrades to text-as-summary if not JSON."""
    cand = (resp.json().get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in ((cand.get("content") or {}).get("parts") or []))
    text = text.strip()
    if not text:
        return None
    sources = _sources(cand)
    obj = _extract_json(text)
    if obj and (obj.get("summary") or obj.get("angles")):
        summary = str(obj.get("summary") or "").strip()
        angles = _clean_angles(obj.get("angles"))
        conf = str(obj.get("confidence") or "").strip().lower()
        conf = conf if conf in ("low", "medium", "high") else ""
        watch = str(obj.get("watch") or "").strip()
        if not summary and not angles:
            return None
        return {"summary": summary, "text": summary, "angles": angles,
                "confidence": conf, "watch": watch, "sources": sources}
    # not JSON (or unparseable) → fall back to the whole text as a plain summary, no angles
    return {"summary": text, "text": text, "angles": [], "confidence": "", "watch": "",
            "sources": sources}


def _err(resp) -> str:
    try:
        e = resp.json().get("error") or {}
        return e.get("message") or resp.text[:160]
    except Exception:  # noqa: BLE001
        return (getattr(resp, "text", "") or "")[:160]


def brief(facts: dict, timeout: float = 25.0) -> dict | None:
    """A betting-tuned, grounded, structured brief.

    Returns ``{"summary","text","angles":[{market,lean,read,why}],"confidence","watch","sources":[...]}``
    on success, ``{"error": ...}`` on an API failure (so the UI can show why), or ``None`` only when
    there's no key / no facts. Falls back to no-search synthesis on ANY error from the grounded call."""
    key = api_key()
    if not key or not facts:
        return None
    prompt = f"{_SYSTEM}\n\nCARD DATA:\n{_render(facts)}\n\nReturn the JSON now:"
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
