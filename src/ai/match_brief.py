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
    "You are a sharp, concise football analyst writing a 2-3 sentence match brief for a "
    "prediction/betting dashboard. Use ONLY the facts provided. Do NOT invent or add anything "
    "not explicitly given — no history, head-to-head trends, form not listed, injuries, reasons "
    "for absences, tactics, or statistics. If a fact isn't provided, omit it; never speculate "
    "about WHY a player is missing. Be specific, readable, neutral. Lead with the model's lean, "
    "weave in the standout lineup/value angle if present, and end with the single best bet if one "
    "is given. No preamble, no bullet points, no headers — just the 2-3 sentences."
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


def brief(facts: dict, timeout: float = 20.0) -> str | None:
    """A grounded 2-3 sentence brief from ``facts``, or None (no key / error / empty)."""
    key = api_key()
    if not key or not facts:
        return None
    prompt = f"{_SYSTEM}\n\nFACTS:\n{_render(facts)}\n\nWrite the brief now:"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200, "topP": 0.9}}
    try:
        r = requests.post(_ENDPOINT.format(model=_model()), params={"key": key},
                          json=body, timeout=timeout)
        if r.status_code != 200:
            return None
        cand = (r.json().get("candidates") or [{}])[0]
        parts = ((cand.get("content") or {}).get("parts") or [{}])
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or None
    except Exception:  # noqa: BLE001 — any failure degrades to no brief
        return None
