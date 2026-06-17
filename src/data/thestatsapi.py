"""TheStatsAPI client — 2026 World Cup odds (incl. BTTS + Pinnacle), xG, and lineups.

Replaces the dead API-Football path (its free tier was locked to 2022-24 with no
national-team data). TheStatsAPI covers the 2026 WC (`comp_6107`): per-match odds with
opening+closing prices across markets (1X2, **BTTS**, totals, Asian handicap, DNB),
`expected_goals` via /stats, and confirmed starting XIs via /lineups.

Auth: ``Authorization: Bearer <key>`` **and a browser ``User-Agent``** — Cloudflare 1010-
blocks the default urllib/requests UA. Key from ``THESTATSAPI_KEY`` (env / .env / secrets/.env
/ Streamlit ``st.secrets``). Everything degrades to a graceful no-op without a key / on error.

Free-of-charge guardrails: a small on-disk cache (``data/raw/thestatsapi/``) with per-endpoint
TTLs + a light request throttle keep us well under the plan's 120 req/min, and one server-side
cache is shared across all dashboard users.
"""
from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path

import requests

from ..config import load_secrets, path_for, load_config

BASE = "https://api.thestatsapi.com/api/football"
WC_COMP = "comp_6107"                       # FIFA World Cup
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_MIN_GAP = 0.6                              # seconds between live calls (≈100/min < 120 cap)
_last_call = [0.0]


def api_key() -> str | None:
    import os
    load_secrets()
    return os.environ.get("THESTATSAPI_KEY") or None


def is_available() -> bool:
    return bool(api_key())


def _cache_dir(cfg=None) -> Path:
    d = path_for("data_raw", cfg or load_config()) / "thestatsapi"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(path: str, ttl: float = 3600.0, cfg=None, **params) -> dict | None:
    """GET {BASE}{path} with Bearer+UA, disk cache (``ttl`` seconds), throttle + 429 retry.
    Returns the parsed JSON dict, or None on any failure (never raises)."""
    key = api_key()
    if not key:
        return None
    qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
    url = f"{BASE}{path}" + (f"?{qs}" if qs else "")
    cache = _cache_dir(cfg) / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if ttl > 0 and cache.exists() and (time.time() - cache.stat().st_mtime) < ttl:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    headers = {"Authorization": f"Bearer {key}", "User-Agent": _UA,
               "Accept": "application/json"}
    for _ in range(4):
        wait = _MIN_GAP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = requests.get(url, headers=headers, timeout=25)
        except Exception:  # noqa: BLE001 — network/firewall → no-op
            return None
        if r.status_code == 429:                 # rate limited — back off and retry
            time.sleep(6)
            continue
        if r.status_code != 200:
            return None                          # 404 (no odds/lineup yet) etc. → no-op
        try:
            data = r.json()
        except Exception:  # noqa: BLE001
            return None
        if ttl > 0:
            try:
                cache.write_text(json.dumps(data), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
        return data
    return None


# ------------------------------------------------------------------ listings
def matches(competition_id: str = WC_COMP, season_id: str | None = None,
            status: str | None = None, date_from: str | None = None,
            date_to: str | None = None, ttl: float = 900.0, cfg=None) -> list[dict]:
    """All matches for a competition (+optional season/status/date), paginated."""
    out, page = [], 1
    while True:
        d = _get("/matches", ttl=ttl, cfg=cfg, competition_id=competition_id,
                 season_id=season_id, status=status, date_from=date_from,
                 date_to=date_to, per_page=100, page=page)
        if not d or not d.get("data"):
            break
        out.extend(d["data"])
        meta = d.get("meta") or {}
        if page >= (meta.get("total_pages") or page):
            break
        page += 1
    return out


def competition_seasons(competition_id: str = WC_COMP, cfg=None) -> list[dict]:
    d = _get(f"/competitions/{competition_id}/seasons", ttl=86400.0, cfg=cfg)
    return (d or {}).get("data", []) if d else []


# --------------------------------------------------------------- per-match
def _unwrap(d: dict | None) -> dict:
    """TheStatsAPI wraps single-resource responses in ``data``."""
    if not isinstance(d, dict):
        return {}
    return d.get("data", d) if isinstance(d.get("data"), dict) else d


def match_xg(match_id: str, ttl: float = 86400.0, cfg=None) -> tuple[float, float] | None:
    """(home_xg, away_xg) from /stats overview, or None if absent."""
    d = _unwrap(_get(f"/matches/{match_id}/stats", ttl=ttl, cfg=cfg))
    eg = ((d.get("overview") or {}).get("expected_goals") or {}).get("all") or {}
    h, a = eg.get("home"), eg.get("away")
    return (float(h), float(a)) if h is not None and a is not None else None


def match_odds(match_id: str, ttl: float = 1800.0, cfg=None) -> dict | None:
    """Raw odds payload: ``{bookmakers: [{bookmaker, markets:{match_odds, btts,
    total_goals, asian_handicap, draw_no_bet, ...}}]}`` with opening+last_seen per price."""
    return _unwrap(_get(f"/matches/{match_id}/odds", ttl=ttl, cfg=cfg)) or None


def match_lineups(match_id: str, ttl: float = 600.0, cfg=None) -> dict | None:
    """Confirmed XI payload (``confirmed``, ``home/away`` with ``formation``,
    ``starting_xi``, ``substitutes``), or None until it's announced (~75 min pre-KO)."""
    d = _unwrap(_get(f"/matches/{match_id}/lineups", ttl=ttl, cfg=cfg))
    return d if d.get("home") else None


def connectivity_check(cfg=None) -> str:
    """'ok' / 'no_key' / 'unreachable' — for the dashboard status chip."""
    if not api_key():
        return "no_key"
    d = _get("/competitions", ttl=0.0, cfg=cfg, search="World Cup")
    return "ok" if d and d.get("data") else "unreachable"
