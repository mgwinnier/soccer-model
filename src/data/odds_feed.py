"""TheStatsAPI **Premium Odds Feed** client — pre-match sharp lines (1X2 / totals / spreads).

A separate product from the football API (base ``/api/odds-feed``), it serves a single *unnamed
sharp* pre-match line with ``recorded_at`` + ``cutoff_at`` (kickoff). We use it as the sharp
reference and as the entry/closing source for a forward CLV tracker. Same ``THESTATSAPI_KEY``
(Bearer + browser User-Agent). Honest no-op without the key / on error.

Limitations on the current plan (verified): ``/odds-movements`` (the opening→closing timeline) is
``403 ADDON_REQUIRED`` (Growth/Scale tier), and ``/odds`` returns no rows once a match is settled.
So CLV must be built by **snapshotting** the live pre-match line ourselves over time, not pulled
retrospectively. The line is unnamed (no Pinnacle/book label).

Markets exposed by ``/odds``: ``moneyline`` (1X2 = price_1/price_x/price_2), ``totals``,
``spread``, ``home_totals``, ``away_totals``. No BTTS (that comes from Action Network).
"""
from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path

import requests

from .thestatsapi import api_key, _UA, _MIN_GAP, _last_call
from .odds import devig
from . import fixture_map
from ..config import path_for, load_config

BASE = "https://api.thestatsapi.com/api/odds-feed"
SOCCER_SPORT = "psp_4153"
WC_LEAGUE = "plg_61384"            # FIFA - World Cup (men's)


def is_available() -> bool:
    return bool(api_key())


def _cache_dir(cfg=None) -> Path:
    d = path_for("data_raw", cfg or load_config()) / "odds_feed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(path: str, ttl: float = 300.0, cfg=None, **params) -> dict | None:
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
    headers = {"Authorization": f"Bearer {key}", "User-Agent": _UA, "Accept": "application/json"}
    for _ in range(3):
        wait = _MIN_GAP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = requests.get(url, headers=headers, timeout=25)
        except Exception:  # noqa: BLE001
            return None
        if r.status_code == 429:
            time.sleep(6)
            continue
        if r.status_code != 200:
            return None
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


def events(starts_from: str | None = None, starts_to: str | None = None,
           live: int | None = 0, league_id: str = WC_LEAGUE, ttl: float = 600.0,
           cfg=None) -> list[dict]:
    """WC events (``pmt_`` ids, ``home``/``away``/``start_time``). ``live=0`` = prematch."""
    d = _get("/events", ttl=ttl, cfg=cfg, league_id=league_id, live=live,
             starts_from=starts_from, starts_to=starts_to, limit=200)
    return (d or {}).get("data", []) or []


def _rows(match_id: str, market: str, period: int = 0, ttl: float = 300.0, cfg=None) -> list[dict]:
    d = _get("/odds", ttl=ttl, cfg=cfg, match_id=match_id, market=market, period=period)
    return [o for o in ((d or {}).get("data", {}) or {}).get("odds", [])
            if o.get("market") == market and o.get("period") == period]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sharp_1x2(match_id: str, ttl: float = 300.0, cfg=None) -> dict | None:
    """Latest sharp 1X2 for a match: decimals + de-vigged fair probs + ``cutoff_at``, or None."""
    rows = _rows(match_id, "moneyline", 0, ttl=ttl, cfg=cfg)
    if not rows:
        return None
    o = rows[0]
    h, d, a = _f(o.get("price_1")), _f(o.get("price_x")), _f(o.get("price_2"))
    if not (h and d and a):
        return None
    fair = devig([1 / h, 1 / d, 1 / a])
    return {"dec": {"H": h, "D": d, "A": a},
            "fair": {"H": fair[0], "D": fair[1], "A": fair[2]} if fair else None,
            "cutoff_at": o.get("cutoff_at"), "recorded_at": o.get("recorded_at")}


def sharp_total(match_id: str, line: float = 2.5, ttl: float = 300.0, cfg=None) -> dict | None:
    """Latest sharp over/under at ``line``: decimals + de-vigged fair, or None."""
    rows = _rows(match_id, "totals", 0, ttl=ttl, cfg=cfg)
    pick = next((o for o in rows if _f(o.get("line")) == line), None)
    if not pick:
        return None
    over, under = _f(pick.get("price_1")), _f(pick.get("price_2"))
    if not (over and under):
        return None
    fair = devig([1 / over, 1 / under])
    return {"line": line, "dec": {"over": over, "under": under},
            "fair": {"over": fair[0], "under": fair[1]} if fair else None,
            "cutoff_at": pick.get("cutoff_at")}


def sharp_1x2_for_fixture(home: str, away: str, date, evts: list[dict] | None = None,
                          ttl: float = 300.0, cfg=None) -> dict | None:
    """Resolve a fixture to its odds-feed event (by team pair + date) and return its sharp 1X2."""
    if evts is None:
        ymd = fixture_map._to_ymd(date) or ""
        evts = events(starts_from=ymd, starts_to=ymd, cfg=cfg) if ymd else events(cfg=cfg)
    eid = fixture_map.find_match_id(home, away, fixture_map._to_ymd(date), evts)
    return sharp_1x2(eid, ttl=ttl, cfg=cfg) if eid else None


def connectivity_check(cfg=None) -> str:
    """'ok' / 'no_key' / 'no_subscription' / 'unreachable' for a status chip."""
    if not api_key():
        return "no_key"
    d = _get("/sports", ttl=0.0, cfg=cfg)
    if d and d.get("data"):
        return "ok"
    return "unreachable"
