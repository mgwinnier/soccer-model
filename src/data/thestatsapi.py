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


def match_stats(match_id: str, ttl: float = 86400.0, cfg=None) -> dict:
    """Full match-stats overview as ``{stat_key: {"home": x, "away": y}}`` (numeric only), or {}.

    Real fields from /stats: ball_possession, expected_goals, big_chances, total_shots,
    shots_on_target, goalkeeper_saves, corner_kicks, fouls, yellow_cards, red_cards, passes,
    accurate_passes, tackles, free_kicks. Only stats with both home & away present are kept."""
    d = _unwrap(_get(f"/matches/{match_id}/stats", ttl=ttl, cfg=cfg))
    ov = d.get("overview") or {}
    out: dict = {}
    for key, node in ov.items():
        allv = node.get("all") if isinstance(node, dict) else None
        if isinstance(allv, dict) and allv.get("home") is not None and allv.get("away") is not None:
            out[key] = {"home": allv["home"], "away": allv["away"]}
    return out


def match_odds(match_id: str, ttl: float = 1800.0, cfg=None) -> dict | None:
    """Raw odds payload: ``{bookmakers: [{bookmaker, markets:{match_odds, btts,
    total_goals, asian_handicap, draw_no_bet, ...}}]}`` with opening+last_seen per price."""
    return _unwrap(_get(f"/matches/{match_id}/odds", ttl=ttl, cfg=cfg)) or None


# Books preferred for *betting* prices, sharpest first (lower vig / closest to true).
_BOOK_PREF = ("pinnacle", "betfair", "bet365")


def _price(node) -> float | None:
    """A price node is ``{opening, last_seen}`` (decimal strings). Take the closing/current
    (``last_seen``), falling back to ``opening``. Returns a float decimal or None."""
    if not isinstance(node, dict):
        return None
    for k in ("last_seen", "opening"):
        v = node.get(k)
        if v not in (None, "", "0", "0.0"):
            try:
                d = float(v)
                return d if d > 1.0 else None
            except (TypeError, ValueError):
                return None
    return None


def _pick_book(payload: dict, market: str) -> dict | None:
    """Choose one bookmaker that quotes ``market``: a sharp book if present, else the first."""
    books = (payload or {}).get("bookmakers") or []
    have = [b for b in books if isinstance(b, dict) and market in (b.get("markets") or {})]
    if not have:
        return None
    for name in _BOOK_PREF:
        for b in have:
            if name in str(b.get("bookmaker", "")).lower():
                return b
    return have[0]


def btts_prices(payload: dict | None) -> dict | None:
    """Both-Teams-To-Score yes/no decimal prices from an odds payload, or None.

    Single-book (a real, bettable two-sided market — no mixing books across sides) chosen by
    ``_pick_book``. Returns ``{"book", "yes", "no"}`` with decimal floats, only when *both*
    sides are present (so de-vig is valid)."""
    b = _pick_book(payload or {}, "btts")
    if not b:
        return None
    mk = (b.get("markets") or {}).get("btts") or {}
    yes, no = _price(mk.get("yes")), _price(mk.get("no"))
    if yes is None or no is None:
        return None
    return {"book": b.get("bookmaker"), "yes": yes, "no": no}


def match_lineups(match_id: str, ttl: float = 600.0, cfg=None) -> dict | None:
    """Confirmed XI payload (``confirmed``, ``home/away`` with ``formation``,
    ``starting_xi``, ``substitutes``), or None until it's announced (~75 min pre-KO)."""
    d = _unwrap(_get(f"/matches/{match_id}/lineups", ttl=ttl, cfg=cfg))
    return d if d.get("home") else None


def match_id_for_fixture(home: str, away: str, date, *, competition_id: str = WC_COMP,
                         season_id: str | None = None, day_tol: int = 1, cfg=None) -> str | None:
    """Resolve an (home, away, date) fixture to a TheStatsAPI ``mt_`` id via a date-windowed
    ``/matches`` pull + the shape-tolerant matcher. None when no key / no match."""
    from . import fixture_map
    from datetime import datetime, timedelta
    ymd = fixture_map._to_ymd(date)
    if not ymd:
        return None
    try:
        d0 = datetime.strptime(ymd, "%Y-%m-%d")
    except ValueError:
        return None
    lo = (d0 - timedelta(days=day_tol)).strftime("%Y-%m-%d")
    hi = (d0 + timedelta(days=day_tol)).strftime("%Y-%m-%d")
    cands = matches(competition_id=competition_id, season_id=season_id,
                    date_from=lo, date_to=hi, cfg=cfg)
    return fixture_map.find_match_id(home, away, ymd, cands, day_tol=day_tol)


def xg_for_fixture(home: str, away: str, date, *, competition_id: str = WC_COMP,
                   season_id: str | None = None, day_tol: int = 1,
                   cfg=None) -> tuple[float, float] | None:
    """(home_xg, away_xg) for a fixture by team names + ``date``, or None. Honest no-op."""
    mid = match_id_for_fixture(home, away, date, competition_id=competition_id,
                               season_id=season_id, day_tol=day_tol, cfg=cfg)
    return match_xg(mid, cfg=cfg) if mid else None


def stats_for_fixture(home: str, away: str, date, *, competition_id: str = WC_COMP,
                      season_id: str | None = None, day_tol: int = 1, cfg=None) -> dict:
    """Full match-stats overview for a played fixture by team names + ``date``, or {}."""
    mid = match_id_for_fixture(home, away, date, competition_id=competition_id,
                               season_id=season_id, day_tol=day_tol, cfg=cfg)
    return match_stats(mid, cfg=cfg) if mid else {}


def match_player_ratings(match_id: str, ttl: float = 86400.0, cfg=None) -> dict:
    """{player_id: {name, rating, started, minutes, position, team_id}} for a match, or {}.
    One call returns both squads' per-player match ratings (e.g. 6.82)."""
    d = _get(f"/matches/{match_id}/player-stats", ttl=ttl, cfg=cfg)
    rows = d.get("data") if isinstance(d, dict) else (d if isinstance(d, list) else None)
    out: dict = {}
    for p in rows or []:
        pid = p.get("player_id")
        if pid:
            out[pid] = {"name": p.get("player_name"), "rating": p.get("rating"),
                        "started": bool(p.get("started")), "minutes": p.get("minutes_played"),
                        "position": p.get("position"), "team_id": p.get("team_id")}
    return out


_SEASON = [None]


def team_squad(team_id: str, ttl: float = 86400.0, cfg=None) -> list[dict]:
    """National-team (or club) squad with market values: [{id, name, position, market_value}].
    From /teams/{id}/players (the SquadPlayer model). Empty on miss."""
    d = _get(f"/teams/{team_id}/players", ttl=ttl, cfg=cfg, per_page=100)
    rows = d.get("data") if isinstance(d, dict) else (d if isinstance(d, list) else None)
    out = []
    for p in rows or []:
        if p.get("id") and p.get("name"):
            out.append({"id": p["id"], "name": p["name"], "position": p.get("position"),
                        "market_value": p.get("market_value")})
    return out


def current_season_id(cfg=None) -> str | None:
    """The competition's current season id (``is_current``), cached for the process."""
    if _SEASON[0]:
        return _SEASON[0]
    for s in competition_seasons(cfg=cfg):
        if s.get("is_current"):
            _SEASON[0] = s.get("id")
            break
    return _SEASON[0]


def connectivity_check(cfg=None) -> str:
    """'ok' / 'no_key' / 'unreachable' — for the dashboard status chip."""
    if not api_key():
        return "no_key"
    d = _get("/competitions", ttl=0.0, cfg=cfg, search="World Cup")
    return "ok" if d and d.get("data") else "unreachable"
