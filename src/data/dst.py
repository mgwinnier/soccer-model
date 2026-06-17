"""Digital Sports Tech (DST) odds for the user's PayPerHead book skin (``sb=buckeye2``).

The book's straight lines are served by an **open DST host** (``bv2-us.digitalsportstech.com``) —
no auth, no login token, no Cloudflare. So we read the user's *actual book odds* with **zero
account risk**: it's the same public widget feed the site embeds; the bookie sees nothing (no
login from a new IP, no extra calls to their authed backend). This is the bettable line that
matters — value is measured vs the price the user actually gets.

Flow (GET, ``sb``-scoped, no auth):
  ``/api/sgmGames?sb=&league=fifa&sport=football``        -> games (provider id + team titles)
  ``/api/sgmMarkets/gfm/grouped?sb=&gameId=<providerId>`` -> grouped markets, decimal odds inline

Markets parsed: moneyline (3-way ``to win``: type 1=home / x=draw / 2=away), ``total goals``
(over/under ladder), ``both teams to score`` (yes/no). Everything degrades to None/{}.
"""
from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path

import requests

from .team_names import normalize_team
from ..config import path_for, load_config

BASE = "https://bv2-us.digitalsportstech.com/api"
DEFAULT_SB = "buckeye2"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_HDRS = {"User-Agent": _UA, "Accept": "application/json",
         "Origin": "https://sun22.ag", "Referer": "https://sun22.ag/"}


def _cache_dir(cfg=None) -> Path:
    d = path_for("data_raw", cfg or load_config()) / "dst"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(path: str, ttl: float = 300.0, cfg=None, **params):
    qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
    url = f"{BASE}{path}" + (f"?{qs}" if qs else "")
    cache = _cache_dir(cfg) / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if ttl > 0 and cache.exists() and (time.time() - cache.stat().st_mtime) < ttl:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    try:
        r = requests.get(url, headers=_HDRS, timeout=20)
    except Exception:  # noqa: BLE001
        return None
    if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
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


def games(sb: str = DEFAULT_SB, league: str = "fifa", sport: str = "football",
          ttl: float = 600.0, cfg=None) -> list[dict]:
    d = _get("/sgmGames", ttl=ttl, cfg=cfg, sb=sb, league=league, sport=sport)
    return d if isinstance(d, list) else []


def _team_title(side) -> str | None:
    if isinstance(side, list) and side:
        return side[0].get("title")
    return None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_markets(grouped: list[dict]) -> dict:
    """Grouped markets -> tidy odds dict (decimals). Keys present only when the market exists."""
    out: dict = {}
    by_stat = {g.get("statistic"): g.get("markets", []) for g in (grouped or [])}
    # moneyline (3-way "to win"): condition.type 1=home, x=draw, 2=away
    ml = {}
    for mm in by_stat.get("to win", []):
        c = (mm.get("condition") or [{}])[0]
        code = {"1": "H", "x": "D", "2": "A"}.get(str(c.get("type")))
        if code and _f(mm.get("odds")):
            ml[code] = _f(mm["odds"])
    if {"H", "D", "A"} <= set(ml):
        out["moneyline"] = ml
    # both teams to score
    btts = {}
    for mm in by_stat.get("both teams to score", []):
        c = (mm.get("condition") or [{}])[0]
        side = str(c.get("type")).lower()
        if side in ("yes", "no") and _f(mm.get("odds")):
            btts[side] = _f(mm["odds"])
    if "yes" in btts and "no" in btts:
        out["btts"] = btts
    # total goals ladder: condition.value=line, type=over/under
    totals: dict = {}
    for mm in by_stat.get("total goals", []):
        c = (mm.get("condition") or [{}])[0]
        line, side = _f(c.get("value")), str(c.get("type")).lower()
        if line is not None and side in ("over", "under") and _f(mm.get("odds")):
            totals.setdefault(line, {})[side] = _f(mm["odds"])
    out["totals"] = {ln: v for ln, v in totals.items() if "over" in v and "under" in v}
    return out


def _index(gms: list[dict]) -> dict:
    """{frozenset(lowercased normalized pair): game} — DST team titles are lowercase."""
    idx = {}
    for g in gms:
        h, a = _team_title(g.get("team1")), _team_title(g.get("team2"))
        if not (h and a):
            continue
        key = frozenset({(normalize_team(h) or h).lower(), (normalize_team(a) or a).lower()})
        if len(key) == 2:
            idx[key] = g
    return idx


def book_odds(home: str, away: str, sb: str = DEFAULT_SB, gms: list[dict] | None = None,
              index: dict | None = None, cfg=None) -> dict | None:
    """The user's book odds for a fixture (moneyline/totals/BTTS as decimals + ``home``/``away``
    orientation), or None. Matched on the unordered, case-insensitive normalized team pair."""
    idx = index if index is not None else _index(gms if gms is not None else games(sb=sb, cfg=cfg))
    key = frozenset({(normalize_team(home) or home).lower(), (normalize_team(away) or away).lower()})
    g = idx.get(key)
    if not g:
        return None
    parsed = parse_markets(game_markets(g, sb=sb, cfg=cfg))
    if not parsed.get("moneyline") and not parsed.get("btts") and not parsed.get("totals"):
        return None
    # orient the moneyline to the REQUESTED (home, away): DST type 1=their team1. If our `home`
    # is their team2, swap H<->A so H is always the queried home. Totals/BTTS are symmetric.
    dst_home = (normalize_team(_team_title(g.get("team1")) or "") or "").lower()
    flipped = dst_home == (normalize_team(away) or away).lower()
    if flipped and "moneyline" in parsed:
        ml = parsed["moneyline"]
        parsed["moneyline"] = {"H": ml.get("A"), "D": ml.get("D"), "A": ml.get("H")}
    parsed["sb"] = sb
    parsed["flipped"] = flipped
    return parsed


def game_markets(game: dict, sb: str = DEFAULT_SB, ttl: float = 300.0, cfg=None) -> list[dict]:
    provs = game.get("providers") or []
    pid = provs[0].get("id") if provs else None
    if pid is None:
        return []
    d = _get("/sgmMarkets/gfm/grouped", ttl=ttl, cfg=cfg, sb=sb, gameId=pid, legacy="1")
    return d if isinstance(d, list) else []


def connectivity_check(sb: str = DEFAULT_SB, cfg=None) -> str:
    """'ok' / 'unreachable' — DST is open (no key needed)."""
    return "ok" if games(sb=sb, ttl=0.0, cfg=cfg) else "unreachable"
