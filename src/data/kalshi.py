"""Kalshi prediction-exchange odds — public market data, no auth.

Kalshi is a CFTC-regulated exchange: a Yes contract at ask ``p`` (dollars 0–1) pays $1, so the **ask
≈ the implied probability you pay** and EV is clean: ``model_p / ask − 1`` (no de-vig). Two surfaces we use:

* **Per-match 1X2** — series ``KXWCGAME``: one event per fixture (``KXWCGAME-26JUN17GHAPAN`` = Ghana v
  Panama) with three markets ``-{HOME3}`` / ``-TIE`` / ``-{AWAY3}``. Resolves on 90'+stoppage (no ET/pens)
  → exactly our Match Result (H/D/A).
* **Tournament winner futures** — series ``KXMENWORLDCUP`` (event ``KXMENWORLDCUP-26``, one Yes per team).

Prices live in the ``*_dollars`` fields (``yes_bid_dollars``/``yes_ask_dollars``/``last_price_dollars`` +
``previous_yes_ask_dollars`` for movement). Public GETs — no key. Disk-cached, honest None when absent.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import requests

from .team_names import normalize_team

BASE = os.environ.get("KALSHI_BASE") or "https://api.elections.kalshi.com/trade-api/v2"
WC_GAME_SERIES = "KXWCGAME"          # per-match 1X2
WC_WINNER_EVENT = "KXMENWORLDCUP-26"  # tournament winner futures
# Per-match market series (all share the same event suffix, e.g. ``26JUN17GHAPAN``):
SERIES = {"winner": "KXWCGAME", "btts": "KXWCBTTS", "total": "KXWCTOTAL", "spread": "KXWCSPREAD"}
_UA = "Mozilla/5.0 (soccer-model; +https://github.com/mgwinnier/soccer-model)"

# Kalshi spells a few nations differently than our canonical names — map before normalize_team.
_KALSHI_ALIASES = {
    "turkiye": "Turkey", "korea republic": "South Korea", "ir iran": "Iran",
    "congo dr": "DR Congo", "czechia": "Czech Republic", "bosnia and herzegovina": "Bosnia and Herzegovina",
    "curacao": "Curacao", "cape verde": "Cape Verde", "ivory coast": "Ivory Coast",
}


def _norm(name: str | None) -> str | None:
    if not name:
        return None
    base = _KALSHI_ALIASES.get(str(name).strip().lower(), name)
    return normalize_team(base)


def _f(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _cache_dir(cfg=None) -> Path:
    root = Path(__file__).resolve().parents[2]
    d = root / "data" / "raw" / "kalshi"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(path: str, ttl: float = 20.0, cfg=None, **params):
    """Cached GET against the public Kalshi market-data API. Follows the cursor to page fully."""
    qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
    url = f"{BASE}{path}" + (f"?{qs}" if qs else "")
    cache = _cache_dir(cfg) / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if ttl > 0 and cache.exists() and (time.time() - cache.stat().st_mtime) < ttl:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    try:
        r = requests.get(url, headers={"User-Agent": _UA, "Accept": "application/json"}, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:  # noqa: BLE001
        return None
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def _markets(ttl: float = 20.0, cfg=None, **params) -> list[dict]:
    """All markets matching params, paging through the cursor."""
    out, cursor, guard = [], None, 0
    while guard < 20:
        guard += 1
        data = _get("/markets", ttl=ttl, cfg=cfg, cursor=cursor, **params)
        if not data:
            break
        out.extend(data.get("markets") or [])
        cursor = data.get("cursor") or None
        if not cursor:
            break
    return out


def _price(m: dict) -> dict:
    """One contract's Yes prices (dollars 0–1) + its ticker, from the real ``*_dollars`` fields."""
    return {"bid": _f(m.get("yes_bid_dollars")), "ask": _f(m.get("yes_ask_dollars")),
            "last": _f(m.get("last_price_dollars")), "prev_ask": _f(m.get("previous_yes_ask_dollars")),
            "ticker": m.get("ticker")}


def _side(m: dict, side: str) -> dict:
    """Prices for the Yes or No side of a market — the side you'd BUY. Kalshi exposes both
    ``yes_*`` and ``no_*`` dollar fields, so 'Under'/'No'/'away +line' is a real tradeable price."""
    return {"bid": _f(m.get(f"{side}_bid_dollars")), "ask": _f(m.get(f"{side}_ask_dollars")),
            "prev_ask": _f(m.get(f"previous_{side}_ask_dollars")), "ticker": m.get("ticker")}


def _suffix(event_ticker: str | None) -> str | None:
    """The shared game key after the series prefix, e.g. KXWCBTTS-26JUN17GHAPAN -> 26JUN17GHAPAN."""
    if not event_ticker or "-" not in event_ticker:
        return event_ticker
    return event_ticker.split("-", 1)[1]


def ask_decimal(price: float | None) -> float | None:
    """Decimal odds of buying a Yes contract at ``price`` dollars (pays $1)."""
    return (1.0 / price) if (price and price > 0) else None


def wc_game_markets(ttl: float = 20.0, cfg=None) -> list[dict]:
    return _markets(ttl=ttl, cfg=cfg, series_ticker=WC_GAME_SERIES, status="open", limit=1000)


def match_winner(home: str, away: str, markets: list[dict] | None = None, cfg=None,
                 ttl: float = 20.0) -> dict | None:
    """Per-match 1X2 from Kalshi, oriented to the caller's home/away. Returns
    ``{"H":{bid,ask,last,prev_ask,ticker}, "D":..., "A":..., "event", "home", "away"}`` or None.
    Matched by team NAME (not event order), so the orientation is always correct."""
    nh, na = _norm(home), _norm(away)
    if not (nh and na):
        return None
    markets = markets if markets is not None else wc_game_markets(ttl=ttl, cfg=cfg)
    by_event: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        by_event[m.get("event_ticker")].append(m)
    for ev, ms in by_event.items():
        teams, tie = {}, None
        for m in ms:
            sub = (m.get("yes_sub_title") or "").strip()
            if sub.lower() == "tie":
                tie = m
            else:
                key = _norm(sub)
                if key:
                    teams[key] = m
        if tie is not None and nh in teams and na in teams:
            return {"H": _price(teams[nh]), "D": _price(tie), "A": _price(teams[na]),
                    "event": ev, "home": home, "away": away}
    return None


def all_match_markets(ttl: float = 30.0, cfg=None) -> list[dict]:
    """All open per-match markets across winner / BTTS / total / spread (one cached call each)."""
    out = []
    for s in (SERIES["winner"], SERIES["btts"], SERIES["total"], SERIES["spread"]):
        out.extend(_markets(ttl=ttl, cfg=cfg, series_ticker=s, status="open", limit=1000))
    return out


def match_book(home: str, away: str, markets: list[dict] | None = None, cfg=None,
               ttl: float = 30.0) -> dict | None:
    """Full per-fixture Kalshi book — Match Result + BTTS + Totals + Spread — oriented to the
    caller's home/away. Each leaf is the BUY-side ``{bid, ask, prev_ask, ticker}``. None if the
    fixture isn't listed. Markets are joined across series by their shared event suffix."""
    nh, na = _norm(home), _norm(away)
    if not (nh and na):
        return None
    markets = markets if markets is not None else all_match_markets(ttl=ttl, cfg=cfg)
    by_suffix: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        by_suffix[_suffix(m.get("event_ticker"))].append(m)
    # find the game whose WINNER markets name both teams
    target = None
    for suf, ms in by_suffix.items():
        teams = set()
        for m in ms:
            if (m.get("event_ticker") or "").startswith(SERIES["winner"] + "-"):
                sub = (m.get("yes_sub_title") or "").strip()
                if sub and sub.lower() != "tie":
                    teams.add(_norm(sub))
        if nh in teams and na in teams:
            target = suf
            break
    if target is None:
        return None
    book = {"home": home, "away": away, "event": target,
            "moneyline": {}, "btts": {}, "totals": {}, "spread": {}}
    for m in by_suffix[target]:
        ev = m.get("event_ticker") or ""
        if ev.startswith(SERIES["winner"] + "-"):
            sub = (m.get("yes_sub_title") or "").strip()
            code = "D" if sub.lower() == "tie" else ("H" if _norm(sub) == nh
                                                     else ("A" if _norm(sub) == na else None))
            if code:
                book["moneyline"][code] = _side(m, "yes")
        elif ev.startswith(SERIES["btts"] + "-"):
            book["btts"] = {"yes": _side(m, "yes"), "no": _side(m, "no")}
        elif ev.startswith(SERIES["total"] + "-"):
            line = _f(m.get("floor_strike"))
            if line is not None:                       # Yes = Over line, No = Under line
                book["totals"][line] = {"over": _side(m, "yes"), "under": _side(m, "no")}
        elif ev.startswith(SERIES["spread"] + "-"):
            line = _f(m.get("floor_strike"))           # "{team} wins by over {line}"
            tm = _norm((m.get("yes_sub_title") or "").split(" wins by")[0].strip())
            if line is not None and tm in (nh, na):
                book["spread"].setdefault(line, {})["home" if tm == nh else "away"] = {
                    "yes": _side(m, "yes"), "no": _side(m, "no")}
    return book


def price_for(book: dict | None, market: str, selection: str, home: str, away: str) -> dict | None:
    """The BUY-side price ``{bid, ask, prev_ask}`` for one of our bet selections, or None.

    Handles all four markets: Match Result (Yes on the outcome), BTTS (Yes/No), Total Goals
    (Over = Yes / Under = No at the line), Spread (team −line = that team's 'wins by over line'
    Yes; team +line = the No side of the opponent's 'wins by over line')."""
    if not book:
        return None
    if market == "Match Result":
        code = "H" if selection == home else ("A" if selection == away else "D")
        return (book.get("moneyline") or {}).get(code)
    if market == "BTTS":
        return (book.get("btts") or {}).get("yes" if "Yes" in selection else "no")
    if market == "Total Goals":
        mt = re.search(r"[\d.]+", selection)
        if not mt:
            return None
        t = (book.get("totals") or {}).get(float(mt.group())) or {}
        return t.get("over" if selection.startswith("Over") else "under")
    if market == "Spread":
        mt = re.search(r"[-+]?\d+(?:\.\d+)?", selection)
        if not mt:
            return None
        signed = float(mt.group())
        sp = (book.get("spread") or {}).get(abs(signed))
        if not sp:
            return None
        team_is_home = selection.startswith(home)
        if signed < 0:                                 # favored: buy that team's "wins by over" Yes
            d = sp.get("home" if team_is_home else "away")
            return (d or {}).get("yes")
        d = sp.get("away" if team_is_home else "home")  # underdog +line: No of opponent's market
        return (d or {}).get("no")
    return None


def winner_futures(ttl: float = 60.0, cfg=None) -> dict:
    """Tournament-winner Yes prices by normalized team: ``{team: {bid,ask,last,prev_ask,ticker}}``."""
    out = {}
    for m in _markets(ttl=ttl, cfg=cfg, event_ticker=WC_WINNER_EVENT, limit=200):
        team = _norm(m.get("yes_sub_title"))
        if team:
            out[team] = _price(m)
    return out


def signal(model_p: float | None, bid: float | None, ask: float | None,
           buy_edge: float = 0.05, sell_edge: float = 0.05) -> dict:
    """BUY / SELL / HOLD on one contract vs the model, using the exchange's two-sided prices.

    * BUY  — model_p ≥ ask + buy_edge  (the model thinks Yes is underpriced; buy at the ask).
    * SELL — bid ≥ model_p + sell_edge (the market prices it richer than the model; sell into the
             bid if long, or buy the cheap No side). ``ev_sell`` is the model EV of the No side.
    * HOLD — otherwise (the edge is inside the bid/ask spread → not worth the friction).
    """
    ev_buy = (model_p / ask - 1.0) if (model_p is not None and ask and ask > 0) else None
    # No side: cost = 1 - bid (sell Yes at bid == buy No at 1-bid), model prob of No = 1 - model_p
    ev_sell = ((1.0 - model_p) / (1.0 - bid) - 1.0) if (model_p is not None and bid is not None
                                                        and 0 < bid < 1) else None
    spread = (ask - bid) if (ask is not None and bid is not None) else None
    action = "HOLD"
    if model_p is not None and ask is not None and (model_p - ask) >= buy_edge:
        action = "BUY"
    elif model_p is not None and bid is not None and (bid - model_p) >= sell_edge:
        action = "SELL"
    return {"action": action, "ev_buy": ev_buy, "ev_sell": ev_sell, "spread": spread}
