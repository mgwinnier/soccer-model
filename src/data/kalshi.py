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
import time
from collections import defaultdict
from pathlib import Path

import requests

from .team_names import normalize_team

BASE = os.environ.get("KALSHI_BASE") or "https://api.elections.kalshi.com/trade-api/v2"
WC_GAME_SERIES = "KXWCGAME"          # per-match 1X2
WC_WINNER_EVENT = "KXMENWORLDCUP-26"  # tournament winner futures
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
    """One contract's prices (dollars 0–1) + its ticker, from the real ``*_dollars`` fields."""
    return {"bid": _f(m.get("yes_bid_dollars")), "ask": _f(m.get("yes_ask_dollars")),
            "last": _f(m.get("last_price_dollars")), "prev_ask": _f(m.get("previous_yes_ask_dollars")),
            "ticker": m.get("ticker")}


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
