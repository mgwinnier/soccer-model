"""Reader for the Action Network BTTS feed harvested on the VPS.

The harvester (`scripts/actionnetwork_harvest.py`) runs on the VPS (Action Network's internal API
is firewall-blocked locally and 403s cloud IPs) and writes ``data/feeds/actionnetwork_btts.json``:
pre-match World Cup BTTS yes/no from US books, per game, plus the best price across books. This
module loads that feed and answers ``btts_prices_for(home, away, date)`` for the value layer.

Why this is the *bettable* BTTS line (vs TheStatsAPI's): it's **pre-match** and from US books
(DraftKings/FanDuel/BetMGM/…) — the books you'd actually bet — with line-shopping (best price).
TheStatsAPI only has a settled Bet365 line after the whistle. Honest: no key/feed/match → None
(the value layer then falls back to the TheStatsAPI settled line or shows model-only).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import PROJECT_ROOT
from .odds import american_to_decimal
from .team_names import normalize_team

DEFAULT_FEED = PROJECT_ROOT / "data" / "feeds" / "actionnetwork_btts.json"


def load_feed(path: str | Path | None = None, url: str | None = None) -> dict | None:
    """Load the harvested feed. Prefers a live URL (``ACTIONNETWORK_FEED_URL`` env, e.g. the
    VPS-served JSON) when set, else the committed file. Returns None if unavailable/unparsable."""
    url = url or os.environ.get("ACTIONNETWORK_FEED_URL")
    if url:
        try:
            import requests
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001 — fall through to the committed file
            pass
    p = Path(path) if path else DEFAULT_FEED
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _pair(home, away) -> frozenset:
    return frozenset({normalize_team(home), normalize_team(away)})


def _index(feed: dict | None) -> dict:
    out: dict = {}
    for g in (feed or {}).get("games", []) or []:
        key = _pair(g.get("home"), g.get("away"))
        if None not in key and len(key) == 2:
            out[key] = g
    return out


def btts_prices_for(home: str, away: str, date=None, feed: dict | None = None,
                    index: dict | None = None) -> dict | None:
    """Best-across-books BTTS for a fixture: ``{yes, no}`` decimals (+ books, source, n_books),
    or None. Matched on the unordered normalized team pair (date is informational — the feed is
    the current slate). Never fabricates: returns None when the game isn't in the feed."""
    idx = index if index is not None else _index(feed)
    g = idx.get(_pair(home, away))
    if not g:
        return None
    best = g.get("btts_best") or {}
    yd, nd = american_to_decimal(best.get("yes")), american_to_decimal(best.get("no"))
    if yd is None or nd is None:
        return None
    return {"yes": yd, "no": nd, "yes_book": best.get("yes_book"),
            "no_book": best.get("no_book"), "source": "actionnetwork",
            "n_books": len(g.get("btts") or {}), "generated_at": (feed or {}).get("generated_at")}
