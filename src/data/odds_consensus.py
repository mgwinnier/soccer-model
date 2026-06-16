"""Multi-book market consensus from ESPN's already-cached summary JSON.

A single-book bettor can't line-shop, but the other ~10 books ESPN reports are a
free **read-only reference**. ``odds.py::_pick_odds_entry`` keeps only one book
(Bet365) and discards the rest — but the raw `odds` array (every book, with
provider names) is still sitting in each cached ``summary_*.json``. We re-extract
it here (no re-harvest), de-vig each book's *own* line, and take the **median**
fair probability per outcome across the *other* books as the market consensus.

The one structural edge a single-book bettor has is to bet only when their book's
price beats that consensus — a soft / stale line. ``code_edge_vs_consensus``
quantifies it: positive means our book's price is +EV under the consensus's own
estimate of the truth, i.e. our book is lagging the market in our favour.
"""
from __future__ import annotations

import json

import numpy as np

from ..config import load_config, path_for
from .odds import (american_to_decimal, decimal_to_prob, devig, _num)


def _norm_provider(name: str | None) -> str:
    """Collapse a provider name to a dedup key ('Caesars (NJ)' -> 'caesars')."""
    if not isinstance(name, str) or not name:   # None / NaN-float / empty -> no key
        return ""
    s = name.lower()
    for cut in ("(", " - ", " sportsbook"):
        i = s.find(cut)
        if i != -1:
            s = s[:i]
    return s.strip().replace(" ", "")


def extract_book_prices(data: dict) -> list[dict]:
    """Every usable book's moneyline / total / spread from a summary JSON.

    Searches both ``pickcenter`` (recent) and ``odds`` (finished, multi-book),
    dedups by normalized provider, and skips in-play ('- Live') variants — those
    are not closing prices and would distort a consensus."""
    pools = list(data.get("pickcenter") or []) + list(data.get("odds") or [])
    books: list[dict] = []
    seen: set[str] = set()
    for o in pools:
        ho, ao = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
        if ho.get("moneyLine") is None or ao.get("moneyLine") is None:
            continue
        name = (o.get("provider") or {}).get("name") or ""
        if "live" in name.lower():
            continue
        key = _norm_provider(name)
        if not key or key in seen:
            continue
        seen.add(key)
        books.append({
            "provider": name,
            "ml_home": ho.get("moneyLine"), "ml_away": ao.get("moneyLine"),
            "ml_draw": (o.get("drawOdds") or {}).get("moneyLine"),
            "total_line": _num(o.get("overUnder")),
            "ou_over_odds": o.get("overOdds"), "ou_under_odds": o.get("underOdds"),
            "spread_home_line": _num(o.get("spread")),
            "spread_home_odds": ho.get("spreadOdds"),
            "spread_away_odds": ao.get("spreadOdds"),
        })
    return books


def _fair_3way(b: dict) -> list[float] | None:
    raw = [decimal_to_prob(american_to_decimal(b.get(k)))
           for k in ("ml_home", "ml_draw", "ml_away")]
    return devig(raw, "proportional")


def _fair_2way(a_odds, b_odds) -> list[float] | None:
    raw = [decimal_to_prob(american_to_decimal(a_odds)),
           decimal_to_prob(american_to_decimal(b_odds))]
    return devig(raw, "proportional")


def consensus(books: list[dict], exclude: str | None = None) -> dict | None:
    """Median de-vigged fair probabilities across the *other* books.

    Returns ``{"moneyline": {"H","D","A","n"}, "totals": {"line","over","under","n"}}``
    (each section omitted when fewer than 2 books support it). ``exclude`` is the
    bettor's own book (matched on normalized provider) so they don't anchor to
    themselves."""
    ex = _norm_provider(exclude) if exclude else None
    pool = [b for b in books if _norm_provider(b["provider"]) != ex] if ex else books
    out: dict = {}

    # Moneyline (3-way)
    fairs = [f for b in pool if (f := _fair_3way(b)) is not None]
    if len(fairs) >= 2:
        med = np.median(np.array(fairs), axis=0)
        med = med / med.sum()
        out["moneyline"] = {"H": float(med[0]), "D": float(med[1]),
                            "A": float(med[2]), "n": len(fairs)}

    # Totals (2-way) — only books sharing the modal over/under line are comparable
    lines = [b["total_line"] for b in pool if b.get("total_line") is not None
             and b.get("ou_over_odds") is not None and b.get("ou_under_odds") is not None]
    if lines:
        modal = max(set(lines), key=lines.count)
        ou = [f for b in pool
              if b.get("total_line") == modal
              and (f := _fair_2way(b.get("ou_over_odds"), b.get("ou_under_odds"))) is not None]
        if len(ou) >= 2:
            med = np.median(np.array(ou), axis=0)
            med = med / med.sum()
            out["totals"] = {"line": float(modal), "over": float(med[0]),
                             "under": float(med[1]), "n": len(ou)}
    return out or None


def consensus_prob_for_code(cons: dict | None, code) -> float | None:
    """Consensus fair probability for a grade code (see bet_grade._candidates).

    Moneyline 'H'/'D'/'A' and totals ('over'|'under', line) are supported; spread
    is deliberately not (lines vary too much across books to form a consensus, and
    the spread market is disabled by default)."""
    if not cons:
        return None
    if code in ("H", "D", "A"):
        ml = cons.get("moneyline")
        return ml[code] if ml else None
    if isinstance(code, tuple):
        kind, line = code
        tot = cons.get("totals")
        if tot and kind in ("over", "under") and abs(tot["line"] - float(line)) < 1e-9:
            return tot[kind]
    return None


def code_edge_vs_consensus(cons: dict | None, code, decimal_odds: float | None) -> float | None:
    """EV of taking ``code`` at ``decimal_odds`` under the consensus's own estimate.

    ``q·d − 1`` where ``q`` = consensus fair prob and ``d`` = our book's decimal
    price. Positive ⇒ our book offers a longer price than the market thinks fair
    (off-consensus favourable); None when no consensus is available for the code."""
    q = consensus_prob_for_code(cons, code)
    if q is None or decimal_odds is None:
        return None
    return q * decimal_odds - 1.0


def load_cached_summary(game_id, league: str, cfg: dict | None = None) -> dict | None:
    """Read a cached summary JSON from disk only — never hits the network, so it's
    safe to call thousands of times inside a backtest."""
    cfg = cfg or load_config()
    cache = path_for("data_raw", cfg) / "odds" / f"summary_{league}_{game_id}.json"
    if not cache.exists():
        return None
    try:
        return json.loads(cache.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def match_consensus(game_id, league: str, exclude: str | None = "bet365",
                    cfg: dict | None = None, allow_fetch: bool = False) -> dict | None:
    """Consensus for one game from its cached summary.

    Offline by default (safe inside a backtest). With ``allow_fetch=True`` it
    triggers ``odds.fetch_summary_odds`` first to populate the cache for a live
    fixture whose summary hasn't been pulled yet."""
    data = load_cached_summary(game_id, league, cfg)
    if data is None and allow_fetch:
        from .odds import fetch_summary_odds
        try:
            fetch_summary_odds(game_id, league=league, cfg=cfg)  # side effect: caches JSON
        except Exception:  # noqa: BLE001 — best-effort; no consensus is acceptable
            return None
        data = load_cached_summary(game_id, league, cfg)
    if data is None:
        return None
    return consensus(extract_book_prices(data), exclude=exclude)
