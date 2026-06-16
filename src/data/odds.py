"""Live odds + fixtures from free, key-less public endpoints.

Primary source is ESPN's hidden JSON API, which returns *both* the fixture list
and bookmaker moneyline (home / draw / away) prices in a single call — verified
live against the 2026 World Cup. We convert American prices to decimal, strip the
bookmaker's overround ("de-vig") to recover the market's true implied
probabilities, and expose them for the model-vs-market comparison in the UI.

    ESPN:  https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=YYYYMMDD

Bovada is supported as an optional second line; The Odds API as an optional keyed
fallback. Everything degrades gracefully — no source, no crash.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from ..config import load_config, path_for
from .team_names import normalize_team

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
)
ESPN_SUMMARY = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary"
)
_HEADERS = {"User-Agent": "Mozilla/5.0 (soccer-model/1.0)"}


def fetch_summary_odds(game_id: str, league: str = "fifa.world",
                       cfg: dict | None = None, use_cache: bool = True) -> dict | None:
    """Retained pre-match odds for a (possibly finished) game via the summary
    endpoint's ``pickcenter`` block — the scoreboard nulls these post-match, the
    summary page keeps them. Returns moneyline H/D/A + totals + spread (American)."""
    cfg = cfg or load_config()
    cache_dir = path_for("data_raw", cfg) / "odds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"summary_{league}_{game_id}.json"
    data = None
    if use_cache and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    if data is None:
        resp = requests.get(ESPN_SUMMARY.format(league=league),
                            params={"event": game_id}, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
    # Recent games carry odds in `pickcenter`; finished games keep them in the
    # multi-book `odds` array instead. Search both and prefer Bet365.
    o = _pick_odds_entry(data)
    if o is None:
        return None
    ho, ao = o.get("homeTeamOdds", {}), o.get("awayTeamOdds", {})
    return {
        "provider": (o.get("provider") or {}).get("name"),
        "ml_home": ho.get("moneyLine"), "ml_away": ao.get("moneyLine"),
        "ml_draw": (o.get("drawOdds") or {}).get("moneyLine"),
        "total_line": _num(o.get("overUnder")),
        "ou_over_odds": o.get("overOdds"), "ou_under_odds": o.get("underOdds"),
        "spread_home_line": _num(o.get("spread")),
        "spread_home_odds": ho.get("spreadOdds"), "spread_away_odds": ao.get("spreadOdds"),
    }


def _pick_odds_entry(data: dict) -> dict | None:
    """Choose a usable odds entry, preferring Bet365, from pickcenter or odds."""
    pools = list(data.get("pickcenter") or []) + list(data.get("odds") or [])
    usable = [o for o in pools
              if (o.get("homeTeamOdds") or {}).get("moneyLine") is not None]
    if not usable:
        return None
    for o in usable:
        name = ((o.get("provider") or {}).get("name") or "").lower().replace(" ", "")
        if "bet365" in name:
            return o
    return usable[0]


# ----------------------------------------------------------------- odds math
def american_to_decimal(american: float | str | None) -> float | None:
    if american is None or american == "":
        return None
    a = float(american)
    if a == 0:                       # 0 is not a valid American price (bad feed data)
        return None
    if a > 0:
        return 1.0 + a / 100.0
    return 1.0 + 100.0 / abs(a)


def decimal_to_prob(dec: float | None) -> float | None:
    """Raw implied probability (still includes the bookmaker margin)."""
    if dec is None or dec <= 0:
        return None
    return 1.0 / dec


def devig(probs: list[float | None], method: str = "proportional") -> list[float] | None:
    """Remove the overround from a set of raw implied probabilities.

    ``proportional`` simply renormalises to sum 1 (multiplicative margin).
    ``shin`` solves Shin's model, which attributes part of the margin to
    insider trading and is gentler on favourites. Returns None if any leg is
    missing.
    """
    if any(p is None for p in probs):
        return None
    p = np.array(probs, dtype=float)
    booksum = p.sum()
    if booksum <= 0:
        return None
    if method == "proportional" or len(p) < 2:
        return list(p / booksum)
    # Shin (1992) iterative solution for z (insider-trading proportion)
    z = 0.0
    for _ in range(100):
        denom = booksum - z * (booksum - 1) if booksum != 1 else booksum
        sqrt_term = np.sqrt(z ** 2 + 4 * (1 - z) * p ** 2 / booksum)
        true = (sqrt_term - z) / (2 * (1 - z)) if z < 1 else p / booksum
        s = true.sum()
        new_z = z + (s - 1.0) * 0.5
        if abs(new_z - z) < 1e-9:
            break
        z = min(max(new_z, 0.0), 0.2)
    true = true / true.sum()
    return list(true)


# -------------------------------------------------------------- ESPN parsing
def _parse_espn_event(ev: dict) -> dict | None:
    comp = ev["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    if "home" not in teams or "away" not in teams:
        return None
    home = normalize_team(teams["home"]["team"].get("displayName"))
    away = normalize_team(teams["away"]["team"].get("displayName"))
    row = {
        "game_id": ev.get("id"),
        "date": pd.to_datetime(ev.get("date")),
        "home_team": home, "away_team": away,
        "venue": comp.get("venue", {}).get("fullName"),
        "status": comp.get("status", {}).get("type", {}).get("state"),
        "home_score": _to_int(teams["home"].get("score")),
        "away_score": _to_int(teams["away"].get("score")),
        "ml_home": None, "ml_away": None, "ml_draw": None, "provider": None,
        "ml_home_open": None, "ml_away_open": None,
        "total_line": None, "ou_over_odds": None, "ou_under_odds": None,
        "spread_home_line": None, "spread_home_odds": None, "spread_away_odds": None,
    }
    # Finished matches sometimes carry "odds": [null]; skip null entries.
    odds_list = [x for x in (comp.get("odds") or []) if x]
    if odds_list:
        o = odds_list[0]
        row["provider"] = (o.get("provider") or {}).get("name")
        ml = o.get("moneyline") or {}
        row["ml_home"] = _ml_close(ml.get("home"))
        row["ml_away"] = _ml_close(ml.get("away"))
        row["ml_draw"] = (o.get("drawOdds") or {}).get("moneyLine")
        # Opening moneyline (for line-movement / sharp-money signal)
        row["ml_home_open"] = _ml_open(ml.get("home"))
        row["ml_away_open"] = _ml_open(ml.get("away"))
        # Totals (over/under): line lives as "o2.5"/"u2.5"
        total = o.get("total") or {}
        over_odds, over_line = _close_field(total.get("over"))
        under_odds, _ = _close_field(total.get("under"))
        row["ou_over_odds"] = over_odds
        row["ou_under_odds"] = under_odds
        row["total_line"] = _num(over_line)
        # Spread / handicap: home line e.g. "-1.5"
        spread = o.get("pointSpread") or {}
        sh_odds, sh_line = _close_field(spread.get("home"))
        sa_odds, _ = _close_field(spread.get("away"))
        row["spread_home_odds"] = sh_odds
        row["spread_away_odds"] = sa_odds
        row["spread_home_line"] = _num(sh_line)
    return row


def _ml_close(side: dict | None):
    if not side:
        return None
    close = side.get("close") or side.get("open") or {}
    return close.get("odds")


def _ml_open(side: dict | None):
    if not side:
        return None
    return (side.get("open") or {}).get("odds")


def _close_field(side: dict | None) -> tuple:
    """Return (odds, line) from a side's close (fallback open) sub-object."""
    if not side:
        return None, None
    close = side.get("close") or side.get("open") or {}
    return close.get("odds"), close.get("line")


def _num(line_str):
    """'o2.5'/'u2.5'/'-1.5'/'+1.5' -> float, else None."""
    if line_str is None:
        return None
    import re
    m = re.search(r"[-+]?\d*\.?\d+", str(line_str))
    return float(m.group()) if m else None


def _to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def fetch_espn(date: str, league: str = "fifa.world",
               cfg: dict | None = None, use_cache: bool = True) -> list[dict]:
    """Fetch one day's fixtures+odds. ``date`` = 'YYYY-MM-DD' or 'YYYYMMDD'."""
    cfg = cfg or load_config()
    ymd = date.replace("-", "")
    cache_dir = path_for("data_raw", cfg) / "odds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"espn_{league}_{ymd}.json"
    data = None
    if use_cache and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    if data is None:
        resp = requests.get(ESPN_SCOREBOARD.format(league=league),
                            params={"dates": ymd, "limit": 200},
                            headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
    out = []
    for ev in data.get("events", []):
        try:
            parsed = _parse_espn_event(ev)
            if parsed:
                out.append(parsed)
        except Exception:  # noqa: BLE001  — never let one bad event kill the feed
            continue
    return out


def fetch_espn_range(start: str, end: str, league: str = "fifa.world",
                     cfg: dict | None = None, use_cache: bool = True) -> list[dict]:
    """Fetch a *date range* in one call (ESPN's reliable form for past dates).

    ESPN's single-date param is flaky for past dates; the ``YYYYMMDD-YYYYMMDD``
    range form reliably returns finished + upcoming matches with odds.
    """
    cfg = cfg or load_config()
    a = pd.Timestamp(start).strftime("%Y%m%d")
    b = pd.Timestamp(end).strftime("%Y%m%d")
    cache_dir = path_for("data_raw", cfg) / "odds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"espn_{league}_{a}_{b}.json"
    data = None
    if use_cache and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    if data is None:
        # ESPN intermittently throttles a range request that *includes today* to
        # "today only" (~3 events). Spaced retries fix the live case. For purely
        # historical ranges the response is authoritative, so we never retry
        # (otherwise sparse past windows waste ~6s each on pointless retries).
        import time
        today_ymd = datetime.utcnow().strftime("%Y%m%d")
        is_live = b >= today_ymd
        best = {"events": []}
        for attempt in range(4):
            resp = requests.get(ESPN_SCOREBOARD.format(league=league),
                                params={"dates": f"{a}-{b}", "limit": 400},
                                headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            d = resp.json()
            if len(d.get("events", [])) > len(best.get("events", [])):
                best = d
            if a == b or not is_live or len(best.get("events", [])) > 4:
                break
            time.sleep(1.5)
        data = best
        if data.get("events"):
            cache.write_text(json.dumps(data), encoding="utf-8")
    out = []
    for ev in data.get("events", []):
        try:
            parsed = _parse_espn_event(ev)
            if parsed:
                out.append(parsed)
        except Exception:  # noqa: BLE001
            continue
    return out


def fetch_fixtures(start: str, days: int = 3, league: str = "fifa.world",
                   cfg: dict | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Fixtures + de-vigged market probabilities over a date window."""
    cfg = cfg or load_config()
    end = (pd.Timestamp(start) + timedelta(days=days - 1)).strftime("%Y-%m-%d")
    try:
        rows = fetch_espn_range(start, end, league, cfg, use_cache)
    except Exception:  # noqa: BLE001
        rows = []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # de-vig moneylines into market probabilities
    raw = df.apply(
        lambda r: [
            decimal_to_prob(american_to_decimal(r["ml_home"])),
            decimal_to_prob(american_to_decimal(r["ml_draw"])),
            decimal_to_prob(american_to_decimal(r["ml_away"])),
        ],
        axis=1,
    )
    fair = raw.apply(lambda p: devig(p, "proportional"))
    df["mkt_home"] = fair.apply(lambda f: f[0] if f else np.nan)
    df["mkt_draw"] = fair.apply(lambda f: f[1] if f else np.nan)
    df["mkt_away"] = fair.apply(lambda f: f[2] if f else np.nan)
    df["overround"] = raw.apply(
        lambda p: sum(x for x in p if x) - 1 if all(x for x in p) else np.nan
    )

    # de-vig totals (over/under) and spread (home/away) two-way markets
    def _devig2(a_odds, b_odds):
        pa = decimal_to_prob(american_to_decimal(a_odds))
        pb = decimal_to_prob(american_to_decimal(b_odds))
        fair2 = devig([pa, pb], "proportional")
        return (fair2[0], fair2[1]) if fair2 else (np.nan, np.nan)

    ou = df.apply(lambda r: _devig2(r["ou_over_odds"], r["ou_under_odds"]), axis=1)
    df["mkt_over"] = ou.apply(lambda t: t[0])
    df["mkt_under"] = ou.apply(lambda t: t[1])
    sp = df.apply(lambda r: _devig2(r["spread_home_odds"], r["spread_away_odds"]), axis=1)
    df["mkt_spread_home"] = sp.apply(lambda t: t[0])
    df["mkt_spread_away"] = sp.apply(lambda t: t[1])

    # Line movement: change in raw implied P(home) from open to close. Positive =
    # the home price shortened (money came in on home) — a sharp-money signal.
    def _move(open_am, close_am):
        po = decimal_to_prob(american_to_decimal(open_am))
        pc = decimal_to_prob(american_to_decimal(close_am))
        return (pc - po) if (po and pc) else np.nan
    df["home_line_move"] = df.apply(lambda r: _move(r["ml_home_open"], r["ml_home"]), axis=1)
    return df.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else datetime.utcnow().strftime("%Y-%m-%d")
    fx = fetch_fixtures(start, days=3, use_cache=False)
    if fx.empty:
        print("no fixtures found")
    else:
        cols = ["date", "home_team", "away_team", "provider",
                "mkt_home", "mkt_draw", "mkt_away", "overround"]
        with pd.option_context("display.width", 200):
            print(fx[cols].to_string(index=False))
