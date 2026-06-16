"""Confirmed lineups + injuries via API-Football (optional, key-gated).

This is deliberately **dormant unless** an ``API_FOOTBALL_KEY`` environment
variable is set (free tier: api-football.com). It never fabricates: with no key
it returns empty structures and callers fall back to squad-value strength. When a
key is present it provides, for an imminent fixture:

  * the **confirmed starting XI** (available ~30-40 min before kickoff), and
  * an **availability index** — the share of a squad's key players who are
    injured/suspended — usable as a strength penalty.

Honest scope (told to the user in the README): predicted XIs are not reliably
free, so lineup strength only sharpens *imminent* matches; forecasts of future /
knockout matches use squad value + availability, not a hypothetical XI.
"""
from __future__ import annotations

import os
from functools import lru_cache

import requests

from .team_names import normalize_team

_BASE = "https://v3.football.api-sports.io"


def api_key() -> str | None:
    from ..config import load_secrets
    load_secrets()  # ensure .env is loaded
    return os.environ.get("API_FOOTBALL_KEY")


def is_available() -> bool:
    return bool(api_key())


def _get(path: str, params: dict) -> dict | None:
    key = api_key()
    if not key:
        return None
    try:
        r = requests.get(f"{_BASE}/{path}", params=params,
                         headers={"x-apisports-key": key}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=256)
def injuries(team_api_id: int, season: int) -> tuple:
    """Return a tuple of injured player names for a team/season (cached)."""
    data = _get("injuries", {"team": team_api_id, "season": season})
    if not data:
        return tuple()
    names = []
    for item in data.get("response", []):
        p = item.get("player", {})
        if p.get("name"):
            names.append(p["name"])
    return tuple(sorted(set(names)))


@lru_cache(maxsize=256)
def _team_id(team_name: str) -> int | None:
    """Resolve a national team's API-Football id from its name (cached)."""
    data = _get("teams", {"name": team_name})
    if not data:
        return None
    resp = data.get("response", [])
    if not resp:
        return None
    return resp[0].get("team", {}).get("id")


def injured_players(team_name: str, season: int = 2026) -> list[str]:
    """Injured player names for a team *by name* (resolves id internally).

    Empty list when no key, no match, or no injuries — callers then apply a
    neutral (1.0) availability multiplier.
    """
    if not is_available():
        return []
    tid = _team_id(normalize_team(team_name) or team_name)
    if tid is None:
        return []
    return list(injuries(tid, season))


def connectivity_check() -> str:
    """Diagnose whether the API is actually reachable (key set != reachable).

    Returns one of: 'no_key', 'ok', 'blocked_by_network', 'api_error: ...',
    'non_json_response', 'network_error: ...'.
    """
    if not is_available():
        return "no_key"
    try:
        r = requests.get(f"{_BASE}/timezone",
                         headers={"x-apisports-key": api_key()}, timeout=15)
    except Exception as exc:  # noqa: BLE001
        return f"network_error: {type(exc).__name__}"
    ctype = r.headers.get("content-type", "")
    if "application/json" in ctype:
        j = r.json()
        errs = j.get("errors")
        if errs:
            return f"api_error: {errs}"
        return "ok"
    low = r.text.lower()
    if "ubiquiti" in low or "blocked" in low or "firewall" in low:
        return "blocked_by_network"
    return "non_json_response"


def fixture_lineups(fixture_api_id: int) -> dict[str, list[str]]:
    """Confirmed starting XIs for a fixture: {team_name: [player, ...]}."""
    data = _get("fixtures/lineups", {"fixture": fixture_api_id})
    if not data:
        return {}
    out: dict[str, list[str]] = {}
    for side in data.get("response", []):
        team = normalize_team(side.get("team", {}).get("name"))
        xi = [p.get("player", {}).get("name")
              for p in side.get("startXI", []) if p.get("player")]
        if team:
            out[team] = [n for n in xi if n]
    return out


def availability_penalty(injured: list[str], key_players: dict[str, float]) -> float:
    """Fraction of squad strength missing = sum(rating of injured key players) /
    sum(all key-player ratings). 0.0 when fully available / no data."""
    if not key_players:
        return 0.0
    injured_set = {n.lower() for n in injured}
    total = sum(key_players.values())
    if total <= 0:
        return 0.0
    out = sum(r for n, r in key_players.items() if n.lower() in injured_set)
    return out / total
