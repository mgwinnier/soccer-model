"""Reader for the committed WC squad market-value snapshot (``data/feeds/wc_squad_values.json``).

Market values barely change, so a static snapshot lets us attach player values to **ESPN lineups**
with zero live API calls. Matching is accent/case-insensitive on the player name (ESPN and
TheStatsAPI spell them slightly differently), with a last-name fallback.
"""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

from ..config import PROJECT_ROOT
from .team_names import normalize_team

FEED = PROJECT_ROOT / "data" / "feeds" / "wc_squad_values.json"


def _norm(s: str | None) -> str:
    if not s:
        return ""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(s.lower().split())


@lru_cache(maxsize=1)
def _load(path: str | None = None) -> dict:
    p = Path(path) if path else FEED
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("teams", {})
    except Exception:  # noqa: BLE001
        return {}


def _team(team: str) -> dict | None:
    teams = _load()
    key = normalize_team(team) or team
    if key in teams:
        return teams[key]
    nk = _norm(key)
    return next((v for k, v in teams.items() if _norm(k) == nk), None)


def _index(t: dict) -> dict:
    """{normalized full name: player} + {normalized last name: player} (last-name only if unique)."""
    full, last_counts, last = {}, {}, {}
    for p in t.get("players", []):
        n = _norm(p.get("name"))
        if n:
            full[n] = p
            ln = n.split()[-1]
            last_counts[ln] = last_counts.get(ln, 0) + 1
            last[ln] = p
    return {"full": full, "last": {ln: p for ln, p in last.items() if last_counts[ln] == 1}}


def player_value(team: str, name: str) -> float | None:
    t = _team(team)
    if not t:
        return None
    idx = _index(t)
    n = _norm(name)
    p = idx["full"].get(n) or (idx["last"].get(n.split()[-1]) if n else None)
    return (p or {}).get("market_value")


def total_value(team: str) -> float | None:
    t = _team(team)
    return t.get("total_value") if t else None


def key_absentees(team: str, xi_names: list[str], top_n: int = 6) -> list[dict]:
    """Squad's top-``top_n`` players by market value who are NOT in the confirmed XI
    (the value-based 'key player out' flag — works from match 1). [{name, market_value}]."""
    t = _team(team)
    if not t:
        return []
    valued = sorted((p for p in t.get("players", []) if p.get("market_value")),
                    key=lambda p: p["market_value"], reverse=True)[:top_n]
    in_xi = {_norm(n) for n in xi_names}
    in_xi_last = {n.split()[-1] for n in in_xi if n}
    out = []
    for p in valued:
        n = _norm(p.get("name"))
        if n in in_xi or (n and n.split()[-1] in in_xi_last):
            continue
        out.append({"name": p["name"], "market_value": p["market_value"]})
    return out


def available() -> bool:
    return bool(_load())
