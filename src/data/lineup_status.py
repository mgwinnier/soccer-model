"""Confirmed-XI status + a 'regular starter missing today' flag, from TheStatsAPI.

Why this definition of "major player out": there is no usable *current* per-player rating for
national teams (the squad endpoint is club-only; the local FIFA CSVs are 2017-2023 and name a
mostly-retired XI — using them would fabricate flags). The honest, current, self-contained signal
is **comparison to the team's previous match**: a player who *started* the team's last World Cup
match but is **not** in today's confirmed XI — captured by player **id** (no fragile name match).
That surfaces real injuries/suspensions/big rotations without inventing anything.

Timing: TheStatsAPI posts the confirmed XI ~75 min before kickoff (and for finished games). Until
then ``lineup_status`` reports ``posted=False`` — never a guessed XI.
"""
from __future__ import annotations

from . import thestatsapi as ts
from . import fixture_map


def _starters(lineup_side: dict | None) -> list[dict]:
    return [(p) for p in ((lineup_side or {}).get("starting_xi") or []) if p.get("id")]


def _recent_starter_ids(team_id: str, before_utc: str, cands: list[dict],
                        cfg=None) -> dict[str, str]:
    """{player_id: name} who started the team's most recent FINISHED match before ``before_utc``.
    Empty when the team has no prior match (e.g. matchday 1) or its XI isn't retrievable."""
    prior = []
    for c in cands:
        st = str(c.get("status") or "").lower()
        if "fin" not in st:
            continue
        tids = {(c.get("home_team") or {}).get("id"), (c.get("away_team") or {}).get("id")}
        if team_id not in tids:
            continue
        d = str(c.get("utc_date") or "")
        if d and before_utc and d >= before_utc:
            continue
        prior.append((d, c))
    if not prior:
        return {}
    prior.sort(key=lambda x: x[0], reverse=True)
    last = prior[0][1]
    lu = ts.match_lineups(fixture_map.match_id_of(last), cfg=cfg)
    if not lu:
        return {}
    side = "home" if (last.get("home_team") or {}).get("id") == team_id else "away"
    return {p["id"]: p.get("name") for p in _starters(lu.get(side))}


def _side_status(team_id: str, side_lineup: dict, before_utc: str,
                 cands: list[dict], cfg=None) -> dict:
    xi = _starters(side_lineup)
    today_ids = {p["id"] for p in xi}
    prior = _recent_starter_ids(team_id, before_utc, cands, cfg=cfg)
    missing = [name for pid, name in prior.items() if pid not in today_ids and name]
    return {"formation": (side_lineup or {}).get("formation"),
            "xi": [p.get("name") for p in xi], "xi_count": len(xi),
            "missing_starters": missing, "had_prior_xi": bool(prior)}


def lineup_status(home: str, away: str, date, cfg=None) -> dict | None:
    """Confirmed-XI status for a fixture, or None if it can't be resolved.

    ``{"posted": False, "status": ...}`` until the team sheet is published (~75 min pre-KO);
    once posted, per-side ``formation``/``xi``/``missing_starters`` (regular starters from the
    team's previous match not in today's XI). Best-effort — any miss degrades to None/posted=False.
    """
    if not ts.is_available():
        return None
    try:
        sid = ts.current_season_id(cfg=cfg)
        cands = ts.matches(competition_id=ts.WC_COMP, season_id=sid, cfg=cfg)
        m = fixture_map.find_match(home, away, fixture_map._to_ymd(date), cands)
        if not m:
            return None
        mid = fixture_map.match_id_of(m)
        lu = ts.match_lineups(mid, cfg=cfg) if mid else None
        if not lu or not lu.get("home"):
            return {"posted": False, "status": m.get("status")}
        before = str(m.get("utc_date") or "")
        h_id = (m.get("home_team") or {}).get("id")
        a_id = (m.get("away_team") or {}).get("id")
        return {"posted": True, "status": m.get("status"),
                "home": _side_status(h_id, lu.get("home"), before, cands, cfg=cfg),
                "away": _side_status(a_id, lu.get("away"), before, cands, cfg=cfg)}
    except Exception:  # noqa: BLE001 — never break a card over a lineup lookup
        return None
