"""Confirmed-XI status + player ratings + a 'regular starter missing today' flag (TheStatsAPI).

Why this definition of "major player out": there is no usable *current* per-player rating for
national teams from a squad list (the squad endpoint is club-only; the local FIFA CSVs are
2017-2023 and name a mostly-retired XI). The honest, current, self-contained signals come from
the team's own recent matches, matched by player **id** (no fragile name match):

  * **ratings** — each player's match rating (e.g. 6.82) from `/matches/{id}/player-stats`, so we
    can list today's XI with their recent-form ratings (today's game isn't played yet at lineup
    time, so the ratings shown are from previous matches);
  * **missing regular starter** — a player who *started* the team's most recent prior World Cup
    match (``started`` in that match's player-stats) but isn't in today's confirmed XI.

Timing: the confirmed XI posts ~75 min before kickoff (and for finished games). Until then
``lineup_status`` reports ``posted=False`` — never a guessed XI.
"""
from __future__ import annotations

from . import thestatsapi as ts
from . import fixture_map

_K = 3  # how many recent matches to average ratings over


def _starters(side: dict | None) -> list[dict]:
    return [p for p in ((side or {}).get("starting_xi") or []) if p.get("id")]


def _finished_for(team_id: str, before_utc: str, cands: list[dict]) -> list[dict]:
    out = []
    for c in cands:
        if "fin" not in str(c.get("status") or "").lower():
            continue
        ids = {(c.get("home_team") or {}).get("id"), (c.get("away_team") or {}).get("id")}
        if team_id not in ids:
            continue
        d = str(c.get("utc_date") or "")
        if d and before_utc and d >= before_utc:
            continue
        out.append(c)
    out.sort(key=lambda c: str(c.get("utc_date") or ""), reverse=True)
    return out


def _team_recent(team_id: str, before_utc: str, cands: list[dict], cfg=None):
    """From the team's last ``_K`` prior matches: per-player rating list (recent first), names,
    and the most-recent match's starters (the prior XI). One player-stats call per match."""
    fin = _finished_for(team_id, before_utc, cands)[:_K]
    hist: dict[str, list] = {}
    names: dict[str, str] = {}
    prior_xi: dict[str, str] = {}
    used = []
    for i, c in enumerate(fin):
        pr = ts.match_player_ratings(fixture_map.match_id_of(c), cfg=cfg)
        h, a = fixture_map._team_names(c)
        opp = a if (c.get("home_team") or {}).get("id") == team_id else h
        used.append({"date": str(c.get("utc_date") or "")[:10], "opp": opp})
        for pid, info in pr.items():
            if info.get("team_id") != team_id:
                continue
            names[pid] = info.get("name")
            if info.get("rating") is not None:
                hist.setdefault(pid, []).append(info["rating"])
            if i == 0 and info.get("started"):
                prior_xi[pid] = info.get("name")
    return hist, names, prior_xi, used


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def _side_status(team_id: str, side_lineup: dict, before_utc: str, cands: list[dict],
                 cfg=None, today_ratings: dict | None = None) -> dict:
    xi = _starters(side_lineup)
    today_ids = {p["id"] for p in xi}
    hist, names, prior_xi, used = _team_recent(team_id, before_utc, cands, cfg=cfg)
    xi_rows = []
    for p in xi:
        rl = hist.get(p["id"], [])
        today = (today_ratings or {}).get(p["id"], {}).get("rating")
        xi_rows.append({"name": p.get("name"), "position": p.get("position"),
                        "recent": rl, "avg": _avg(rl), "last": (rl[0] if rl else None),
                        "today": today})
    missing = [{"name": nm, "avg": _avg(hist.get(pid, []))}
               for pid, nm in prior_xi.items() if pid not in today_ids and nm]
    return {"formation": (side_lineup or {}).get("formation"), "xi": xi_rows,
            "missing_starters": missing, "had_prior_xi": bool(prior_xi),
            "matches_used": used}


def lineup_status(home: str, away: str, date, cfg=None) -> dict | None:
    """Confirmed-XI status for a fixture, or None if unresolved.

    ``{"posted": False, "status": ...}`` until the team sheet posts (~75 min pre-KO). Once posted,
    per-side ``formation`` + ``xi`` (each starter with recent-match ``avg``/``last``/``recent``
    ratings, and ``today`` rating when the game is played) + ``missing_starters`` (regular starters
    from the previous match not in today's XI, with their recent avg). Best-effort -> None/posted
    False on any miss. Never fabricates a lineup or a rating."""
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
        played = "fin" in str(m.get("status") or "").lower()
        today_r = ts.match_player_ratings(mid, cfg=cfg) if played else None
        h_id = (m.get("home_team") or {}).get("id")
        a_id = (m.get("away_team") or {}).get("id")
        return {"posted": True, "status": m.get("status"), "played": played,
                "home": _side_status(h_id, lu.get("home"), before, cands, cfg, today_r),
                "away": _side_status(a_id, lu.get("away"), before, cands, cfg, today_r)}
    except Exception:  # noqa: BLE001 — never break a card over a lineup lookup
        return None
