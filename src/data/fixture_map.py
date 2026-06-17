"""Map an ESPN fixture (home, away, date) onto a TheStatsAPI ``match_id``.

Every TheStatsAPI feature we want for a *specific* game — live xG, the real opening/closing
odds (incl. BTTS), the confirmed XI — is keyed by TheStatsAPI's own ``mt_`` match id, not by
team name. Our fixtures come from ESPN (and history from ``matches.parquet``) keyed by
normalized team names + a date. This module bridges the two.

Design constraints (deliberate, per the project's validate-first rule):
* **Shape-tolerant.** The exact field names on a ``/matches`` list item were not captured to a
  fixture yet (the local sandbox is firewall-blocked from the API). ``_team_names`` therefore
  tries the documented/likely shapes (``home``/``away`` as strings or ``{name}`` dicts,
  ``home_team``/``away_team``, ``teams.home/away``, ``participants[{is_home,name}]``) rather than
  hard-coding one. ``scripts/capture_thestatsapi_shapes.py`` locks the real shape from the VPS.
* **No fabrication.** If no match in the window has *both* normalized team names equal to the
  fixture's, we return ``None`` — never a best-guess id. A wrong id would silently attach the
  wrong game's xG/odds, which is exactly the failure mode to avoid.
* Names are compared through ``team_names.normalize_team`` so "USA"/"United States",
  "South Korea"/"Korea Republic", etc. line up across the two sources.
"""
from __future__ import annotations

from datetime import date as _date, datetime, timedelta

from .team_names import normalize_team

# Candidate id keys on a match item, in priority order (``mt_...``).
_ID_KEYS = ("match_id", "id", "uuid")


def _as_name(v) -> str | None:
    """A team field may be a bare string or a ``{"name"/"title"/...}`` dict."""
    if isinstance(v, str):
        return v or None
    if isinstance(v, dict):
        for k in ("name", "title", "team", "short_name", "common_name"):
            if isinstance(v.get(k), str) and v[k]:
                return v[k]
    return None


def _team_names(match: dict) -> tuple[str | None, str | None]:
    """(home, away) raw names from a match item, across the shapes the API might use."""
    if not isinstance(match, dict):
        return (None, None)
    # 1) flat home/away (string or dict)
    for hk, ak in (("home", "away"), ("home_team", "away_team"),
                   ("home_team_name", "away_team_name"), ("hometeam", "awayteam")):
        if hk in match or ak in match:
            return (_as_name(match.get(hk)), _as_name(match.get(ak)))
    # 2) nested teams object
    teams = match.get("teams")
    if isinstance(teams, dict):
        return (_as_name(teams.get("home")), _as_name(teams.get("away")))
    # 3) participants list with a home/away flag
    parts = match.get("participants") or match.get("competitors")
    if isinstance(parts, list):
        home = away = None
        for p in parts:
            if not isinstance(p, dict):
                continue
            side = str(p.get("side") or p.get("home_away") or "").lower()
            is_home = p.get("is_home")
            nm = _as_name(p.get("team") if isinstance(p.get("team"), (str, dict)) else p) \
                or _as_name(p)
            if is_home is True or side in ("home", "h"):
                home = nm
            elif is_home is False or side in ("away", "a"):
                away = nm
        if home or away:
            return (home, away)
    return (None, None)


def match_id_of(match: dict) -> str | None:
    for k in _ID_KEYS:
        v = match.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _date_of(match: dict) -> str | None:
    """Best-effort YYYY-MM-DD from whatever date field the item carries."""
    for k in ("utc_date", "date", "match_date", "kickoff", "kickoff_at", "start_time",
              "starts_at", "datetime", "scheduled_at"):
        v = match.get(k)
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]
    return None


def _to_ymd(d) -> str | None:
    if isinstance(d, str):
        return d[:10] if len(d) >= 10 else None
    if isinstance(d, (datetime, _date)):
        return d.strftime("%Y-%m-%d")
    return None


def _pair(home, away) -> frozenset:
    return frozenset({normalize_team(home), normalize_team(away)})


def find_match(home: str, away: str, date, candidates: list[dict],
               day_tol: int = 1) -> dict | None:
    """Return the candidate match whose normalized team **pair** equals (home, away) and whose
    date is within ``day_tol`` days, or ``None``. Order-independent (host/home ambiguity across
    sources is common for neutral-site WC games), so we match on the unordered pair and prefer
    the closest date. Never guesses — a missing/garbled name pair yields ``None``."""
    want = _pair(home, away)
    if None in want or len(want) < 2:      # unknown/missing name -> refuse to guess
        return None
    target = _to_ymd(date)
    best, best_gap = None, None
    for m in candidates:
        h, a = _team_names(m)
        if _pair(h, a) != want:
            continue
        md = _date_of(m)
        if target and md:
            try:
                gap = abs((datetime.strptime(md, "%Y-%m-%d")
                           - datetime.strptime(target, "%Y-%m-%d")).days)
            except ValueError:
                gap = day_tol + 1
            if gap > day_tol:
                continue
        else:
            gap = 0
        if best is None or gap < best_gap:
            best, best_gap = m, gap
    return best


def find_match_id(home: str, away: str, date, candidates: list[dict],
                  day_tol: int = 1) -> str | None:
    m = find_match(home, away, date, candidates, day_tol=day_tol)
    return match_id_of(m) if m else None
