"""Official FIFA match-centre lineups — the authoritative, earliest source.

FIFA's public v3 API (``api.fifa.com/api/v3``, no auth) is the origin of the official team sheet, so
it carries the starting XI as early as or earlier than ESPN's third-party chain — plus a *projected*
XI hours before kickoff. We label projected vs confirmed via ``OfficialityStatus`` so a projection is
never shown as confirmed (the project's no-fabrication rule).

Live match: ``/live/football/{comp}/{season}/{stage}/{match}`` — ``HomeTeam``/``AwayTeam`` carry
``Players`` (``Status==1`` = the 11 starters, ``Status==2`` = bench), ``Tactics`` = formation,
``Captain``, ``ShirtNumber``, ``Position`` (0 GK / 1 DEF / 2 MID / 3 FWD). The fixture → match-id map
comes from ``/calendar/matches`` (each entry's ``Home``/``Away`` have ``TeamName`` + ``IdMatch`` +
``IdStage``). Disk-cached; honest None when a fixture/XI isn't posted.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

from .team_names import normalize_team

BASE = os.environ.get("FIFA_BASE") or "https://api.fifa.com/api/v3"
WC_COMPETITION = os.environ.get("FIFA_WC_COMPETITION") or "17"        # FIFA World Cup
WC_SEASON = os.environ.get("FIFA_WC_SEASON") or "285023"             # 2026
_UA = "Mozilla/5.0 (soccer-model; +https://github.com/mgwinnier/soccer-model)"
_HEADERS = {"User-Agent": _UA, "Accept": "application/json", "Referer": "https://www.fifa.com/"}
_POS = {0: "G", 1: "D", 2: "M", 3: "F"}

# FIFA's official names mostly match ours; a few need aliasing before normalize_team.
_FIFA_ALIASES = {"congo dr": "DR Congo", "korea republic": "South Korea", "ir iran": "Iran",
                 "türkiye": "Turkey", "turkiye": "Turkey", "côte d'ivoire": "Ivory Coast",
                 "chinese taipei": "Taiwan"}


def _norm(name) -> str | None:
    if not name:
        return None
    return normalize_team(_FIFA_ALIASES.get(str(name).strip().lower(), name))


def _en(x):
    """A FIFA localized field (list of {Locale, Description}) -> the English string."""
    if isinstance(x, list):
        for d in x:
            if str(d.get("Locale", "")).lower().startswith("en"):
                return d.get("Description")
        return x[0].get("Description") if x else None
    return x


def _cache_dir() -> Path:
    d = Path(__file__).resolve().parents[2] / "data" / "raw" / "fifa"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(path: str, ttl: float, **params):
    qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
    url = f"{BASE}{path}" + (f"?{qs}" if qs else "")
    cache = _cache_dir() / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if ttl > 0 and cache.exists() and (time.time() - cache.stat().st_mtime) < ttl:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:  # noqa: BLE001
        return None
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def calendar(ttl: float = 900.0) -> list[dict]:
    """All WC fixtures with team names + match/stage ids (cached 15 min; the schedule is stable)."""
    d = _get("/calendar/matches", ttl=ttl, idCompetition=WC_COMPETITION,
             idSeason=WC_SEASON, count=500, language="en")
    return (d or {}).get("Results") or []


def match_ref(home: str, away: str, cal: list[dict] | None = None) -> dict | None:
    """Resolve a fixture to its FIFA ids ``{comp, season, stage, match, home, away, date}`` by the
    unordered normalized team pair, or None."""
    nh, na = _norm(home), _norm(away)
    if not (nh and na):
        return None
    for m in (cal if cal is not None else calendar()):
        h = _norm(_en((m.get("Home") or {}).get("TeamName")))
        a = _norm(_en((m.get("Away") or {}).get("TeamName")))
        if {h, a} == {nh, na} and None not in (h, a):
            return {"comp": m.get("IdCompetition") or WC_COMPETITION,
                    "season": m.get("IdSeason") or WC_SEASON,
                    "stage": m.get("IdStage"), "match": m.get("IdMatch"),
                    "home": home, "away": away, "date": m.get("Date")}
    return None


def _parse_team(team: dict) -> dict | None:
    """A live HomeTeam/AwayTeam -> ``{team, formation, xi:[{name,pos,jersey,captain}]}`` (Status==1)."""
    xi = []
    for p in team.get("Players") or []:
        if p.get("Status") != 1:                 # 1 = starting XI, 2 = bench
            continue
        xi.append({"name": _en(p.get("PlayerName")) or p.get("ShortName"),
                   "pos": _POS.get(p.get("Position"), "?"),
                   "jersey": p.get("ShirtNumber"),
                   "captain": bool(p.get("Captain"))})
    if not xi:
        return None
    xi.sort(key=lambda x: (x["pos"] not in _POS.values(), list(_POS.values()).index(x["pos"])
                           if x["pos"] in _POS.values() else 9, x["jersey"] or 99))
    return {"team": _norm(_en(team.get("TeamName"))) or _en(team.get("TeamName")),
            "formation": team.get("Tactics"), "xi": xi}


def lineups(home: str, away: str, ref: dict | None = None, ttl: float = 60.0) -> dict | None:
    """FIFA starting XIs for a fixture, oriented to the caller's home/away. Returns
    ``{home, away, confirmed, officiality, source:'FIFA', kickoff}`` once both XIs are posted
    (projected OR confirmed — ``confirmed`` says which), else None.

    ``confirmed`` is True only when ``OfficialityStatus >= 1`` — until then it's FIFA's projected XI,
    surfaced early but clearly labeled, never passed off as the official sheet."""
    ref = ref or match_ref(home, away)
    if not ref or not ref.get("match"):
        return None
    live = _get(f"/live/football/{ref['comp']}/{ref['season']}/{ref['stage']}/{ref['match']}", ttl=ttl)
    if not live:
        return None
    nh = _norm(home)
    th, ta = live.get("HomeTeam") or {}, live.get("AwayTeam") or {}
    # orient by team name (FIFA's Home/Away may not match our home/away)
    if _norm(_en(th.get("TeamName"))) == nh:
        home_t, away_t = th, ta
    elif _norm(_en(ta.get("TeamName"))) == nh:
        home_t, away_t = ta, th
    else:
        home_t, away_t = th, ta
    ph, pa = _parse_team(home_t), _parse_team(away_t)
    if not (ph and pa):
        return None
    off = live.get("OfficialityStatus")
    return {"home": ph, "away": pa, "confirmed": bool(off and off >= 1),
            "officiality": off, "source": "FIFA", "kickoff": live.get("Date")}
