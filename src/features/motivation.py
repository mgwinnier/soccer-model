"""World-Cup / Euro game-state ("motivation") features for final group matches.

A generic Elo/DC model is blind to *why* a team plays its last group match the way
it does — a side already through may rotate, a side that only needs a draw may sit
deep, a dead rubber means nothing to either team. The market is also thinnest there.
These features make the incentive structure explicit for the **final group round**,
the only matchday where standings create such incentives.

Leak-free by construction: a final-round match's standings are reconstructed only
from the *earlier* group matches (matchdays 1–2). The two final-round matches in a
group are simultaneous, so the other final result is unknown — we enumerate all
joint outcomes to decide what each team needs, rather than peeking at it.

Scope: top-2-advance group stages only (see ``data/group_structures.py``). Every
other match — group matchdays 1–2, knockouts, qualifiers, friendlies — gets NaN,
so the model only keys off these signals in the context where they're defined.

The enumeration uses points plus a nominal ±1 goal-difference swing per result; it
captures the qualification structure, not exact goal-line tiebreaks (an honest
approximation for a feature, noted here rather than overstated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.group_structures import COVERED, team_to_group

FEATURE_COLS = [
    "is_final_group_match",
    "home_needs_win", "away_needs_win",
    "home_draw_enough", "away_draw_enough",
    "home_already_q", "away_already_q",
    "home_eliminated", "away_eliminated",
    "home_gd_incentive", "away_gd_incentive",
    "dead_rubber",
]


def _standings(teams: list[str], results: list[tuple]) -> dict:
    tbl = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
    for h, a, hg, ag in results:
        tbl[h]["gf"] += hg
        tbl[a]["gf"] += ag
        tbl[h]["gd"] += hg - ag
        tbl[a]["gd"] += ag - hg
        if hg > ag:
            tbl[h]["pts"] += 3
        elif hg < ag:
            tbl[a]["pts"] += 3
        else:
            tbl[h]["pts"] += 1
            tbl[a]["pts"] += 1
    return tbl


def _rank_key(s: dict) -> tuple:
    return (s["pts"], s["gd"], s["gf"])


def _top2_after(pre: dict, fixtures: list[tuple], r1: str, r2: str) -> set:
    """Top-2 team set after applying result codes r1,r2 to the two final fixtures
    (nominal ±1 GD swing for a decisive result)."""
    s = {t: dict(v) for t, v in pre.items()}
    for (h, a), r in zip(fixtures, (r1, r2)):
        if r == "H":
            s[h]["pts"] += 3; s[h]["gd"] += 1; s[h]["gf"] += 1; s[a]["gd"] -= 1
        elif r == "A":
            s[a]["pts"] += 3; s[a]["gd"] += 1; s[a]["gf"] += 1; s[h]["gd"] -= 1
        else:
            s[h]["pts"] += 1; s[a]["pts"] += 1
    order = sorted(s, key=lambda t: _rank_key(s[t]), reverse=True)
    return set(order[:2])


def _enumerate_needs(pre: dict, fixtures: list[tuple]) -> dict:
    """Per-team qualification needs from enumerating all 3x3 joint final results."""
    outcomes = ("H", "D", "A")
    top2 = {(r1, r2): _top2_after(pre, fixtures, r1, r2)
            for r1 in outcomes for r2 in outcomes}
    res: dict[str, dict] = {}
    for fi, (h, a) in enumerate(fixtures):
        for team, is_home in ((h, True), (a, False)):
            win_r = "H" if is_home else "A"
            lose_r = "A" if is_home else "H"

            def frac(rr: str) -> float:
                vals = [team in top2[(rr, ro) if fi == 0 else (ro, rr)]
                        for ro in outcomes]
                return sum(vals) / len(vals)

            p_win, p_draw, p_lose = frac(win_r), frac("D"), frac(lose_r)
            already = p_win == 1 and p_draw == 1 and p_lose == 1
            elim = p_win == 0 and p_draw == 0 and p_lose == 0
            res[team] = {
                "already_q": already,
                "eliminated": elim,
                "draw_enough": (not already) and p_draw == 1,
                "needs_win": (not already) and (not elim) and p_draw < 1,
                "gd_incentive": (not already) and (not elim) and 0 < p_win < 1,
            }
    return res


def _group_final_round(gm: pd.DataFrame):
    """Given one group's matches (sorted), return (final_matches, pre_results).

    final_matches: list of rows where both teams are playing their 3rd group game.
    pre_results: [(home, away, hg, ag)] from the earlier (non-final) matches.
    Returns (None, None) if the group's data is incomplete."""
    prior: dict[str, int] = {}
    final_rows, pre_rows = [], []
    for r in gm.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if prior.get(h, 0) >= 2 and prior.get(a, 0) >= 2:
            final_rows.append(r)
        else:
            pre_rows.append(r)
        prior[h] = prior.get(h, 0) + 1
        prior[a] = prior.get(a, 0) + 1
    if len(final_rows) != 2 or len(pre_rows) < 4:
        return None, None
    pre_results = [(r.home_team, r.away_team, int(r.home_score), int(r.away_score))
                   for r in pre_rows]
    return final_rows, pre_results


def compute_motivation_features(matches: pd.DataFrame,
                                cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    out = pd.DataFrame(index=matches.set_index("match_id").index,
                       columns=FEATURE_COLS, dtype="float64")

    for (tournament, year), groups in COVERED.items():
        t2g = team_to_group(tournament, year)
        sub = matches[(matches["tournament"] == tournament)
                      & (matches["date"].dt.year == year)
                      & matches["home_team"].isin(t2g)
                      & matches["away_team"].isin(t2g)]
        if sub.empty:
            continue
        # keep only intra-group matches (both teams in the same group)
        same = sub["home_team"].map(t2g) == sub["away_team"].map(t2g)
        sub = sub[same]
        for g, teams in groups.items():
            gm = sub[sub["home_team"].map(t2g) == g].sort_values(["date", "match_id"])
            if gm.empty:
                continue
            final_rows, pre_results = _group_final_round(gm)
            if final_rows is None:
                continue
            present = {t for r in pre_results for t in (r[0], r[1])}
            if not set(teams) <= present:
                continue  # a team never appears in the earlier results — skip group
            pre = _standings(teams, pre_results)
            fixtures = [(r.home_team, r.away_team) for r in final_rows]
            needs = _enumerate_needs(pre, fixtures)
            for r in final_rows:
                h, a = r.home_team, r.away_team
                nh, na = needs[h], needs[a]
                dead = ((nh["already_q"] or nh["eliminated"])
                        and (na["already_q"] or na["eliminated"]))
                out.loc[r.match_id, FEATURE_COLS] = [
                    1.0,
                    float(nh["needs_win"]), float(na["needs_win"]),
                    float(nh["draw_enough"]), float(na["draw_enough"]),
                    float(nh["already_q"]), float(na["already_q"]),
                    float(nh["eliminated"]), float(na["eliminated"]),
                    float(nh["gd_incentive"]), float(na["gd_incentive"]),
                    float(dead),
                ]
    return out
