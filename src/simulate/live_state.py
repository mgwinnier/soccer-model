"""Live 2026 group state: standings, qualification odds, and per-team stakes.

This is the **format-correct** companion to the (historical, top-2-only)
``features/motivation.py``. The 2026 World Cup advances the top 2 of each of the
12 groups **plus the 8 best third-placed teams**, so a side sitting 3rd is usually
still alive and an "already-through" side still chases goal difference. We never
hard-code "needs"; instead we:

  • build current **standings** from played results (dataset ∪ live ESPN scores),
  • mark **top-2 clinch / still-alive** by enumerating the group's remaining games
    (top-2 is always safe regardless of best-thirds, so that flag is exact), and
  • read each team's **P(reach knockouts)** straight from the Monte-Carlo simulator
    with the played results locked in — which natively handles the best-third math.

P(advance) is the honest, format-correct "stakes" signal for a match card: a team
at 95% has little to play for; a team at 40% is fighting for its life.
"""
from __future__ import annotations

from itertools import combinations, product

import pandas as pd

from ..config import load_config, path_for
from ..data.team_names import normalize_team
from .bracket_2026 import GROUPS, load_played_results, team_to_group


def fetch_live_results(cfg: dict | None = None, use_cache: bool = True) -> dict:
    """Played 2026 WC results: the cleaned dataset ∪ finished ESPN games.

    ESPN is best-effort — if it's unreachable we just use the dataset, so the
    dashboard never hard-fails on a network blip."""
    cfg = cfg or load_config()
    played = dict(load_played_results(cfg))
    try:
        from datetime import date
        from ..data.odds import fetch_espn_range
        t2g = team_to_group()
        evs = fetch_espn_range("2026-06-01", date.today().strftime("%Y-%m-%d"),
                               league="fifa.world", cfg=cfg, use_cache=use_cache)
        for e in evs:
            if e.get("status") != "post" or e.get("home_score") is None:
                continue
            h, a = e["home_team"], e["away_team"]
            if t2g.get(h) and t2g.get(h) == t2g.get(a):     # same-group games only
                played[frozenset((h, a))] = (h, int(e["home_score"]), int(e["away_score"]))
    except Exception:  # noqa: BLE001 — dataset-only is an acceptable fallback
        pass
    return played


def _elo_ratings(cfg: dict) -> dict:
    import json
    p = path_for("data_processed", cfg) / "elo_ratings.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


def standings(cfg: dict | None = None, played: dict | None = None) -> dict[str, pd.DataFrame]:
    """Per-group standings DataFrame, sorted by the simulator's tiebreaker
    (points → GD → GF → Elo). Columns: Pos, team, P, W, D, L, GF, GA, GD, Pts."""
    cfg = cfg or load_config()
    played = fetch_live_results(cfg) if played is None else played
    t2g = team_to_group()
    elo = _elo_ratings(cfg)
    out: dict[str, pd.DataFrame] = {}
    for g, teams in GROUPS.items():
        tn = [normalize_team(t) for t in teams]
        rec = {t: dict(P=0, W=0, D=0, L=0, GF=0, GA=0) for t in tn}
        for key, (hteam, hg, ag) in played.items():
            if t2g.get(hteam) != g:
                continue
            pair = tuple(key)
            ateam = pair[1] if pair[0] == hteam else pair[0]
            if hteam not in rec or ateam not in rec:
                continue
            rec[hteam]["P"] += 1; rec[ateam]["P"] += 1
            rec[hteam]["GF"] += hg; rec[hteam]["GA"] += ag
            rec[ateam]["GF"] += ag; rec[ateam]["GA"] += hg
            if hg > ag:
                rec[hteam]["W"] += 1; rec[ateam]["L"] += 1
            elif hg < ag:
                rec[hteam]["L"] += 1; rec[ateam]["W"] += 1
            else:
                rec[hteam]["D"] += 1; rec[ateam]["D"] += 1
        rows = []
        for t in tn:
            r = rec[t]
            rows.append({"team": t, "P": r["P"], "W": r["W"], "D": r["D"], "L": r["L"],
                         "GF": r["GF"], "GA": r["GA"], "GD": r["GF"] - r["GA"],
                         "Pts": r["W"] * 3 + r["D"], "_elo": elo.get(t, 1500.0)})
        df = (pd.DataFrame(rows)
              .sort_values(["Pts", "GD", "GF", "_elo"], ascending=False)
              .drop(columns="_elo").reset_index(drop=True))
        df.insert(0, "Pos", range(1, len(df) + 1))
        out[g] = df
    return out


def clinch_flags(cfg: dict | None = None, played: dict | None = None) -> dict[str, dict]:
    """Per-team top-2 status by enumerating the group's remaining games.

    ``clinched_top2`` (advances no matter what) and ``alive_top2`` (top-2 still
    possible) are **exact** — top-2 always advances in any format. Best-third
    qualification is cross-group and left to the simulator's P(advance)."""
    cfg = cfg or load_config()
    played = fetch_live_results(cfg) if played is None else played
    st_ = standings(cfg, played)
    flags: dict[str, dict] = {}
    for g, df in st_.items():
        teams = list(df["team"])
        base = {r["team"]: {"pts": r["Pts"], "gd": r["GD"], "gf": r["GF"]}
                for _, r in df.iterrows()}
        rem = [pair for pair in combinations(teams, 2)
               if frozenset(pair) not in played]
        top2 = {t: 0 for t in teams}
        total = 0
        for outcomes in product("HDA", repeat=len(rem)):
            s = {t: dict(v) for t, v in base.items()}
            for (h, a), o in zip(rem, outcomes):
                if o == "H":
                    s[h]["pts"] += 3; s[h]["gd"] += 1; s[h]["gf"] += 1; s[a]["gd"] -= 1
                elif o == "A":
                    s[a]["pts"] += 3; s[a]["gd"] += 1; s[a]["gf"] += 1; s[h]["gd"] -= 1
                else:
                    s[h]["pts"] += 1; s[a]["pts"] += 1
            order = sorted(teams, key=lambda t: (s[t]["pts"], s[t]["gd"], s[t]["gf"]),
                           reverse=True)
            for t in order[:2]:
                top2[t] += 1
            total += 1
        for t in teams:
            c = top2[t]
            flags[t] = {"clinched_top2": total > 0 and c == total,
                        "alive_top2": c > 0}
    return flags


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def team_summary(state: dict, team: str) -> dict | None:
    """Compact per-team state for a match card, or None if not a 2026 group team."""
    g = team_to_group().get(team)
    if not g or g not in state["standings"]:
        return None
    df = state["standings"][g]
    hit = df[df["team"] == team]
    if hit.empty:
        return None
    row = hit.iloc[0]
    qrow = state["qual"][state["qual"]["team"] == team]
    p_adv = float(qrow["advance"].iloc[0]) if not qrow.empty else float("nan")
    fl = state["clinch"].get(team, {})
    played, pos = int(row["P"]), int(row["Pos"])
    if fl.get("clinched_top2"):
        status = "✓ qualified (top 2)"
    elif played >= 3 and pos <= 2:
        status = "✓ advanced"
    elif played >= 3 and pos == 3:
        status = "3rd — best-third cutoff pending"
    elif played >= 3 and pos == 4:
        status = "eliminated"
    elif not fl.get("alive_top2", True):
        status = "top-2 out — best-third hopes only"
    else:
        status = "alive — fighting to advance"
    return {"group": g, "pos": pos, "pos_str": _ordinal(pos), "pts": int(row["Pts"]),
            "played": played, "gd": int(row["GD"]), "p_advance": p_adv, "status": status}


def live_match_frame(cfg: dict | None = None) -> pd.DataFrame:
    """The freshest match table for strength estimation: ``matches.parquet`` plus
    any finished 2026 ESPN games not yet ingested into the dataset. Best-effort —
    if the ESPN delta can't be built it returns the parquet alone (which already
    fixes the Elo-vs-JSON staleness). New rows carry only the core columns the Elo
    engine / Dixon-Coles need; everything else is NaN."""
    from ..data.clean import load_matches
    cfg = cfg or load_config()
    base = load_matches(cfg)
    try:
        from .bracket_2026 import HOST_TEAMS
        t2g = team_to_group()
        played = fetch_live_results(cfg)
        wc = base[(base["tournament"] == "FIFA World Cup")
                  & (base["date"] >= pd.Timestamp("2026-06-01"))]
        have = {frozenset((r.home_team, r.away_team)) for r in wc.itertuples(index=False)}
        wc_imp = float(wc["importance"].iloc[0]) if len(wc) else 1.0
        delta_date = base["date"].max() + pd.Timedelta(days=1)
        rows, nid = [], int(base["match_id"].max()) + 1
        for key, (hteam, hg, ag) in played.items():
            if key in have:
                continue
            pair = tuple(key)
            ateam = pair[1] if pair[0] == hteam else pair[0]
            if not (t2g.get(hteam) and t2g.get(hteam) == t2g.get(ateam)):
                continue
            rows.append({"match_id": nid, "date": delta_date, "home_team": hteam,
                         "away_team": ateam, "home_score": int(hg), "away_score": int(ag),
                         "tournament": "FIFA World Cup", "importance": wc_imp,
                         "neutral": hteam not in HOST_TEAMS})
            nid += 1
        if rows:
            add = pd.DataFrame(rows).reindex(columns=base.columns)
            for c in rows[0]:
                add[c] = [r[c] for r in rows]
            base = pd.concat([base, add], ignore_index=True)
    except Exception:  # noqa: BLE001 — parquet-only is an acceptable fallback
        return load_matches(cfg)
    return base


def live_elo(cfg: dict | None = None, frame: pd.DataFrame | None = None) -> dict:
    """Elo ratings recomputed from scratch over the freshest results (so 2026 games
    are reflected and never double-counted). Sub-second over ~32k matches."""
    from ..features.elo import compute_elo_features
    cfg = cfg or load_config()
    frame = live_match_frame(cfg) if frame is None else frame
    _, engine = compute_elo_features(frame, cfg)
    return dict(engine.ratings)


def live_state(cfg: dict | None = None, n_iter: int = 20000,
               simulator=None, use_cache: bool = True) -> dict:
    """Bundle everything the UI needs: played set, standings, clinch flags, and the
    simulator's per-team qualification/championship odds with results locked in."""
    cfg = cfg or load_config()
    played = fetch_live_results(cfg, use_cache=use_cache)
    if simulator is None:
        from .tournament import TournamentSimulator
        simulator = TournamentSimulator(cfg, live=True)   # strength reflects 2026 form
    qual = simulator.run(n_iter=n_iter, played=played)
    return {"played": played, "standings": standings(cfg, played),
            "clinch": clinch_flags(cfg, played), "qual": qual,
            "n_played": len(played)}
