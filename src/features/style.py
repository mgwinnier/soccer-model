"""xG-based style/quality features, leak-free via as-of joins.

International xG exists only for elite tournaments, so rather than a dense rolling
window we keep a slowly-updating **trailing xG rating** per team (mean of its last
N xG matches) and, for any later fixture, attach each side's most recent rating
strictly *before* kickoff (``merge_asof`` with ``allow_exact_matches=False`` — so a
team never sees its own current-match xG). Teams with no prior xG match get NaN,
which LightGBM handles natively.

Exposed per match (all NaN-tolerant):
  home/away_xgf, home/away_xga  — trailing xG for / against
  xg_rating_diff                — (home xgf-xga) − (away xgf-xga)
  xg_attack_vs_defense          — home attacking xG vs away conceded xG (matchup)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.fbref import attach_xg_to_matches

_WINDOW = 5  # last N xG matches


def _team_xg_ratings(matches_with_xg: pd.DataFrame) -> pd.DataFrame:
    """Per (team, date) trailing xG-for/against over the last N xG matches."""
    xgm = matches_with_xg.dropna(subset=["home_xg", "away_xg"])
    if xgm.empty:
        return pd.DataFrame(columns=["team", "date", "xgf", "xga"])
    home = pd.DataFrame({
        "team": xgm["home_team"], "date": xgm["date"],
        "xgf": xgm["home_xg"], "xga": xgm["away_xg"],
    })
    away = pd.DataFrame({
        "team": xgm["away_team"], "date": xgm["date"],
        "xgf": xgm["away_xg"], "xga": xgm["home_xg"],
    })
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])
    g = long.groupby("team", group_keys=False)
    long["xgf"] = g["xgf"].apply(lambda s: s.rolling(_WINDOW, min_periods=1).mean())
    long["xga"] = g["xga"].apply(lambda s: s.rolling(_WINDOW, min_periods=1).mean())
    return long.sort_values("date").reset_index(drop=True)


def _asof_side(matches: pd.DataFrame, ratings: pd.DataFrame, team_col: str) -> pd.DataFrame:
    left = matches[["match_id", "date", team_col]].rename(columns={team_col: "team"})
    left = left.sort_values("date")
    merged = pd.merge_asof(
        left, ratings, on="date", by="team",
        direction="backward", allow_exact_matches=False,
    )
    return merged.set_index("match_id")[["xgf", "xga"]]


def compute_style_features(matches: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    # Reuse xG already attached upstream (build.py) to avoid a double-merge;
    # only fetch+attach if the caller passed plain matches.
    if {"home_xg", "away_xg"}.issubset(matches.columns):
        mx = matches
    else:
        mx = attach_xg_to_matches(matches, cfg)
    ratings = _team_xg_ratings(mx)
    idx = matches.set_index("match_id").index
    out = pd.DataFrame(index=idx)
    if ratings.empty:
        for c in ["home_xgf", "home_xga", "away_xgf", "away_xga",
                  "xg_rating_diff", "xg_attack_vs_defense"]:
            out[c] = np.nan
        return out

    h = _asof_side(matches, ratings, "home_team").add_prefix("home_")
    a = _asof_side(matches, ratings, "away_team").add_prefix("away_")
    out = out.join(h).join(a)
    out["xg_rating_diff"] = (out["home_xgf"] - out["home_xga"]) - (
        out["away_xgf"] - out["away_xga"])
    out["xg_attack_vs_defense"] = out["home_xgf"] - out["away_xga"]
    return out
