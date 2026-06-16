"""Rolling recent-form features, computed without look-ahead leakage.

Each match is exploded into two team-perspective rows; for every team we look
only at its *previous* N matches (``.shift(1)`` before rolling) to compute
points-per-game, goals for/against, and rest days. Results are folded back onto
the match as ``home_*`` / ``away_*`` columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config


def _long_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode each match into two team-perspective rows."""
    home = pd.DataFrame({
        "match_id": matches["match_id"], "date": matches["date"],
        "team": matches["home_team"], "gf": matches["home_score"],
        "ga": matches["away_score"], "side": "home",
    })
    away = pd.DataFrame({
        "match_id": matches["match_id"], "date": matches["date"],
        "team": matches["away_team"], "gf": matches["away_score"],
        "ga": matches["home_score"], "side": "away",
    })
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.select(
        [long.gf > long.ga, long.gf == long.ga], [3, 1], default=0
    )
    return long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def compute_form_features(
    matches: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    """Return per-match rolling-form features indexed by ``match_id``."""
    cfg = cfg or load_config()
    windows = cfg["features"]["form_windows"]
    long = _long_form(matches)
    g = long.groupby("team", sort=False, group_keys=False)

    # Rest days since the team's previous match
    long["rest_days"] = g["date"].diff().dt.days

    for w in windows:
        # shift(1) so the current match is excluded -> no leakage
        long[f"ppg_{w}"] = (
            g["points"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        long[f"gf_{w}"] = (
            g["gf"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        long[f"ga_{w}"] = (
            g["ga"].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    feat_cols = ["rest_days"] + [
        f"{m}_{w}" for w in windows for m in ("ppg", "gf", "ga")
    ]

    home = long[long.side == "home"].set_index("match_id")[feat_cols]
    away = long[long.side == "away"].set_index("match_id")[feat_cols]
    home = home.add_prefix("home_")
    away = away.add_prefix("away_")
    out = home.join(away, how="outer")

    # Differential convenience features (home minus away)
    for w in windows:
        out[f"ppg_diff_{w}"] = out[f"home_ppg_{w}"] - out[f"away_ppg_{w}"]
    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]
    return out
