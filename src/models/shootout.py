"""Penalty-shootout model for knockout ties.

Shootouts are close to a coin flip, but not exactly: on 542 shootouts since 1990
the higher-Elo side wins ~55%, and an Elo-difference logistic beats the 50/50
baseline on log loss (0.6846 vs 0.6931). We fit that slope and use it to resolve
simulated knockout draws, replacing the old flat ``P(draw)·0.5`` with a
strength-aware ``P(draw)·P(win shootout)``. The effect is modest by design —
overconfidence here would be wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import load_config, path_for
from ..data.team_names import normalize_team


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ShootoutModel:
    """P(team A beats team B on penalties) from their Elo gap."""

    def __init__(self, slope: float = 0.0011, clip: tuple[float, float] = (0.30, 0.70)):
        # default slope ~ fitted value; overwritten by .fit()
        self.slope = slope
        self.clip = clip

    @classmethod
    def fit_from_history(cls, elo_ratings: dict[str, float],
                         cfg: dict | None = None) -> "ShootoutModel":
        cfg = cfg or load_config()
        path = path_for("data_raw", cfg) / "shootouts.csv"
        if not path.exists():
            return cls()  # defaults; sim still works
        s = pd.read_csv(path, parse_dates=["date"])
        s = s[s["date"] >= pd.Timestamp(cfg["data"]["min_date"])].copy()
        for c in ["home_team", "away_team", "winner"]:
            s[c] = s[c].map(normalize_team)
        s = s.dropna(subset=["home_team", "away_team", "winner"])

        def elo(t):
            return elo_ratings.get(t, 1500.0)

        s["ediff"] = s.apply(lambda r: elo(r.home_team) - elo(r.away_team), axis=1)
        s["home_won"] = (s["winner"] == s["home_team"]).astype(int)
        if len(s) < 50 or s["home_won"].nunique() < 2:
            return cls()
        # symmetric model: no intercept, slope on Elo difference only
        lr = LogisticRegression(fit_intercept=False, C=1.0)
        lr.fit(s[["ediff"]], s["home_won"])
        model = cls(slope=float(lr.coef_[0, 0]))
        return model

    def prob_a_wins(self, elo_a: float, elo_b: float) -> float:
        p = float(_sigmoid(np.array(self.slope * (elo_a - elo_b))))
        return min(max(p, self.clip[0]), self.clip[1])

    def prob_matrix(self, elo: np.ndarray) -> np.ndarray:
        """(n, n) matrix of P(row team beats column team on penalties)."""
        diff = elo[:, None] - elo[None, :]
        p = _sigmoid(self.slope * diff)
        return np.clip(p, self.clip[0], self.clip[1])
