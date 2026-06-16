"""World-Football-style Elo rating engine.

Processes matches in chronological order and, for every match, records the two
teams' ratings *before* the match (so the rating is a valid, leak-free feature)
then updates them from the result. The update follows the well-known World
Football Elo formula:

    R' = R + K · G · (W − We)
    We = 1 / (1 + 10^(−Δ/400)),   Δ = R_home − R_away (+ home_adv if not neutral)
    G  = goal-difference multiplier (bigger wins move ratings more)
    K  = base K scaled by match importance

The engine is reusable: ``EloEngine`` can be seeded with existing ratings and
fed hypothetical fixtures, which the tournament simulator relies on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import load_config


def goal_diff_multiplier(goal_diff: int) -> float:
    """World Football Elo margin-of-victory multiplier."""
    g = abs(int(goal_diff))
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def expected_score(rating_a: float, rating_b: float, home_adv: float = 0.0) -> float:
    """Expected score (win prob + half draw prob) of A vs B."""
    delta = rating_a - rating_b + home_adv
    return 1.0 / (1.0 + 10.0 ** (-delta / 400.0))


@dataclass
class EloEngine:
    base_rating: float = 1500.0
    k_factor: float = 40.0
    home_advantage: float = 65.0
    ratings: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "EloEngine":
        cfg = cfg or load_config()
        e = cfg["elo"]
        return cls(
            base_rating=e["base_rating"],
            k_factor=e["k_factor"],
            home_advantage=e["home_advantage"],
        )

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.base_rating)

    def expected_home(self, home: str, away: str, neutral: bool) -> float:
        ha = 0.0 if neutral else self.home_advantage
        return expected_score(self.rating(home), self.rating(away), ha)

    def update_one(
        self, home: str, away: str, home_score: int, away_score: int,
        neutral: bool, importance: float = 1.0,
    ) -> tuple[float, float]:
        """Apply one result; returns the pre-match (home, away) ratings."""
        rh, ra = self.rating(home), self.rating(away)
        ha = 0.0 if neutral else self.home_advantage
        we = expected_score(rh, ra, ha)
        if home_score > away_score:
            w = 1.0
        elif home_score < away_score:
            w = 0.0
        else:
            w = 0.5
        k = self.k_factor * importance * goal_diff_multiplier(home_score - away_score)
        change = k * (w - we)
        self.ratings[home] = rh + change
        self.ratings[away] = ra - change
        return rh, ra


def compute_elo_features(
    matches: pd.DataFrame, cfg: dict | None = None
) -> tuple[pd.DataFrame, EloEngine]:
    """Walk all matches once, returning per-match pre-ratings + final engine.

    Returned frame is indexed like ``matches`` with columns:
    ``home_elo, away_elo, elo_diff, elo_exp_home`` (all leak-free / pre-match).
    """
    cfg = cfg or load_config()
    engine = EloEngine.from_config(cfg)
    matches = matches.sort_values("date")

    home_elo = np.empty(len(matches))
    away_elo = np.empty(len(matches))
    exp_home = np.empty(len(matches))

    for i, row in enumerate(matches.itertuples(index=False)):
        neutral = bool(row.neutral)
        exp_home[i] = engine.expected_home(row.home_team, row.away_team, neutral)
        rh, ra = engine.update_one(
            row.home_team, row.away_team, int(row.home_score),
            int(row.away_score), neutral, float(row.importance),
        )
        home_elo[i] = rh
        away_elo[i] = ra

    feats = pd.DataFrame(
        {
            "match_id": matches["match_id"].to_numpy(),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "elo_exp_home": exp_home,
        }
    )
    return feats.set_index("match_id"), engine
