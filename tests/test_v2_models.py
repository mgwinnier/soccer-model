"""Tests for v2 additions: shootout model, xG Dixon-Coles, leak-free xG style."""
import numpy as np
import pandas as pd
import pytest

from src.models.shootout import ShootoutModel
from src.models.xg_dixon_coles import XgDixonColesModel
from src.features.style import _team_xg_ratings, _asof_side


# ----------------------------------------------------------------- shootout
def test_shootout_bounds_and_symmetry():
    m = ShootoutModel(slope=0.0011, clip=(0.30, 0.70))
    p = m.prob_a_wins(2000, 1600)   # stronger team A
    assert 0.5 < p <= 0.70
    # symmetric: P(A) + P(B) == 1 before clipping at equal-ish gaps
    assert m.prob_a_wins(1800, 1700) + m.prob_a_wins(1700, 1800) == pytest.approx(1.0, abs=1e-6)
    # equal teams -> coin flip
    assert m.prob_a_wins(1700, 1700) == pytest.approx(0.5)


def test_shootout_clip():
    m = ShootoutModel(slope=0.01, clip=(0.30, 0.70))
    assert m.prob_a_wins(3000, 1000) == 0.70   # clipped
    assert m.prob_a_wins(1000, 3000) == 0.30


# -------------------------------------------------------------- xG Dixon-Coles
def _synth_xg_matches():
    rng = np.random.default_rng(1)
    rows = []
    base = pd.Timestamp("2018-01-01")
    strength = {"Strong": 2.2, "Medium": 1.2, "Weak": 0.4}
    teams = list(strength)
    for d in range(300):
        a, b = rng.choice(teams, size=2, replace=False)
        ga, gb = rng.poisson(strength[a]), rng.poisson(strength[b])
        rows.append({
            "home_team": a, "away_team": b, "home_score": ga, "away_score": gb,
            "home_xg": strength[a], "away_xg": strength[b],
            "date": base + pd.Timedelta(days=d * 3), "neutral": True, "importance": 1.0,
        })
    return pd.DataFrame(rows)


def test_xg_dc_blends_target_and_predicts():
    dc = XgDixonColesModel(xg_weight=0.6, xi=0.0).fit(_synth_xg_matches())
    fx = pd.DataFrame([{"home_team": "Strong", "away_team": "Weak", "neutral": True}])
    p = dc.predict_proba(fx)[0]
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[2]  # stronger team favoured


def test_xg_dc_falls_back_to_goals_without_xg():
    df = _synth_xg_matches().drop(columns=["home_xg", "away_xg"])
    dc = XgDixonColesModel(xg_weight=0.6, xi=0.0).fit(df)  # must not crash
    fx = pd.DataFrame([{"home_team": "Strong", "away_team": "Weak", "neutral": True}])
    assert dc.predict_proba(fx)[0].sum() == pytest.approx(1.0)


# --------------------------------------------------------- leak-free xG style
def test_xg_style_asof_is_leak_free():
    # Two xG matches for team A; the as-of join must NOT use a match's own xG.
    mx = pd.DataFrame({
        "home_team": ["A", "A"], "away_team": ["B", "C"],
        "home_xg": [2.0, 3.0], "away_xg": [1.0, 0.5],
        "date": pd.to_datetime(["2022-01-01", "2022-06-01"]),
    })
    matches = pd.DataFrame({
        "match_id": [10, 11],
        "home_team": ["A", "A"], "away_team": ["B", "C"],
        "date": pd.to_datetime(["2022-01-01", "2022-06-01"]),
    })
    ratings = _team_xg_ratings(mx)
    side = _asof_side(matches, ratings, "home_team")
    # first match: no prior xG -> NaN (cannot see its own 2.0)
    assert np.isnan(side.loc[10, "xgf"])
    # second match: sees only the first match's xG (2.0), not its own 3.0
    assert side.loc[11, "xgf"] == pytest.approx(2.0)
