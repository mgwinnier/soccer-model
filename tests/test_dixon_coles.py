"""Tests for the Dixon-Coles model on a controlled synthetic dataset."""
import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import DixonColesModel, _dc_tau
from src.models.base import scoreline_to_outcome_probs


def _synthetic_matches() -> pd.DataFrame:
    """Strong beats Medium beats Weak, repeated, with dated neutral games."""
    rng = np.random.default_rng(0)
    rows = []
    base = pd.Timestamp("2015-01-01")
    strength = {"Strong": 2.2, "Medium": 1.2, "Weak": 0.4}
    teams = list(strength)
    for d in range(300):
        a, b = rng.choice(teams, size=2, replace=False)
        ga = rng.poisson(strength[a])
        gb = rng.poisson(strength[b])
        rows.append({
            "home_team": a, "away_team": b, "home_score": ga, "away_score": gb,
            "date": base + pd.Timedelta(days=d * 3), "neutral": True,
            "importance": 1.0,
        })
    return pd.DataFrame(rows)


def test_fit_and_probabilities_sum_to_one():
    dc = DixonColesModel(xi=0.0).fit(_synthetic_matches())
    fx = pd.DataFrame([{"home_team": "Strong", "away_team": "Weak", "neutral": True}])
    p = dc.predict_proba(fx)[0]
    assert p.shape == (3,)
    assert p.sum() == pytest.approx(1.0)


def test_stronger_team_is_favoured():
    dc = DixonColesModel(xi=0.0).fit(_synthetic_matches())
    fx = pd.DataFrame([{"home_team": "Strong", "away_team": "Weak", "neutral": True}])
    p = dc.predict_proba(fx)[0]
    assert p[0] > p[2]  # P(home win) > P(away win)
    lam, mu = dc.expected_goals("Strong", "Weak", neutral=True)
    assert lam > mu


def test_scoreline_matrix_normalised():
    dc = DixonColesModel(xi=0.0).fit(_synthetic_matches())
    mat = dc.scoreline_matrix(1.5, 1.1)
    assert mat.sum() == pytest.approx(1.0)
    h, d, a = scoreline_to_outcome_probs(mat)
    assert h + d + a == pytest.approx(1.0)


def test_dc_tau_unaffected_for_high_scores():
    h = np.array([3]); a = np.array([2])
    assert _dc_tau(h, a, 1.5, 1.1, 0.1)[0] == pytest.approx(1.0)
