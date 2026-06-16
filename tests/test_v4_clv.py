"""Tests for CLV bet-code/result logic and the bivariate-Poisson model."""
import numpy as np
import pandas as pd
import pytest

from src.predict import clv
from src.predict.value import _type_key, _recenter_matches
from src.predict.betting import evaluate_bet
from src.models.bivariate_poisson import _bp_matrix, BivariatePoissonModel


# ----------------------------------------------------------------- CLV codes
def test_bet_code_match_result():
    a = {"spread": {"home_line": -1.5}}
    assert clv._bet_code("Match Result", "France", "France", "Spain", a) == "H"
    assert clv._bet_code("Match Result", "Spain", "France", "Spain", a) == "A"
    assert clv._bet_code("Match Result", "Draw", "France", "Spain", a) == "D"


def test_bet_code_totals_and_spread():
    a = {"spread": {"home_line": -1.5}}
    assert clv._bet_code("Total Goals", "Over 2.5", "France", "Spain", a) == "over@2.5"
    assert clv._bet_code("Total Goals", "Under 2.5", "France", "Spain", a) == "under@2.5"
    assert clv._bet_code("Spread", "France -1.5", "France", "Spain", a) == "cover_home@-1.5"
    assert clv._bet_code("Spread", "Spain +1.5", "France", "Spain", a) == "cover_away@-1.5"


def test_result_grading():
    assert clv._result("H", 2, 0) == "win"
    assert clv._result("A", 2, 0) == "loss"
    assert clv._result("D", 1, 1) == "win"
    assert clv._result("over@2.5", 2, 1) == "win"     # 3 > 2.5
    assert clv._result("under@2.5", 1, 1) == "win"    # 2 < 2.5
    assert clv._result("cover_home@-1.5", 2, 0) == "win"   # margin 2 > 1.5
    assert clv._result("cover_home@-1.5", 1, 0) == "loss"  # margin 1 < 1.5
    assert clv._result("cover_home@-1.0", 1, 0) == "push"  # margin exactly 1


# -------------------------------------------------------- bivariate Poisson
def test_recenter_removes_systematic_lean(monkeypatch):
    # force the per-slate fallback (no persisted historical bias)
    import src.models.market_bias as mbmod
    monkeypatch.setattr(mbmod, "load_default", lambda cfg=None: mbmod.MarketBias({}))
    # 3 matches where the model is consistently +8% on "Under" vs the market
    matches = []
    for i in range(3):
        bets = [
            evaluate_bet("Total Goals", "Over 2.5", -110, 0.42, 0.50, 1000, 0.5),
            evaluate_bet("Total Goals", "Under 2.5", -110, 0.58, 0.50, 1000, 0.5),
        ]
        matches.append({"home": "A", "away": "B", "bets": bets})
    _recenter_matches(matches, 1000, 0.5, shrink=1.0)
    # after full recentering, the model's mean edge per role should be ~0
    import numpy as np
    under_edges = [b.model_p - b.fair_p for m in matches for b in m["bets"]
                   if b.selection.startswith("Under")]
    assert abs(np.mean(under_edges)) < 1e-9


def test_type_key():
    assert _type_key("Match Result", "France", "France", "Spain") == "MR:H"
    assert _type_key("Match Result", "Spain", "France", "Spain") == "MR:A"
    assert _type_key("Total Goals", "Under 2.5", "France", "Spain") == "TG:under"
    assert _type_key("Spread", "Spain +1.5", "France", "Spain") == "SP:away"


def test_bp_matrix_normalised():
    mat = _bp_matrix(1.5, 1.1, 0.2, 10)
    assert mat.sum() == pytest.approx(1.0)


def test_bp_reduces_to_independent_when_lambda3_zero():
    lam1, lam2 = 1.4, 1.0
    mat = _bp_matrix(lam1, lam2, 0.0, 8)
    k = np.arange(9)
    from scipy.special import gammaln
    pois = lambda lm: np.exp(-lm) * lm ** k / np.exp(gammaln(k + 1))
    indep = np.outer(pois(lam1), pois(lam2))
    indep /= indep.sum()
    assert np.allclose(mat, indep, atol=1e-9)


def test_bp_model_fits_and_predicts():
    rng = np.random.default_rng(0)
    rows = []
    base = pd.Timestamp("2015-01-01")
    strength = {"Strong": 2.0, "Weak": 0.6}
    for d in range(200):
        a, b = ("Strong", "Weak") if d % 2 else ("Weak", "Strong")
        rows.append({"home_team": a, "away_team": b,
                     "home_score": rng.poisson(strength[a]),
                     "away_score": rng.poisson(strength[b]),
                     "date": base + pd.Timedelta(days=d), "neutral": True,
                     "importance": 1.0})
    df = pd.DataFrame(rows)
    m = BivariatePoissonModel(xi=0.0).fit(df)
    assert m.lambda3_ >= 0.0
    fx = pd.DataFrame([{"home_team": "Strong", "away_team": "Weak", "neutral": True}])
    p = m.predict_proba(fx)[0]
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[2]
