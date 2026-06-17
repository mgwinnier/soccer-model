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


def test_bet_code_and_result_btts():
    assert clv._bet_code("BTTS", "Both Teams Score: Yes", "France", "Spain", {}) == "btts_yes"
    assert clv._bet_code("BTTS", "Both Teams Score: No", "France", "Spain", {}) == "btts_no"
    assert clv._result("btts_yes", 3, 1) == "win"     # both scored
    assert clv._result("btts_yes", 2, 0) == "loss"    # only one scored
    assert clv._result("btts_no", 1, 0) == "win"      # one side blanked
    assert clv._result("btts_no", 2, 2) == "loss"     # both scored
    # BTTS has no ESPN closing line -> no close price
    assert clv._closing_price("btts_yes", {"ml_home": 100}) is None


def test_result_unknown_code_never_raises():
    # the exact legacy bug: a BTTS ticket stored with code "?" used to crash grade()
    assert clv._result("?", 3, 1) is None
    assert clv._result("cover_away@None", 3, 1) is None   # malformed line, no crash
    assert clv._result("weird", 0, 0) is None


def test_repair_code_heals_legacy_btts():
    assert clv._repair_code("BTTS", "?", "Both Teams Score: Yes") == "btts_yes"
    assert clv._repair_code("BTTS", float("nan"), "Both Teams Score: No") == "btts_no"
    assert clv._repair_code("Match Result", "H", "France") == "H"   # good code untouched
    assert clv._repair_code("Spread", "?", "France -1.5") is None   # unrecoverable -> None


def test_grade_does_not_crash_on_bad_ticket(monkeypatch, tmp_path):
    """grade() must settle every gradable ticket and skip a '?' BTTS ticket without crashing —
    the regression that silently froze the live Tracker."""
    import pandas as pd
    cfg = {"paths": {"reports": str(tmp_path), "models": str(tmp_path)}}
    open_df = pd.DataFrame([
        {"game_id": 111, "match_date": "2026-06-17", "match": "A v B", "market": "Match Result",
         "code": "H", "segment": "MR:H", "system": "", "selection": "A", "american": 120,
         "decimal": 2.2, "model_p": 0.55, "fair_p": 0.5, "ev": 0.1, "snapshot_time": "t"},
        {"game_id": 111, "match_date": "2026-06-17", "match": "A v B", "market": "BTTS",
         "code": "?", "segment": "?", "system": "", "selection": "Both Teams Score: Yes",
         "american": 150, "decimal": 2.5, "model_p": 0.5, "fair_p": 0.45, "ev": 0.1,
         "snapshot_time": "t"},
        {"game_id": 222, "match_date": "2026-06-17", "match": "C v D", "market": "Total Goals",
         "code": "over@2.5", "segment": "TG:over", "system": "", "selection": "Over 2.5",
         "american": 110, "decimal": 2.1, "model_p": 0.52, "fair_p": 0.48, "ev": 0.08,
         "snapshot_time": "t"},
    ])
    open_df.to_csv(clv._open_path(cfg), index=False)
    # A v B finished 3-1 (both scored, home win); C v D not finished yet
    monkeypatch.setattr(clv, "fetch_espn_range", lambda *a, **k: [
        {"game_id": "111", "status": "post", "home_score": 3, "away_score": 1}])
    monkeypatch.setattr(clv, "fetch_summary_odds", lambda *a, **k: None)
    n = clv.grade(cfg=cfg, now="2026-06-17T18:00:00Z")
    led = pd.read_csv(clv._ledger_path(cfg))
    rem = pd.read_csv(clv._open_path(cfg))
    assert n == 2                                   # MR:H + BTTS both settled
    codes = set(led["code"])
    assert "btts_yes" in codes and "H" in codes     # the '?' BTTS healed to btts_yes
    assert led.loc[led["code"] == "btts_yes", "result"].iloc[0] == "win"   # 3-1 -> both scored
    assert led["closing_decimal"].isna().all()      # no ESPN close mocked -> CLV blank
    assert set(rem["game_id"]) == {222}             # unfinished C v D stays open, no crash


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
