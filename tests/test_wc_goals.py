"""Tests for the World-Cup scoring-environment correction.

The model is trained on all internationals and under-projects WC goals vs ACTUAL
results (backtested across 1998-2022). ``MatchPredictor.WC_GOALS_SCALE`` corrects the
expected goals toward real WC scoring — not toward the Vegas line. These tests pin the
mechanism, not exact numbers (which depend on the fitted model).
"""
import numpy as np

from src.predict.predict_match import MatchPredictor


def test_scale_is_a_meaningful_uplift():
    # measured WC under-projection is ~-0.35 goals on a ~2.2 base -> ~15% uplift
    assert 1.05 < MatchPredictor.WC_GOALS_SCALE < 1.30


def test_analyze_applies_the_wc_scale_to_expected_goals():
    mp = MatchPredictor()
    home, away = "Brazil", "Morocco"
    raw = mp.dc.expected_goals(home, away, True)          # un-corrected DC goals
    a = mp.analyze(home, away, neutral=True)
    eg = a["expected_goals"]
    s = MatchPredictor.WC_GOALS_SCALE
    # the analyzed expected goals are the raw DC goals scaled by WC_GOALS_SCALE
    assert eg[0] == round(raw[0] * s, 2)
    assert eg[1] == round(raw[1] * s, 2)
    # and the correction genuinely raises the projected total
    assert sum(eg) > sum(raw)


def test_outcome_probs_still_normalize():
    mp = MatchPredictor()
    a = mp.analyze("Spain", "Cape Verde", neutral=True)
    p = a["probs"]
    assert abs(p["H"] + p["D"] + p["A"] - 1.0) < 1e-6


def test_no_market_line_needed():
    # the correction must work without any Vegas total (Team Explorer path):
    # analyze takes no goals/line argument and still corrects toward real WC scoring.
    mp = MatchPredictor()
    a = mp.analyze("Argentina", "Mexico", neutral=True)
    assert sum(a["expected_goals"]) > 0
