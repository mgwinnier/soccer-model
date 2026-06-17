"""Tests for the v13 upset/high-variance surface: signals block + the τ dial."""
import numpy as np

from src.predict.predict_match import MatchPredictor, _temper, _variance_signals


def test_temper_identity_at_one():
    p = np.array([0.6, 0.25, 0.15])
    assert np.allclose(_temper(p, 1.0), p)


def test_temper_flattens_and_normalizes():
    p = np.array([0.85, 0.11, 0.04])
    q = _temper(p, 1.5)
    assert abs(q.sum() - 1.0) < 1e-9          # still a distribution
    assert q[0] < p[0] and q[2] > p[2]        # favorite down, underdog up
    assert int(np.argmax(q)) == int(np.argmax(p))  # the PICK is unchanged


def test_variance_signals_bounded():
    s = _variance_signals(np.array([0.4, 0.3, 0.3]), 0.35, 2.7)
    for k in ("upset_risk", "draw_risk", "competitiveness", "shootout_potential"):
        assert 0.0 <= s[k] <= 1.0
    assert s["upset_risk"] == 0.3             # min(pH, pA)
    assert isinstance(s["high_upset"], bool) and isinstance(s["high_scoring"], bool)


def test_competitiveness_higher_for_closer_game():
    close = _variance_signals(np.array([0.34, 0.33, 0.33]), 0.3, 2.5)["competitiveness"]
    lop = _variance_signals(np.array([0.85, 0.1, 0.05]), 0.3, 2.5)["competitiveness"]
    assert close > lop


def test_analyze_has_signals_and_temp_is_no_op_by_default():
    mp = MatchPredictor()
    base = mp.analyze("Spain", "Cape Verde", neutral=True)
    assert "signals" in base and "upset_risk" in base["signals"]
    # τ=1.0 (default) must reproduce the untempered probabilities exactly
    again = mp.analyze("Spain", "Cape Verde", neutral=True, upset_temp=1.0)
    assert base["probs"] == again["probs"]


def test_dial_raises_underdog_without_flipping_pick():
    mp = MatchPredictor()
    base = mp.analyze("Spain", "Cape Verde", neutral=True)["probs"]
    hot = mp.analyze("Spain", "Cape Verde", neutral=True, upset_temp=1.6)["probs"]
    assert hot["A"] > base["A"]                       # Cape Verde (underdog) gets more
    assert max(base, key=base.get) == max(hot, key=hot.get)  # Spain still favored
