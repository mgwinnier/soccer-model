"""Tests for v11: the probability-edge gate that fixes the underdog/longshot lean."""
import pytest

from src.predict.betting import qualifies, expected_value


def test_gate_rejects_leverage_longshot():
    # +800 dog, model 14% vs fair 8%: big EV + a real edge, but past the odds cap → out
    assert expected_value(0.14, 9.0) > 0.03            # clears a flat EV bar
    assert qualifies(0.14, 0.08, 9.0) is False         # but the gate rejects (dec > 6)


def test_gate_rejects_tiny_edge_even_with_high_ev():
    # model 9% vs fair 8.5% at +1000: +EV by leverage but only a 0.5pp disagreement
    assert qualifies(0.09, 0.085, 11.0) is False


def test_gate_keeps_a_real_favorite_edge():
    # heavy favorite the model genuinely rates higher: 80% vs fair 72% at -300
    assert qualifies(0.80, 0.72, 1.333) is True


def test_gate_keeps_a_real_near_even_edge():
    # model 45% vs fair 40% at +135 (dec 2.35): real +EV and a real 5pp edge
    assert expected_value(0.45, 2.35) >= 0.03
    assert qualifies(0.45, 0.40, 2.35) is True


def test_gate_rejects_negative_ev_and_nan():
    assert qualifies(0.45, 0.40, 2.0) is False          # break-even 50% > model 45% → -EV
    assert qualifies(0.5, 0.4, float("nan")) is False    # no usable price
    assert qualifies(0.5, 0.4, None) is False
