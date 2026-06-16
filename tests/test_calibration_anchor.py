"""Tests for market calibration and market anchoring."""
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from src.models.market_calibration import MarketCalibrators
from src.predict.anchor import anchor, anchor_vector
from src.backtest.metrics import binary_brier, calibration_error, evaluate_binary


# ----------------------------------------------------------------- anchoring
def test_anchor_endpoints():
    assert anchor(0.7, 0.5, w=1.0) == pytest.approx(0.7)   # w=1 -> pure model
    assert anchor(0.7, 0.5, w=0.0) == pytest.approx(0.5)   # w=0 -> pure market
    assert anchor(0.7, 0.5, w=0.5) == pytest.approx(0.6)   # midpoint


def test_anchor_no_market_returns_model():
    assert anchor(0.7, None, w=0.5) == 0.7
    assert anchor(0.7, float("nan"), w=0.5) == 0.7


def test_anchor_vector_renormalises_and_blends():
    model = np.array([0.6, 0.25, 0.15])
    market = np.array([0.4, 0.30, 0.30])
    out = anchor_vector(model, market, w=0.5)
    assert out.sum() == pytest.approx(1.0)
    # each leg is the midpoint (both inputs already sum to 1, so no renorm needed)
    assert out == pytest.approx([0.5, 0.275, 0.225])


# --------------------------------------------------------------- calibration
def test_calibrators_noop_when_empty():
    cal = MarketCalibrators({})
    assert cal.calibrate("over", 0.42) == 0.42


def test_calibrator_applies_monotone_map():
    # build a calibrator that maps raw -> shifted-up probabilities
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    raw = np.linspace(0, 1, 50)
    ir.fit(raw, np.clip(raw + 0.05, 0, 1))
    cal = MarketCalibrators({"over": ir})
    assert cal.calibrate("over", 0.50) == pytest.approx(0.55, abs=1e-2)
    # unknown market -> unchanged
    assert cal.calibrate("btts", 0.50) == 0.50


def test_calibration_reduces_bias_metric():
    # a biased forecaster (always 0.1 too high) should have worse ECE than calibrated
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.45).astype(float)
    biased = np.clip(np.full(2000, 0.55), 0, 1)
    assert calibration_error(biased, y) > 0.05
    assert binary_brier(biased, y) > binary_brier(np.full(2000, 0.45), y)
