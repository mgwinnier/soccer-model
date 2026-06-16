"""Tests for feature attribution and walk-forward calibration."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.feature_importance import gbm_importances, GROUPS
from src.backtest.metrics import labels_from_results
from src.models.market_calibration import fit_calibrators
from src.models.market_bias import default_path


def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)                      # determines the outcome
    noise = rng.normal(size=n)                       # pure noise
    p_home = 1 / (1 + np.exp(-1.6 * signal))
    u = rng.random(n)
    result = np.where(u < p_home * 0.85, "H",
                      np.where(u < p_home * 0.85 + 0.15, "D", "A"))
    return pd.DataFrame({"signal": signal, "noise": noise, "result": result,
                         "date": pd.Timestamp("2020-01-01")})


def test_permutation_importance_signal_beats_noise():
    train, hold = _synthetic(1500, 0), _synthetic(600, 1)
    y_hold = labels_from_results(hold["result"])
    base, gain, perm = gbm_importances(train, hold, ["signal", "noise"], y_hold, n_repeats=3)
    # shuffling the signal hurts RPS; shuffling noise barely moves it
    assert perm["signal"] > perm["noise"]
    assert perm["signal"] > 0.0
    assert abs(perm["noise"]) < perm["signal"] / 3


def test_feature_groups_cover_the_user_list():
    for g in ("elo", "form", "market_value", "travel", "altitude", "head_to_head"):
        assert g in GROUPS


# ------------------------------------------------ walk-forward calibration
def test_calibration_as_of_uses_no_future_data():
    # before any modeled history exists -> no calibrators can be fit (so none can
    # have leaked from the future). Also must not touch the deployed file.
    before = default_path().with_name("market_calibrators.joblib")
    stamp = before.stat().st_mtime if before.exists() else None
    cal = fit_calibrators(as_of="2005-01-01", save=False)
    assert cal.models == {}                                   # nothing to fit pre-2005
    assert (before.stat().st_mtime if before.exists() else None) == stamp  # untouched
