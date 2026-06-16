"""Market anchoring — blend the (calibrated) model toward the sharp price.

`P_final = w·P_model + (1−w)·P_market_fair`, where `P_market_fair` is the
de-vigged bookmaker probability. The bookmaker line aggregates far more
information than our model, so anchoring toward it is how real betting models
behave: you only deviate from a sharp consensus by a little, and *that small,
calibrated deviation is your edge*. `w` is the weight on the model (default 0.5);
lower `w` = trust the market more = fewer, smaller edges.

Setting `w = 1` recovers the independent model; `w = 0` just echoes the market.
"""
from __future__ import annotations

import numpy as np

DEFAULT_W = 0.5


def anchor(model_p: float, market_fair_p: float | None, w: float = DEFAULT_W) -> float:
    """Blend one probability toward the market. If no market price, return model."""
    if market_fair_p is None or (isinstance(market_fair_p, float) and np.isnan(market_fair_p)):
        return model_p
    return w * model_p + (1.0 - w) * market_fair_p


def anchor_vector(model_p: np.ndarray, market_fair: np.ndarray,
                  w: float = DEFAULT_W) -> np.ndarray:
    """Blend a probability vector (e.g. 1X2) toward the market, renormalised."""
    model_p = np.asarray(model_p, float)
    if market_fair is None or np.any(pd_isna(market_fair)):
        return model_p
    market_fair = np.asarray(market_fair, float)
    blended = w * model_p + (1.0 - w) * market_fair
    s = blended.sum()
    return blended / s if s > 0 else model_p


def pd_isna(x) -> np.ndarray:
    return np.isnan(np.asarray(x, float))
