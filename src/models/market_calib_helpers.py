"""Pooled per-market model probabilities / outcomes for calibration fitting.

Pools predictions across goal/spread lines into one array per market type so a
single isotonic map per market can be fit (the bias is line-agnostic)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..predict.predict_match import _over_prob, _cover_prob

TOTAL_LINES = [1.5, 2.5, 3.5]
SPREAD_LINES = [-1.5, -0.5, 0.5, 1.5]


def market_probs_pooled(dc, test: pd.DataFrame) -> dict[str, np.ndarray]:
    neutral = test["neutral"].astype(bool).to_numpy()
    lam, mu = dc._lambdas(test["home_team"], test["away_team"], neutral)
    n = len(test)
    over_cols = {l: np.empty(n) for l in TOTAL_LINES}
    cover_cols = {l: np.empty(n) for l in SPREAD_LINES}
    btts = np.empty(n)
    for i in range(n):
        mat = dc.scoreline_matrix(lam[i], mu[i])
        for l in TOTAL_LINES:
            over_cols[l][i] = _over_prob(mat, l)
        for l in SPREAD_LINES:
            cover_cols[l][i] = _cover_prob(mat, l)[0]
        btts[i] = mat[1:, 1:].sum()
    return {
        "over": np.concatenate([over_cols[l] for l in TOTAL_LINES]),
        "cover": np.concatenate([cover_cols[l] for l in SPREAD_LINES]),
        "btts": btts,
    }


def outcomes_pooled(test: pd.DataFrame) -> dict[str, np.ndarray]:
    hs = test["home_score"].to_numpy()
    as_ = test["away_score"].to_numpy()
    tot, margin = hs + as_, hs - as_
    over = np.concatenate([(tot > l).astype(float) for l in TOTAL_LINES])
    cover = np.concatenate([
        np.where(np.abs(margin + l) < 1e-9, np.nan, (margin + l > 0).astype(float))
        for l in SPREAD_LINES])
    btts = ((hs > 0) & (as_ > 0)).astype(float)
    return {"over": over, "cover": cover, "btts": btts}
