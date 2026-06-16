"""Elo-based W/D/L predictor (Base B).

The Elo *engine* (``features/elo.py``) produces a rating difference; this wraps
it into calibrated three-way probabilities by fitting a multinomial logistic
regression on the "effective" rating gap (rating difference plus a home-field
bonus on non-neutral games). Doubles as a strong, transparent benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import load_config
from .base import OUTCOMES, results_to_labels


class EloModel:
    def __init__(self, home_advantage: float | None = None, cfg: dict | None = None):
        cfg = cfg or load_config()
        self.home_advantage = (
            home_advantage if home_advantage is not None else cfg["elo"]["home_advantage"]
        )
        self.clf = LogisticRegression(C=1.0, max_iter=1000)

    def _effective_diff(self, df: pd.DataFrame) -> np.ndarray:
        bonus = np.where(df["neutral"].astype(bool), 0.0, self.home_advantage)
        return (df["elo_diff"].to_numpy() + bonus).reshape(-1, 1)

    def fit(self, df: pd.DataFrame) -> "EloModel":
        X = self._effective_diff(df)
        y = results_to_labels(df["result"])
        self.clf.fit(X, y)
        self._classes = list(self.clf.classes_)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._effective_diff(df)
        raw = self.clf.predict_proba(X)
        # reorder to canonical [H, D, A]
        out = np.zeros((len(df), 3))
        for col, cls in enumerate(self._classes):
            out[:, cls] = raw[:, col]
        return out
