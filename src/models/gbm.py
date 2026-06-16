"""LightGBM multinomial W/D/L model (Base C).

Consumes the full engineered feature set (Elo, form, squad value, context) and
predicts three-way outcome probabilities. LightGBM handles missing values
natively, which matters because squad value and travel distance are sparse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .base import results_to_labels

# Meta / identifier columns that must never be fed to the model.
_NON_FEATURES = {
    "match_id", "date", "home_team", "away_team", "home_score", "away_score",
    "result", "tournament", "is_world_cup",
    # post-match outcomes — would leak the result if used as features
    "home_xg", "away_xg",
}


def _load_tuned_params() -> dict:
    """Optuna-tuned hyperparameters, if a tuning run has persisted them."""
    import json
    from ..config import path_for
    p = path_for("models") / "gbm_params.json"
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric model-input columns (everything that isn't meta/identifier)."""
    cols = []
    for c in df.columns:
        if c in _NON_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


class GBMModel:
    def __init__(self, **params):
        defaults = dict(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            max_depth=-1, min_child_samples=40, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            objective="multiclass", num_class=3, n_jobs=-1, verbose=-1,
        )
        defaults.update(_load_tuned_params())   # Optuna-tuned, if present
        defaults.update(params)                 # explicit args win
        self.params = defaults
        self.clf = LGBMClassifier(**defaults)
        self.features_: list[str] = []

    def fit(self, df: pd.DataFrame, features: list[str] | None = None) -> "GBMModel":
        self.features_ = features or model_feature_columns(df)
        X = df[self.features_].astype(float)
        y = results_to_labels(df["result"])
        self.clf.fit(X, y)
        self._classes = list(self.clf.classes_)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.features_).astype(float)
        raw = self.clf.predict_proba(X)
        out = np.zeros((len(df), 3))
        for col, cls in enumerate(self._classes):
            out[:, cls] = raw[:, col]
        return out

    def importances(self) -> pd.Series:
        return pd.Series(self.clf.feature_importances_, index=self.features_) \
            .sort_values(ascending=False)
