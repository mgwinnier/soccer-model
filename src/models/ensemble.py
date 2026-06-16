"""Stacked ensemble of Dixon-Coles + Elo + LightGBM, blended out-of-fold.

Naive stacking fails here: if the meta-learner is trained on base models fit to a
*subset* of data but then deployed with base models refit on *all* data, the base
probability distributions shift and the meta weights are miscalibrated (we
observed the ensemble underperforming its own members). The fix is **time-series
out-of-fold (OOF) cross-fitting**:

  * walk forward over K temporal folds; for each fold, train the bases on all
    earlier data and predict the held-out fold — these OOF predictions match the
    quality the bases will have at deployment (trained on the past, scoring the
    future);
  * learn blend weights on the *pooled* OOF predictions by directly minimising
    RPS over the simplex (non-negative weights summing to one) — far more stable
    than a 9-input logistic;
  * optionally fit an isotonic calibrator on the OOF blend;
  * refit the bases on the entire window for deployment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from ..config import load_config
from ..backtest.metrics import ranked_probability_score, labels_from_results
from .dixon_coles import DixonColesModel
from .xg_dixon_coles import XgDixonColesModel
from .elo_model import EloModel
from .gbm import GBMModel

# Base members, in canonical order. xg_dixon_coles is included unconditionally —
# if xG data is absent it degrades to the goals model and the OOF weight optimiser
# simply down-weights it, so there is no harm in always offering it.
_MEMBERS = ["dixon_coles", "xg_dixon_coles", "elo", "gbm"]


class EnsembleModel:
    def __init__(self, cfg: dict | None = None, n_folds: int = 5,
                 members: list[str] | None = None):
        self.cfg = cfg or load_config()
        self.calibration = self.cfg["ensemble"].get("calibration", "none")
        self.n_folds = n_folds
        # Default members come from config (ablation-driven). xg_dixon_coles is
        # built and available but excluded by default: the ablation showed sparse
        # WC/Euro-only xG is neutral-to-noise for World Cup RPS. Add it back via
        # config `ensemble.members` or members=[...] if richer xG data arrives.
        self.members = members or self.cfg["ensemble"].get("members") or list(_MEMBERS)
        self.models_: dict[str, object] = {}
        self.weights_ = np.full(len(self.members), 1 / len(self.members))
        self.iso: list[IsotonicRegression] | None = None

    # -------------------------------------------------------------- helpers
    def _new_member(self, name: str):
        if name == "dixon_coles":
            return DixonColesModel.from_config(self.cfg)
        if name == "xg_dixon_coles":
            return XgDixonColesModel.from_config(self.cfg)
        if name == "elo":
            return EloModel(cfg=self.cfg)
        if name == "gbm":
            return GBMModel()
        raise ValueError(f"unknown ensemble member: {name}")

    def _fit_bases(self, df: pd.DataFrame) -> dict[str, object]:
        return {name: self._new_member(name).fit(df) for name in self.members}

    def _member_probs(self, models: dict[str, object], df) -> dict[str, np.ndarray]:
        return {name: models[name].predict_proba(df) for name in self.members}

    def _blend(self, member_probs: dict[str, np.ndarray]) -> np.ndarray:
        out = sum(self.weights_[i] * member_probs[m] for i, m in enumerate(self.members))
        return out / out.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> "EnsembleModel":
        train = train.sort_values("date").reset_index(drop=True)
        n = len(train)

        # ---- 1) OOF predictions via expanding-window temporal folds ----
        bounds = [int(n * (i + 1) / (self.n_folds + 1)) for i in range(self.n_folds + 1)]
        oof_probs = {m: [] for m in self.members}
        oof_y = []
        for f in range(self.n_folds):
            lo, hi = bounds[f], bounds[f + 1]
            tr, va = train.iloc[:lo], train.iloc[lo:hi]
            if len(tr) < 1000 or len(va) == 0:
                continue
            models = self._fit_bases(tr)
            mp = self._member_probs(models, va)
            for m in self.members:
                oof_probs[m].append(mp[m])
            oof_y.append(labels_from_results(va["result"]))
        y = np.concatenate(oof_y)
        oof = {m: np.vstack(oof_probs[m]) for m in self.members}

        # ---- 2) optimise simplex weights to minimise RPS ----
        self.weights_ = self._optimise_weights(oof, y)

        # ---- 3) refit bases on everything for deployment ----
        self.models_ = self._fit_bases(train)

        # ---- 4) optional isotonic calibration on the OOF blend ----
        if self.calibration == "isotonic":
            blend = sum(self.weights_[i] * oof[m] for i, m in enumerate(self.members))
            blend = blend / blend.sum(axis=1, keepdims=True)
            self.iso = []
            for k in range(3):
                ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                ir.fit(blend[:, k], (y == k).astype(float))
                self.iso.append(ir)
        return self

    def _optimise_weights(self, oof: dict, y: np.ndarray) -> np.ndarray:
        mats = [oof[m] for m in self.members]
        k = len(mats)

        def objective(w):
            w = np.clip(w, 0, None)
            s = w.sum()
            if s == 0:
                return 1.0
            w = w / s
            blend = sum(w[i] * mats[i] for i in range(k))
            blend = blend / blend.sum(axis=1, keepdims=True)
            return ranked_probability_score(blend, y)

        starts = [np.full(k, 1 / k)] + list(np.eye(k))
        best, best_val = None, np.inf
        for start in starts:
            res = minimize(objective, start, method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 3000})
            if res.fun < best_val:
                best_val, best = res.fun, res.x
        w = np.clip(best, 0, None)
        return w / w.sum()

    # -------------------------------------------------------------- predict
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        mp = self._member_probs(self.models_, df)
        probs = self._blend(mp)
        if self.iso is not None:
            cal = np.column_stack([self.iso[k].predict(probs[:, k]) for k in range(3)])
            cal = np.clip(cal, 1e-6, None)
            probs = cal / cal.sum(axis=1, keepdims=True)
        return probs

    def base_predictions(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return self._member_probs(self.models_, df)

    @property
    def weights(self) -> dict[str, float]:
        return {m: float(self.weights_[i]) for i, m in enumerate(self.members)}
