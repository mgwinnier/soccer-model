"""Optuna hyperparameter tuning for the LightGBM model (temporal, leak-free).

Searches LightGBM params to minimise Ranked Probability Score on a *time-ordered*
validation split (train on the past, score the future). Best params persist to
``data/models/gbm_params.json`` and are auto-loaded by ``GBMModel``. Optuna is an
optional dependency — if absent, the model just uses its sensible defaults.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..backtest.metrics import ranked_probability_score, labels_from_results
from .gbm import GBMModel, model_feature_columns


def tune(n_trials: int = 40, cfg: dict | None = None,
         valid_start: str = "2019-01-01", valid_end: str = "2023-01-01") -> dict:
    try:
        import optuna
    except ImportError:
        print("[gbm_tune] optuna not installed — `pip install optuna` to tune.")
        return {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = cfg or load_config()
    feats = pd.read_parquet(path_for("data_processed", cfg) / "features.parquet")
    feats = feats.sort_values("date")
    train = feats[feats["date"] < pd.Timestamp(valid_start)]
    valid = feats[(feats["date"] >= pd.Timestamp(valid_start))
                  & (feats["date"] < pd.Timestamp(valid_end))]
    cols = model_feature_columns(feats)
    y_valid = labels_from_results(valid["result"])

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 900, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 120),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        m = GBMModel(**params).fit(train, features=cols)
        return ranked_probability_score(m.predict_proba(valid), y_valid)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # baseline (default params) for an honest comparison
    base = GBMModel().fit(train, features=cols)
    base_rps = ranked_probability_score(base.predict_proba(valid), y_valid)

    ensure_dirs(cfg)
    out = path_for("models", cfg) / "gbm_params.json"
    if study.best_value < base_rps:
        json.dump(study.best_params, open(out, "w"), indent=2)
        print(f"[gbm_tune] tuned RPS {study.best_value:.4f} < default {base_rps:.4f} "
              f"-> saved {out}")
    else:
        print(f"[gbm_tune] tuned RPS {study.best_value:.4f} not better than default "
              f"{base_rps:.4f} -> keeping defaults (nothing saved)")
    return study.best_params


if __name__ == "__main__":
    tune()
