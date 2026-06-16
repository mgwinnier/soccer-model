"""Proper scoring rules for three-way (H/D/A) match predictions.

The **Ranked Probability Score** is the primary football metric: unlike log loss
it is *sensitive to ordinal distance* — predicting a draw when the home team wins
is penalised less than predicting an away win, which matches the natural
Home-win > Draw > Away-win ordering. Lower is better for every metric here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.base import OUTCOME_INDEX

_EPS = 1e-15


def _onehot(labels: np.ndarray) -> np.ndarray:
    oh = np.zeros((len(labels), 3))
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


def ranked_probability_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean RPS over ordered categories [H, D, A]. Range [0, 1], lower better."""
    e = _onehot(labels)
    cp = np.cumsum(probs[:, :-1], axis=1)   # cumulative predicted (first r-1)
    ce = np.cumsum(e[:, :-1], axis=1)        # cumulative observed
    return float(np.mean(np.sum((cp - ce) ** 2, axis=1) / (probs.shape[1] - 1)))


def log_loss_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], _EPS, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    e = _onehot(labels)
    return float(np.mean(np.sum((probs - e) ** 2, axis=1)))


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == labels))


def labels_from_results(results: pd.Series) -> np.ndarray:
    return results.map(OUTCOME_INDEX).to_numpy()


# ---------------------------------------------------------------- binary markets
def binary_brier(p: np.ndarray, y: np.ndarray) -> float:
    """Brier score for a yes/no market. p = P(yes), y in {0,1}. Lower is better."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def binary_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Calibration table: mean predicted vs observed frequency per probability bin."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "mean_predicted": float(p[sel].mean()),
            "observed_freq": float(y[sel].mean()),
            "n": int(sel.sum()),
        })
    return pd.DataFrame(rows)


def calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: n-weighted mean |predicted − observed| over bins."""
    tbl = reliability_table(p, y, n_bins)
    if tbl.empty:
        return float("nan")
    w = tbl["n"] / tbl["n"].sum()
    return float((w * (tbl["mean_predicted"] - tbl["observed_freq"]).abs()).sum())


def evaluate_binary(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """All binary-market metrics at once."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return {
        "brier": binary_brier(p, y),
        "log_loss": binary_log_loss(p, y),
        "ece": calibration_error(p, y),
        "mean_pred": float(p.mean()),
        "base_rate": float(y.mean()),
        "n": int(len(y)),
    }


def evaluate(probs: np.ndarray, results: pd.Series) -> dict[str, float]:
    """All metrics at once. ``results`` is a Series of 'H'/'D'/'A'."""
    y = labels_from_results(results)
    return {
        "rps": ranked_probability_score(probs, y),
        "log_loss": log_loss_score(probs, y),
        "brier": brier_score(probs, y),
        "accuracy": accuracy(probs, y),
        "n": int(len(y)),
    }
