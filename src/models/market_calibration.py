"""Per-market probability calibration (isotonic), fit leak-free on history.

The markets backtest showed the derived Totals/Spread/BTTS probabilities carry a
small but systematic bias (the model slightly over-predicts unders / underdog
covers). We remove it with an isotonic map per market type, fit on **out-of-fold,
walk-forward** predictions (each year predicted by a model trained only on earlier
years), pooling across lines. Calibrators persist to ``data/models/`` and are
applied in `MatchPredictor.analyze()` and the betting backtest.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ..config import load_config, path_for
from ..data.clean import load_matches
from ..models.dixon_coles import DixonColesModel
from .market_calib_helpers import market_probs_pooled, outcomes_pooled

_START_YEAR = 2009


class MarketCalibrators:
    """Isotonic maps keyed by market type: 'over', 'cover', 'btts'."""

    def __init__(self, models: dict | None = None):
        self.models = models or {}

    def calibrate(self, market: str, p):
        m = self.models.get(market)
        if m is None:
            return p
        arr = np.atleast_1d(p).astype(float)
        out = np.clip(m.predict(arr), 1e-4, 1 - 1e-4)
        return float(out[0]) if np.ndim(p) == 0 else out

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, path)

    @classmethod
    def load(cls, path: Path) -> "MarketCalibrators":
        return cls(joblib.load(path)) if path.exists() else cls()


def _collect(cfg: dict, as_of: str | None):
    matches = load_matches(cfg).sort_values("date").reset_index(drop=True)
    if as_of:
        matches = matches[matches["date"] < pd.Timestamp(as_of)]
    years = range(_START_YEAR, int(matches["date"].dt.year.max()) + 1)
    pools = {"over": [[], []], "cover": [[], []], "btts": [[], []], "mr": [[], []]}
    for y in years:
        train = matches[matches["date"] < pd.Timestamp(f"{y}-01-01")]
        test = matches[(matches["date"] >= pd.Timestamp(f"{y}-01-01"))
                       & (matches["date"] < pd.Timestamp(f"{y + 1}-01-01"))]
        if len(train) < 2000 or test.empty:
            continue
        dc = DixonColesModel.from_config(cfg).fit(train)
        mp = market_probs_pooled(dc, test)
        oc = outcomes_pooled(test)
        for k in ("over", "cover", "btts"):
            pools[k][0].append(mp[k])
            pools[k][1].append(oc[k])
        # Match Result: pool H/D/A selection probabilities vs whether they hit, so the
        # isotonic learns the favorite-longshot curve (favorites win MORE than the raw
        # model says, longshots LESS) and corrects it regardless of role.
        p3 = dc.predict_proba(test)                 # (n, 3) = [H, D, A]
        res = test["result"].to_numpy()
        pools["mr"][0].append(np.concatenate([p3[:, 0], p3[:, 1], p3[:, 2]]))
        pools["mr"][1].append(np.concatenate(
            [res == "H", res == "D", res == "A"]).astype(float))
    return pools


def fit_calibrators(cfg: dict | None = None, as_of: str | None = None,
                    save: bool = True) -> MarketCalibrators:
    cfg = cfg or load_config()
    pools = _collect(cfg, as_of)
    models = {}
    for k, (ps, ys) in pools.items():
        if not ps:
            continue
        p = np.concatenate(ps)
        y = np.concatenate(ys)
        keep = ~np.isnan(y)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        ir.fit(p[keep], y[keep])
        models[k] = ir
    cal = MarketCalibrators(models)
    if save:
        cal.save(path_for("models", cfg) / "market_calibrators.joblib")
        print(f"[calibration] fit {list(models)} -> "
              f"{path_for('models', cfg) / 'market_calibrators.joblib'}")
    return cal


def load_default(cfg: dict | None = None) -> MarketCalibrators:
    cfg = cfg or load_config()
    return MarketCalibrators.load(path_for("models", cfg) / "market_calibrators.joblib")


if __name__ == "__main__":
    fit_calibrators()
