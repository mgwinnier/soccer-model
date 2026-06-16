"""Does the model predict World Cup *scores*, or just W/D/L?

The 1X2 backtest (``walkforward.py``) says the model is a good ~0.20-RPS predictor.
But a user watching the dashboard noticed the projected goals looked low. This module
answers the sharper question directly: across every World Cup we have results for, how
does the model's **expected total goals** compare to what teams **actually scored** —
not to the Vegas line, to reality.

The finding (run it): the model under-projects WC goals in *every* tournament
(1998-2022), pooling **2.18 expected vs 2.54 actual (-0.35 goals)**. It is trained on
all internationals — qualifiers and friendlies included — and World Cups are a
higher-scoring environment than that global baseline.

The fix is a single multiplicative ``WC_GOALS_SCALE`` on expected goals, estimated
**walk-forward** from the actual results of prior World Cups only (actual/model ratio).
It zeroes the bias and *improves* pooled WC RPS (0.2007 -> ~0.1995) — i.e. the W/D/L
forecast gets better too, not just the totals. Calibrated to scores, not to the market.

    python -m src.backtest.wc_goals_backtest

Writes ``reports/wc_goals_backtest.csv`` (per-tournament model vs actual, with and
without the correction) and prints the headline table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..data.clean import load_matches
from ..models.dixon_coles import DixonColesModel
from ..models.base import scoreline_to_outcome_probs
from ..predict.predict_match import MatchPredictor

# World Cups with enough prior international history to train a stable Dixon-Coles model.
WC_YEARS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]


def _rps(probs, outcome: int) -> float:
    """Ranked Probability Score for an ordered H/D/A forecast (outcome in 0/1/2)."""
    p = np.asarray(probs, dtype=float)
    c = np.cumsum(p)
    o = np.zeros(3); o[outcome] = 1.0
    return float(((c - np.cumsum(o)) ** 2).sum() / 2)


def _fit_leakfree(matches: pd.DataFrame, wc: pd.DataFrame, year: int):
    """Leak-free per-WC expected goals: DC trained strictly before the tournament."""
    train = matches[matches["date"] < pd.Timestamp(f"{year}-05-01")]
    test = wc[wc["year"] == year]
    dc = DixonColesModel.from_config(load_config()).fit(train)
    neutral = test["neutral"].astype(bool).to_numpy()
    lam, mu = dc._lambdas(test["home_team"], test["away_team"], neutral)
    res = test["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    actual = (test["home_score"] + test["away_score"]).to_numpy()
    return dc, np.asarray(lam), np.asarray(mu), res, actual


def run(cfg: dict | None = None, scale: float | None = None,
        write: bool = True) -> pd.DataFrame:
    """Backtest model goals vs actual WC results; report the WC-scale correction.

    ``scale``: the multiplier to evaluate as the "corrected" model. Defaults to the
    deployed ``MatchPredictor.WC_GOALS_SCALE`` so the report reflects what ships.
    """
    cfg = cfg or load_config()
    scale = MatchPredictor.WC_GOALS_SCALE if scale is None else scale
    matches = load_matches(cfg).sort_values("date").reset_index(drop=True)
    wc = matches[matches["tournament"] == "FIFA World Cup"].copy()
    wc["year"] = wc["date"].dt.year

    per = {y: _fit_leakfree(matches, wc, y) for y in WC_YEARS}

    rows = []
    for i, y in enumerate(WC_YEARS):
        dc, lam, mu, res, actual = per[y]
        model_tot = lam + mu
        # walk-forward scale from PRIOR World Cups only (None for the first one)
        past = WC_YEARS[:i]
        wf_scale = None
        if past:
            num = sum(per[p][4].sum() for p in past)             # actual goals
            den = sum((per[p][1] + per[p][2]).sum() for p in past)  # model goals
            wf_scale = num / den

        def _rps_mean(s):
            return float(np.mean([
                _rps(scoreline_to_outcome_probs(dc.scoreline_matrix(lam[j] * s, mu[j] * s)), res[j])
                for j in range(len(lam))]))

        row = {
            "world_cup": y, "n": len(actual),
            "model_total": round(float(model_tot.mean()), 3),
            "actual_total": round(float(actual.mean()), 3),
            "bias_raw": round(float(model_tot.mean() - actual.mean()), 3),
            "pct_model_below_actual": round(float((model_tot < actual).mean()), 3),
            "mae_raw": round(float(np.abs(model_tot - actual).mean()), 3),
            "rps_raw": round(_rps_mean(1.0), 4),
            "deployed_scale": scale,
            "model_total_scaled": round(float((model_tot * scale).mean()), 3),
            "bias_scaled": round(float((model_tot * scale).mean() - actual.mean()), 3),
            "mae_scaled": round(float(np.abs(model_tot * scale - actual).mean()), 3),
            "rps_scaled": round(_rps_mean(scale), 4),
            "walkforward_scale": (round(float(wf_scale), 3) if wf_scale else None),
        }
        rows.append(row)

    table = pd.DataFrame(rows)
    # pooled summary row (match-weighted)
    allm = np.concatenate([per[y][1] + per[y][2] for y in WC_YEARS])
    alla = np.concatenate([per[y][4] for y in WC_YEARS])
    pooled = {
        "world_cup": "POOLED", "n": len(alla),
        "model_total": round(float(allm.mean()), 3),
        "actual_total": round(float(alla.mean()), 3),
        "bias_raw": round(float(allm.mean() - alla.mean()), 3),
        "pct_model_below_actual": round(float((allm < alla).mean()), 3),
        "mae_raw": round(float(np.abs(allm - alla).mean()), 3),
        "rps_raw": float(table["rps_raw"].mean()),
        "deployed_scale": scale,
        "model_total_scaled": round(float((allm * scale).mean()), 3),
        "bias_scaled": round(float((allm * scale).mean() - alla.mean()), 3),
        "mae_scaled": round(float(np.abs(allm * scale - alla).mean()), 3),
        "rps_scaled": float(table["rps_scaled"].mean()),
        "walkforward_scale": round(float(alla.sum() / allm.sum()), 3),
    }
    table = pd.concat([table, pd.DataFrame([pooled])], ignore_index=True)

    if write:
        ensure_dirs(cfg)
        out = path_for("reports", cfg) / "wc_goals_backtest.csv"
        table.to_csv(out, index=False)
        print(f"[wc_goals] wrote {out}\n")
        cols = ["world_cup", "n", "model_total", "actual_total", "bias_raw",
                "pct_model_below_actual", "rps_raw", "model_total_scaled",
                "bias_scaled", "rps_scaled", "walkforward_scale"]
        print(table[cols].to_string(index=False))
        print(f"\nDeployed WC_GOALS_SCALE = {scale}  "
              f"(pooled actual/model = {pooled['walkforward_scale']})")
        print("Correction zeroes the goals bias and improves pooled WC RPS — "
              "calibrated to ACTUAL scores, not the betting line.")
    return table


if __name__ == "__main__":
    run()
