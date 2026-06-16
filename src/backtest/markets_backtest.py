"""Backtest the *derived* markets — Totals, Spread, BTTS — on the full history.

The 1X2 market is backtested elsewhere; the totals/spread/BTTS probabilities the
betting layer ships were never validated. But we have the goals for every match,
so we can grade them directly. Walk forward year by year: fit the Dixon-Coles
model on everything *before* each year, then for every match that year read
`P(over L)`, `P(home covers L)`, and `P(BTTS)` off the scoreline matrix and grade
against what actually happened.

The headline columns are **mean_pred vs base_rate** (systematic bias) and **ECE**
(calibration error). If the model says "58% under" but unders only land 52% of the
time, that gap is exactly the false "edge" the dashboard was flagging.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..data.clean import load_matches
from ..models.dixon_coles import DixonColesModel
from ..predict.predict_match import _over_prob, _cover_prob
from .metrics import evaluate_binary, reliability_table

TOTAL_LINES = [1.5, 2.5, 3.5]
SPREAD_LINES = [-1.5, -0.5, 0.5, 1.5]      # home handicap lines
START_YEAR = 2009                           # enough history before this to train


def _market_probs(dc: DixonColesModel, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-match model probabilities for every graded market."""
    n = len(test)
    cols = {f"over_{l}": np.empty(n) for l in TOTAL_LINES}
    cols.update({f"cover_{l}": np.empty(n) for l in SPREAD_LINES})
    cols["btts"] = np.empty(n)
    neutral = test["neutral"].astype(bool).to_numpy()
    lam, mu = dc._lambdas(test["home_team"], test["away_team"], neutral)
    for i in range(n):
        mat = dc.scoreline_matrix(lam[i], mu[i])
        for l in TOTAL_LINES:
            cols[f"over_{l}"][i] = _over_prob(mat, l)
        for l in SPREAD_LINES:
            cols[f"cover_{l}"][i] = _cover_prob(mat, l)[0]   # P(home covers)
        cols["btts"][i] = mat[1:, 1:].sum()
    return cols


def _outcomes(test: pd.DataFrame) -> dict[str, np.ndarray]:
    hs = test["home_score"].to_numpy()
    as_ = test["away_score"].to_numpy()
    tot = hs + as_
    margin = hs - as_
    out = {f"over_{l}": (tot > l).astype(float) for l in TOTAL_LINES}
    # home covers line L when margin + L > 0; pushes (integer) dropped from grading
    for l in SPREAD_LINES:
        adj = margin + l
        out[f"cover_{l}"] = np.where(np.abs(adj) < 1e-9, np.nan, (adj > 0).astype(float))
    out["btts"] = ((hs > 0) & (as_ > 0)).astype(float)
    return out


def run_markets_backtest(cfg: dict | None = None, write: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    matches = load_matches(cfg).sort_values("date").reset_index(drop=True)
    years = range(START_YEAR, int(matches["date"].dt.year.max()) + 1)

    preds: dict[str, list] = {}
    obs: dict[str, list] = {}
    for y in years:
        train = matches[matches["date"] < pd.Timestamp(f"{y}-01-01")]
        test = matches[(matches["date"] >= pd.Timestamp(f"{y}-01-01"))
                       & (matches["date"] < pd.Timestamp(f"{y + 1}-01-01"))]
        if len(train) < 2000 or test.empty:
            continue
        dc = DixonColesModel.from_config(cfg).fit(train)
        mp = _market_probs(dc, test)
        oc = _outcomes(test)
        for k in mp:
            preds.setdefault(k, []).append(mp[k])
            obs.setdefault(k, []).append(oc[k])

    rows = []
    reliab = {}
    for k in preds:
        p = np.concatenate(preds[k])
        y = np.concatenate(obs[k])
        keep = ~np.isnan(y)              # drop spread pushes
        p, y = p[keep], y[keep]
        market = ("Total Goals" if k.startswith("over") else
                  "Spread" if k.startswith("cover") else "BTTS")
        line = k.split("_", 1)[1] if "_" in k else ""
        m = evaluate_binary(p, y)
        m.update({"market": market, "selection": k, "line": line,
                  "bias": m["mean_pred"] - m["base_rate"]})
        rows.append(m)
        if k in ("over_2.5", "btts", "cover_-0.5"):
            reliab[k] = reliability_table(p, y)

    table = pd.DataFrame(rows)[
        ["market", "selection", "line", "n", "mean_pred", "base_rate", "bias",
         "brier", "ece", "log_loss"]]

    if write:
        ensure_dirs(cfg)
        out = path_for("reports", cfg) / "markets_calibration.csv"
        table.to_csv(out, index=False)
        for k, r in reliab.items():
            r.to_csv(path_for("reports", cfg) / f"reliability_{k}.csv", index=False)
        print(f"[markets] wrote {out}\n")
        print(table.to_string(index=False, formatters={
            "mean_pred": "{:.3f}".format, "base_rate": "{:.3f}".format,
            "bias": "{:+.3f}".format, "brier": "{:.4f}".format,
            "ece": "{:.4f}".format, "log_loss": "{:.4f}".format}))
    return table


if __name__ == "__main__":
    run_markets_backtest()
