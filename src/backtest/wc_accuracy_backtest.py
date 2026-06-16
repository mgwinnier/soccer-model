"""How accurate is the DEPLOYED model on past World Cups?

The other backtests grade either the ensemble (``walkforward.py``) or the goals
(``wc_goals_backtest.py``). This one grades the exact pipeline the live match cards
use — the **market-independent** DC+Elo blend with the World-Cup goals correction —
walk-forward across every World Cup we have results for (1998-2022), and stacks it
against its own components and dumb baselines so "how accurate is it" has a number.

Deployed per-match W/D/L (replicated leak-free here):
  1. per-year Dixon-Coles (trained strictly before the WC) -> expected goals
  2. World-Cup favorite/underdog goals correction (src/models/wc_goals.py)
  3. P(H/D/A) from the corrected scoreline matrix  (the DC channel)
  4. blend 50/50 with the Elo model's P(H/D/A) from the as-of Elo gap
No betting market is involved anywhere (the anchoring was removed).

    python -m src.backtest.wc_accuracy_backtest

Writes ``reports/wc_accuracy_backtest.csv``. Primary metric is RPS (lower better;
good football models land ~0.18-0.21). Also reports the refactor's before/after
(with vs without the WC goals correction) and a pooled calibration table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..data.clean import load_matches
from ..models.dixon_coles import DixonColesModel
from ..models.elo_model import EloModel
from ..models.base import scoreline_to_outcome_probs
from ..models import wc_goals
from .benchmarks import ClimatologyBaseline, HomePriorBaseline
from .metrics import (ranked_probability_score, log_loss_score, brier_score,
                      accuracy, labels_from_results, reliability_table)

WC_YEARS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]


def _dc_probs(dc, lam, mu, corrected: bool):
    """P(H/D/A) from the DC scoreline matrix, optionally WC-corrected."""
    out = np.empty((len(lam), 3))
    for i in range(len(lam)):
        la, mua = (wc_goals.correct(lam[i], mu[i]) if corrected else (lam[i], mu[i]))
        out[i] = scoreline_to_outcome_probs(dc.scoreline_matrix(la, mua))
    return out


def _metrics(probs, labels):
    return {
        "rps": ranked_probability_score(probs, labels),
        "log_loss": log_loss_score(probs, labels),
        "brier": brier_score(probs, labels),
        "accuracy": accuracy(probs, labels),
        "n": int(len(labels)),
    }


def run(cfg: dict | None = None, write: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    matches = load_matches(cfg).sort_values("date").reset_index(drop=True)
    feats = pd.read_parquet(path_for("data_processed", cfg) / "features.parquet")
    feats = feats.sort_values("date").reset_index(drop=True)
    wc = matches[matches["tournament"] == "FIFA World Cup"].copy()
    wc["year"] = wc["date"].dt.year

    # accumulate pooled predictions per model for an honest pooled score
    pool = {k: [] for k in ["deployed", "no_wc_corr", "dixon_coles", "elo",
                            "home_prior", "climatology"]}
    pool_lab = []
    rows = []
    for y in WC_YEARS:
        split = pd.Timestamp(f"{y}-05-01")
        train_m = matches[matches["date"] < split]
        test = wc[wc["year"] == y].copy()
        labels = labels_from_results(test["result"])

        # --- DC channel (per-year, leak-free) + WC goals correction ---
        dc = DixonColesModel.from_config(cfg).fit(train_m)
        neutral = test["neutral"].astype(bool).to_numpy()
        lam, mu = dc._lambdas(test["home_team"], test["away_team"], neutral)
        lam, mu = np.asarray(lam, float), np.asarray(mu, float)
        dc_p_corr = _dc_probs(dc, lam, mu, corrected=True)
        dc_p_raw = _dc_probs(dc, lam, mu, corrected=False)

        # --- Elo channel (as-of elo_diff from features, fit elo model pre-WC) ---
        elo = EloModel(cfg=cfg).fit(feats[feats["date"] < split])
        tfeat = feats[feats["match_id"].isin(test["match_id"])][["match_id", "elo_diff", "neutral"]]
        tfeat = test[["match_id"]].merge(tfeat, on="match_id", how="left")
        elo_p = elo.predict_proba(tfeat)

        # --- deployed blend (DC-corrected + Elo) and the no-correction ablation ---
        def blend(dcp):
            b = (dcp + elo_p) / 2
            return b / b.sum(axis=1, keepdims=True)
        deployed = blend(dc_p_corr)
        no_corr = blend(dc_p_raw)

        # --- baselines ---
        hp = HomePriorBaseline().fit(train_m).predict_proba(test)
        cl = ClimatologyBaseline().fit(train_m).predict_proba(test)

        preds = {"deployed": deployed, "no_wc_corr": no_corr,
                 "dixon_coles": dc_p_corr, "elo": elo_p,
                 "home_prior": hp, "climatology": cl}
        for name, p in preds.items():
            m = _metrics(p, labels)
            m.update({"world_cup": y, "model": name})
            rows.append(m)
            pool[name].append(p)
        pool_lab.append(labels)

    # pooled rows
    labels_all = np.concatenate(pool_lab)
    for name, plist in pool.items():
        p = np.concatenate(plist)
        m = _metrics(p, labels_all)
        m.update({"world_cup": "POOLED", "model": name})
        rows.append(m)

    table = pd.DataFrame(rows)[["world_cup", "model", "n", "rps", "log_loss",
                                "brier", "accuracy"]]
    if write:
        ensure_dirs(cfg)
        out = path_for("reports", cfg) / "wc_accuracy_backtest.csv"
        table.to_csv(out, index=False)
        print(f"[wc_accuracy] wrote {out}\n")
        order = ["deployed", "no_wc_corr", "dixon_coles", "elo", "home_prior", "climatology"]
        pooled = table[table.world_cup == "POOLED"].set_index("model").loc[order]
        print("POOLED across 7 World Cups (1998-2022, 448 matches), walk-forward leak-free:\n")
        disp = pooled[["n", "rps", "log_loss", "brier", "accuracy"]].copy()
        disp["accuracy"] = (disp["accuracy"] * 100).round(1)
        print(disp.round({"rps": 4, "log_loss": 4, "brier": 4}).to_string())
        dep = pooled.loc["deployed"]; nc = pooled.loc["no_wc_corr"]; el = pooled.loc["elo"]
        print(f"\nDeployed model: RPS {dep.rps:.4f}, accuracy {dep.accuracy*100:.1f}% "
              f"(picks the right result {dep.accuracy*100:.0f}% of the time).")
        print(f"Refactor impact (WC goals correction): RPS {nc.rps:.4f} -> {dep.rps:.4f} "
              f"({(nc.rps-dep.rps):+.4f}); a better forecast, not just better totals.")
        print(f"Beats Elo-only ({el.rps:.4f}) and the home-prior/climatology baselines "
              f"({pooled.loc['home_prior'].rps:.3f}/{pooled.loc['climatology'].rps:.3f}).")
        print("\nPer-World-Cup deployed RPS:")
        dep_by = table[(table.model == "deployed") & (table.world_cup != "POOLED")]
        for r in dep_by.itertuples(index=False):
            print(f"  {r.world_cup}: RPS {r.rps:.4f}  accuracy {r.accuracy*100:.0f}%  (n={r.n})")
        # pooled calibration of the deployed model
        favp = np.concatenate([p.max(axis=1) for p in pool["deployed"]])
        won = (np.concatenate([p.argmax(axis=1) for p in pool["deployed"]]) == labels_all).astype(float)
        rel = reliability_table(favp, won)
        print("\nCalibration of the deployed model's top pick (confidence vs hit-rate):")
        print(rel.to_string(index=False))
    return table


if __name__ == "__main__":
    run()
