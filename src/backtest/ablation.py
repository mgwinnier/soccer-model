"""Ablation study — prove (or disprove) that each addition lowers RPS.

Two walk-forward ablations over the past World Cups (data frozen at each kickoff):

  A) **Ensemble members** — Elo only → DC only → DC+Elo+GBM (v1) → +xG-DC (v2).
  B) **GBM feature blocks** — cumulatively add elo → form → squad → context →
     xG/style and measure the GBM's pooled RPS.

Nothing is kept on faith: if a block doesn't lower RPS, this is the evidence to
drop it. Results are written to ``reports/ablation.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..models.ensemble import EnsembleModel
from ..models.gbm import GBMModel, model_feature_columns
from .metrics import ranked_probability_score, labels_from_results

# GBM feature blocks, identified by column-name predicates (cumulative).
_BLOCKS = [
    ("elo", lambda c: c.startswith("elo_") or c in ("home_elo", "away_elo")),
    ("form", lambda c: any(c.startswith(p) for p in ("ppg_", "gf_", "ga_", "rest"))
        or c.endswith(("_5", "_10", "_20"))),
    ("squad", lambda c: "squad_value" in c),
    ("context", lambda c: any(k in c for k in
        ("travel", "altitude", "h2h", "neutral", "cross_conf", "importance"))),
    ("xg_style", lambda c: c.startswith("xg_") or "xgf" in c or "xga" in c),
]


def _wc_rows(features: pd.DataFrame, start: str) -> pd.DataFrame:
    s = pd.Timestamp(start)
    mask = ((features["tournament"] == "FIFA World Cup")
            & (features["date"] >= s) & (features["date"] <= s + pd.Timedelta(days=40)))
    return features[mask].copy()


def _pooled_rps(preds: list[np.ndarray], labels: list[np.ndarray]) -> float:
    return ranked_probability_score(np.vstack(preds), np.concatenate(labels))


def ablate_members(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    variants = {
        "Elo only": ["elo"],
        "DC goals only": ["dixon_coles"],
        "DC+Elo+GBM (v1)": ["dixon_coles", "elo", "gbm"],
        "+ xG-DC (v2)": ["dixon_coles", "xg_dixon_coles", "elo", "gbm"],
    }
    pooled = {name: ([], []) for name in variants}
    for _, start in cfg["backtest"]["world_cups"]:
        train = features[features["date"] < pd.Timestamp(start)]
        test = _wc_rows(features, start)
        if test.empty:
            continue
        y = labels_from_results(test["result"])
        for name, members in variants.items():
            ens = EnsembleModel(cfg, members=members).fit(train)
            pooled[name][0].append(ens.predict_proba(test))
            pooled[name][1].append(y)
    rows = [{"ablation": "members", "variant": name,
             "rps": _pooled_rps(p, l), "n": int(sum(len(a) for a in l))}
            for name, (p, l) in pooled.items() if p]
    return pd.DataFrame(rows)


def ablate_gbm_features(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    all_feats = model_feature_columns(features)
    cumulative: list[str] = []
    block_sets = []
    for name, pred in _BLOCKS:
        cols = [c for c in all_feats if pred(c) and c not in cumulative]
        cumulative = cumulative + cols
        block_sets.append((name, list(cumulative)))

    pooled = {name: ([], []) for name, _ in block_sets}
    for _, start in cfg["backtest"]["world_cups"]:
        train = features[features["date"] < pd.Timestamp(start)]
        test = _wc_rows(features, start)
        if test.empty:
            continue
        y = labels_from_results(test["result"])
        for name, cols in block_sets:
            if not cols:
                continue
            gbm = GBMModel().fit(train, features=cols)
            pooled[name][0].append(gbm.predict_proba(test))
            pooled[name][1].append(y)
    rows = [{"ablation": "gbm_features", "variant": f"+{name}",
             "rps": _pooled_rps(p, l), "n": int(sum(len(a) for a in l))}
            for name, (p, l) in pooled.items() if p]
    return pd.DataFrame(rows)


def run_ablation(cfg: dict | None = None, write: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    features = pd.read_parquet(path_for("data_processed", cfg) / "features.parquet")
    members = ablate_members(features, cfg)
    gbm_blocks = ablate_gbm_features(features, cfg)
    out = pd.concat([members, gbm_blocks], ignore_index=True)
    if write:
        ensure_dirs(cfg)
        p = path_for("reports", cfg) / "ablation.csv"
        out.to_csv(p, index=False)
        print(f"[ablation] wrote {p}\n")
        print(out.to_string(index=False, formatters={"rps": "{:.4f}".format}))
    return out


if __name__ == "__main__":
    run_ablation()
