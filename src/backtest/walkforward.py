"""Walk-forward backtest over past World Cups.

For each tournament we *freeze* all data at its start date, fit the full ensemble
(and baselines) on everything prior, then predict every match of that World Cup.
This is the honest test: the model only ever sees the past. We report per-model
RPS / log-loss / Brier / accuracy per tournament and pooled, plus a calibration
table on the pooled ensemble predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..models.ensemble import EnsembleModel
from .benchmarks import ClimatologyBaseline, HomePriorBaseline
from .metrics import evaluate, labels_from_results


def _wc_test_rows(features: pd.DataFrame, year: int, start: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.Timedelta(days=40)
    mask = (
        (features["tournament"] == "FIFA World Cup")
        & (features["date"] >= start_ts)
        & (features["date"] <= end_ts)
    )
    return features[mask].copy()


def backtest_world_cups(
    features: pd.DataFrame | None = None, cfg: dict | None = None,
    write: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-model results table, pooled calibration table)."""
    cfg = cfg or load_config()
    if features is None:
        features = pd.read_parquet(path_for("data_processed", cfg) / "features.parquet")

    rows = []
    pooled_probs, pooled_labels = [], []
    for year, start in cfg["backtest"]["world_cups"]:
        train = features[features["date"] < pd.Timestamp(start)].copy()
        test = _wc_test_rows(features, year, start)
        if len(test) == 0:
            print(f"[backtest] {year}: no test matches found, skipping")
            continue

        ens = EnsembleModel(cfg).fit(train)
        clim = ClimatologyBaseline().fit(train)
        homep = HomePriorBaseline().fit(train)

        preds = {
            "climatology": clim.predict_proba(test),
            "home_prior": homep.predict_proba(test),
            **ens.base_predictions(test),
            "ENSEMBLE": ens.predict_proba(test),
        }
        for name, p in preds.items():
            m = evaluate(p, test["result"])
            rows.append({"world_cup": year, "model": name, **m})

        pooled_probs.append(preds["ENSEMBLE"])
        pooled_labels.append(labels_from_results(test["result"]))
        print(f"[backtest] {year}: {len(test)} matches | "
              f"ensemble weights={ {k: round(v,2) for k,v in ens.weights.items()} }")

    table = pd.DataFrame(rows)
    # pooled summary across all world cups (sample-size weighted)
    def _wavg(g: pd.DataFrame) -> pd.Series:
        w = g["n"].to_numpy()
        return pd.Series({
            "rps": np.average(g["rps"], weights=w),
            "log_loss": np.average(g["log_loss"], weights=w),
            "brier": np.average(g["brier"], weights=w),
            "accuracy": np.average(g["accuracy"], weights=w),
            "n": int(g["n"].sum()),
        })

    pooled = (
        pd.concat({m: _wavg(g) for m, g in table.groupby("model")}, axis=1)
        .T.reset_index().rename(columns={"index": "model"})
        .sort_values("rps").reset_index(drop=True)
    )

    calib = _calibration_table(
        np.vstack(pooled_probs), np.concatenate(pooled_labels)
    )

    if write:
        ensure_dirs(cfg)
        rep = path_for("reports", cfg)
        table.to_csv(rep / "backtest_by_worldcup.csv", index=False)
        pooled.to_csv(rep / "backtest_pooled.csv", index=False)
        calib.to_csv(rep / "calibration.csv", index=False)
        print(f"\n[backtest] POOLED across World Cups (lower RPS is better):")
        print(pooled.to_string(index=False,
              formatters={"rps": "{:.4f}".format, "log_loss": "{:.4f}".format,
                          "brier": "{:.4f}".format, "accuracy": "{:.3f}".format}))
    return table, pooled, calib


def _calibration_table(probs: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Reliability: bin predicted prob of the realised class vs observed freq."""
    pred_top = probs.max(axis=1)
    hit = (np.argmax(probs, axis=1) == labels).astype(float)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(pred_top, bins) - 1, 0, 9)
    rows = []
    for b in range(10):
        sel = idx == b
        if sel.sum() == 0:
            continue
        rows.append({
            "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "mean_predicted": float(pred_top[sel].mean()),
            "observed_freq": float(hit[sel].mean()),
            "n": int(sel.sum()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    backtest_world_cups()
