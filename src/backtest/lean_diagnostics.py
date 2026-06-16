"""Diagnose the model's underdog/draw/under lean — measure before fixing (v11).

Reuses the leak-free historical predictions (`odds_history_backtest.build_predictions`)
and reports, on the out-of-sample test slice:

  1. **+EV split** — of the bets the model currently flags (ev ≥ min_ev after the
     deployed recentering), what share are favorites vs underdogs, over vs under.
     This is the imbalance the user sees.
  2. **Reliability by odds bucket** — mean model probability vs realized win-rate,
     bucketed favorite → longshot. Shows where the model is over/under-confident
     (expect: over-confident on longshots, hence the junk +EV flags).
  3. **Totals bias** — mean model P(over) vs the actual over rate (the real
     under-bias to correct).

Run:  python -m src.backtest.lean_diagnostics
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..models.market_bias import fit_market_bias
from .odds_history_backtest import build_predictions


def _bucket(dec: float) -> str:
    if dec < 1.6:
        return "1 heavy fav (<1.6)"
    if dec < 2.0:
        return "2 fav (1.6-2.0)"
    if dec < 3.0:
        return "3 pickem (2.0-3.0)"
    if dec < 5.0:
        return "4 dog (3.0-5.0)"
    return "5 longshot (5.0+)"


def run(cfg: dict | None = None, split: str = "2023-01-01",
        min_ev: float = 0.03, write: bool = True) -> dict:
    cfg = cfg or load_config()
    preds = build_predictions(cfg, anchor_w=0.5, split=split)
    if preds.empty:
        print("[lean] no predictions — harvest odds first")
        return {}
    train = preds[preds["date"] < pd.Timestamp(split)]
    test = preds[preds["date"] >= pd.Timestamp(split)].copy()
    bias = fit_market_bias(train["type"], train["model_p"], train["fair_p"])
    test["adj"] = [bias.recenter(t, p) for t, p in zip(test["type"], test["model_p"])]
    test["ev"] = test["adj"] * (test["dec"] - 1) - (1 - test["adj"])
    test["won"] = (test["result"] == "win").astype(float)

    flagged = test[test["ev"] >= min_ev]
    mr = flagged[flagged["type"].str.startswith("MR")]
    n_fav = int((mr["dec"] < 2.0).sum())
    n_dog = int((mr["dec"] >= 2.0).sum())
    n_over = int((flagged["type"] == "TG:over").sum())
    n_under = int((flagged["type"] == "TG:under").sum())
    split_tbl = {
        "MR favorite (<2.0)": n_fav, "MR underdog (>=2.0)": n_dog,
        "TG over": n_over, "TG under": n_under,
        "MR:H": int((flagged["type"] == "MR:H").sum()),
        "MR:D": int((flagged["type"] == "MR:D").sum()),
        "MR:A": int((flagged["type"] == "MR:A").sum()),
    }

    mrall = test[test["type"].str.startswith("MR")].copy()
    mrall["bucket"] = mrall["dec"].map(_bucket)
    rel = (mrall.groupby("bucket")
           .agg(n=("won", "size"), model_p=("adj", "mean"), actual=("won", "mean"))
           .reset_index().sort_values("bucket"))
    rel["gap"] = rel["model_p"] - rel["actual"]

    over = test[test["type"] == "TG:over"]
    tot = {"n": int(len(over)),
           "model_over": float(over["adj"].mean()) if len(over) else float("nan"),
           "actual_over": float(over["won"].mean()) if len(over) else float("nan")}
    tot["bias"] = tot["model_over"] - tot["actual_over"]

    # ---- effect of the probability-edge gate (the v11 fix) — sweep settings ----
    from ..predict.betting import qualifies

    def _roi(df):
        s = df[df["result"] != "push"].dropna(subset=["dec"])
        if len(s) == 0:
            return float("nan"), 0
        pnl = np.where(s["result"] == "win", s["dec"] - 1, -1.0)
        return float(pnl.mean()), len(s)

    sweep = []
    for maxd in (8.0, 5.0, 4.0, 3.0):
        for edge in (0.01, 0.02, 0.03):
            mask = [qualifies(p, f, d, min_ev=min_ev, min_prob_edge=edge, max_decimal=maxd)
                    for p, f, d in zip(test["adj"], test["fair_p"], test["dec"])]
            gg = test[mask]
            gmr = gg[gg["type"].str.startswith("MR")]
            roi, n = _roi(gg)
            sweep.append({"max_dec": maxd, "min_edge": edge, "n": len(gg),
                          "dog%": round(100 * (gmr["dec"] >= 2.0).mean(), 0) if len(gmr) else 0,
                          "roi%": round(roi * 100, 1), "n_settled": n})
    gate = {"before": _roi(flagged), "sweep": pd.DataFrame(sweep)}

    if write:
        ensure_dirs(cfg)
        rel.to_csv(path_for("reports", cfg) / "lean_diagnostics.csv", index=False)
    return {"split": split_tbl, "reliability": rel, "totals": tot, "gate": gate,
            "n_flagged": len(flagged), "n_test": len(test)}


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    r = run()
    if not r:
        raise SystemExit
    print(f"\n=== LEAN DIAGNOSTIC  ({r['n_flagged']} flagged +EV bets of {r['n_test']} test) ===\n")
    s = r["split"]
    mr_tot = s["MR favorite (<2.0)"] + s["MR underdog (>=2.0)"]
    tg_tot = s["TG over"] + s["TG under"]
    print("1) WHERE THE +EV FLAGS LAND:")
    if mr_tot:
        print(f"   Match Result: {s['MR favorite (<2.0)']} favorites vs "
              f"{s['MR underdog (>=2.0)']} underdogs  "
              f"({s['MR underdog (>=2.0)']/mr_tot*100:.0f}% dogs)")
    if tg_tot:
        print(f"   Totals: {s['TG over']} over vs {s['TG under']} under  "
              f"({s['TG under']/tg_tot*100:.0f}% under)")
    print(f"   By role: H={s['MR:H']}  D={s['MR:D']}  A={s['MR:A']}")
    print("\n2) RELIABILITY by odds bucket (model prob vs actual win rate; +gap = over-confident):")
    for _, row in r["reliability"].iterrows():
        print(f"   {row['bucket']:20s} n={int(row['n']):4d}  model {row['model_p']*100:5.1f}%  "
              f"actual {row['actual']*100:5.1f}%  gap {row['gap']*100:+5.1f}pp")
    t = r["totals"]
    print(f"\n3) TOTALS bias: model P(over) {t['model_over']*100:.1f}% vs actual "
          f"{t['actual_over']*100:.1f}%  → bias {t['bias']*100:+.1f}pp "
          f"({'under' if t['bias']<0 else 'over'}-predicts overs)")
    gt = r["gate"]
    rb, nb = gt["before"]
    print(f"\n4) PROBABILITY-EDGE GATE — sweep (BEFORE: {rb*100:+.1f}% ROI on {nb} settled bets):")
    print(gt["sweep"].to_string(index=False))
    print("\nRead: the model is broadly well-calibrated (the 1X2 calibrator came out ≈ identity); "
          "the dog/under lean is EV-leverage on long odds. The gate demands a real disagreement, "
          "rebalancing the flags and cutting the losing longshot bets.")
