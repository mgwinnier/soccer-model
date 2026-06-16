"""Stress-test the probability-edge gate — especially out-of-sample on the 2022 WC.

The gate (`betting.qualifies`: EV ≥ min_ev AND model_p − fair_p ≥ min_edge AND
decimal ≤ max_decimal) was tuned on the **2023+** international slice. The **2022
World Cup is a clean held-out set** — the gate never saw it — so it's the honest
test of whether the gate's apparent benefit replicates, or was overfit.

Three views:
  1. 2022 WC: ungated vs gated ROI (overall + by market) with bootstrap CIs.
  2. Robustness: does the gain hold across a grid of (min_edge, max_decimal), or is
     it a knife-edge at the tuned values?
  3. The honest verdict — one tournament is still a tiny, variance-heavy sample.

Run:  python -m src.backtest.gate_stress
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..models.market_bias import fit_market_bias
from ..predict.betting import qualifies
from .odds_history_backtest import build_predictions, _bootstrap_roi


def _grade(df: pd.DataFrame):
    s = df[df["result"] != "push"]
    if len(s) == 0:
        return None
    pnl = np.where(s["result"] == "win", s["dec"] - 1, -1.0)
    m, lo, hi = _bootstrap_roi(pnl)
    return {"bets": len(df), "wins": int((df["result"] == "win").sum()),
            "roi": m, "lo": lo, "hi": hi, "units": float(pnl.sum())}


def _gmask(df, min_ev, edge, maxd):
    return [qualifies(p, f, d, min_ev, edge, maxd)
            for p, f, d in zip(df["adj"], df["fair_p"], df["dec"])]


def stress(cfg: dict | None = None, split: str = "2022-06-01", min_ev: float = 0.03,
           min_edge: float = 0.02, max_dec: float = 6.0) -> dict:
    cfg = cfg or load_config()
    preds = build_predictions(cfg, anchor_w=0.5, split=split)
    if preds.empty:
        return {}
    train = preds[preds["date"] < pd.Timestamp(split)]
    bias = fit_market_bias(train["type"], train["model_p"], train["fair_p"])
    wc = preds[(preds["league"] == "fifa.world") & (preds["date"].dt.year == 2022)
               & (preds["date"] >= pd.Timestamp(split))].copy()
    wc["adj"] = [bias.recenter(t, p) for t, p in zip(wc["type"], wc["model_p"])]
    wc["ev"] = wc["adj"] * (wc["dec"] - 1) - (1 - wc["adj"])

    ungated = wc[wc["ev"] >= min_ev]
    gated = wc[_gmask(wc, min_ev, min_edge, max_dec)]
    out = {"ungated": _grade(ungated), "gated": _grade(gated),
           "by_market": {}, "n_wc": len(wc)}
    for mk in ["Match Result", "Total Goals", "Spread"]:
        out["by_market"][mk] = {
            "ungated": _grade(ungated[ungated["market"] == mk]),
            "gated": _grade(gated[gated["market"] == mk])}

    sweep = []
    for me in (0.01, 0.02, 0.03, 0.04):
        for md in (4.0, 5.0, 6.0, 8.0):
            r = _grade(wc[_gmask(wc, min_ev, me, md)])
            if r:
                sweep.append({"min_edge": me, "max_dec": md, "bets": r["bets"],
                              "roi%": round(r["roi"] * 100, 1),
                              "ci_lo%": round(r["lo"] * 100, 1)})
    out["sweep"] = pd.DataFrame(sweep)
    return out


def _fmt(r):
    if not r:
        return "—"
    return (f"{r['bets']:3d} bets, {r['wins']}W, {r['units']:+.1f}u, "
            f"ROI {r['roi']*100:+.1f}% [CI {r['lo']*100:+.1f}, {r['hi']*100:+.1f}]")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    r = stress()
    if not r:
        raise SystemExit("no predictions")
    print(f"\n=== GATE STRESS TEST — 2022 World Cup (held-out; gate tuned on 2023+) ===")
    print(f"({r['n_wc']} candidate WC bets)\n")
    print("OVERALL")
    print("  ungated :", _fmt(r["ungated"]))
    print("  GATED   :", _fmt(r["gated"]))
    for mk, d in r["by_market"].items():
        print(f"\n{mk}")
        print("  ungated :", _fmt(d["ungated"]))
        print("  GATED   :", _fmt(d["gated"]))
    print("\nRobustness across gate settings (does the gain hold or is it a knife-edge?):")
    print(r["sweep"].to_string(index=False))
    print("\nHonest read: a +ROI that HOLDS across the grid AND on this held-out tournament "
          "is supportive; but it's ONE tournament (tiny n, wide CIs) — not proof. The CI "
          "must clear 0 to claim an edge.")
