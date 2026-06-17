"""Forward-validation tracker for the squad-market-value XI signal (2026 World Cup).

Market-value XI strength is principled but **cannot be RPS-backtested** — there's no historical
confirmed-lineup data. So this is the only honest test: as the 2026 WC plays out, does the team
with the **higher-value starting XI** actually win, and do **weakened favorites** (a low share of
their squad value on the pitch) get upset more? Directional, small-sample — every number is
reported with its n and labelled "not validated".

For each played 2026 WC game it reconstructs both confirmed XIs (ESPN ``fetch_lineups``), computes
each side's XI value-share from the committed squad snapshot, and joins the model's pre-match
probabilities + the actual result. Writes ``reports/xi_value_2026.csv`` (read by the Performance
page) and prints the directional table.

Run:  python -m src.backtest.xi_value_2026
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for
from ..predict import value as value_mod
from ..data.odds import fetch_lineups
from ..data import squad_values as sv


def collect(start: str = "2026-06-01", days: int = 60, cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    res = value_mod.build_bets(start, days=days, bankroll=1000, kelly_fraction=0.25,
                               cfg=cfg, use_cache=False)
    rows = []
    for m in res["matches"]:
        if not m.get("played") or not m.get("game_id"):
            continue
        el = fetch_lineups(str(m["game_id"]))
        if not el:
            continue
        hn = [p.get("name") for p in (el.get("home") or {}).get("xi", [])]
        an = [p.get("name") for p in (el.get("away") or {}).get("xi", [])]
        hv, hs = sv.xi_value(m["home"], hn)
        av, as_ = sv.xi_value(m["away"], an)
        if hs is None or as_ is None:
            continue
        probs = m["analysis"]["probs"]
        rows.append({"date": str(m["date"])[:10], "match": f"{m['home']} v {m['away']}",
                     "result": m["result"], "p_home": probs["H"], "p_away": probs["A"],
                     "home_val": hv, "away_val": av, "home_share": hs, "away_share": as_})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    out: dict = {"n": int(len(df))}
    if df.empty:
        return out
    dec = df[df["result"] != "D"].copy()
    out["n_decisive"] = int(len(dec))
    if len(dec):
        val_fav = np.where(dec["home_val"] >= dec["away_val"], "H", "A")
        mdl_fav = np.where(dec["p_home"] >= dec["p_away"], "H", "A")
        out["value_fav_winrate"] = float((val_fav == dec["result"]).mean())
        out["model_fav_winrate"] = float((mdl_fav == dec["result"]).mean())
    # favorite XI value-share -> upset rate (favorite = higher model prob; upset = didn't win)
    fav = np.where(df["p_home"] >= df["p_away"], "H", "A")
    fav_share = np.where(fav == "H", df["home_share"], df["away_share"])
    upset = (fav != df["result"]).to_numpy()
    buckets = []
    for lo, hi in [(0.0, 0.60), (0.60, 0.70), (0.70, 1.01)]:
        mask = (fav_share >= lo) & (fav_share < hi)
        if mask.sum():
            buckets.append({"band": f"{lo*100:.0f}-{min(hi,1)*100:.0f}%",
                            "n": int(mask.sum()), "upset_rate": float(upset[mask].mean())})
    out["fav_share_buckets"] = buckets
    return out


def report(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = collect(cfg=cfg)
    if df.empty:
        print("No played 2026 games with both a confirmed XI and squad values yet.")
        return df
    s = summarize(df)
    p = path_for("reports", cfg) / "xi_value_2026.csv"
    df.to_csv(p, index=False)
    print(f"\nXI value-share forward check — {s['n']} played games "
          f"({s.get('n_decisive', 0)} decisive)")
    if "value_fav_winrate" in s:
        print(f"  higher-VALUE XI won {s['value_fav_winrate']*100:.0f}% of decisive games "
              f"(model favorite won {s['model_fav_winrate']*100:.0f}%)")
    print("  favorite's XI value-share -> upset rate:")
    for b in s.get("fav_share_buckets", []):
        print(f"    {b['band']:>8} share: upset {b['upset_rate']*100:.0f}% (n={b['n']})")
    print("\nDIRECTIONAL — small sample, NOT a validated edge. Accrues as the WC plays out.")
    print(f"wrote {p}")
    return df


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    report()
