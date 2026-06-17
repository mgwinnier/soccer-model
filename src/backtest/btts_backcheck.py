"""Back-check the model's Both-Teams-To-Score calls on the 2026 World Cup so far.

The model has always computed P(BTTS) but never had a line to price it against. TheStatsAPI now
provides the real Bet365 BTTS yes/no (closing) for every played 2026 match, so we can finally ask
the honest question: **does betting the model's BTTS pick actually have value at the real price?**

What it does (leak-aware, honest about the tiny sample):
  * pulls every *played* 2026 WC match with the model's market-independent P(BTTS) and the real
    BTTS closing line (one ``/odds`` per game via the fixture mapper),
  * grades the model's *recommended* side (the +EV one under the deployed gate) AND, separately,
    a flat both-sides baseline, at the offered decimal price,
  * reports n, record, ROI (flat + quarter-Kelly units) with a bootstrap 95% CI, and a calibration
    line (mean model P(yes) vs the actual BTTS rate).

Sample size is ~20-50 games — far too small to claim an edge. The verdict line says so explicitly;
a CI through zero is reported as "no demonstrated edge (variance)", never spun as a system.

Run:  python -m src.backtest.btts_backcheck
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for
from ..predict import value as value_mod
from ..predict.betting import qualifies
from .odds_history_backtest import _bootstrap_roi


def _pnl(dec: float, won: bool) -> float:
    return (dec - 1.0) if won else -1.0


def collect(start: str = "2026-06-01", days: int = 60, cfg=None) -> pd.DataFrame:
    """One row per (played match × BTTS side) with model_p, decimal, fair, EV, won."""
    cfg = cfg or load_config()
    res = value_mod.build_bets(start, days=days, bankroll=1000, kelly_fraction=0.25,
                               cfg=cfg, use_cache=False, api_markets=True)
    rows = []
    for m in res["matches"]:
        if not m.get("played"):
            continue
        both = (m["home_score"] > 0 and m["away_score"] > 0)
        for b in m["bets"]:
            if b.market != "BTTS":
                continue
            is_yes = "Yes" in b.selection
            won = both if is_yes else (not both)
            rows.append({
                "date": str(m["date"])[:10], "match": f"{m['home']} v {m['away']}",
                "side": "Yes" if is_yes else "No", "model_p": b.model_p,
                "fair_p": b.fair_p, "decimal": b.decimal, "ev": b.ev,
                "kelly_used": b.kelly_used, "both_scored": both, "won": won,
                "score": f"{m['home_score']}-{m['away_score']}", "book": m.get("btts_book"),
            })
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, label: str, min_ev: float = 0.03,
               min_prob_edge: float = 0.02) -> dict:
    pnl = np.array([_pnl(r.decimal, bool(r.won)) for r in df.itertuples()])
    roi, lo, hi = _bootstrap_roi(pnl) if len(pnl) else (float("nan"),) * 3
    units = float(pnl.sum())                          # flat 1u stakes
    k = df["kelly_used"].to_numpy()
    kelly_units = float((k * pnl).sum() * 100) if len(k) else 0.0  # 1u = 1% bankroll
    return {"segment": label, "bets": int(len(df)), "wins": int(df["won"].sum()),
            "roi": roi, "roi_lo": lo, "roi_hi": hi, "units": round(units, 2),
            "kelly_units": round(kelly_units, 2)}


def report(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = collect(cfg=cfg)
    if df.empty:
        print("No played BTTS games with odds found — is THESTATSAPI_KEY set and reachable?")
        return pd.DataFrame()

    # the model's *recommendation* = the +EV side that clears the deployed gate
    rec = df[df.apply(lambda r: qualifies(r["model_p"], r["fair_p"], r["decimal"],
                                          0.03, 0.02, 6.0), axis=1)]
    rows = [
        _summarize(df, "All BTTS sides (flat baseline)"),
        _summarize(df[df["side"] == "Yes"], "BTTS Yes"),
        _summarize(df[df["side"] == "No"], "BTTS No"),
    ]
    if len(rec):
        rows.append(_summarize(rec, "Model recommendation (+EV, gated)"))
    out = pd.DataFrame(rows)

    # calibration: model P(yes) vs actual BTTS rate
    yes = df[df["side"] == "Yes"]
    n_games = len(yes)
    actual_rate = float(yes["both_scored"].mean()) if n_games else float("nan")
    model_mean = float(yes["model_p"].mean()) if n_games else float("nan")

    p = path_for("reports", cfg) / "btts_backcheck.csv"
    out.to_csv(p, index=False)

    print(f"\nBTTS back-check — {n_games} played 2026 WC games (book: "
          f"{yes['book'].dropna().iloc[0] if n_games and yes['book'].notna().any() else 'n/a'})")
    print("=" * 72)
    show = out.copy()
    for c in ("roi", "roi_lo", "roi_hi"):
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    print(show.to_string(index=False))
    print(f"\nCalibration: model mean P(BTTS yes) = {model_mean*100:.1f}% vs "
          f"actual both-scored rate = {actual_rate*100:.1f}% ({n_games} games)")

    if len(rec):
        r = rows[-1]
        verdict = ("NO demonstrated edge (CI through 0 — variance at this sample)"
                   if not (r["roi_lo"] > 0) else
                   "positive ROI with CI>0 — but n is tiny; treat as a forward candidate, not proof")
        print(f"\nVERDICT (model's {r['bets']} gated BTTS picks): ROI {r['roi']*100:+.1f}% "
              f"[{r['roi_lo']*100:+.1f}%, {r['roi_hi']*100:+.1f}%] → {verdict}")
    else:
        print("\nVERDICT: the model flagged NO +EV BTTS bet on the played games "
              "(the line and the model agree — no value, the honest common case).")
    print(f"\nwrote {p}")
    return out


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    report()
