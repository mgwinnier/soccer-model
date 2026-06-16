"""Closing-Line Value (CLV) tracking — the honest proof of edge.

Two steps that run over time:

  1. **snapshot()** — record every bet the model *recommends right now* (calibrated
     + anchored, EV ≥ threshold) as an open "ticket" at the price currently offered.
     Dedupes, so a bet is logged once at our entry price.
  2. **grade()** — for tickets whose match has finished, fetch the **closing** line
     (ESPN summary endpoint) and the result, then compute:
        • CLV  = our_decimal / closing_decimal − 1  (did we beat the close?)
        • P&L  = settled at our price.

Beating the closing line consistently is the single best predictor of a real edge —
far more reliable than short-run ROI. The ledger accumulates as matches play.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..data.odds import (fetch_espn_range, fetch_summary_odds, american_to_decimal)
from ..models.segment_gate import segment_from_code, disabled_set
from .value import build_bets

OPEN_COLS = ["game_id", "match_date", "match", "market", "code", "segment",
             "system", "selection", "american", "decimal", "model_p", "fair_p", "ev",
             "snapshot_time"]


def _system_tag(market: str, decimal: float | None) -> str:
    """Named-system tag for forward tracking. ``pickem_ml_2_3`` = the v8 candidate
    (Match Result at decimal odds [2.0, 3.0)) — recorded for OBSERVATION only; it
    failed the pre-registered family-wise bar, so it is not a deployed bet filter.
    Forward CLV on these picks is the one clean test the backtest can't fake."""
    if market == "Match Result" and decimal is not None and 2.0 <= decimal < 3.0:
        return "pickem_ml_2_3"
    return ""


def _open_path(cfg):
    return path_for("reports", cfg) / "clv_open.csv"


def _ledger_path(cfg):
    return path_for("reports", cfg) / "clv_ledger.csv"


def _bet_code(market: str, selection: str, home: str, away: str, analysis: dict):
    """Reconstruct a grade-able code from a BetEval's market+selection."""
    if market == "Match Result":
        return "H" if selection == home else ("A" if selection == away else "D")
    if market == "Total Goals":
        side, line = selection.split(" ")
        return f"{side.lower()}@{line}"          # over@2.5 / under@2.5
    if market == "Spread":
        line = analysis.get("spread", {}).get("home_line")
        return (f"cover_home@{line}" if selection.startswith(home)
                else f"cover_away@{line}")
    return "?"


def snapshot(start: str | None = None, days: int = 3, min_ev: float = 0.03,
             anchor_w: float | None = None, cfg: dict | None = None,
             now: str | None = None) -> int:
    """Log newly-recommended +EV bets as open tickets. Returns # added."""
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    start = start or datetime.utcnow().strftime("%Y-%m-%d")
    now = now or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    res = build_bets(start, days=days, bankroll=1000, cfg=cfg,
                     use_cache=False, anchor_w=anchor_w)
    existing = pd.read_csv(_open_path(cfg)) if _open_path(cfg).exists() else \
        pd.DataFrame(columns=OPEN_COLS)
    seen = set(zip(existing["game_id"].astype(str), existing["market"],
                   existing["selection"])) if len(existing) else set()
    # Also dedup against already-SETTLED bets, so a graded ticket is never recorded
    # (and re-graded) a second time on a later sync — the cause of ledger duplicates.
    if _ledger_path(cfg).exists():
        led = pd.read_csv(_ledger_path(cfg))
        if len(led):
            seen |= set(zip(led["game_id"].astype(str), led["market"], led["selection"]))
    disabled = disabled_set(cfg)
    rows = []
    for m in res["matches"]:
        for b in m["bets"]:
            if b.ev is None or pd.isna(b.ev) or b.ev < min_ev \
                    or b.decimal is None or pd.isna(b.decimal):
                continue   # skip un-priced bets (odds nulled once a match kicks off)
            code = _bet_code(b.market, b.selection, m["home"], m["away"], m["analysis"])
            segment = segment_from_code(code)
            if segment in disabled:
                continue  # killed segment — never snapshot it
            key = (str(m["game_id"]), b.market, b.selection)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "game_id": m["game_id"],
                "match_date": pd.to_datetime(m["date"]).strftime("%Y-%m-%d"),
                "match": f"{m['home']} v {m['away']}", "market": b.market,
                "code": code, "segment": segment,
                "system": _system_tag(b.market, b.decimal),
                "selection": b.selection, "american": b.american, "decimal": b.decimal,
                "model_p": round(b.model_p, 4), "fair_p": b.fair_p,
                "ev": round(b.ev, 4), "snapshot_time": now,
            })
    if rows:
        out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(_open_path(cfg), index=False)
    print(f"[clv] snapshot: +{len(rows)} new tickets ({len(seen)} tracked total)")
    return len(rows)


def _closing_price(code: str, od: dict):
    """Closing decimal odds for a ticket code from summary odds."""
    if code in ("H", "D", "A"):
        am = {"H": od["ml_home"], "D": od["ml_draw"], "A": od["ml_away"]}[code]
    elif code.startswith("over@"):
        am = od.get("ou_over_odds")
    elif code.startswith("under@"):
        am = od.get("ou_under_odds")
    elif code.startswith("cover_home@"):
        am = od.get("spread_home_odds")
    elif code.startswith("cover_away@"):
        am = od.get("spread_away_odds")
    else:
        return None
    return american_to_decimal(am)


def _result(code: str, hs: int, as_: int) -> str:
    tot, margin = hs + as_, hs - as_
    if code in ("H", "D", "A"):
        actual = "H" if hs > as_ else ("D" if hs == as_ else "A")
        return "win" if code == actual else "loss"
    kind, line = code.split("@")
    line = float(line)
    if kind == "over":
        return "push" if tot == line else ("win" if tot > line else "loss")
    if kind == "under":
        return "push" if tot == line else ("win" if tot < line else "loss")
    adj = margin + line
    if kind == "cover_home":
        return "push" if abs(adj) < 1e-9 else ("win" if adj > 0 else "loss")
    return "push" if abs(adj) < 1e-9 else ("win" if adj < 0 else "loss")


def grade(cfg: dict | None = None, now: str | None = None) -> int:
    """Settle finished tickets: compute CLV + P&L, move them to the ledger."""
    cfg = cfg or load_config()
    op = _open_path(cfg)
    if not op.exists():
        print("[clv] no open tickets")
        return 0
    opendf = pd.read_csv(op)
    if opendf.empty:
        return 0
    # results for the date span covered by open tickets
    lo = opendf["match_date"].min()
    hi = (pd.Timestamp(opendf["match_date"].max()) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    scores = {e["game_id"]: e for e in fetch_espn_range(lo, hi, cfg=cfg, use_cache=False)
              if e["status"] == "post" and e["home_score"] is not None}

    graded, remaining = [], []
    for t in opendf.to_dict("records"):
        ev = scores.get(str(t["game_id"])) or scores.get(t["game_id"])
        if not ev:
            remaining.append(t)
            continue
        od = fetch_summary_odds(str(t["game_id"]), cfg=cfg)
        close_dec = _closing_price(t["code"], od) if od else None
        res = _result(t["code"], ev["home_score"], ev["away_score"])
        pnl = 0.0 if res == "push" else (t["decimal"] - 1 if res == "win" else -1.0)
        clv = (t["decimal"] / close_dec - 1.0) if close_dec else np.nan
        graded.append({**t, "segment": t.get("segment") or segment_from_code(t["code"]),
                       "system": t.get("system") or _system_tag(t.get("market", ""), t.get("decimal")),
                       "score": f"{ev['home_score']}-{ev['away_score']}",
                       "closing_decimal": round(close_dec, 3) if close_dec else None,
                       "clv": round(clv, 4) if close_dec else None,
                       "result": res, "pnl": round(pnl, 3),
                       "graded_time": now or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
    if graded:
        led = pd.DataFrame(graded)
        if _ledger_path(cfg).exists():
            led = pd.concat([pd.read_csv(_ledger_path(cfg)), led], ignore_index=True)
        led.to_csv(_ledger_path(cfg), index=False)
        pd.DataFrame(remaining, columns=opendf.columns).to_csv(op, index=False)
    print(f"[clv] graded {len(graded)} tickets, {len(remaining)} still open")
    return len(graded)


def report(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    if not _ledger_path(cfg).exists():
        return {"n": 0}
    led = pd.read_csv(_ledger_path(cfg))
    if "segment" not in led.columns:
        led["segment"] = led["code"].map(segment_from_code)
    settled = led[led["result"] != "push"]
    with_clv = led.dropna(subset=["clv"])

    def _rollup(group_col):
        out = {}
        for key, g in led.groupby(group_col):
            if key in ("", None) or (isinstance(key, float) and pd.isna(key)):
                continue
            gc = g.dropna(subset=["clv"])
            gs = g[g["result"] != "push"]
            out[key] = {
                "n": int(len(g)),
                "avg_clv": float(gc["clv"].mean()) if len(gc) else float("nan"),
                "pct_positive_clv": float((gc["clv"] > 0).mean()) if len(gc) else float("nan"),
                "roi": float(g["pnl"].sum() / len(gs)) if len(gs) else float("nan"),
                "units": float(g["pnl"].sum()),
            }
        return out

    by_segment = _rollup("segment")
    # Forward record of the v8 candidate (observational; not a deployed filter)
    by_system = _rollup("system") if "system" in led.columns else {}
    return {
        "n": len(led),
        "avg_clv": float(with_clv["clv"].mean()) if len(with_clv) else float("nan"),
        "pct_positive_clv": float((with_clv["clv"] > 0).mean()) if len(with_clv) else float("nan"),
        "roi": float(led["pnl"].sum() / len(settled)) if len(settled) else float("nan"),
        "record": f"{int((settled.result=='win').sum())}-{int((settled.result=='loss').sum())}",
        "units": float(led["pnl"].sum()),
        "by_segment": by_segment,
        "by_system": by_system,
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cmd = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
    if cmd == "snapshot":
        snapshot()
    elif cmd == "grade":
        grade()
    elif cmd == "kill":
        from ..models.segment_gate import evaluate_kill_switches
        g = evaluate_kill_switches(now=datetime.utcnow().strftime("%Y-%m-%d"))
        print(f"[clv] disabled segments: {sorted(g.disabled_set())}")
        for s, info in g.disabled.items():
            print(f"  {s}: {info['reason']}")
    elif cmd == "report":
        r = report()
        if r["n"] == 0:
            print("No graded CLV tickets yet.")
        else:
            print(f"CLV track record: {r['n']} bets, record {r['record']}, "
                  f"{r['units']:+.1f}u (ROI {r['roi']*100:+.1f}%)")
            print(f"Avg CLV {r['avg_clv']*100:+.2f}% · beat the close "
                  f"{r['pct_positive_clv']*100:.0f}% of the time")
            print("\nBy segment (CLV is the leading edge indicator):")
            for s, v in sorted(r.get("by_segment", {}).items()):
                print(f"  {s:8s}: {v['n']:3d} bets, avg CLV {v['avg_clv']*100:+.2f}%, "
                      f"ROI {v['roi']*100:+.1f}%, {v['units']:+.1f}u")
