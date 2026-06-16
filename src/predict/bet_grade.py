"""Grade the model's betting recommendations on matches already played.

ESPN's *summary* endpoint keeps each finished game's pre-match odds (the
scoreboard nulls them), so we can finally do the real test: train the model only
on data **before** the tournament, have it recommend +EV bets at the **actual
closing prices** across Match Result / Totals / Spread, then grade those picks
against what happened and tally P&L.

Honest notes: these are the *current independent* model's recommendations (before
the v4 market-anchoring/calibration work), and one tournament's group stage is a
tiny sample — this measures what the model *would have done*, not a guaranteed edge.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import load_config, path_for
from ..data.odds import (fetch_espn_range, fetch_summary_odds, american_to_decimal,
                         decimal_to_prob, devig)
from ..models.base import scoreline_to_outcome_probs
from ..predict.predict_match import _over_prob, _cover_prob
from ..predict.odds_backtest import _AsOfPredictor
from ..predict.betting import expected_value, kelly_fraction
from ..predict.anchor import anchor
from ..simulate.bracket_2026 import HOST_TEAMS


def _grade(code, hs: int, as_: int) -> str:
    tot, margin = hs + as_, hs - as_
    if code == "H":
        return "win" if hs > as_ else "loss"
    if code == "D":
        return "win" if hs == as_ else "loss"
    if code == "A":
        return "win" if as_ > hs else "loss"
    kind, line = code
    if kind == "over":
        return "push" if tot == line else ("win" if tot > line else "loss")
    if kind == "under":
        return "push" if tot == line else ("win" if tot < line else "loss")
    adj = margin + line
    if kind == "cover_home":
        return "push" if abs(adj) < 1e-9 else ("win" if adj > 0 else "loss")
    return "push" if abs(adj) < 1e-9 else ("win" if adj < 0 else "loss")   # cover_away


def _candidates(home, away, probs, mat, od, calibrators=None, anchor_w=None):
    """Yield (market, selection, grade_code, american, model_p, fair_p).

    Applies optional per-market calibration then market-anchoring to the model
    probability of each selection (EV is computed per-selection downstream)."""
    def adj(p, fair, market_cal=None):
        if calibrators is not None and market_cal is not None:
            p = calibrators.calibrate(market_cal, p)
        if anchor_w is not None and fair is not None:
            p = anchor(p, fair, anchor_w)
        return p

    # Match Result (anchored only; 1X2 handled by the ensemble's own calibration)
    fair = devig([decimal_to_prob(american_to_decimal(od[k]))
                  for k in ("ml_home", "ml_draw", "ml_away")], "proportional")
    fair = fair or [None, None, None]
    yield ("Match Result", home, "H", od["ml_home"], adj(probs[0], fair[0]), fair[0])
    yield ("Match Result", "Draw", "D", od["ml_draw"], adj(probs[1], fair[1]), fair[1])
    yield ("Match Result", away, "A", od["ml_away"], adj(probs[2], fair[2]), fair[2])
    # Totals — calibrate P(over), under = 1 − calibrated over
    if od.get("ou_over_odds") is not None and od.get("total_line") is not None:
        L = od["total_line"]
        p_over_cal = calibrators.calibrate("over", _over_prob(mat, L)) \
            if calibrators is not None else _over_prob(mat, L)
        fo = devig([decimal_to_prob(american_to_decimal(od["ou_over_odds"])),
                    decimal_to_prob(american_to_decimal(od["ou_under_odds"]))], "proportional")
        fo = fo or [None, None]
        yield ("Total Goals", f"Over {L}", ("over", L), od["ou_over_odds"],
               adj(p_over_cal, fo[0]), fo[0])
        yield ("Total Goals", f"Under {L}", ("under", L), od["ou_under_odds"],
               adj(1 - p_over_cal, fo[1]), fo[1])
    # Spread — calibrate P(home covers), away = 1 − that
    if od.get("spread_home_odds") is not None and od.get("spread_home_line") is not None:
        line = od["spread_home_line"]
        ph = _cover_prob(mat, line)[0]
        ph_cal = calibrators.calibrate("cover", ph) if calibrators is not None else ph
        fs = devig([decimal_to_prob(american_to_decimal(od["spread_home_odds"])),
                    decimal_to_prob(american_to_decimal(od["spread_away_odds"]))], "proportional")
        fs = fs or [None, None]
        yield ("Spread", f"{home} {line:+g}", ("cover_home", line),
               od["spread_home_odds"], adj(ph_cal, fs[0]), fs[0])
        yield ("Spread", f"{away} {-line:+g}", ("cover_away", line),
               od["spread_away_odds"], adj(1 - ph_cal, fs[1]), fs[1])


def _grade_type(code) -> str:
    """Map a grading code to a directional role key for recentering."""
    if code in ("H", "D", "A"):
        return f"MR:{code}"
    kind, _ = code
    if kind in ("over", "under"):
        return f"TG:{kind}"
    return "SP:home" if kind == "cover_home" else "SP:away"


def run(cutoff: str = "2026-06-11", end: str | None = None, min_ev: float = 0.05,
        cfg: dict | None = None, calibrators=None, anchor_w=None,
        pred=None, recenter: bool = False, recenter_shrink: float = 0.8) -> dict:
    cfg = cfg or load_config()
    end = end or date.today().strftime("%Y-%m-%d")
    pred = pred or _AsOfPredictor(cutoff, cfg)
    events = [e for e in fetch_espn_range(cutoff, end, cfg=cfg, use_cache=True)
              if e["status"] == "post" and e["home_score"] is not None and e.get("game_id")]

    picks, cands = [], []
    for ev in events:
        home, away = ev["home_team"], ev["away_team"]
        if home not in pred.known or away not in pred.known:
            continue
        od = fetch_summary_odds(ev["game_id"], cfg=cfg)
        if not od or od.get("ml_home") is None:
            continue
        neutral = home not in HOST_TEAMS
        probs = pred.predict(home, away, neutral)
        lam, mu = pred.dc.expected_goals(home, away, neutral)
        mat = pred.dc.scoreline_matrix(lam, mu)
        hs, as_ = ev["home_score"], ev["away_score"]

        pick_code = ["H", "D", "A"][int(np.argmax(probs))]
        picks.append({"match": f"{home} v {away}", "score": f"{hs}-{as_}",
                      "pick": {"H": home, "D": "Draw", "A": away}[pick_code],
                      "result": _grade(pick_code, hs, as_)})

        for market, sel, code, am, mp, fair in _candidates(
                home, away, probs, mat, od, calibrators, anchor_w):
            if am is None or mp is None:
                continue
            cands.append({"match": f"{home} v {away}", "score": f"{hs}-{as_}",
                          "market": market, "selection": sel, "code": code,
                          "american": am, "model_p": mp, "fair_p": fair,
                          "type": _grade_type(code), "hs": hs, "as_": as_})

    # market-relative recentering: remove the slate-wide per-role tilt vs market
    if recenter and cands:
        from collections import defaultdict
        edges = defaultdict(list)
        for c in cands:
            if c["fair_p"] is not None:
                edges[c["type"]].append(c["model_p"] - c["fair_p"])
        bias = {k: float(np.mean(v)) for k, v in edges.items() if v}
        for c in cands:
            c["model_p"] = min(max(c["model_p"] - recenter_shrink * bias.get(c["type"], 0.0),
                                   1e-4), 1 - 1e-4)

    ledger = []
    for c in cands:
        dec = american_to_decimal(c["american"])
        ev_val = expected_value(c["model_p"], dec)
        if ev_val < min_ev:
            continue
        res = _grade(c["code"], c["hs"], c["as_"])
        pnl = 0.0 if res == "push" else (dec - 1.0 if res == "win" else -1.0)
        ledger.append({
            "match": c["match"], "score": c["score"], "market": c["market"],
            "selection": c["selection"], "american": c["american"],
            "model_p": round(c["model_p"], 3),
            "fair_p": round(c["fair_p"], 3) if c["fair_p"] else None,
            "ev": round(ev_val, 3), "result": res, "pnl": round(pnl, 3),
            "kelly": round(kelly_fraction(c["model_p"], dec), 3),
        })

    return {"ledger": pd.DataFrame(ledger), "picks": pd.DataFrame(picks),
            "n_matches": len(picks)}


def _summary(res: dict, kelly_fraction: float = 0.5, bankroll: float = 1000.0):
    led, picks = res["ledger"], res["picks"]
    out = []
    if not picks.empty:
        acc = (picks["result"] == "win").mean()
        out.append(f"Straight 1X2 picks: {int((picks.result=='win').sum())}/"
                   f"{len(picks)} correct ({acc*100:.0f}%)")
    if led.empty:
        out.append("No +EV recommendations cleared the threshold.")
        return "\n".join(out), led
    graded = led[led["result"] != "push"]
    wins = (graded["result"] == "win").sum()
    flat_pnl = led["pnl"].sum()           # 1 unit flat per bet (1u = 1% bankroll)
    roi = flat_pnl / len(graded) if len(graded) else 0

    # Half-Kelly staking, sized in units (1 unit = 1% of bankroll)
    dec = led["american"].map(american_to_decimal)
    stake_u = led["kelly"] * kelly_fraction * 100.0
    kpnl_u = np.where(led["result"] == "push", 0.0,
                      np.where(led["result"] == "win", stake_u * (dec - 1), -stake_u))
    k_total, k_staked = float(np.sum(kpnl_u)), float(stake_u.sum())
    k_roi = k_total / k_staked if k_staked else 0

    out.append(f"\nRecommended +EV bets: {len(led)} "
               f"({wins} win / {len(graded)-wins} loss / {(led.result=='push').sum()} push)")
    out.append(f"  Flat 1u/bet : {flat_pnl:+.2f}u  (ROI {roi*100:+.1f}%)")
    out.append(f"  Half-Kelly  : {k_total:+.2f}u staked {k_staked:.1f}u "
               f"(ROI {k_roi*100:+.1f}%)   [1 unit = 1% of bankroll]")
    # by market (flat)
    for mk, g in led.groupby("market"):
        gg = g[g["result"] != "push"]
        w = (gg["result"] == "win").sum()
        out.append(f"  {mk:13s}: {len(g):2d} bets, {w} won, "
                   f"{g['pnl'].sum():+.2f}u flat (ROI {g['pnl'].sum()/max(len(gg),1)*100:+.0f}%)")
    return "\n".join(out), led


def _asof_calibrators(cutoff: str, cfg: dict):
    """Calibrators fit ONLY on pre-cutoff data (leak-free for the re-grade), cached."""
    from ..models.market_calibration import fit_calibrators, MarketCalibrators
    p = path_for("models", cfg) / f"market_calibrators_asof_{cutoff}.joblib"
    if p.exists():
        return MarketCalibrators.load(p)
    cal = fit_calibrators(cfg, as_of=cutoff, save=False)
    cal.save(p)
    return cal


def compare_modes(cutoff: str = "2026-06-11", cfg: dict | None = None):
    cfg = cfg or load_config()
    pred = _AsOfPredictor(cutoff, cfg)               # fit once, reuse
    cal = _asof_calibrators(cutoff, cfg)
    modes = [
        ("INDEPENDENT (current model)", dict()),
        ("CALIBRATED (markets de-biased)", dict(calibrators=cal)),
        ("CALIBRATED + ANCHORED to market (w=0.5)", dict(calibrators=cal, anchor_w=0.5)),
        ("CALIBRATED + ANCHORED + RECENTERED (deployed)",
         dict(calibrators=cal, anchor_w=0.5, recenter=True)),
    ]
    for label, kw in modes:
        res = run(cutoff, cfg=cfg, pred=pred, **kw)
        print(f"\n{'='*64}\n{label}\n{'='*64}")
        text, _ = _summary(res)
        print(text)
    return pred


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cutoff = sys.argv[1] if len(sys.argv) > 1 else "2026-06-11"
    print(f"\nMODEL BETTING RECORD on 2026 WC matches since {cutoff} "
          f"(trained only on pre-{cutoff} data, real DraftKings closing prices)")
    compare_modes(cutoff)
