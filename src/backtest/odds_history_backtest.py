"""Large-sample betting backtest on harvested historical international odds.

For every harvested match we rebuild the model **leak-free as of that date**
(per-year Dixon-Coles + continuous as-of Elo), produce the deployed
recommendation pipeline (calibration + anchoring), grade each +EV pick at the
Bet365 closing price, and aggregate ROI with a **bootstrap confidence interval** —
the honest verdict on whether the edge is real, over thousands of bets rather than
a handful.

The market-relative de-bias (recentering) is fit **walk-forward**: estimated on an
earlier train slice and applied out-of-sample to the later test slice, so the ROI
isn't flattered by in-sample fitting. The *deployed* bias (for live betting) is fit
on everything and persisted to ``data/models/market_bias.json``.
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..data.clean import load_matches
from ..data.odds_history import load_odds_history
from ..models.dixon_coles import DixonColesModel
from ..features.elo import EloEngine
from ..models.elo_model import EloModel
from ..models.base import scoreline_to_outcome_probs
from ..models.market_bias import fit_market_bias, MarketBias, default_path
from ..predict.bet_grade import _candidates, _grade, _grade_type
from ..predict.betting import expected_value
from ..data.odds import american_to_decimal
from ..data.odds_consensus import match_consensus, code_edge_vs_consensus

NEUTRAL_LEAGUES = {"fifa.world", "uefa.euro", "conmebol.america", "concacaf.gold",
                   "afc.asian", "caf.nations", "fifa.cwc"}


# ----------------------------------------------------- leak-free as-of model
def _elo_timeline(matches: pd.DataFrame, cfg: dict):
    """Per-team sorted (date, rating-after) history from the Elo engine."""
    eng = EloEngine.from_config(cfg)
    tl: dict[str, list] = {}
    for r in matches.sort_values("date").itertuples(index=False):
        eng.update_one(r.home_team, r.away_team, int(r.home_score), int(r.away_score),
                       bool(r.neutral), float(r.importance))
        for t in (r.home_team, r.away_team):
            tl.setdefault(t, [[], []])
            tl[t][0].append(r.date)
            tl[t][1].append(eng.rating(t))
    return tl


def _asof(tl, team, date, base=1500.0) -> float:
    rec = tl.get(team)
    if not rec:
        return base
    i = bisect.bisect_left(rec[0], date) - 1     # latest strictly before `date`
    return rec[1][i] if i >= 0 else base


def build_predictions(cfg: dict | None = None, anchor_w: float = 0.5,
                      test_from: str = "2019-01-01", split: str = "2023-01-01") -> pd.DataFrame:
    cfg = cfg or load_config()
    oh = load_odds_history(cfg)
    if oh.empty:
        return pd.DataFrame()
    oh = oh.sort_values("date").reset_index(drop=True)
    matches = load_matches(cfg)
    tl = _elo_timeline(matches, cfg)
    # WALK-FORWARD calibration: fit the market calibrators only on data BEFORE the
    # test split, so graded (>= split) matches never inform their own calibration.
    # (The deployed all-history calibrators are used live, not here.)
    from ..models.market_calibration import fit_calibrators
    calibrators = fit_calibrators(cfg, as_of=split, save=False)
    print(f"[odds-bt] calibration fit as-of {split} (walk-forward) — "
          f"separate from the deployed all-history calibrators")

    # Elo logistic + per-year DC trained only on pre-test data (leak-free)
    feats = pd.read_parquet(path_for("data_processed", cfg) / "features.parquet")
    elo_model = EloModel(cfg=cfg).fit(feats[feats["date"] < pd.Timestamp(test_from)]
                                      .dropna(subset=["elo_diff"]))
    dc_cache: dict[int, DixonColesModel] = {}

    def yearly_dc(year: int) -> DixonColesModel:
        if year not in dc_cache:
            tr = matches[matches["date"] < pd.Timestamp(f"{year}-01-01")]
            dc_cache[year] = DixonColesModel.from_config(cfg).fit(tr)
        return dc_cache[year]

    rows = []
    for r in oh.itertuples(index=False):
        home, away = r.home_team, r.away_team
        dc = yearly_dc(pd.Timestamp(r.date).year)
        if home not in dc._tidx or away not in dc._tidx:
            continue
        neutral = r.league in NEUTRAL_LEAGUES
        ed = _asof(tl, home, r.date) - _asof(tl, away, r.date)
        elo_p = elo_model.predict_proba(pd.DataFrame([{"elo_diff": ed, "neutral": neutral}]))[0]
        lam, mu = dc.expected_goals(home, away, neutral)
        mat = dc.scoreline_matrix(lam, mu)
        dc_p = np.array(scoreline_to_outcome_probs(mat))
        blend = (dc_p + elo_p) / 2
        blend /= blend.sum()
        # favorite-longshot calibration (same map MatchPredictor applies), renormalized
        if calibrators.models.get("mr") is not None:
            c3 = np.array([calibrators.calibrate("mr", float(x)) for x in blend])
            if c3.sum() > 0:
                blend = c3 / c3.sum()
        od = {k: getattr(r, k) for k in (
            "ml_home", "ml_away", "ml_draw", "total_line", "ou_over_odds",
            "ou_under_odds", "spread_home_line", "spread_home_odds", "spread_away_odds")}
        hs, as_ = int(r.home_score), int(r.away_score)
        # Read-only multi-book consensus from the cached summary (no re-harvest):
        # the off-consensus split below asks whether our book beating the market
        # actually predicts winning bets. Exclude our own book so it can't anchor.
        cons = match_consensus(r.game_id, r.league, exclude="bet365", cfg=cfg)
        # World-Cup scoring-environment correction (mirrors the deployed predictor):
        # the model under-projects WC goals vs ACTUAL results, so for WC matches scale
        # expected goals by the historical actual/model ratio and rebuild the matrix.
        # Non-WC internationals are left untouched. Calibrated to scores, not the line.
        if r.league == "fifa.world" and (lam + mu) > 0:
            from ..models.wc_goals import correct as _wc_correct
            lam, mu = _wc_correct(lam, mu)
            mat = dc.scoreline_matrix(lam, mu)
            dc_p = np.array(scoreline_to_outcome_probs(mat))
            blend = (dc_p + elo_p) / 2
            blend /= blend.sum()
            if calibrators.models.get("mr") is not None:
                c3 = np.array([calibrators.calibrate("mr", float(x)) for x in blend])
                if c3.sum() > 0:
                    blend = c3 / c3.sum()
        for market, sel, code, am, mp, fair in _candidates(
                home, away, blend, mat, od, calibrators, anchor_w):
            if am is None or mp is None:
                continue
            dec = american_to_decimal(am)
            if dec is None:
                continue
            rows.append({
                "date": r.date, "league": r.league, "game_id": getattr(r, "game_id", None),
                "market": market,
                "type": _grade_type(code), "american": am, "dec": dec,
                "model_p": mp, "fair_p": fair, "result": _grade(code, hs, as_),
                "cons_edge": code_edge_vs_consensus(cons, code, dec),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- aggregation
def _bootstrap_roi(pnl: np.ndarray, n: int = 2000, seed: int = 0):
    if len(pnl) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), size=(n, len(pnl)))
    rois = pnl[idx].mean(axis=1)
    return float(pnl.mean()), float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def _grade_block(test: pd.DataFrame, bias: MarketBias, min_ev: float,
                 min_prob_edge: float = 0.0, max_decimal: float | None = None):
    adj = np.array([bias.recenter(t, p) for t, p in zip(test["type"], test["model_p"])])
    dec = test["dec"].to_numpy()
    ev = adj * (dec - 1) - (1 - adj)
    if min_prob_edge or max_decimal:
        # mirror the deployed bet gate (prob-edge floor + longshot cap)
        from ..predict.betting import qualifies
        fair = (test["fair_p"].to_numpy() if "fair_p" in test.columns
                else np.full(len(test), np.nan))
        keep = np.array([qualifies(a, (f if f == f else None), d, min_ev, min_prob_edge,
                                   max_decimal) for a, f, d in zip(adj, fair, dec)])
    else:
        keep = ev >= min_ev
    sel = test[keep].copy()
    if sel.empty:
        return None
    pnl = np.where(sel["result"] == "push", 0.0,
                   np.where(sel["result"] == "win", sel["dec"] - 1, -1.0))
    settled = sel["result"] != "push"
    mean, lo, hi = _bootstrap_roi(pnl[settled.to_numpy()])
    return {"bets": len(sel), "wins": int((sel["result"] == "win").sum()),
            "roi": mean, "roi_lo": lo, "roi_hi": hi, "units": float(pnl.sum())}


def backtest(cfg: dict | None = None, anchor_w: float = 0.5, min_ev: float = 0.03,
             split: str = "2023-01-01", write: bool = True,
             off_consensus_threshold: float = 0.0) -> dict:
    cfg = cfg or load_config()
    preds = build_predictions(cfg, anchor_w=anchor_w, split=split)
    if preds.empty:
        print("[odds-bt] no predictions — harvest odds first")
        return {}
    train = preds[preds["date"] < pd.Timestamp(split)]
    test = preds[preds["date"] >= pd.Timestamp(split)]
    bias = fit_market_bias(train["type"], train["model_p"], train["fair_p"])

    out = {"n_predictions": len(preds), "n_train": len(train), "n_test": len(test)}
    overall = _grade_block(test, bias, min_ev)
    out["overall"] = overall
    by_market = {}
    for mk, g in test.groupby("market"):
        r = _grade_block(g, bias, min_ev)
        if r:
            by_market[mk] = r
    out["by_market"] = by_market

    # Off-consensus split — the headline single-book edge test. Of the model's +EV
    # bets, do the ones where our book beat the market consensus (cons_edge > thr)
    # actually win more than the ones where it didn't? If yes, the lagging-book
    # signal is real; if both CIs straddle 0, it isn't.
    ce = test.get("cons_edge")
    if ce is not None:
        has = ce.notna()
        fav = _grade_block(test[has & (ce > off_consensus_threshold)], bias, min_ev)
        unfav = _grade_block(test[has & (ce <= off_consensus_threshold)], bias, min_ev)
        out["off_consensus"] = {
            "favorable": fav, "unfavorable": unfav,
            "n_with_consensus": int(has.sum()), "n_test": len(test),
            "threshold": off_consensus_threshold,
        }

    # deployed bias: fit on EVERYTHING, persist for live use
    deployed = fit_market_bias(preds["type"], preds["model_p"], preds["fair_p"])
    if write:
        ensure_dirs(cfg)
        deployed.save(default_path(cfg))
        rows = [{"segment": "OVERALL", **overall}] if overall else []
        rows += [{"segment": mk, **r} for mk, r in by_market.items()]
        oc = out.get("off_consensus") or {}
        if oc.get("favorable"):
            rows.append({"segment": "OFF-CONSENSUS favorable", **oc["favorable"]})
        if oc.get("unfavorable"):
            rows.append({"segment": "OFF-CONSENSUS unfavorable", **oc["unfavorable"]})
        pd.DataFrame(rows).to_csv(path_for("reports", cfg) / "odds_history_backtest.csv",
                                  index=False)
    return out


def wc2022_report(cfg: dict | None = None, min_ev: float = 0.03,
                  split: str = "2022-06-01", write: bool = True) -> dict:
    """Betting backtest on the **2022 World Cup only** (out-of-sample: split before the
    tournament). OVERALL + by-market ROI with bootstrap CI → reports/wc2022_backtest.csv.
    For the dashboard's Performance page — a WC app should show WC results."""
    cfg = cfg or load_config()
    # Mirror the DEPLOYED model: market-INDEPENDENT (no anchoring, no market-bias
    # recentering) + the WC goals correction (applied inside build_predictions for
    # fifa.world) + the deployed bet gate (prob-edge floor + longshot cap).
    preds = build_predictions(cfg, anchor_w=1.0, split=split)
    if preds.empty:
        print("[wc2022] no predictions — harvest odds first")
        return {}
    bias = MarketBias({})            # no recentering — the model is market-independent
    wc = preds[(preds["league"] == "fifa.world")
               & (preds["date"].dt.year == 2022)
               & (preds["date"] >= pd.Timestamp(split))].copy()
    out = {"overall": _grade_block(wc, bias, min_ev, 0.02, 6.0), "by_market": {}}
    for mk, g in wc.groupby("market"):
        r = _grade_block(g, bias, min_ev, 0.02, 6.0)
        if r:
            out["by_market"][mk] = r
    if write:
        ensure_dirs(cfg)
        rows = [{"segment": "OVERALL", **out["overall"]}] if out["overall"] else []
        rows += [{"segment": mk, **r} for mk, r in out["by_market"].items()]
        pd.DataFrame(rows).to_csv(path_for("reports", cfg) / "wc2022_backtest.csv", index=False)
    return out


def _fmt(r):
    return (f"{r['bets']:4d} bets, {r['wins']} won, {r['units']:+.1f}u, "
            f"ROI {r['roi']*100:+.1f}%  [95% CI {r['roi_lo']*100:+.1f}%, {r['roi_hi']*100:+.1f}%]")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "wc2022":
        r = wc2022_report()
        if r.get("overall"):
            print("2022 WORLD CUP betting backtest (out-of-sample):")
            print("OVERALL  :", _fmt(r["overall"]))
            for mk, b in r.get("by_market", {}).items():
                print(f"{mk:11s}:", _fmt(b))
        raise SystemExit
    res = backtest()
    if not res:
        raise SystemExit
    print(f"\nHISTORICAL ODDS BACKTEST — {res['n_predictions']} candidate bets "
          f"({res['n_test']} in out-of-sample test, bet at Bet365 close)\n")
    if res.get("overall"):
        print("OVERALL  :", _fmt(res["overall"]))
    for mk, r in res.get("by_market", {}).items():
        print(f"{mk:9s}:", _fmt(r))

    oc = res.get("off_consensus")
    if oc:
        print(f"\nOFF-CONSENSUS split (our book vs market consensus on the same +EV "
              f"bets; consensus available for {oc['n_with_consensus']}/{oc['n_test']}):")
        if oc.get("favorable"):
            print("  book BEATS consensus :", _fmt(oc["favorable"]))
        if oc.get("unfavorable"):
            print("  book ≤ consensus     :", _fmt(oc["unfavorable"]))
        print("  → real single-book edge only if 'beats' ROI CI clears 0 AND beats "
              "the '≤' bucket. Historical prices are Bet365's, not necessarily yours.")
    print("\nVerdict: a market's edge is real only if its ROI 95% CI is clearly above 0.")
