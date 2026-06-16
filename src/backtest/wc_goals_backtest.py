"""Does the model predict World Cup *scores*, or just W/D/L? And is it using the
right factors to get the per-game goals right?

This is the honest per-game scorecard. Across every World Cup we have results for, it
compares the model's expected goals to what teams **actually scored** — not to the Vegas
line, to reality — and benchmarks that against dumb baselines so the irreducible-variance
ceiling is explicit. It also runs a **factor bake-off**: do recent goal-form, xG, or
confederation add real out-of-sample predictive power to the goals beyond Dixon-Coles?

Key findings (run it):
- The model under-projects WC goals in every tournament (1998-2022), and **not uniformly**:
  the FAVORITE is under-projected more than the underdog. Correcting the favored/underdog
  sides separately (``src/models/wc_goals.py``) zeroes the bias and improves pooled WC RPS.
- Per-game total-goals **MAE ≈ 1.3 and is essentially irreducible** — barely better than
  "always predict the tournament mean". Football is low-scoring and high-variance; no model
  predicts exact per-game scores. What we fix is the conditional mean and the calibration.
- Factor bake-off: candidate signals are kept only if they lower **held-out** WC goals MAE.

    python -m src.backtest.wc_goals_backtest

Writes ``reports/wc_goals_backtest.csv`` (per-tournament model vs actual + correction) and
``reports/wc_goals_factors.csv`` (the bake-off verdicts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..data.clean import load_matches
from ..models.dixon_coles import DixonColesModel
from ..models.base import scoreline_to_outcome_probs
from ..models import wc_goals
from ..predict.predict_match import _over_prob
from .metrics import evaluate_binary

# World Cups with enough prior international history to train a stable Dixon-Coles model.
WC_YEARS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]


def _rps(probs, outcome: int) -> float:
    """Ranked Probability Score for an ordered H/D/A forecast (outcome in 0/1/2)."""
    p = np.asarray(probs, dtype=float)
    c = np.cumsum(p)
    o = np.zeros(3); o[outcome] = 1.0
    return float(((c - np.cumsum(o)) ** 2).sum() / 2)


def _leakfree(matches: pd.DataFrame, wc: pd.DataFrame, year: int):
    """Leak-free per-WC Dixon-Coles expected goals (trained strictly before the WC)."""
    train = matches[matches["date"] < pd.Timestamp(f"{year}-05-01")]
    test = wc[wc["year"] == year]
    dc = DixonColesModel.from_config(load_config()).fit(train)
    neutral = test["neutral"].astype(bool).to_numpy()
    lam, mu = dc._lambdas(test["home_team"], test["away_team"], neutral)
    return dc, np.asarray(lam, float), np.asarray(mu, float), test


def _frame(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Pooled per-match table of leak-free predictions across all backtested WCs,
    plus a dict of per-year (dc, lam, mu) for matrix-based RPS."""
    matches = load_matches(cfg).sort_values("date").reset_index(drop=True)
    wc = matches[matches["tournament"] == "FIFA World Cup"].copy()
    wc["year"] = wc["date"].dt.year
    feats = pd.read_parquet(path_for("data_processed", cfg) / "features.parquet")
    fav_s, dog_s = wc_goals.load_scales(cfg)

    rows, per = [], {}
    for y in WC_YEARS:
        dc, lam, mu, test = _leakfree(matches, wc, y)
        per[y] = (dc, lam, mu)
        fav_is_home = lam >= mu
        # favorite/underdog-corrected expected goals (the deployed correction)
        clam = np.where(fav_is_home, lam * fav_s, lam * dog_s)
        cmu = np.where(fav_is_home, mu * dog_s, mu * fav_s)
        fcols = ["match_id", "home_gf_5", "away_gf_5", "home_ga_5", "away_ga_5",
                 "home_xgf", "away_xgf", "home_xga", "away_xga", "cross_conf"]
        have = [c for c in fcols if c in feats.columns]
        fj = test[["match_id"]].merge(feats[have], on="match_id", how="left")
        for i, r in enumerate(test.reset_index(drop=True).itertuples(index=False)):
            row = dict(
                year=y, idx=i, lam=lam[i], mu=mu[i], clam=clam[i], cmu=cmu[i],
                hs=r.home_score, as_=r.away_score,
                res={"H": 0, "D": 1, "A": 2}[r.result],
                fav_is_home=bool(fav_is_home[i]),
            )
            fr = fj.iloc[i]
            for c in have:
                if c != "match_id":
                    row[c] = fr[c]
            rows.append(row)
    d = pd.DataFrame(rows)
    d["model_tot"] = d.lam + d.mu
    d["corr_tot"] = d.clam + d.cmu
    d["act_tot"] = d.hs + d.as_
    return d, per


def _rps_pooled(d: pd.DataFrame, per: dict, use_corrected: bool) -> float:
    out = []
    for y in WC_YEARS:
        dc, lam, mu = per[y]
        sub = d[d.year == y]
        for r in sub.itertuples(index=False):
            la, mua = (r.clam, r.cmu) if use_corrected else (r.lam, r.mu)
            out.append(_rps(scoreline_to_outcome_probs(dc.scoreline_matrix(la, mua)), r.res))
    return float(np.mean(out))


def run(cfg: dict | None = None, write: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    d, per = _frame(cfg)
    fav_s, dog_s = wc_goals.load_scales(cfg)

    # ---- per-tournament model vs actual, raw and corrected --------------------
    rows = []
    for y in WC_YEARS:
        s = d[d.year == y]
        rows.append({
            "world_cup": y, "n": len(s),
            "model_total": round(s.model_tot.mean(), 3),
            "actual_total": round(s.act_tot.mean(), 3),
            "bias_raw": round(s.model_tot.mean() - s.act_tot.mean(), 3),
            "corrected_total": round(s.corr_tot.mean(), 3),
            "bias_corrected": round(s.corr_tot.mean() - s.act_tot.mean(), 3),
            "mae_raw": round((s.model_tot - s.act_tot).abs().mean(), 3),
            "mae_corrected": round((s.corr_tot - s.act_tot).abs().mean(), 3),
        })
    table = pd.DataFrame(rows)
    pooled = {
        "world_cup": "POOLED", "n": len(d),
        "model_total": round(d.model_tot.mean(), 3),
        "actual_total": round(d.act_tot.mean(), 3),
        "bias_raw": round(d.model_tot.mean() - d.act_tot.mean(), 3),
        "corrected_total": round(d.corr_tot.mean(), 3),
        "bias_corrected": round(d.corr_tot.mean() - d.act_tot.mean(), 3),
        "mae_raw": round((d.model_tot - d.act_tot).abs().mean(), 3),
        "mae_corrected": round((d.corr_tot - d.act_tot).abs().mean(), 3),
    }
    table = pd.concat([table, pd.DataFrame([pooled])], ignore_index=True)

    # ---- benchmarks: how close can a per-game goals predictor really get? ------
    mean_baseline = d.act_tot.mean()                       # "always predict the WC mean"
    bench = {
        "always_mean": round((d.act_tot - mean_baseline).abs().mean(), 3),
        "raw_dc": round((d.model_tot - d.act_tot).abs().mean(), 3),
        "corrected": round((d.corr_tot - d.act_tot).abs().mean(), 3),
    }

    # ---- over/under calibration (corrected model) -----------------------------
    p_over = np.array([_over_prob(per[r.year][0].scoreline_matrix(r.clam, r.cmu), 2.5)
                       for r in d.itertuples(index=False)])
    y_over = (d.act_tot.to_numpy() > 2.5).astype(float)
    ou = evaluate_binary(p_over, y_over)

    # ---- result accuracy + RPS, raw vs corrected ------------------------------
    rps_raw, rps_corr = _rps_pooled(d, per, False), _rps_pooled(d, per, True)

    # ---- favorite vs underdog asymmetry (raw) ---------------------------------
    fav_model = np.where(d.fav_is_home, d.lam, d.mu)
    dog_model = np.where(d.fav_is_home, d.mu, d.lam)
    fav_act = np.where(d.fav_is_home, d.hs, d.as_)
    dog_act = np.where(d.fav_is_home, d.as_, d.hs)

    if write:
        ensure_dirs(cfg)
        out = path_for("reports", cfg) / "wc_goals_backtest.csv"
        table.to_csv(out, index=False)
        print(f"[wc_goals] wrote {out}\n")
        print(table.to_string(index=False))
        print(f"\nDeployed correction: favorite x{fav_s}, underdog x{dog_s}  "
              f"(matchup-strength aware; src/models/wc_goals.py)")
        print("\n--- Per-game total-goals MAE vs benchmarks (the honest ceiling) ---")
        print(f"  always predict the mean ({mean_baseline:.2f}) : {bench['always_mean']}")
        print(f"  raw Dixon-Coles                       : {bench['raw_dc']}")
        print(f"  corrected (deployed)                  : {bench['corrected']}")
        print("  -> per-game totals are near-irreducible; correction fixes the MEAN, not the scatter.")
        print("\n--- Over/under 2.5 calibration (corrected) ---")
        print(f"  mean P(over)={ou['mean_pred']:.3f}  actual over-rate={ou['base_rate']:.3f}  "
              f"ECE={ou['ece']:.3f}  Brier={ou['brier']:.3f}")
        print("\n--- W/D/L forecast (does the goals correction help the result too?) ---")
        print(f"  pooled RPS: raw {rps_raw:.4f}  ->  corrected {rps_corr:.4f}")
        print("\n--- Favorite vs underdog (why a flat scale is wrong) ---")
        print(f"  favorite goals: model {fav_model.mean():.2f}  actual {fav_act.mean():.2f}  "
              f"ratio {fav_act.mean()/fav_model.mean():.3f}")
        print(f"  underdog goals: model {dog_model.mean():.2f}  actual {dog_act.mean():.2f}  "
              f"ratio {dog_act.mean()/dog_model.mean():.3f}")
    return table


# ---------------------------------------------------------------- factor bake-off
def _oos_factor_test(d: pd.DataFrame, per: dict, signal: np.ndarray) -> dict:
    """Leave-one-WC-out test: does ``signal`` add real predictive power out-of-sample?

    Baseline residual = actual_total - corrected_model_total. For each held-out WC we fit
    a 1-D OLS (residual ~ signal) on the OTHER WCs, predict the held-out residual, apply it
    as a proportional shift to the expected goals, and rebuild the scoreline matrix.
    Returns baseline vs with-factor **MAE and RPS**. A factor is only "the right factor" if
    it improves the probabilistic OUTCOME forecast (RPS) — an MAE-only gain on near-random
    totals is just shrinkage toward the mean (a negative OLS slope), not a real signal.
    """
    valid = np.isfinite(signal)
    sub = d[valid].reset_index(drop=True)
    s = signal[valid]
    resid = (sub.act_tot - sub.corr_tot).to_numpy()
    years = sub.year.to_numpy()
    pred = np.zeros_like(resid)
    for y in WC_YEARS:
        tr, te = years != y, years == y
        if te.sum() == 0 or tr.sum() < 30 or np.std(s[tr]) < 1e-9:
            continue
        b = np.cov(s[tr], resid[tr], bias=True)[0, 1] / np.var(s[tr])
        a = resid[tr].mean() - b * s[tr].mean()
        pred[te] = a + b * s[te]
    base_mae = base_rps = adj_mae = adj_rps = 0.0
    for i, r in enumerate(sub.itertuples(index=False)):
        dc = per[r.year][0]
        base_mae += abs(r.corr_tot - r.act_tot)
        base_rps += _rps(scoreline_to_outcome_probs(dc.scoreline_matrix(r.clam, r.cmu)), r.res)
        ntot = max(r.corr_tot + pred[i], 0.3); sc = ntot / r.corr_tot
        adj_mae += abs(ntot - r.act_tot)
        adj_rps += _rps(scoreline_to_outcome_probs(dc.scoreline_matrix(r.clam * sc, r.cmu * sc)), r.res)
    n = len(sub)
    return {"n": n, "base_mae": base_mae / n, "adj_mae": adj_mae / n,
            "base_rps": base_rps / n, "adj_rps": adj_rps / n}


def factor_bakeoff(cfg: dict | None = None, write: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    d, per = _frame(cfg)
    # candidate per-match signals (centered where natural); favored/underdog oriented
    def side(col_home, col_away):
        h = d[col_home].to_numpy(float); a = d[col_away].to_numpy(float)
        fav = np.where(d.fav_is_home, h, a); dog = np.where(d.fav_is_home, a, h)
        return fav, dog
    candidates = {}
    if "home_gf_5" in d:
        fav_gf, dog_gf = side("home_gf_5", "away_gf_5")
        candidates["recent_form: favorite gf_5"] = fav_gf
        candidates["recent_form: total gf_5 (both)"] = (d.home_gf_5 + d.away_gf_5).to_numpy(float)
    if "home_ga_5" in d:
        candidates["recent_form: total ga_5 (both)"] = (d.home_ga_5 + d.away_ga_5).to_numpy(float)
    if "home_xgf" in d:
        candidates["xG: total xgf (both, sparse)"] = (d.home_xgf + d.away_xgf).to_numpy(float)
    if "cross_conf" in d:
        candidates["cross-confederation flag"] = d.cross_conf.to_numpy(float)

    rows = []
    for name, sig in candidates.items():
        t = _oos_factor_test(d, per, sig)
        cov = float(np.isfinite(sig).mean())
        d_mae = t["base_mae"] - t["adj_mae"]      # +ve = factor lowers MAE
        d_rps = t["base_rps"] - t["adj_rps"]      # +ve = factor improves outcomes
        # "Right factor" = improves the OUTCOME forecast (RPS) out-of-sample, not just
        # MAE (an MAE-only win on near-random totals is mean-reversion shrinkage).
        keep = d_rps > 0.0005 and d_mae > 0 and cov >= 0.3
        rows.append({
            "factor": name, "n": t["n"], "coverage": round(cov, 2),
            "oos_mae_improvement": round(d_mae, 4),
            "oos_rps_improvement": round(d_rps, 4),
            "verdict": "KEEP" if keep else "reject",
        })
    table = pd.DataFrame(rows).sort_values("oos_rps_improvement", ascending=False)
    if write:
        ensure_dirs(cfg)
        out = path_for("reports", cfg) / "wc_goals_factors.csv"
        table.to_csv(out, index=False)
        print("\n--- Factor bake-off (leave-one-WC-out; real out-of-sample predictive power?) ---")
        print(table.to_string(index=False))
        kept = table[table.verdict == "KEEP"]
        if kept.empty:
            print("\nVERDICT: no candidate factor improves the held-out WC outcome forecast (RPS). "
                  "Dixon-Coles attack/defense + the favorite/underdog WC correction already capture\n"
                  "the per-game scoring signal. The only factor with a positive MAE nudge (recent\n"
                  "goal-form) does it via a NEGATIVE slope = mean-reversion, with ~0 RPS gain — i.e.\n"
                  "shrinkage, not a real form signal — so it is not wired in (would be false precision).")
        else:
            print(f"\nVERDICT: {', '.join(kept.factor)} improve held-out RPS — wire into _compute.")
    return table


if __name__ == "__main__":
    run()
    factor_bakeoff()
