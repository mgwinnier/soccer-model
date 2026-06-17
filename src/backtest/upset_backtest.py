"""Upsets & high-variance: are the meters honest, do real signals help, what does the dial cost?

Three honest questions, on the 7-World-Cup walk-forward backtest (1998-2022, leak-free):

1. METERS (Component 1) — are the surfaced "upset risk" and "shootout potential" meaningful?
   Bucket past games by the model's predicted upset_risk / over-3.5 prob and check the ACTUAL
   underdog-win rate / 4+ goal rate rises across buckets. If monotonic, the meters track reality.

2. SIGNALS (Component 2) — do causal upset signals the live predictor ignores (rest/fatigue,
   dead-rubber motivation, favorite-already-qualified, travel burden) actually predict upsets
   out-of-sample? Leave-one-World-Cup-out: fit a favorite→underdog nudge from the signal on the
   other WCs, apply to the held-out WC, keep ONLY if pooled held-out RPS improves. (Prior: most
   are weak; injuries — already wired live — are the likely keeper.)

3. DIAL (Component 3) — what does the "upset sensitivity" τ cost? Sweep τ and report pooled RPS,
   accuracy, and upset recall (share of real upsets where the model gave the underdog ≥ 30%), so
   each notch's accuracy cost is explicit.

    python -m src.backtest.upset_backtest

Writes reports/wc_variance_meters.csv, reports/upset_signals.csv, reports/upset_dial.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..data.clean import load_matches
from ..models.dixon_coles import DixonColesModel
from ..models.elo_model import EloModel
from ..models.base import scoreline_to_outcome_probs
from ..models import wc_goals
from ..predict.predict_match import _over_prob, _temper

WC_YEARS = [1998, 2002, 2006, 2010, 2014, 2018, 2022]
_SIG_COLS = ["home_rest_days", "away_rest_days", "home_travel_km", "away_travel_km",
             "venue_altitude_m", "dead_rubber", "home_already_q", "away_already_q"]


def _rps(p, o):
    p = np.asarray(p, float); c = np.cumsum(p)
    t = np.zeros(3); t[o] = 1
    return float(((c - np.cumsum(t)) ** 2).sum() / 2)


def _frame(cfg: dict) -> pd.DataFrame:
    """Per-game leak-free deployed blend + outcomes + upset-signal features, all WCs."""
    matches = load_matches(cfg).sort_values("date").reset_index(drop=True)
    feats = pd.read_parquet(path_for("data_processed", cfg) / "features.parquet")
    wc = matches[matches["tournament"] == "FIFA World Cup"].copy()
    wc["year"] = wc["date"].dt.year
    rows = []
    for y in WC_YEARS:
        split = pd.Timestamp(f"{y}-05-01")
        dc = DixonColesModel.from_config(cfg).fit(matches[matches["date"] < split])
        elo = EloModel(cfg=cfg).fit(feats[feats["date"] < split])
        test = wc[wc["year"] == y]
        neutral = test["neutral"].astype(bool).to_numpy()
        lam, mu = dc._lambdas(test["home_team"], test["away_team"], neutral)
        lam, mu = np.asarray(lam, float), np.asarray(mu, float)
        have = [c for c in _SIG_COLS if c in feats.columns]
        fj = test[["match_id"]].merge(feats[["match_id", "elo_diff", "neutral"] + have],
                                      on="match_id", how="left")
        elo_p = elo.predict_proba(fj[["elo_diff", "neutral"]])
        for i, r in enumerate(test.reset_index(drop=True).itertuples(index=False)):
            la, muu = wc_goals.correct(float(lam[i]), float(mu[i]))
            mat = dc.scoreline_matrix(la, muu)
            dcp = np.array(scoreline_to_outcome_probs(mat))
            b = (dcp + elo_p[i]) / 2; b = b / b.sum()
            res = {"H": 0, "D": 1, "A": 2}[r.result]
            row = {"year": y, "pH": b[0], "pD": b[1], "pA": b[2], "res": res,
                   "total": int(r.home_score + r.away_score),
                   "over35_pred": float(_over_prob(mat, 3.5))}
            fr = fj.iloc[i]
            for c in have:
                row[c] = fr[c]
            rows.append(row)
    d = pd.DataFrame(rows)
    # favorite/underdog orientation (favorite = higher win prob between H and A)
    d["fav_home"] = d["pH"] >= d["pA"]
    d["fav_p"] = np.where(d["fav_home"], d["pH"], d["pA"])
    d["und_p"] = np.where(d["fav_home"], d["pA"], d["pH"])
    d["upset_risk"] = d[["pH", "pA"]].min(axis=1)
    d["fav_lost"] = np.where(d["fav_home"], d["res"] == 2, d["res"] == 0).astype(int)  # underdog won outright
    d["upset"] = d["fav_lost"]              # outright upset
    return d


# ----------------------------------------------------------- Component 1: meters
def meters(d: pd.DataFrame, cfg: dict, write=True) -> pd.DataFrame:
    rows = []
    # upset_risk -> actual underdog-win rate
    d2 = d.copy()
    d2["bucket"] = pd.cut(d2["upset_risk"], [0, 0.12, 0.20, 0.28, 0.5])
    for b, g in d2.groupby("bucket", observed=True):
        rows.append({"meter": "upset_risk", "bucket": str(b), "n": len(g),
                     "mean_pred": round(g["upset_risk"].mean(), 3),
                     "actual_rate": round(g["upset"].mean(), 3)})
    # shootout_potential -> actual over-3.5 rate
    d2["sbucket"] = pd.qcut(d2["over35_pred"], 4, duplicates="drop")
    for b, g in d2.groupby("sbucket", observed=True):
        rows.append({"meter": "shootout(over3.5)", "bucket": str(b), "n": len(g),
                     "mean_pred": round(g["over35_pred"].mean(), 3),
                     "actual_rate": round((g["total"] > 3.5).mean(), 3)})
    t = pd.DataFrame(rows)
    if write:
        ensure_dirs(cfg)
        t.to_csv(path_for("reports", cfg) / "wc_variance_meters.csv", index=False)
    return t


# ------------------------------------------------------- Component 2: signal gate
def _signal_series(d: pd.DataFrame) -> dict:
    """Candidate signals oriented so HIGHER = favorite more vulnerable (upset more likely)."""
    fav_home = d["fav_home"].to_numpy()
    def fav(col_h, col_a):
        return np.where(fav_home, d[col_h], d[col_a]).astype(float)
    def und(col_h, col_a):
        return np.where(fav_home, d[col_a], d[col_h]).astype(float)
    sig = {}
    if "home_rest_days" in d:
        sig["fav_rest_deficit"] = und("home_rest_days", "away_rest_days") - fav("home_rest_days", "away_rest_days")
    if "home_travel_km" in d:
        sig["fav_travel_burden"] = (fav("home_travel_km", "away_travel_km")
                                    - und("home_travel_km", "away_travel_km")) / 1000.0
    if "venue_altitude_m" in d:
        sig["altitude_km"] = d["venue_altitude_m"].to_numpy(float) / 1000.0
    if "dead_rubber" in d:
        sig["dead_rubber"] = d["dead_rubber"].to_numpy(float)
    if "home_already_q" in d:
        sig["fav_already_qualified"] = fav("home_already_q", "away_already_q")
    return sig


def _apply_nudge(d: pd.DataFrame, s: np.ndarray, k: float) -> list:
    """Shift k·s of probability from the favorite to the underdog per game; return RPS list."""
    out = []
    for i, r in enumerate(d.itertuples(index=False)):
        b = np.array([r.pH, r.pD, r.pA])
        if np.isfinite(s[i]):
            shift = np.clip(k * s[i], -0.30, 0.30) * (r.fav_p if k * s[i] > 0 else r.und_p)
            if r.fav_home:
                b[0] -= shift; b[2] += shift
            else:
                b[2] -= shift; b[0] += shift
            b = np.clip(b, 1e-6, None); b = b / b.sum()
        out.append(_rps(b, r.res))
    return out


def signal_gate(d: pd.DataFrame, cfg: dict, write=True) -> pd.DataFrame:
    sigs = _signal_series(d)
    years = d["year"].to_numpy()
    base_rps = float(np.mean([_rps([r.pH, r.pD, r.pA], r.res) for r in d.itertuples(index=False)]))
    grid = np.linspace(-0.15, 0.15, 13)
    rows = []
    for name, raw in sigs.items():
        s = np.asarray(raw, float)
        valid = np.isfinite(s)
        cov = valid.mean()
        # standardize on observed support
        mu_, sd_ = np.nanmean(s[valid]), np.nanstd(s[valid]) or 1.0
        sz = (s - mu_) / sd_
        sz[~valid] = np.nan
        # leave-one-WC-out: pick k on train folds (min RPS), score held-out
        held = np.zeros(len(d))
        for y in WC_YEARS:
            tr, te = years != y, years == y
            if te.sum() == 0:
                continue
            dtr = d[tr].reset_index(drop=True); str_ = sz[tr]
            best_k, best = 0.0, 1e9
            for k in grid:
                rr = np.mean(_apply_nudge(dtr, np.nan_to_num(str_, nan=0.0), k))
                if rr < best:
                    best, best_k = rr, k
            dte = d[te].reset_index(drop=True)
            held[te] = _apply_nudge(dte, np.nan_to_num(sz[te], nan=0.0), best_k)
        held_rps = float(held.mean())
        # upset-subset RPS (games that were actually upsets)
        up = d["upset"].to_numpy().astype(bool)
        rows.append({"signal": name, "coverage": round(float(cov), 2),
                     "baseline_rps": round(base_rps, 4),
                     "held_out_rps": round(held_rps, 4),
                     "rps_improvement": round(base_rps - held_rps, 4),
                     "verdict": "KEEP" if (base_rps - held_rps) > 0.0005 and cov >= 0.3 else "reject"})
    t = pd.DataFrame(rows).sort_values("rps_improvement", ascending=False)
    if write:
        ensure_dirs(cfg)
        t.to_csv(path_for("reports", cfg) / "upset_signals.csv", index=False)
    return t


# --------------------------------------------------------- Component 3: τ dial
def dial_sweep(d: pd.DataFrame, cfg: dict, write=True) -> pd.DataFrame:
    taus = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0]
    up = d["upset"].to_numpy().astype(bool)
    rows = []
    for tau in taus:
        rps = []; correct = 0; und_caught = 0
        for r in d.itertuples(index=False):
            b = _temper(np.array([r.pH, r.pD, r.pA]), tau)
            rps.append(_rps(b, r.res))
            correct += int(np.argmax(b) == r.res)
            # underdog prob the (tempered) model assigns
            und = min(b[0], b[2])
            if r.upset and und >= 0.30:
                und_caught += 1
        recall = und_caught / max(int(up.sum()), 1)
        rows.append({"tau": tau, "rps": round(float(np.mean(rps)), 4),
                     "accuracy": round(correct / len(d), 3),
                     "upset_recall": round(recall, 3)})
    t = pd.DataFrame(rows)
    if write:
        ensure_dirs(cfg)
        t.to_csv(path_for("reports", cfg) / "upset_dial.csv", index=False)
    return t


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    d = _frame(cfg)
    print(f"[upset] {len(d)} World Cup matches (1998-2022), {int(d['upset'].sum())} outright upsets "
          f"({d['upset'].mean()*100:.0f}% of games)\n")
    print("=== Component 1: are the meters honest? (actual rate should rise with the meter) ===")
    mt = meters(d, cfg)
    print(mt.to_string(index=False))
    for meter in mt["meter"].unique():
        sub = mt[mt["meter"] == meter]
        spread = sub["actual_rate"].iloc[-1] - sub["actual_rate"].iloc[0]
        mono = sub["actual_rate"].is_monotonic_increasing
        verdict = ("STRONG — monotonic, trustworthy" if mono and spread > 0.05 else
                   "WEAK — real but noisy; can't cleanly pick individual games" if spread > 0.03 else
                   "no usable per-game signal")
        print(f"  {meter}: top-vs-bottom actual spread {spread:+.3f}  ->  {verdict}")
    print("\n=== Component 2: do upset signals help out-of-sample? (leave-one-WC-out) ===")
    print(signal_gate(d, cfg).to_string(index=False))
    print("  (KEEP only if it lowers held-out RPS; else honest negative — DC+Elo already prices it in.)")
    print("\n=== Component 3: the upset-sensitivity dial tradeoff ===")
    print(dial_sweep(d, cfg).to_string(index=False))
    print("  tau=1.0 is deployed (no change). Higher tau -> more upset recall but worse RPS/accuracy.")


if __name__ == "__main__":
    run()
