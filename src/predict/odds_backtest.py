"""Backtest the model on matches already played this tournament.

Honest, leak-free design: the predictor is fit **only on data before the cutoff**
(default 2026-06-11, the World Cup kickoff), then scored against the **actual
results** of every match played since — RPS, log-loss, and pick accuracy, with a
per-match table of model probabilities vs what happened.

Important data limitation (handled honestly): **ESPN nulls the moneyline odds of a
match once it finishes**, so pre-match closing lines for already-played games are
not recoverable retroactively. The model-vs-Vegas comparison therefore runs on
*upcoming* matches (`src/predict/value.py`), where odds still exist. Here we score
the model against reality; where a played match still carries odds, we include the
market too.

Small-sample caveat: a handful of group-stage matches cannot prove an edge — this
is a live scoreboard of model accuracy, not a profitability claim.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.clean import load_matches
from ..data.odds import fetch_espn_range, american_to_decimal, devig, decimal_to_prob
from ..features.elo import compute_elo_features
from ..models.dixon_coles import DixonColesModel
from ..models.elo_model import EloModel
from ..models.base import OUTCOMES
from ..backtest.metrics import evaluate
from ..simulate.bracket_2026 import HOST_TEAMS

VALUE_THRESHOLD = 0.05


class _AsOfPredictor:
    """DC + Elo blend trained strictly on matches before ``cutoff``."""

    def __init__(self, cutoff: str, cfg: dict):
        self.cfg = cfg
        matches = load_matches(cfg)
        pre = matches[matches["date"] < pd.Timestamp(cutoff)].copy()
        self.dc = DixonColesModel.from_config(cfg).fit(pre)
        elo_feats, engine = compute_elo_features(pre, cfg)
        self.ratings = engine.ratings
        feat = pre.set_index("match_id").join(elo_feats)
        feat = feat.dropna(subset=["elo_diff"])
        self.elo_model = EloModel(cfg=cfg).fit(feat)
        self.known = set(self.dc.teams_)

    def predict(self, home: str, away: str, neutral: bool) -> np.ndarray | None:
        if home not in self.known or away not in self.known:
            return None
        fx = pd.DataFrame([{"home_team": home, "away_team": away, "neutral": neutral}])
        dc_p = self.dc.predict_proba(fx)[0]
        ediff = self.ratings.get(home, 1500) - self.ratings.get(away, 1500)
        elo_p = self.elo_model.predict_proba(
            pd.DataFrame([{"elo_diff": ediff, "neutral": neutral}]))[0]
        blend = (dc_p + elo_p) / 2
        return blend / blend.sum()


def _played_matches(start: str, end: str, cfg: dict) -> list[dict]:
    rows = []
    for ev in fetch_espn_range(start, end, cfg=cfg, use_cache=False):
        if ev["status"] != "post":          # only finished matches
            continue
        if ev["home_score"] is None:        # need a final score; odds optional
            continue
        rows.append(ev)
    return rows


def run(cutoff: str = "2026-06-11", end: str | None = None,
        cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    end = end or date.today().strftime("%Y-%m-%d")
    pred = _AsOfPredictor(cutoff, cfg)
    events = _played_matches(cutoff, end, cfg)

    recs = []
    for ev in events:
        home, away = ev["home_team"], ev["away_team"]
        neutral = home not in HOST_TEAMS
        model = pred.predict(home, away, neutral)
        if model is None:
            continue
        hs, as_ = ev["home_score"], ev["away_score"]
        outcome = "H" if hs > as_ else ("D" if hs == as_ else "A")
        # market is only present if ESPN still carries odds (upcoming games)
        market = None
        if ev["ml_home"] is not None:
            dec = [american_to_decimal(ev["ml_home"]),
                   american_to_decimal(ev["ml_draw"]),
                   american_to_decimal(ev["ml_away"])]
            market = devig([decimal_to_prob(x) for x in dec], "proportional")
        recs.append({
            "date": pd.to_datetime(ev["date"]).strftime("%m-%d"),
            "home": home, "away": away, "score": f"{hs}-{as_}", "result": outcome,
            "model": model,
            "model_pick": OUTCOMES[int(np.argmax(model))],
            "hit": OUTCOMES[int(np.argmax(model))] == outcome,
            "p_outcome": float(model[OUTCOMES.index(outcome)]),
            "has_market": market is not None,
        })

    if not recs:
        return {"n": 0}

    df = pd.DataFrame(recs)
    y = df["result"]
    model_m = evaluate(np.vstack(df["model"]), y)
    summary = {
        "n": len(df),
        "model_rps": model_m["rps"], "model_acc": model_m["accuracy"],
        "model_logloss": model_m["log_loss"],
        "n_with_market": int(df["has_market"].sum()),
        "table": df,
    }
    return summary


def _fmt(p):
    return "/".join(f"{x*100:.0f}" for x in p)


if __name__ == "__main__":
    import sys
    try:                                  # render accents/em-dash on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    cutoff = sys.argv[1] if len(sys.argv) > 1 else "2026-06-11"
    s = run(cutoff)
    if s["n"] == 0:
        print("No finished matches found in range.")
        raise SystemExit
    df = s["table"]
    print(f"\nMODEL vs REALITY — {s['n']} matches played since {cutoff}")
    print(f"(predictor trained ONLY on pre-{cutoff} data — leak-free)\n")
    show = df.copy()
    show["model %(H/D/A)"] = show["model"].map(_fmt)
    show["P(actual)"] = (show["p_outcome"] * 100).round(0).astype(int).astype(str) + "%"
    show["hit?"] = show["hit"].map({True: "HIT", False: "-"})
    cols = ["date", "home", "away", "score", "result", "model %(H/D/A)",
            "model_pick", "P(actual)", "hit?"]
    with pd.option_context("display.width", 240, "display.max_rows", 100):
        print(show[cols].to_string(index=False))
    hits = int(df["hit"].sum())
    print(f"\n--- scorecard (model trained only on pre-tournament data) ---")
    print(f"  Correct W/D/L picks : {hits}/{s['n']}  ({hits/s['n']*100:.0f}%)")
    print(f"  RPS                 : {s['model_rps']:.4f}  (≈0.20 is bookmaker-grade)")
    print(f"  Log-loss            : {s['model_logloss']:.4f}")
    print(f"\n  Note: ESPN nulls odds once a match ends, so pre-match Vegas lines for")
    print(f"  these games are gone. For model-vs-Vegas on UPCOMING games with live")
    print(f"  odds + value picks, run:  python -m src.predict.value {cutoff}")
