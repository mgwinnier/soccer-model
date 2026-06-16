"""World-Cup scoring-environment correction — matchup-strength aware.

The model is trained on all internationals (qualifiers and friendlies included) and
measurably under-projects World Cup goals vs ACTUAL results — and crucially **not
uniformly**. Backtesting expected goals vs real scores across 7 World Cups (1998-2022,
walk-forward leak-free) shows the **favorite** (the side with the higher expected goals)
is under-projected MORE than the underdog: pooled actual/model ratios are **1.19x** for
the favored side vs **1.12x** for the underdog. Strong teams convert their dominance into
goals at the World Cup more than the global model expects.

So instead of one flat multiplier we scale the favored side and the underdog side by
separate factors. This zeroes the goals bias AND improves pooled WC RPS (0.1992 -> 0.1983
walk-forward) over a single flat scale — the W/D/L forecast sharpens too, because
amplifying the favorite's goals correctly widens the win probability. Calibrated to
ACTUAL scores, not the betting line, so it needs no market input and applies to every WC
prediction (match cards, Team Explorer, the tournament simulator).

Estimation + validation: ``src/backtest/wc_goals_backtest.py``. The deployed constants
below are the pooled actual/model side ratios over all 7 backtested World Cups; a
walk-forward refit can persist overrides to ``data/models/wc_goals_correction.json``.
"""
from __future__ import annotations

import json

from ..config import load_config, path_for

# Pooled actual/model side-goal ratios over World Cups 1998-2022 (see backtest).
WC_FAV_SCALE = 1.186   # favored side  (higher expected goals)
WC_DOG_SCALE = 1.116   # underdog side (lower expected goals)


def load_scales(cfg: dict | None = None) -> tuple[float, float]:
    """(fav_scale, dog_scale), from data/models/wc_goals_correction.json if present,
    else the deployed defaults above. Mirrors the market_bias.json load pattern."""
    try:
        cfg = cfg or load_config()
        p = path_for("models", cfg) / "wc_goals_correction.json"
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            return float(d["fav_scale"]), float(d["dog_scale"])
    except Exception:
        pass
    return WC_FAV_SCALE, WC_DOG_SCALE


def save_scales(fav_scale: float, dog_scale: float, cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    p = path_for("models", cfg) / "wc_goals_correction.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"fav_scale": float(fav_scale), "dog_scale": float(dog_scale)},
              open(p, "w", encoding="utf-8"), indent=2)


def correct(lam: float, mu: float, fav_scale: float | None = None,
            dog_scale: float | None = None) -> tuple[float, float]:
    """Scale a single match's expected goals by the favorite/underdog WC multipliers.

    The favored side is whichever of (lam, mu) is larger; it gets ``fav_scale``, the
    other gets ``dog_scale``. Defaults to the deployed constants when not given.
    """
    if fav_scale is None:
        fav_scale = WC_FAV_SCALE
    if dog_scale is None:
        dog_scale = WC_DOG_SCALE
    if lam >= mu:
        return lam * fav_scale, mu * dog_scale
    return lam * dog_scale, mu * fav_scale
