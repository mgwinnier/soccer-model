"""Data-grounded market-relative de-bias.

The model systematically sits a few points off the sharp line in one direction
(it backs unders/underdogs/draws). We estimate that **per-role tilt** —
`mean(model_prob − Bet365 de-vigged prob)` for each role (home/draw/away,
over/under, spread sides) — on a large historical-odds sample, then subtract it so
deployed bets reflect only *game-specific* disagreement, not the blanket lean.

This replaces the noisy per-16-game estimate in ``value._recenter_matches`` with a
stable vector learned from thousands of real prices.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import load_config, path_for, ensure_dirs


class MarketBias:
    def __init__(self, bias: dict[str, float] | None = None, shrink: float = 0.8):
        self.bias = bias or {}
        self.shrink = shrink

    def recenter(self, type_key: str, model_p: float) -> float:
        adj = model_p - self.shrink * self.bias.get(type_key, 0.0)
        return min(max(adj, 1e-4), 1 - 1e-4)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"shrink": self.shrink, "bias": self.bias},
                  open(path, "w", encoding="utf-8"), indent=2)

    @classmethod
    def load(cls, path: Path) -> "MarketBias":
        if not path.exists():
            return cls({})
        d = json.load(open(path, encoding="utf-8"))
        return cls(d.get("bias", {}), d.get("shrink", 0.8))


def fit_market_bias(type_keys, model_p, fair_p, shrink: float = 0.8) -> MarketBias:
    """Estimate per-role mean(model − market) bias from aligned arrays."""
    import numpy as np
    type_keys = list(type_keys)
    model_p = np.asarray(model_p, float)
    fair_p = np.asarray(fair_p, float)
    bias: dict[str, list] = {}
    for tk, mp, fp in zip(type_keys, model_p, fair_p):
        if fp is None or (isinstance(fp, float) and np.isnan(fp)):
            continue
        bias.setdefault(tk, []).append(mp - fp)
    return MarketBias({k: float(np.mean(v)) for k, v in bias.items() if v}, shrink)


def default_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return path_for("models", cfg) / "market_bias.json"


def load_default(cfg: dict | None = None) -> MarketBias:
    return MarketBias.load(default_path(cfg))
