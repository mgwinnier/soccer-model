"""xG-informed Dixon-Coles (Base D).

Identical machinery to the goals-based Dixon-Coles, but the regression target is
**blended with expected goals where xG exists**: `target = (1-w)·goals + w·xG`.
xG is a lower-variance estimate of how many goals a performance "deserved", so on
the elite-tournament matches that carry xG this de-noises the attack/defense
ratings; everywhere else it falls back to actual goals, so the model still trains
on the full history (no shrinkage to a few hundred games).

Added as a 4th ensemble member — the OOF simplex blender will down-weight it to
zero if the ablation says it doesn't help, so including it is safe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from .dixon_coles import DixonColesModel


class XgDixonColesModel(DixonColesModel):
    def __init__(self, xg_weight: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        self.xg_weight = xg_weight

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "XgDixonColesModel":
        cfg = cfg or load_config()
        dc = cfg["dixon_coles"]
        xgw = cfg.get("xg", {}).get("blend_weight", 0.6)
        return cls(xg_weight=xgw, xi=dc["xi"], max_goals=dc["max_goals"],
                   l2_penalty=dc["l2_penalty"])

    def _targets(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        gh = matches["home_score"].to_numpy().astype(float)
        ga = matches["away_score"].to_numpy().astype(float)
        if "home_xg" not in matches.columns:
            return gh, ga
        xh = pd.to_numeric(matches["home_xg"], errors="coerce").to_numpy()
        xa = pd.to_numeric(matches["away_xg"], errors="coerce").to_numpy()
        w = self.xg_weight
        yh = np.where(np.isnan(xh), gh, (1 - w) * gh + w * xh)
        ya = np.where(np.isnan(xa), ga, (1 - w) * ga + w * xa)
        return yh, ya
