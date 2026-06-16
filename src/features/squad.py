"""Squad-strength features from market value (and FIFA ratings if available).

Squad market value is a strong, slowly-varying proxy for team strength. The
scraped values are *current* (2026), so attaching them to a 2010 match would be
anachronistic leakage. We therefore only apply squad value to matches on/after
``valid_from`` (default 2024) and leave it missing otherwise — which is exactly
what an honest pre-2024 model would have had. LightGBM handles the NaNs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for

VALID_FROM = pd.Timestamp("2024-01-01")


def _load_squad_values(cfg: dict) -> pd.DataFrame | None:
    raw = path_for("data_raw", cfg)
    for name in ("squad_values.csv", "squad_values_static.csv"):
        p = raw / name
        if p.exists():
            df = pd.read_csv(p)
            if {"team", "squad_value_eur"}.issubset(df.columns):
                return df.drop_duplicates("team")
    return None


def compute_squad_features(
    matches: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    """Return squad-value features indexed by ``match_id`` (NaN where unknown)."""
    cfg = cfg or load_config()
    idx = matches.set_index("match_id")
    out = pd.DataFrame(index=idx.index)
    sv = _load_squad_values(cfg)
    if sv is None:
        out["home_squad_value_log"] = np.nan
        out["away_squad_value_log"] = np.nan
        out["squad_value_log_diff"] = np.nan
        return out

    val = sv.set_index("team")["squad_value_eur"].to_dict()

    def lookup(team_series: pd.Series) -> pd.Series:
        return team_series.map(val).astype(float)

    hv = lookup(idx["home_team"])
    av = lookup(idx["away_team"])
    # zero out anachronistic application
    recent = idx["date"] >= VALID_FROM
    hv = hv.where(recent, np.nan)
    av = av.where(recent, np.nan)

    out["home_squad_value_log"] = np.log1p(hv)
    out["away_squad_value_log"] = np.log1p(av)
    out["squad_value_log_diff"] = out["home_squad_value_log"] - out["away_squad_value_log"]
    return out
