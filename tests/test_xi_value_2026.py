"""Tests for the 2026 XI-value forward-tracker aggregation (pure, synthetic rows)."""
import pandas as pd

from src.backtest.xi_value_2026 import summarize


def test_summarize_winrate_and_buckets():
    df = pd.DataFrame([
        # home is model+value favorite, strong XI, home won -> value-fav won, no upset (high band)
        {"result": "H", "p_home": 0.70, "p_away": 0.15,
         "home_val": 500, "away_val": 100, "home_share": 0.95, "away_share": 0.90},
        # home is model+value favorite but a weak XI (0.55 share) and LOST -> upset, low band
        {"result": "A", "p_home": 0.65, "p_away": 0.20,
         "home_val": 300, "away_val": 120, "home_share": 0.55, "away_share": 0.95},
    ])
    s = summarize(df)
    assert s["n"] == 2 and s["n_decisive"] == 2
    assert s["value_fav_winrate"] == 0.5          # value-fav (home) won game1, lost game2
    bands = {b["band"]: b for b in s["fav_share_buckets"]}
    assert bands["0-60%"]["upset_rate"] == 1.0    # the 55%-share favorite got upset
    assert bands["70-100%"]["upset_rate"] == 0.0  # the 95%-share favorite held


def test_summarize_empty():
    assert summarize(pd.DataFrame())["n"] == 0
