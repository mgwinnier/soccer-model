"""Tests for v8: pre-registered segment validation + forward CLV system tag."""
import numpy as np
import pandas as pd
import pytest

from src.models.market_bias import MarketBias
from src.backtest.segment_validate import _band_bets, _segment_masks, family_wise_test
from src.predict.clv import _system_tag


def _preds(rows):
    return pd.DataFrame(rows)


def test_band_selector_picks_only_moneyline_2_3():
    # MR in band (kept), MR out of band (dropped), TG in the band range (dropped: not MR)
    df = _preds([
        {"type": "MR:H", "dec": 2.5, "model_p": 0.55, "fair_p": 0.40, "result": "win"},
        {"type": "MR:A", "dec": 2.9, "model_p": 0.45, "fair_p": 0.34, "result": "loss"},
        {"type": "MR:H", "dec": 1.8, "model_p": 0.70, "fair_p": 0.55, "result": "win"},  # below band
        {"type": "MR:A", "dec": 3.4, "model_p": 0.40, "fair_p": 0.29, "result": "loss"},  # above band
        {"type": "TG:over", "dec": 2.5, "model_p": 0.60, "fair_p": 0.40, "result": "win"},  # not MR
    ])
    sel = _band_bets(df, MarketBias({}), band=(2.0, 3.0), min_ev=0.03)
    assert set(sel["type"]) <= {"MR:H", "MR:A"}
    assert sel["dec"].between(2.0, 3.0, inclusive="left").all()
    assert len(sel) == 2


def test_segment_masks_identify_pickem():
    s = _preds([{"type": "MR:H", "dec": 2.5}, {"type": "MR:A", "dec": 6.0},
                {"type": "TG:under", "dec": 1.9}])
    masks = _segment_masks(s)
    assert list(masks["MR_pickem 2.0-3.0"]) == [True, False, False]
    assert list(masks["MR 5.0+"]) == [False, True, False]


def test_family_wise_flags_a_strong_real_edge():
    # 30 pick'em bets that ALL win at dec 2.5 (fair 0.40) -> a huge real edge; a few
    # other-segment bets so the family has >1 member. The candidate must land near the
    # top of the max-null distribution.
    rows = [{"type": "MR:H", "dec": 2.5, "fair_p": 0.40, "result": "win"} for _ in range(30)]
    rows += [{"type": "TG:over", "dec": 1.9, "fair_p": 0.53, "result": "loss"} for _ in range(20)]
    fw = family_wise_test(_preds(rows), n=2000)
    assert fw["pickem_roi"] > 1.0                       # ~ +150% ROI (all win at 2.5)
    assert fw["pickem_percentile_vs_maxnull"] >= 0.95   # clears the family-wise bar


def test_family_wise_does_not_flag_a_fair_segment():
    # pick'em wins at exactly the fair rate (12/30 ≈ 0.40) -> ~0% ROI, must NOT pass.
    res = ["win"] * 12 + ["loss"] * 18
    rows = [{"type": "MR:H", "dec": 2.5, "fair_p": 0.40, "result": r} for r in res]
    rows += [{"type": "MR:A", "dec": 6.0, "fair_p": 0.16, "result": "loss"} for _ in range(20)]
    fw = family_wise_test(_preds(rows), n=2000)
    assert abs(fw["pickem_roi"]) < 0.05                 # ~ break-even
    assert fw["pickem_percentile_vs_maxnull"] < 0.95    # not significant after correction


def test_system_tag():
    assert _system_tag("Match Result", 2.5) == "pickem_ml_2_3"
    assert _system_tag("Match Result", 2.0) == "pickem_ml_2_3"   # band is [2.0, 3.0)
    assert _system_tag("Match Result", 3.0) == ""                # upper bound exclusive
    assert _system_tag("Match Result", 1.8) == ""
    assert _system_tag("Total Goals", 2.5) == ""
    assert _system_tag("Match Result", None) == ""
