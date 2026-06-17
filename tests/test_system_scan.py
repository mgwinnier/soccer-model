"""Tests for the honest system finder (src/backtest/system_scan.py).

These exercise the pure helpers on synthetic data (no slow build_predictions): the
slice library, per-slice metrics, and — most importantly — that the family-wise
max-null bar is calibrated (no-edge data rarely clears it) yet a genuinely-edged
slice DOES clear it (the harness can detect a real edge when one exists)."""
import numpy as np
import pandas as pd

from src.backtest.system_scan import _slices, _metrics, _maxnull_block


def _frame(n, fair=0.5, dec=2.0, win_prob=None, seed=0):
    rng = np.random.default_rng(seed)
    win_prob = fair if win_prob is None else win_prob
    res = np.where(rng.random(n) < win_prob, "win", "loss")
    return pd.DataFrame({
        "date": pd.Timestamp("2024-01-01"), "league": "fifa.world",
        "market": "Match Result", "type": "MR:A",
        "model_p": fair, "fair_p": fair, "dec": dec, "result": res,
        "cons_edge": -0.05, "ev": fair * (dec - 1) - (1 - fair), "edge": 0.03,
    })


def test_metrics_basic():
    df = _frame(100, win_prob=1.0)            # all wins at dec 2.0 -> +100% ROI
    m = _metrics(df)
    assert m["n"] == 100 and m["wins"] == 100
    assert abs(m["roi"] - 1.0) < 1e-9         # win pays dec-1 = 1.0


def test_slices_select_right_rows():
    df = _frame(50)
    df.loc[:9, "dec"] = 3.0                    # 10 dogs
    sl = _slices(df)
    assert sl["side:MR:A"].all()              # all rows are MR:A
    assert sl["odds dog 2.5-4"].sum() == 10   # exactly the 10 with dec=3.0


def test_maxnull_calibrated_on_no_edge():
    # outcomes drawn at the fair prob -> no edge. The max-null bar should be POSITIVE
    # (mining many slices finds spurious +ROI) and the REALIZED no-edge ROI should
    # almost never exceed it.
    df = _frame(400, fair=0.5, dec=2.0, win_prob=0.5, seed=1)
    masks = [np.ones(len(df), bool), (df["dec"] >= 2.5).to_numpy(),
             np.arange(len(df)) % 2 == 0, np.arange(len(df)) % 3 == 0]
    bar = _maxnull_block(df, masks, n=500)
    assert bar > 0.0                          # mining finds spurious edge -> bar above 0
    real_roi = (np.where(df["result"] == "win", df["dec"] - 1, -1.0)).mean()
    assert real_roi <= bar                    # honest no-edge data does NOT clear the bar


def test_maxnull_catches_a_real_edge():
    # inject a genuine edge: bets win 70% at dec 2.0 (true +40% ROI). It must clear the
    # null bar built from the same fair_p=0.5 — proving the harness detects real edges.
    df = _frame(400, fair=0.5, dec=2.0, win_prob=0.70, seed=2)
    masks = [np.ones(len(df), bool)]
    bar = _maxnull_block(df, masks, n=500)
    real_roi = (np.where(df["result"] == "win", df["dec"] - 1, -1.0)).mean()
    assert real_roi > bar                     # a true +40% edge beats the no-skill null
