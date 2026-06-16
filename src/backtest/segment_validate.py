"""Pre-registered validation of the one candidate edge (v8).

Segmenting the out-of-sample bets surfaced a single pattern that cleared zero: the
model's **moneyline picks on near-even matches (decimal odds ∈ [2.0, 3.0))**. But it
was found by mining ~16 slices, so a 95% CI clearing zero is *expected* once by
chance. This module runs the honest tests — with the rule and pass/fail bar frozen
in the plan **before** these ran — and lets the data decide. No deployment unless it
passes all three.

Tests
  1. Family-wise (multiple-comparison) — resample each selected bet's outcome from
     its **market-fair** probability (null = "no skill beyond the market"), record the
     **best-of-all-segments** ROI each draw, and require the pick'em band's real ROI to
     beat the **95th percentile** of that max-null. This is the FWER-correct bar.
  2. Out-of-time replication — evaluate the frozen rule on the **never-mined
     2019–2022** slice (market-bias fit on the disjoint 2023+ slice). The clean test.
  3. Sensitivity — vary band edges / min_ev / anchor_w / recentering; a real edge is
     stable across the neighborhood, not a knife-edge at the default knobs.

Run:  python -m src.backtest.segment_validate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..models.market_bias import fit_market_bias, MarketBias
from .odds_history_backtest import build_predictions, _bootstrap_roi

MINING_SPLIT = "2023-01-01"          # the slice the band was discovered on (>= this)
BAND = (2.0, 3.0)                    # frozen: moneyline decimal-odds band
MIN_EV = 0.03                        # frozen: deployed EV threshold
ANCHOR_W = 0.5                       # frozen: deployed anchoring weight

# Pass/fail bar (pre-registered in the plan)
FWER_BAR = 0.95                      # actual ROI must beat 95th pct of the max-null
REPL_CI_FLOOR = -0.05               # replication CI low must exceed this (no meaningful loss)
SENS_FRACTION = 0.70                # >= 70% of sensitivity grid cells positive


# ---------------------------------------------------------------- helpers
def _ev(bias: MarketBias, df: pd.DataFrame) -> np.ndarray:
    adj = np.array([bias.recenter(t, p) for t, p in zip(df["type"], df["model_p"])])
    return adj * (df["dec"].to_numpy() - 1) - (1 - adj)


def _band_bets(preds: pd.DataFrame, bias: MarketBias, band=BAND, min_ev=MIN_EV) -> pd.DataFrame:
    """The frozen rule: Match Result, decimal odds in [band), model EV >= min_ev."""
    mr = preds[preds["type"].str.startswith("MR")].copy()
    mr = mr[(mr["dec"] >= band[0]) & (mr["dec"] < band[1])]
    mr = mr[_ev(bias, mr) >= min_ev]
    return mr


def _roi_ci(df: pd.DataFrame):
    s = df[df["result"] != "push"]
    if len(s) < 5:
        return {"bets": len(df), "roi": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "units": float(df.get("pnl", pd.Series()).sum())}
    pnl = np.where(s["result"] == "win", s["dec"] - 1, -1.0)
    m, lo, hi = _bootstrap_roi(pnl)
    return {"bets": len(df), "roi": m, "lo": lo, "hi": hi, "units": float(pnl.sum())}


# ------------------------------------------------------ the candidate-segment family
def _segment_masks(sel: pd.DataFrame) -> dict[str, np.ndarray]:
    """The ~12 segments that were actually eyeballed — the multiple-comparison family."""
    t = sel["type"].to_numpy()
    dec = sel["dec"].to_numpy()
    mr = sel["type"].str.startswith("MR").to_numpy()
    masks = {role: (t == role) for role in
             ("MR:H", "MR:D", "MR:A", "TG:over", "TG:under", "SP:home", "SP:away")}
    masks["MR fav<1.5"] = mr & (dec < 1.5)
    masks["MR 1.5-2.0"] = mr & (dec >= 1.5) & (dec < 2.0)
    masks["MR_pickem 2.0-3.0"] = mr & (dec >= 2.0) & (dec < 3.0)   # the candidate
    masks["MR 3.0-5.0"] = mr & (dec >= 3.0) & (dec < 5.0)
    masks["MR 5.0+"] = mr & (dec >= 5.0)
    return masks


def family_wise_test(sel: pd.DataFrame, n: int = 4000, seed: int = 0) -> dict:
    """Null = each selected bet wins at its market-fair rate (no model skill). Compare
    the pick'em band's real ROI to the 95th pct of the best-segment ROI under the null."""
    s = sel[sel["result"] != "push"].reset_index(drop=True)
    fair = s["fair_p"].to_numpy(float)
    dec = s["dec"].to_numpy(float)
    pnl_actual = np.where((s["result"] == "win").to_numpy(), dec - 1, -1.0)
    masks = {k: m for k, m in _segment_masks(s).items() if m.sum() >= 15}

    actual = {k: float(pnl_actual[m].mean()) for k, m in masks.items()}
    rng = np.random.default_rng(seed)
    win = rng.random((n, len(s))) < fair[None, :]
    pnl = np.where(win, (dec - 1)[None, :], -1.0)
    max_null = np.full(n, -np.inf)
    for m in masks.values():
        max_null = np.maximum(max_null, pnl[:, m].mean(axis=1))
    pe = actual["MR_pickem 2.0-3.0"]
    pct = float((max_null < pe).mean())                      # fraction of max-null below actual
    return {"actual": actual, "pickem_roi": pe,
            "max_null_p95": float(np.percentile(max_null, 95)),
            "pickem_percentile_vs_maxnull": pct, "n_bets": int(len(s))}


# ------------------------------------------------------------- run all tests
def run(cfg: dict | None = None, write: bool = True) -> dict:
    cfg = cfg or load_config()
    preds = build_predictions(cfg, anchor_w=ANCHOR_W, split=MINING_SPLIT)
    if preds.empty:
        print("[validate] no predictions — harvest odds first")
        return {}

    mine = preds[preds["date"] >= pd.Timestamp(MINING_SPLIT)].copy()
    bias_mine = fit_market_bias(preds[preds["date"] < pd.Timestamp(MINING_SPLIT)]["type"],
                               preds[preds["date"] < pd.Timestamp(MINING_SPLIT)]["model_p"],
                               preds[preds["date"] < pd.Timestamp(MINING_SPLIT)]["fair_p"])
    # all +EV bets on the mining slice (the family-wise universe)
    sel = mine[_ev(bias_mine, mine) >= MIN_EV].copy()
    fw = family_wise_test(sel)

    # Test 2 — replication on the never-mined 2019-2022 slice (bias fit on disjoint 2023+)
    later = preds[preds["date"] >= pd.Timestamp(MINING_SPLIT)]
    bias_repl = fit_market_bias(later["type"], later["model_p"], later["fair_p"])
    early = preds[preds["date"] < pd.Timestamp(MINING_SPLIT)].copy()
    repl_df = _band_bets(early, bias_repl)
    repl = _roi_ci(repl_df)

    # Test 4 — closing-line corroboration: mean edge vs multi-book consensus on the band
    band_all = _band_bets(mine, bias_mine)
    ce = band_all["cons_edge"].dropna()
    cons = {"n_with_consensus": int(len(ce)),
            "mean_cons_edge": float(ce.mean()) if len(ce) else float("nan")}

    # Test 3 — sensitivity grid (anchor_w needs a rebuild; band/min_ev/recenter are cheap)
    grid = []
    for aw in (0.4, 0.5, 0.6):
        p = preds if aw == ANCHOR_W else build_predictions(cfg, anchor_w=aw, split=MINING_SPLIT)
        m = p[p["date"] >= pd.Timestamp(MINING_SPLIT)].copy()
        tr = p[p["date"] < pd.Timestamp(MINING_SPLIT)]
        b = fit_market_bias(tr["type"], tr["model_p"], tr["fair_p"])
        zero = MarketBias({})
        for band in ((1.8, 2.8), (1.9, 3.1), (2.0, 3.0), (2.1, 2.9)):
            for mev in (0.02, 0.03, 0.05):
                for rc, bs in (("on", b), ("off", zero)):
                    df = _band_bets(m, bs, band=band, min_ev=mev)
                    r = _roi_ci(df)
                    grid.append({"anchor_w": aw, "band": f"{band[0]}-{band[1]}",
                                 "min_ev": mev, "recenter": rc,
                                 "bets": r["bets"], "roi": r["roi"]})
    gdf = pd.DataFrame(grid)
    valid = gdf[gdf["bets"] >= 15]
    frac_pos = float((valid["roi"] > 0).mean()) if len(valid) else float("nan")

    # ---- pre-registered verdict ----
    pass_a = fw["pickem_percentile_vs_maxnull"] >= FWER_BAR
    pass_b = (repl["roi"] > 0) and (repl["lo"] > REPL_CI_FLOOR)
    pass_c = frac_pos >= SENS_FRACTION
    verdict = "PASS" if (pass_a and pass_b and pass_c) else "FAIL"

    out = {"family_wise": fw, "replication": repl, "consensus": cons,
           "sensitivity_frac_positive": frac_pos, "sensitivity": gdf,
           "pass_a_fwer": pass_a, "pass_b_replication": pass_b,
           "pass_c_sensitivity": pass_c, "verdict": verdict}
    if write:
        ensure_dirs(cfg)
        rows = [{"test": "family_wise", "metric": "pickem_pct_vs_maxnull",
                 "value": fw["pickem_percentile_vs_maxnull"], "pass": pass_a},
                {"test": "replication_2019_2022", "metric": "roi",
                 "value": repl["roi"], "pass": pass_b},
                {"test": "sensitivity", "metric": "frac_positive",
                 "value": frac_pos, "pass": pass_c}]
        pd.DataFrame(rows).to_csv(path_for("reports", cfg) / "segment_validation.csv",
                                  index=False)
    return out


def _pct(x):
    return "nan" if x != x else f"{x*100:+.1f}%"


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    r = run()
    if not r:
        raise SystemExit
    fw = r["family_wise"]
    print("\n=== v8 VALIDATION: pick'em moneyline band [2.0,3.0) ===\n")
    print("Candidate-segment ROIs on the mining slice (the multiple-comparison family):")
    for k, v in sorted(fw["actual"].items(), key=lambda kv: -kv[1]):
        mark = "  <-- candidate" if k.startswith("MR_pickem") else ""
        print(f"  {k:20s}: {_pct(v)}{mark}")
    print(f"\n(1) FAMILY-WISE: pick'em ROI {_pct(fw['pickem_roi'])} vs max-null 95th pct "
          f"{_pct(fw['max_null_p95'])} -> percentile {fw['pickem_percentile_vs_maxnull']*100:.1f}% "
          f"(need >= {FWER_BAR*100:.0f}%)  [{'PASS' if r['pass_a_fwer'] else 'FAIL'}]")
    rp = r["replication"]
    print(f"(2) REPLICATION 2019-2022 (never mined): {rp['bets']} bets, ROI {_pct(rp['roi'])} "
          f"[CI {_pct(rp['lo'])},{_pct(rp['hi'])}]  [{'PASS' if r['pass_b_replication'] else 'FAIL'}]")
    print(f"(3) SENSITIVITY: {r['sensitivity_frac_positive']*100:.0f}% of grid cells positive "
          f"(need >= {SENS_FRACTION*100:.0f}%)  [{'PASS' if r['pass_c_sensitivity'] else 'FAIL'}]")
    c = r["consensus"]
    print(f"(4) closing-line corroboration: mean edge vs consensus {_pct(c['mean_cons_edge'])} "
          f"on {c['n_with_consensus']} band bets (info only)")
    print(f"\nVERDICT: {r['verdict']}  — pre-registered: deploy the pick'em filter ONLY on PASS; "
          f"otherwise it's a data-mining artifact and nothing ships.")
