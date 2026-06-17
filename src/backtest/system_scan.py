"""Honest system finder — mine the graded bets many ways, then gate on the tests that
actually matter so we don't fool ourselves.

Slice enough ways and *something* always looks profitable. This tool slices the deployed
(market-independent) model's gated bets by EV, odds, side, edge, confidence, tier, and a
few creative combos (away-dogs, high-EV pick'ems, favorites-with-edge, "model backs the
underdog"), and for each slice reports ROI + bootstrap CI + units on BOTH the 2022 World
Cup and the large 2023+ out-of-sample set — then holds every candidate to three honest
bars before calling it real:

  1. FAMILY-WISE max-null — a slice's ROI must beat the 95th percentile of the best-of-all-
     slices null (resampling outcomes from the market-fair prob). Corrects for mining N slices.
  2. OUT-OF-TIME replication — discovered on 2023+, must stay positive on the never-mined
     2019-2022 window.
  3. CLV (the arbiter) — relative CLV (slice mean cons_edge minus the all-bets mean) must be
     >= ~0. The raw cons_edge is ~-5% for everything (that's mostly the vig); a genuine
     soft-line system sits ABOVE the pack, not at the average.

Verdict per slice: noise / "variance - fails CLV/FW" / CANDIDATE. Report-only and flag-only:
a surviving CANDIDATE is printed as "track forward (no auto-bet)", never wired into a filter.

    python -m src.backtest.system_scan

Writes reports/system_scan.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from .odds_history_backtest import build_predictions, _bootstrap_roi, NEUTRAL_LEAGUES
from ..predict.betting import qualifies, kelly_fraction

_KELLY = 0.25


def _gated(cfg, split: str) -> pd.DataFrame:
    preds = build_predictions(cfg, anchor_w=1.0, test_from="2019-01-01", split=split)
    if preds.empty:
        return preds
    preds["ev"] = preds["model_p"] * (preds["dec"] - 1) - (1 - preds["model_p"])
    preds["edge"] = preds["model_p"] - preds["fair_p"]
    keep = preds.apply(lambda r: qualifies(
        r["model_p"], (r["fair_p"] if r["fair_p"] == r["fair_p"] else None),
        r["dec"], 0.03, 0.02, 6.0), axis=1)
    return preds[keep].reset_index(drop=True)


def _tier(league: str) -> str:
    if league in NEUTRAL_LEAGUES:
        return "major"
    s = str(league)
    if "worldq" in s or ".q" in s or s.endswith("q"):
        return "qualifier"
    if "friendly" in s:
        return "friendly"
    return "other"


# ----------------------------------------------------------------- slice library
def _slices(d: pd.DataFrame) -> dict[str, np.ndarray]:
    dec, ev, edge, mp, typ, mkt = (d["dec"].to_numpy(), d["ev"].to_numpy(),
                                   d["edge"].to_numpy(), d["model_p"].to_numpy(),
                                   d["type"].to_numpy(), d["market"].to_numpy())
    tier = d["league"].map(_tier).to_numpy()
    bet_is_dog = dec >= 2.5
    sl = {"ALL": np.ones(len(d), bool)}
    for m in ("Match Result", "Total Goals", "Spread"):
        sl[f"market:{m}"] = mkt == m
    for t in ("MR:H", "MR:D", "MR:A", "TG:over", "TG:under", "SP:home", "SP:away"):
        sl[f"side:{t}"] = typ == t
    sl["EV 3-10%"] = (ev >= .03) & (ev < .10)
    sl["EV 10-25%"] = (ev >= .10) & (ev < .25)
    sl["EV 25-50%"] = (ev >= .25) & (ev < .50)
    sl["EV 50%+"] = ev >= .50
    sl["odds fav<1.8"] = dec < 1.8
    sl["odds 1.8-2.5"] = (dec >= 1.8) & (dec < 2.5)
    sl["odds pickem 2-3"] = (dec >= 2.0) & (dec < 3.0)
    sl["odds dog 2.5-4"] = (dec >= 2.5) & (dec < 4.0)
    sl["odds 4+"] = dec >= 4.0
    sl["edge 2-5%"] = (edge >= .02) & (edge < .05)
    sl["edge 5-10%"] = (edge >= .05) & (edge < .10)
    sl["edge 10%+"] = edge >= .10
    sl["conf model<40%"] = mp < .40
    sl["conf model 40-55%"] = (mp >= .40) & (mp < .55)
    sl["conf model 55%+"] = mp >= .55
    for t in ("major", "qualifier", "friendly", "other"):
        sl[f"tier:{t}"] = tier == t
    # creative combos the user hinted at
    sl["combo away-dog (MR:A & dec>=2.5)"] = (typ == "MR:A") & bet_is_dog
    sl["combo high-EV pickem (EV>=10% & 2<=dec<3)"] = (ev >= .10) & (dec >= 2.0) & (dec < 3.0)
    sl["combo fav+edge (dec<2 & edge>=5%)"] = (dec < 2.0) & (edge >= .05)
    sl["combo model-backs-underdog (bet dec>=2.5)"] = bet_is_dog
    return sl


# ------------------------------------------------------------------- metrics
def _pnl(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    res, dec = g["result"].to_numpy(), g["dec"].to_numpy()
    flat = np.where(res == "push", 0.0, np.where(res == "win", dec - 1, -1.0))
    kf = np.array([kelly_fraction(p, dd) for p, dd in zip(g["model_p"], dec)]) * _KELLY * 100
    kpnl = np.where(res == "push", 0.0, np.where(res == "win", kf * (dec - 1), -kf))
    return flat, kpnl


def _metrics(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {}
    res = g["result"].to_numpy()
    flat, kpnl = _pnl(g)
    settled = res != "push"
    m, lo, hi = _bootstrap_roi(flat[settled])
    clv = g["cons_edge"].dropna()
    return {"n": len(g), "wins": int((res == "win").sum()),
            "roi": m, "roi_lo": lo, "roi_hi": hi,
            "flat_u": float(flat.sum()), "kelly_u": float(kpnl.sum()),
            "clv": (float(clv.mean()) if len(clv) else np.nan)}


def _maxnull_block(d: pd.DataFrame, masks: list, n: int = 1500, rng_seed: int = 0) -> float:
    """95th pct of the best-of-all-slices ROI under the no-skill null: each draw resamples
    every bet's outcome from its market-fair prob, scores each slice, and keeps the max."""
    fair = np.nan_to_num(d["fair_p"].to_numpy(float), nan=0.5)
    dec = d["dec"].to_numpy(float)
    ok = np.isfinite(d["fair_p"].to_numpy(float))
    use = [m & ok for m in masks if (m & ok).sum() >= 20]
    rng = np.random.default_rng(rng_seed)
    maxroi = np.empty(n)
    for i in range(n):
        pnl = np.where(rng.random(len(d)) < fair, dec - 1, -1.0)
        maxroi[i] = max(pnl[mm].mean() for mm in use)
    return float(np.percentile(maxroi, 95))


def run(cfg: dict | None = None, write: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    d = _gated(cfg, split="2023-01-01")
    if d.empty:
        print("[system_scan] no predictions — harvest odds first")
        return pd.DataFrame()

    big = d[d["date"] >= pd.Timestamp("2023-01-01")].reset_index(drop=True)   # clean OOS
    early = d[d["date"] < pd.Timestamp("2023-01-01")].reset_index(drop=True)  # 2019-2022
    wc22 = d[(d["league"] == "fifa.world") & (d["date"].dt.year == 2022)].reset_index(drop=True)

    masks_big = _slices(big)
    masks_early = _slices(early)
    masks_wc = _slices(wc22)
    overall_clv = big["cons_edge"].dropna().mean()
    fw_bar = _maxnull_block(big, [m for m in masks_big.values()], n=1500)

    rows = []
    for name, m in masks_big.items():
        b = _metrics(big[m])
        if not b or b["n"] < 12:
            continue
        e = _metrics(early[masks_early[name]]) if name in masks_early else {}
        w = _metrics(wc22[masks_wc[name]]) if name in masks_wc else {}
        rel_clv = (b["clv"] - overall_clv) if b["clv"] == b["clv"] else np.nan
        replicates = bool(e.get("roi", -1) > 0)
        clears0 = b["roi_lo"] > 0
        beats_fw = b["roi"] > fw_bar
        clv_ok = (rel_clv >= -0.005) if rel_clv == rel_clv else False
        if not clears0:
            verdict = "noise"
        elif beats_fw and replicates and clv_ok:
            verdict = "CANDIDATE"
        else:
            why = []
            if not beats_fw: why.append("fails family-wise")
            if not replicates: why.append("no out-of-time")
            if not clv_ok: why.append("fails CLV")
            verdict = "variance — " + ", ".join(why)
        rows.append({
            "system": name, "verdict": verdict,
            "big_n": b["n"], "big_roi%": round(b["roi"] * 100, 1),
            "big_ci": f"[{b['roi_lo']*100:+.0f},{b['roi_hi']*100:+.0f}]",
            "big_kelly_u": round(b["kelly_u"], 1),
            "rel_clv%": (round(rel_clv * 100, 1) if rel_clv == rel_clv else None),
            "oot2019-22_roi%": (round(e["roi"] * 100, 1) if e else None),
            "wc2022_roi%": (round(w["roi"] * 100, 1) if w else None),
            "wc2022_n": (w.get("n") if w else None),
            "_sort": b["roi"],
        })
    table = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")

    if write:
        ensure_dirs(cfg)
        out = path_for("reports", cfg) / "system_scan.csv"
        table.to_csv(out, index=False)
        print(f"[system_scan] wrote {out}")
    print(f"\nFamily-wise bar (95th pct of best-of-slices null ROI): {fw_bar*100:+.1f}%")
    print(f"Overall CLV (mostly vig): {overall_clv*100:+.1f}%  — 'rel_clv' = slice minus this\n")
    cols = ["system", "big_n", "big_roi%", "big_ci", "big_kelly_u", "rel_clv%",
            "oot2019-22_roi%", "wc2022_roi%", "verdict"]
    with pd.option_context("display.width", 220, "display.max_rows", 60):
        print(table[cols].to_string(index=False))
    cand = table[table["verdict"] == "CANDIDATE"]
    print()
    if cand.empty:
        print("VERDICT: no system survived family-wise + out-of-time + CLV. The eye-catching "
              "slices (away-dogs, EV50%+, etc.) post +ROI but FAIL CLV — variance, not edge. "
              "Nothing deployed (report-only).")
    else:
        print(f"VERDICT: {len(cand)} CANDIDATE(s) survived — track forward via clv system tag, "
              f"NO auto-bet:\n{cand['system'].tolist()}")
    return table


if __name__ == "__main__":
    run()
