"""Does game-state bias TOTAL GOALS in the final group round? (the totals angle)

1X2 showed nothing (feature_importance: +0.0 mRPS over 66 matches). The more
plausible channel is goals: a dead rubber or a match where both sides are content
with a draw should produce fewer goals (rotation, low intensity, killing the game),
biasing **unders**. This tests that directly on the historical final-group matches.

Honesty up front:
- Sample is ~66 matches, split further by stakes — tiny. A bootstrap CI is reported
  so the noise is visible; treat anything whose CI straddles 0 as "not shown".
- These are **top-2-advance** group stages. The **2026 World Cup advances the 8 best
  third-placed teams**, which structurally removes most dead rubbers — so any effect
  found here is an *upper bound* on what 2026 would show, not a transferable edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.clean import load_matches
from ..features.motivation import compute_motivation_features


def _boot_diff(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 0):
    """Bootstrap 95% CI for mean(a) - mean(b)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = (a[rng.integers(0, len(a), (n, len(a)))].mean(1)
             - b[rng.integers(0, len(b), (n, len(b)))].mean(1))
    return float(np.mean(a) - np.mean(b)), float(np.percentile(diffs, 2.5)), \
        float(np.percentile(diffs, 97.5))


def analyze(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    matches = load_matches(cfg).sort_values("date").reset_index(drop=True)
    mot = compute_motivation_features(matches)
    fin_idx = mot.index[mot["is_final_group_match"] == 1]
    j = matches.set_index("match_id").loc[fin_idx]
    f = mot.loc[fin_idx]

    tg = (j["home_score"] + j["away_score"]).to_numpy(dtype=float)

    # "low stakes" = a dead rubber, OR both sides already content with a draw
    low = ((f["dead_rubber"] == 1)
           | ((f["home_draw_enough"] == 1) & (f["away_draw_enough"] == 1))).to_numpy()
    # "high stakes" = at least one side must win to advance
    high = ((f["home_needs_win"] == 1) | (f["away_needs_win"] == 1)).to_numpy()

    def block(name, mask):
        g = tg[mask]
        return {"bucket": name, "n": int(mask.sum()),
                "mean_total": float(g.mean()) if len(g) else float("nan"),
                "under25_rate": float((g < 2.5).mean()) if len(g) else float("nan")}

    rows = [block("ALL final-group", np.ones(len(tg), bool)),
            block("low-stakes (dead/mutual-draw)", low),
            block("high-stakes (someone needs win)", high)]
    diff, lo, hi = _boot_diff(tg[high], tg[low])   # high − low goals
    return {"rows": rows, "diff_high_minus_low": diff, "ci": (lo, hi),
            "n_low": int(low.sum()), "n_high": int(high.sum())}


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    r = analyze()
    print("\nMOTIVATION → TOTAL GOALS (historical final group rounds, top-2 format)\n")
    print(f"{'bucket':34s} {'n':>4s} {'mean goals':>11s} {'under2.5':>9s}")
    for b in r["rows"]:
        print(f"{b['bucket']:34s} {b['n']:4d} {b['mean_total']:11.2f} "
              f"{b['under25_rate']*100:8.0f}%")
    d, (lo, hi) = r["diff_high_minus_low"], r["ci"]
    print(f"\nhigh-stakes minus low-stakes mean goals: {d:+.2f} "
          f"[95% CI {lo:+.2f}, {hi:+.2f}]  (n_high={r['n_high']}, n_low={r['n_low']})")
    real = (lo > 0)
    print("Read:", "effect is directionally present AND its CI clears 0 — worth a look."
          if real else
          "CI straddles 0 → NOT shown at this sample. The dead-rubber-under intuition "
          "is plausible but unproven here, and 2026's best-third format weakens it further.")
