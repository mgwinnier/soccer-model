"""Honest feature attribution — what actually drives the predictions?

Four complementary views, leading with the robust ones (permutation / drop-out)
rather than raw LightGBM split counts, which are misleading with correlated and
high-cardinality features:

  1. **Ensemble level** — base-model blend weights (DC / Elo / GBM) and the pooled
     RPS cost of dropping each member. (Dixon-Coles "expected goals" is an ensemble
     *member*, not a GBM feature, so this is where its contribution shows up.)
  2. **GBM gain** — total gain per feature.
  3. **GBM permutation importance** — shuffle each feature on a temporal holdout and
     measure the RPS damage. The honest "real predictive power" measure.
  4. **Per-feature-group drop-out** — retrain without each group; the holdout RPS
     increase is that group's marginal value (Elo / form / market value / travel /
     altitude / head-to-head / rest / context / xG).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..models.gbm import GBMModel, model_feature_columns
from ..models.ensemble import EnsembleModel
from ..features.motivation import FEATURE_COLS as _MOTIV_COLS
from .metrics import ranked_probability_score, labels_from_results

TRAIN_END = "2022-01-01"
HOLDOUT_END = "2025-07-01"

# Fine-grained feature groups (covers the user's exact list).
GROUPS = {
    "elo": lambda c: c.startswith("elo_") or c in ("home_elo", "away_elo"),
    "form": lambda c: any(c.startswith(p) for p in ("ppg_", "gf_", "ga_")),
    "rest": lambda c: c.startswith("rest") or "rest" in c,
    "market_value": lambda c: "squad_value" in c,
    "travel": lambda c: "travel" in c,
    "altitude": lambda c: "altitude" in c,
    "head_to_head": lambda c: c.startswith("h2h"),
    "context": lambda c: c in ("neutral", "cross_conf", "importance"),
    "xg_style": lambda c: c.startswith("xg_") or "xgf" in c or "xga" in c,
    "motivation": lambda c: c in set(_MOTIV_COLS),
}


def _split(feats: pd.DataFrame):
    feats = feats.sort_values("date")
    train = feats[feats["date"] < pd.Timestamp(TRAIN_END)]
    hold = feats[(feats["date"] >= pd.Timestamp(TRAIN_END))
                 & (feats["date"] < pd.Timestamp(HOLDOUT_END))]
    return train, hold


def _rps(model, df, y) -> float:
    return ranked_probability_score(model.predict_proba(df), y)


def gbm_importances(train, hold, cols, y_hold, n_repeats=4, seed=0):
    gbm = GBMModel().fit(train, features=cols)
    base = _rps(gbm, hold, y_hold)
    gain = pd.Series(gbm.clf.booster_.feature_importance(importance_type="gain"),
                     index=cols)
    gain = gain / gain.sum()

    rng = np.random.default_rng(seed)
    Xh = hold[cols].astype(float).reset_index(drop=True)
    perm = {}
    for c in cols:
        deltas = []
        for _ in range(n_repeats):
            Xs = Xh.copy()
            Xs[c] = rng.permutation(Xs[c].to_numpy())
            deltas.append(_rps_from_X(gbm, Xs, y_hold) - base)
        perm[c] = float(np.mean(deltas))
    return base, gain, pd.Series(perm)


def _rps_from_X(gbm, X, y):
    """RPS from a raw feature matrix, mapping LightGBM class order to [H,D,A]."""
    raw = gbm.clf.predict_proba(X)
    out = np.zeros((len(X), 3))
    for col, cls in enumerate(gbm._classes):
        out[:, cls] = raw[:, col]
    return ranked_probability_score(out, y)


def group_dropout(train, hold, cols, y_hold, base):
    rows = []
    for name, pred in GROUPS.items():
        drop = [c for c in cols if pred(c)]
        if not drop:
            continue
        keep = [c for c in cols if c not in drop]
        gbm = GBMModel().fit(train, features=keep)
        rps = _rps(gbm, hold, y_hold)
        rows.append({"group": name, "n_features": len(drop),
                     "rps_without": rps, "rps_delta": rps - base})
    return pd.DataFrame(rows).sort_values("rps_delta", ascending=False)


def ensemble_attribution(train, hold, y_hold):
    full = EnsembleModel(members=["dixon_coles", "elo", "gbm"]).fit(train)
    base = _rps(full, hold, y_hold)
    rows = [{"variant": "FULL (DC+Elo+GBM)", "rps": base, "rps_delta": 0.0,
             "weights": {k: round(v, 2) for k, v in full.weights.items()}}]
    for drop in ["dixon_coles", "elo", "gbm"]:
        members = [m for m in ["dixon_coles", "elo", "gbm"] if m != drop]
        ens = EnsembleModel(members=members).fit(train)
        rps = _rps(ens, hold, y_hold)
        rows.append({"variant": f"drop {drop}", "rps": rps, "rps_delta": rps - base,
                     "weights": {k: round(v, 2) for k, v in ens.weights.items()}})
    return full.weights, pd.DataFrame(rows)


def motivation_subset_eval(train, hold, cols) -> dict:
    """Honest, targeted test: does the motivation group lower RPS *on the final
    group matches it actually applies to*? The full-holdout drop-out dilutes it to
    nothing (only ~16 of thousands of holdout matches are covered), so we evaluate
    on that subset directly and report **n** — which is small by construction."""
    mot = [c for c in cols if GROUPS["motivation"](c)]
    if "is_final_group_match" not in hold.columns:
        return {"n": 0}
    sub = hold[hold["is_final_group_match"] == 1]
    if sub.empty or not mot:
        return {"n": 0}
    y_sub = labels_from_results(sub["result"])
    full = GBMModel().fit(train, features=cols)
    without = GBMModel().fit(train, features=[c for c in cols if c not in mot])
    rps_full = _rps(full, sub, y_sub)
    rps_without = _rps(without, sub, y_sub)
    return {"n": int(len(sub)), "n_motiv_features": len(mot),
            "rps_full": rps_full, "rps_without": rps_without,
            "rps_delta": rps_without - rps_full}      # >0 ⇒ motivation helps


def motivation_walkforward_eval(feats, cols) -> dict:
    """Properly-powered, leak-free test pooled over every covered tournament.

    For each tournament we train on matches *before* its group stage and predict
    its final-group matches, with and without the motivation features, then pool
    the predictions and score RPS once. This uses all ~70 final-group matches
    (not just the ~14 that happen to fall in a single holdout), so it's the honest
    read on whether game-state actually helps."""
    from ..data.group_structures import COVERED
    mot = [c for c in cols if GROUPS["motivation"](c)]
    if "is_final_group_match" not in feats.columns or not mot:
        return {"n": 0}
    base_cols = [c for c in cols if c not in mot]
    pf, pw, ys = [], [], []
    n_tourn = 0
    for (tournament, year) in COVERED:
        sub = feats[(feats["tournament"] == tournament)
                    & (feats["date"].dt.year == year)
                    & (feats["is_final_group_match"] == 1)]
        if sub.empty:
            continue
        train = feats[feats["date"] < sub["date"].min()]
        if len(train) < 500:
            continue
        full = GBMModel().fit(train, features=cols)
        without = GBMModel().fit(train, features=base_cols)
        pf.append(full.predict_proba(sub))
        pw.append(without.predict_proba(sub))
        ys.append(labels_from_results(sub["result"]))
        n_tourn += 1
    if not ys:
        return {"n": 0}
    import numpy as np
    Pf, Pw, Y = np.vstack(pf), np.vstack(pw), np.concatenate(ys)
    rps_full = ranked_probability_score(Pf, Y)
    rps_without = ranked_probability_score(Pw, Y)
    return {"n": int(len(Y)), "n_tournaments": n_tourn, "n_motiv_features": len(mot),
            "rps_full": rps_full, "rps_without": rps_without,
            "rps_delta": rps_without - rps_full}


def run(cfg: dict | None = None, write: bool = True) -> dict:
    cfg = cfg or load_config()
    feats = pd.read_parquet(path_for("data_processed", cfg) / "features.parquet")
    train, hold = _split(feats)
    cols = model_feature_columns(feats)
    y_hold = labels_from_results(hold["result"])

    weights, ens_tbl = ensemble_attribution(train, hold, y_hold)
    base, gain, perm = gbm_importances(train, hold, cols, y_hold)
    groups = group_dropout(train, hold, cols, y_hold, base)
    motiv = motivation_subset_eval(train, hold, cols)
    motiv_wf = motivation_walkforward_eval(feats, cols)

    if write:
        ensure_dirs(cfg)
        rep = path_for("reports", cfg)
        imp = pd.DataFrame({"feature": cols, "gain": gain.values,
                            "perm_rps_delta": perm.reindex(cols).values})
        imp.sort_values("perm_rps_delta", ascending=False).to_csv(
            rep / "feature_importance.csv", index=False)
        groups.to_csv(rep / "feature_group_dropout.csv", index=False)
        ens_tbl.drop(columns="weights").to_csv(rep / "ensemble_attribution.csv", index=False)
    return {"weights": weights, "ensemble": ens_tbl, "gain": gain, "perm": perm,
            "groups": groups, "motivation": motiv, "motivation_wf": motiv_wf,
            "n_train": len(train), "n_hold": len(hold)}


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    r = run()
    print(f"\nFEATURE ATTRIBUTION  (train {r['n_train']}, holdout {r['n_hold']} matches; "
          f"lower RPS = better)\n")
    print("=== 1) Ensemble: is it 'just' Elo/DC? ===")
    print("Blend weights:", {k: round(v, 2) for k, v in r["weights"].items()})
    for _, row in r["ensemble"].iterrows():
        print(f"  {row['variant']:22s} RPS {row['rps']:.4f}  "
              f"(drop cost {row['rps_delta']*1000:+.1f} mRPS)")
    print("\n=== 2/3) GBM features: gain vs permutation (top 12 by real predictive power) ===")
    tbl = pd.DataFrame({"gain%": (r["gain"] * 100).round(1),
                        "perm_mRPS": (r["perm"] * 1000).round(2)})
    tbl = tbl.sort_values("perm_mRPS", ascending=False).head(12)
    print(tbl.to_string())
    print("\n=== 4) Feature-GROUP drop-out (RPS increase when removed = real value) ===")
    g = r["groups"].copy()
    g["rps_delta_mRPS"] = (g["rps_delta"] * 1000).round(2)
    print(g[["group", "n_features", "rps_delta_mRPS"]].to_string(index=False))
    print("\nRead: a feature/group only adds real signal if removing it RAISES RPS "
          "(positive delta). Near-zero or negative = noise the model is better without.")

    m = r.get("motivation", {})
    if m.get("n"):
        verdict = ("helps" if m["rps_delta"] > 0 else "does NOT help")
        print(f"\n=== Motivation (final-group-round subset only) ===")
        print(f"n = {m['n']} matches ({m['n_motiv_features']} motivation features). "
              f"RPS with {m['rps_full']:.4f} vs without {m['rps_without']:.4f} "
              f"-> delta {m['rps_delta']*1000:+.1f} mRPS ({verdict}).")
        print(f"HONEST CAVEAT: n={m['n']} is tiny (only WC-2022 final rounds fall in the "
              f"holdout) — directional at best, not significant.")

    w = r.get("motivation_wf", {})
    if w.get("n"):
        verdict = ("helps" if w["rps_delta"] > 0 else "does NOT help")
        print(f"\n=== Motivation (leave-one-tournament-out, pooled — the real test) ===")
        print(f"n = {w['n']} final-group matches across {w['n_tournaments']} tournaments. "
              f"RPS with {w['rps_full']:.4f} vs without {w['rps_without']:.4f} "
              f"-> delta {w['rps_delta']*1000:+.1f} mRPS ({verdict}).")
        print(f"Still a small sample (~{w['n']} matches is all the history affords); "
              f"treat a positive delta as encouraging, not proven.")
