"""Assemble the full per-match feature matrix from all feature modules.

Joins Elo, rolling form, squad value, and context features onto the clean match
table and carries the target columns (result + scores). Also persists the final
Elo ratings (used to seed the 2026 simulator and single-match predictions).
"""
from __future__ import annotations

import json

import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..data.clean import load_matches
from ..data.fbref import attach_xg_to_matches
from .elo import compute_elo_features
from .form import compute_form_features
from .squad import compute_squad_features
from .context import compute_context_features
from .style import compute_style_features
from .motivation import compute_motivation_features

# Columns carried through for modeling / inspection (not used as model inputs).
# NB: home_xg/away_xg are POST-match quantities — carried only as the xG model's
# training target and excluded from any predictive feature set (leakage guard).
META_COLS = [
    "date", "home_team", "away_team", "home_score", "away_score", "result",
    "tournament", "is_world_cup", "neutral", "importance", "home_xg", "away_xg",
]


def build_feature_matrix(
    matches: pd.DataFrame | None = None, cfg: dict | None = None,
    write: bool = True,
) -> pd.DataFrame:
    cfg = cfg or load_config()
    if matches is None:
        matches = load_matches(cfg)
    matches = matches.sort_values("date").reset_index(drop=True)
    # attach post-match xG (NaN where unavailable) for the xG model's target
    matches = attach_xg_to_matches(matches, cfg)

    elo_feats, engine = compute_elo_features(matches, cfg)
    form_feats = compute_form_features(matches, cfg)
    squad_feats = compute_squad_features(matches, cfg)
    ctx_feats = compute_context_features(matches, cfg)
    style_feats = compute_style_features(matches, cfg)
    motiv_feats = compute_motivation_features(matches, cfg)

    base = matches.set_index("match_id")
    feats = (
        base[META_COLS]
        .join(elo_feats, how="left")
        .join(form_feats, how="left")
        .join(squad_feats, how="left")
        .join(ctx_feats.drop(columns=["neutral", "importance"], errors="ignore"),
              how="left")
        .join(style_feats, how="left")
        .join(motiv_feats, how="left")
    )
    feats = feats.reset_index()

    if write:
        ensure_dirs(cfg)
        out = path_for("data_processed", cfg) / "features.parquet"
        feats.to_parquet(out, index=False)
        # persist current ratings for the simulator / live predictions
        ratings_path = path_for("data_processed", cfg) / "elo_ratings.json"
        with open(ratings_path, "w", encoding="utf-8") as fh:
            json.dump(engine.ratings, fh, indent=2, sort_keys=True)
        print(f"[features] wrote {len(feats):,} rows x {feats.shape[1]} cols -> {out}")
        print(f"[features] saved {len(engine.ratings)} Elo ratings -> {ratings_path}")
    return feats


# The model-input feature columns (everything that isn't meta / identifiers).
def feature_columns(feats: pd.DataFrame) -> list[str]:
    exclude = set(["match_id"] + META_COLS)
    return [c for c in feats.columns if c not in exclude]


if __name__ == "__main__":
    build_feature_matrix()
