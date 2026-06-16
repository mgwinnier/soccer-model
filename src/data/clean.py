"""Clean and normalize the raw results into the canonical match table.

Output (``data/processed/matches.parquet``) is the single spine every model and
feature builds on. One row per match, chronologically sortable, with normalized
team names, derived outcome, match-importance weight, and confederations.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from .team_names import normalize_team, confederation_of


def _importance(tournament: str, cfg: dict) -> float:
    weights = cfg["data"]["importance_weights"]
    return float(weights.get(tournament, weights["_default"]))


def load_raw_results(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    raw = path_for("data_raw", cfg)
    path = raw / "results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.data.download` first."
        )
    return pd.read_csv(path, parse_dates=["date"])


def build_matches(cfg: dict | None = None, write: bool = True) -> pd.DataFrame:
    """Produce the canonical, model-ready match table."""
    cfg = cfg or load_config()
    df = load_raw_results(cfg)

    df = df.rename(columns={"country": "venue_country"})
    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    df["venue_country"] = df["venue_country"].map(normalize_team)

    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)

    # Date filter (modern era by default)
    min_date = pd.Timestamp(cfg["data"]["min_date"])
    df = df[df["date"] >= min_date].copy()

    drop = set(cfg["data"].get("drop_tournaments") or [])
    if drop:
        df = df[~df["tournament"].isin(drop)].copy()

    # Derived outcome from the home side's perspective (H / D / A)
    gd = df["home_score"] - df["away_score"]
    df["result"] = pd.cut(
        gd, bins=[-99, -1, 0, 99], labels=["A", "D", "H"]
    ).astype(str)
    df["goal_diff"] = gd
    df["total_goals"] = df["home_score"] + df["away_score"]

    df["importance"] = df["tournament"].map(lambda t: _importance(t, cfg))
    df["is_world_cup"] = df["tournament"].eq("FIFA World Cup")
    df["home_conf"] = df["home_team"].map(confederation_of)
    df["away_conf"] = df["away_team"].map(confederation_of)
    df["cross_conf"] = df["home_conf"] != df["away_conf"]

    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index

    cols = [
        "match_id", "date", "home_team", "away_team", "home_score",
        "away_score", "result", "goal_diff", "total_goals", "tournament",
        "importance", "is_world_cup", "city", "venue_country", "neutral",
        "home_conf", "away_conf", "cross_conf",
    ]
    df = df[cols]

    if write:
        ensure_dirs(cfg)
        out = path_for("data_processed", cfg) / "matches.parquet"
        df.to_parquet(out, index=False)
        print(f"[clean] wrote {len(df):,} matches -> {out}")
    return df


def load_matches(cfg: dict | None = None) -> pd.DataFrame:
    """Load the cleaned match table, building it if absent."""
    cfg = cfg or load_config()
    out = path_for("data_processed", cfg) / "matches.parquet"
    if out.exists():
        return pd.read_parquet(out)
    return build_matches(cfg, write=True)


if __name__ == "__main__":
    build_matches()
