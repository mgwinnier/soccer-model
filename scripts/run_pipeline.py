"""End-to-end pipeline: download -> clean -> features -> backtest -> simulate.

    python scripts/run_pipeline.py                # full run
    python scripts/run_pipeline.py --skip-download --skip-backtest

Each stage is idempotent and can be skipped once its artifacts exist.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, ensure_dirs  # noqa: E402
from src.data import download, clean  # noqa: E402
from src.data.statsbomb import fetch_statsbomb_xg  # noqa: E402
from src.features.build import build_feature_matrix  # noqa: E402
from src.backtest.walkforward import backtest_world_cups  # noqa: E402
from src.backtest.ablation import run_ablation  # noqa: E402
from src.simulate.tournament import run_simulation  # noqa: E402
from src.simulate.bracket_2026 import validate_groups  # noqa: E402
from src.predict.report import build_report  # noqa: E402


def _stage(name: str):
    print(f"\n{'='*60}\n# {name}\n{'='*60}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-xg", action="store_true", help="skip StatsBomb xG fetch")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--iterations", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    t0 = time.time()

    if not args.skip_download:
        _stage("1/7  Download public data")
        download.main()

    if not args.skip_xg:
        _stage("2/7  Fetch xG (StatsBomb open data)")
        fetch_statsbomb_xg(cfg)

    _stage("3/7  Clean match table")
    clean.build_matches(cfg)

    _stage("4/7  Build feature matrix (Elo, form, squad, context, xG/style)")
    build_feature_matrix(cfg=cfg)

    _stage("5/7  Validate 2026 group data")
    validate_groups(cfg)
    print("groups consistent with played matches.")

    if not args.skip_backtest:
        _stage("6/7  Backtest on past World Cups")
        backtest_world_cups(cfg=cfg)
        if not args.skip_ablation:
            print("\n--- ablation: does each block lower RPS? ---")
            run_ablation(cfg=cfg)

    _stage("7/7  Simulate 2026 World Cup")
    fc = run_simulation(cfg=cfg, n_iter=args.iterations)
    build_report(cfg)

    print(f"\nDONE in {time.time()-t0:.0f}s. Top 5 championship odds:")
    for _, r in fc.head(5).iterrows():
        print(f"  {r.team:14s} {r.champion*100:5.1f}%")


if __name__ == "__main__":
    main()
