"""Daily CLV job: grade finished tickets, then snapshot newly-recommended bets.

Run manually:  python scripts/snapshot_odds.py
Schedule it daily (Windows Task Scheduler example):
    schtasks /create /tn "soccer-clv" /tr "python C:\\soccer\\scripts\\snapshot_odds.py" ^
             /sc DAILY /st 12:00
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import clv  # noqa: E402


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("[snapshot_odds] grading finished tickets…")
    clv.grade()
    print("[snapshot_odds] snapshotting new recommendations…")
    clv.snapshot(days=3)
    r = clv.report()
    if r.get("n"):
        print(f"[snapshot_odds] CLV so far: {r['n']} bets, avg CLV "
              f"{r['avg_clv']*100:+.2f}%, ROI {r['roi']*100:+.1f}%")


if __name__ == "__main__":
    main()
