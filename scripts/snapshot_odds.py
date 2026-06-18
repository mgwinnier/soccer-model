"""CLV job: grade finished tickets, then snapshot newly-recommended bets.

Runs unattended on a schedule (GitHub Actions ``.github/workflows/clv-sync.yml`` every ~2h) so the
Tracker self-updates — the workflow commits ``reports/clv_open.csv`` + ``reports/clv_ledger.csv`` and
the live site redeploys. Also runnable manually: ``python scripts/snapshot_odds.py``.
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
    # Each step is independently guarded so a transient network/API hiccup on one never aborts the
    # other — the scheduled CLV job must be robust unattended.
    print("[snapshot_odds] grading finished tickets…")
    try:
        print(f"[snapshot_odds] graded {clv.grade()} ticket(s)")
    except Exception as e:  # noqa: BLE001
        print(f"[snapshot_odds] grade failed: {type(e).__name__}: {e}")
    print("[snapshot_odds] snapshotting new recommendations…")
    try:
        print(f"[snapshot_odds] snapshotted {clv.snapshot(days=3)} new ticket(s)")
    except Exception as e:  # noqa: BLE001
        print(f"[snapshot_odds] snapshot failed: {type(e).__name__}: {e}")
    try:
        r = clv.report()
        if r.get("n"):
            print(f"[snapshot_odds] CLV so far: {r['n']} bets, avg CLV "
                  f"{r['avg_clv']*100:+.2f}%, ROI {r['roi']*100:+.1f}%")
    except Exception as e:  # noqa: BLE001
        print(f"[snapshot_odds] report failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
