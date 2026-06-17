"""CLV-gated segment kill-switches — automatic, honest discipline.

A "segment" is a directional bet role (``MR:H/D/A``, ``TG:over/under``,
``SP:home/away`` — the same keys as ``value._type_key``). Once a segment's
closing-line-value history turns negative over a meaningful sample it gets
**disabled**: the live pipeline stops recommending and snapshotting it. CLV (not
ROI) is the trigger because beating the close is the leading indicator of edge and
stabilises with far fewer bets than realised ROI.

Spreads (``SP:*``) are disabled by default — the v5/v6 backtest showed them the
clearest loser, so they start off until a spread-specific model earns them back.

Mirrors ``market_bias.py``: a small JSON artifact under ``data/models/`` with a
load/save class, so the deployed gate is a single inspectable file.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import load_config, path_for

# Segments off out of the gate, regardless of CLV history. (Empty by default.)
# Spreads were disabled under the old market-ANCHORED pipeline (v5/v6). Re-evaluated on
# the large out-of-sample international set with the current market-INDEPENDENT model +
# gate, spreads are ~break-even (about -0.3% ROI over ~1,460 bets, 95% CI straddling 0) —
# not a proven loser — so they're re-enabled. The CLV kill-switch still disables any
# segment that goes meaningfully negative on a forward sample.
DEFAULT_DISABLED: dict[str, str] = {}


def segment_from_code(code: str) -> str:
    """Map a CLV ledger grade code to a segment key (MR:/TG:/SP:)."""
    if code in ("H", "D", "A"):
        return f"MR:{code}"
    if code.startswith("over@"):
        return "TG:over"
    if code.startswith("under@"):
        return "TG:under"
    if code.startswith("cover_home@"):
        return "SP:home"
    if code.startswith("cover_away@"):
        return "SP:away"
    return code


class SegmentGate:
    def __init__(self, disabled: dict[str, dict] | None = None):
        # {segment: {"reason", "killed_at", "n", "avg_clv"}}
        self.disabled = disabled or {}

    def is_disabled(self, segment: str) -> bool:
        return segment in self.disabled or segment in DEFAULT_DISABLED

    def disabled_set(self) -> set[str]:
        return set(self.disabled) | set(DEFAULT_DISABLED)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"disabled": self.disabled}, open(path, "w", encoding="utf-8"), indent=2)

    @classmethod
    def load(cls, path: Path) -> "SegmentGate":
        if not path.exists():
            return cls({})
        d = json.load(open(path, encoding="utf-8"))
        return cls(d.get("disabled", {}))


def default_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return path_for("models", cfg) / "disabled_segments.json"


def load_gate(cfg: dict | None = None) -> SegmentGate:
    return SegmentGate.load(default_path(cfg))


def disabled_set(cfg: dict | None = None) -> set[str]:
    """The set of currently-disabled segments (persisted kills ∪ defaults)."""
    return load_gate(cfg).disabled_set()


def evaluate_kill_switches(cfg: dict | None = None, min_bets: int = 30,
                           now: str | None = None, write: bool = True) -> SegmentGate:
    """Disable any segment with ≥ ``min_bets`` settled CLV tickets and avg CLV < 0.

    Reads ``reports/clv_ledger.csv``, derives each ticket's segment from its code,
    and persists the disabled set (defaults stay implicit, so spreads remain off)."""
    import pandas as pd
    cfg = cfg or load_config()
    gate = load_gate(cfg)
    ledger = path_for("reports", cfg) / "clv_ledger.csv"
    if ledger.exists():
        led = pd.read_csv(ledger)
        seg = (led["segment"] if "segment" in led.columns
               else led["code"].map(segment_from_code))
        led = led.assign(segment=seg).dropna(subset=["clv"])
        for s, g in led.groupby("segment"):
            if s in DEFAULT_DISABLED:
                continue
            avg_clv = float(g["clv"].mean())
            if len(g) >= min_bets and avg_clv < 0:
                gate.disabled[s] = {
                    "reason": f"avg CLV {avg_clv*100:+.2f}% over {len(g)} bets (< 0)",
                    "killed_at": now or "",
                    "n": int(len(g)), "avg_clv": round(avg_clv, 4),
                }
    if write:
        gate.save(default_path(cfg))
    return gate
