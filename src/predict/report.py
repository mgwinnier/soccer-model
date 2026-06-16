"""Generate a human-readable Markdown report of the forecast + backtest."""
from __future__ import annotations

import pandas as pd

from ..config import load_config, path_for, ensure_dirs


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def build_report(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    rep = path_for("reports", cfg)
    ensure_dirs(cfg)
    lines: list[str] = ["# 2026 World Cup — Model Forecast\n"]

    fc_path = rep / "wc2026_forecast.csv"
    if fc_path.exists():
        fc = pd.read_csv(fc_path)
        lines.append("## Championship odds (top 16)\n")
        lines.append("| Team | Group | Elo | Advance | Reach QF | Reach SF | Final | **Champion** |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for _, r in fc.head(16).iterrows():
            lines.append(
                f"| {r.team} | {r.group} | {int(r.elo)} | {_pct(r.advance)} | "
                f"{_pct(r.reach_qf)} | {_pct(r.reach_sf)} | {_pct(r.reach_final)} | "
                f"**{_pct(r.champion)}** |"
            )
        lines.append("")

    pooled = rep / "backtest_pooled.csv"
    if pooled.exists():
        bt = pd.read_csv(pooled)
        lines.append("## Backtest (pooled over 2010–2022 World Cups)\n")
        lines.append("Lower RPS / log-loss / Brier is better. Climatology is the "
                     "information-free floor.\n")
        lines.append("| Model | RPS | Log-loss | Brier | Accuracy | N |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in bt.iterrows():
            lines.append(
                f"| {r.model} | {r.rps:.4f} | {r.log_loss:.4f} | {r.brier:.4f} | "
                f"{r.accuracy:.3f} | {int(r.n)} |"
            )
        lines.append("")

    calib = rep / "calibration.csv"
    if calib.exists():
        cal = pd.read_csv(calib)
        lines.append("## Calibration (ensemble, pooled WCs)\n")
        lines.append("| Predicted bin | Mean predicted | Observed freq | N |")
        lines.append("|---|---:|---:|---:|")
        for _, r in cal.iterrows():
            lines.append(
                f"| {r['bin']} | {r.mean_predicted:.3f} | {r.observed_freq:.3f} | "
                f"{int(r.n)} |"
            )
        lines.append("")

    text = "\n".join(lines)
    out = rep / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(f"[report] wrote {out}")
    return text


if __name__ == "__main__":
    build_report()
