"""Download raw public data into ``data/raw``.

Tiered by access difficulty so the pipeline *never hard-fails*:

  1. martj42 international results  — raw GitHub CSV, no auth.  REQUIRED.
  2. Kaggle Elo / FIFA player data  — needs the ``kaggle`` CLI + token. OPTIONAL.
  3. Transfermarkt squad values     — best-effort scrape.            OPTIONAL.

Anything optional that is unavailable is simply skipped with a warning; the
feature layer treats the corresponding columns as missing. Our own Elo engine
(``features/elo.py``) means we are never *dependent* on the external Elo feed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

from ..config import load_config, path_for, ensure_dirs


def _log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def fetch_results(cfg: dict | None = None) -> list[Path]:
    """Download the martj42 results CSVs (the required data spine)."""
    cfg = cfg or load_config()
    raw = path_for("data_raw", cfg)
    raw.mkdir(parents=True, exist_ok=True)
    base = cfg["sources"]["results_base_url"].rstrip("/")
    out: list[Path] = []
    for fname in cfg["sources"]["results_files"]:
        url = f"{base}/{fname}"
        dest = raw / fname
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            _log(f"saved {fname} ({len(resp.content):,} bytes)")
            out.append(dest)
        except Exception as exc:  # noqa: BLE001
            if dest.exists():
                _log(f"WARN could not refresh {fname} ({exc}); using cached copy")
                out.append(dest)
            else:
                raise RuntimeError(
                    f"Required file {fname} could not be downloaded and no cache "
                    f"exists: {exc}"
                ) from exc
    return out


def _kaggle_available() -> bool:
    from ..config import load_secrets, PROJECT_ROOT
    load_secrets()  # sets KAGGLE_CONFIG_DIR if secrets/kaggle.json exists
    if shutil.which("kaggle") is None:
        return False
    candidates = [
        PROJECT_ROOT / "secrets" / "kaggle.json",
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    has_file = any(p.exists() for p in candidates if str(p) != "kaggle.json")
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    return has_file or has_env


def fetch_kaggle(cfg: dict | None = None) -> list[Path]:
    """Best-effort Kaggle downloads (Elo + FIFA players). Skips if no token."""
    cfg = cfg or load_config()
    raw = path_for("data_raw", cfg)
    if not _kaggle_available():
        _log("Kaggle CLI/token not found — skipping Kaggle sources "
             "(our own Elo engine covers ratings).")
        return []
    out: list[Path] = []

    def _try_download(slug: str, dest: Path) -> bool:
        dest.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", slug,
                            "-p", str(dest), "--unzip"],
                           check=True, capture_output=True, text=True)
            _log(f"kaggle: downloaded {slug} -> {dest.name}/")
            out.extend(dest.glob("*.csv"))
            return True
        except subprocess.CalledProcessError as exc:
            _log(f"WARN kaggle download failed for {slug}: {exc.stderr[:160]}")
            return False

    for key, subdir in [("kaggle_elo", "elo"), ("kaggle_wc2026_elo", "wc2026_elo")]:
        slug = cfg["sources"].get(key)
        if slug:
            _try_download(slug, raw / subdir)

    # FIFA/FC player ratings: try the current dataset, then fallbacks (FC25→FC24→old)
    fifa_slugs = [cfg["sources"].get("kaggle_fifa_players")] + \
        list(cfg["sources"].get("kaggle_fifa_players_fallbacks") or [])
    for slug in [s for s in fifa_slugs if s]:
        if _try_download(slug, raw / "fifa_players"):
            break
    return out


def fetch_transfermarkt(cfg: dict | None = None) -> Path | None:
    """Best-effort scrape of 2026 WC squad market values.

    Transfermarkt markup is fragile and rate-limited; on any failure we fall
    back to a committed static CSV at ``data/raw/squad_values_static.csv`` if it
    exists, else return None (squad-value features become missing).
    """
    cfg = cfg or load_config()
    raw = path_for("data_raw", cfg)
    static = raw / "squad_values_static.csv"
    url = cfg["sources"].get("transfermarkt_wc2026")
    if not url:
        return static if static.exists() else None
    try:
        from .transfermarkt import scrape_wc2026_squad_values  # lazy import
        df = scrape_wc2026_squad_values(url)
        if df is not None and len(df):
            dest = raw / "squad_values.csv"
            df.to_csv(dest, index=False)
            _log(f"transfermarkt: scraped {len(df)} team values")
            return dest
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN transfermarkt scrape failed ({exc})")
    if static.exists():
        _log("using static squad-value fallback")
        return static
    _log("no squad-value data available — feature will be missing")
    return None


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    _log("=== fetching required results spine ===")
    fetch_results(cfg)
    _log("=== fetching optional Kaggle sources ===")
    fetch_kaggle(cfg)
    _log("=== fetching optional squad values ===")
    fetch_transfermarkt(cfg)
    _log("done.")


if __name__ == "__main__":
    main()
