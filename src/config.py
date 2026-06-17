"""Configuration loading and path helpers.

A single ``load_config()`` reads ``config.yaml`` from the project root and
returns a plain dict. ``PROJECT_ROOT`` and ``ensure_dirs()`` give every module a
consistent view of where data lives, regardless of the current working dir.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_secrets() -> dict[str, bool]:
    """Load project-local credentials so no system env vars are required.

    * ``.env`` (project root) → ``os.environ`` (e.g. ``API_FOOTBALL_KEY``)
    * ``secrets/kaggle.json`` → points ``KAGGLE_CONFIG_DIR`` at ``secrets/`` so the
      Kaggle CLI finds the token inside the project.

    Returns a small status dict so callers/UI can report what's configured.
    """
    # Look for .env in the project root and in secrets/ (both are natural spots).
    for env_path in (PROJECT_ROOT / ".env", PROJECT_ROOT / "secrets" / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and not os.environ.get(key):
                os.environ[key] = val
    # Streamlit Cloud: secrets set in the app's Secrets UI live in ``st.secrets``, not
    # os.environ. Bridge the keys we use across so the cloud picks them up. Guarded so
    # the offline pipeline (no streamlit / no secrets file) is unaffected.
    try:
        import streamlit as _st
        for _k in ("THESTATSAPI_KEY", "API_FOOTBALL_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY",
                   "ACTIONNETWORK_FEED_URL", "GEMINI_API_KEY", "GEMINI_MODEL"):
            if not os.environ.get(_k) and _k in _st.secrets:
                os.environ[_k] = str(_st.secrets[_k])
    except Exception:  # noqa: BLE001 — streamlit absent or no secrets configured
        pass

    # Accept KAGGLE_API_TOKEN as an alias for the CLI's KAGGLE_KEY.
    if os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

    secrets_dir = PROJECT_ROOT / "secrets"
    kaggle_json = secrets_dir / "kaggle.json"
    if kaggle_json.exists():
        os.environ.setdefault("KAGGLE_CONFIG_DIR", str(secrets_dir))

    kaggle_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    return {
        "thestatsapi": bool(os.environ.get("THESTATSAPI_KEY")),
        "api_football": bool(os.environ.get("API_FOOTBALL_KEY")),
        "kaggle": kaggle_json.exists() or kaggle_env,
        # surfaced so we can warn about a token without a username
        "kaggle_key_no_user": bool(os.environ.get("KAGGLE_KEY")
                                   and not os.environ.get("KAGGLE_USERNAME")),
    }


# Load credentials as soon as the package config is imported.
load_secrets()


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML config. Pass a path to override the default."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def path_for(key: str, cfg: dict[str, Any] | None = None) -> Path:
    """Resolve a ``paths.<key>`` config entry to an absolute Path."""
    cfg = cfg or load_config()
    rel = cfg["paths"][key]
    p = PROJECT_ROOT / rel
    return p


def ensure_dirs(cfg: dict[str, Any] | None = None) -> None:
    """Create all configured data/report directories if missing."""
    cfg = cfg or load_config()
    for key in cfg["paths"]:
        path_for(key, cfg).mkdir(parents=True, exist_ok=True)
