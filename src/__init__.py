"""Badass Soccer Model — 2026 World Cup predictor.

A principled ensemble (Dixon-Coles bivariate Poisson + Elo + LightGBM) over
public international-football data, backtested on past World Cups and used to
Monte-Carlo simulate the 2026 tournament.
"""
__version__ = "0.2.0"

# Use the OS (Windows) certificate store for TLS so HTTPS works behind corporate
# / UniFi proxies that do certificate inspection. Verification stays ON.
try:  # pragma: no cover
    import truststore as _truststore
    _truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass
