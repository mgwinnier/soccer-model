"""Match-context features: geography (travel, altitude) and head-to-head.

Travel distance uses country centroids (``data/geo.py``); a team playing in its
own country travels ~0 km, a neutral-site team travels the full distance. Head-
to-head is a leak-free, time-decayed summary of *prior* meetings between the two
teams, oriented to the current home side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.geo import travel_distance_km, altitude_of


def _geo_features(matches: pd.DataFrame) -> pd.DataFrame:
    idx = matches.set_index("match_id")
    out = pd.DataFrame(index=idx.index)
    home_tr, away_tr, alt = [], [], []
    for row in matches.itertuples(index=False):
        venue = row.venue_country
        home_tr.append(travel_distance_km(row.home_team, venue))
        away_tr.append(travel_distance_km(row.away_team, venue))
        alt.append(altitude_of(venue))
    out["home_travel_km"] = home_tr
    out["away_travel_km"] = away_tr
    out["travel_diff_km"] = out["away_travel_km"] - out["home_travel_km"]
    out["venue_altitude_m"] = alt
    return out


def _h2h_features(matches: pd.DataFrame, decay_days: float) -> pd.DataFrame:
    """Time-decayed prior head-to-head, oriented to the current home team."""
    history: dict[tuple[str, str], list[tuple[pd.Timestamp, int, int]]] = {}
    hp_points = np.full(len(matches), np.nan)
    hp_gd = np.full(len(matches), np.nan)
    n_prev = np.zeros(len(matches))

    for i, row in enumerate(matches.itertuples(index=False)):
        a, b = row.home_team, row.away_team
        key = (a, b) if a <= b else (b, a)
        recs = history.get(key)
        if recs:
            weights, pts, gds = [], [], []
            for (d, s0, s1) in recs:
                # orient s0/s1 (canonical) to current home perspective
                if a == key[0]:
                    hgf, hga = s0, s1
                else:
                    hgf, hga = s1, s0
                w = np.exp(-(row.date - d).days / decay_days)
                weights.append(w)
                pts.append(3 if hgf > hga else (1 if hgf == hga else 0))
                gds.append(hgf - hga)
            wsum = sum(weights)
            if wsum > 0:
                hp_points[i] = np.average(pts, weights=weights)
                hp_gd[i] = np.average(gds, weights=weights)
            n_prev[i] = len(recs)
        # append current result to history (canonical orientation)
        if a == key[0]:
            s0, s1 = row.home_score, row.away_score
        else:
            s0, s1 = row.away_score, row.home_score
        history.setdefault(key, []).append((row.date, int(s0), int(s1)))

    return pd.DataFrame(
        {
            "match_id": matches["match_id"].to_numpy(),
            "h2h_home_points": hp_points,
            "h2h_home_gd": hp_gd,
            "h2h_n_prev": n_prev,
        }
    ).set_index("match_id")


def compute_context_features(
    matches: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    cfg = cfg or load_config()
    matches = matches.sort_values("date")
    geo = _geo_features(matches)
    h2h = _h2h_features(matches, float(cfg["features"]["h2h_decay_days"]))
    out = geo.join(h2h, how="outer")
    # simple flags straight off the match row
    flags = matches.set_index("match_id")[["neutral", "cross_conf", "importance"]].copy()
    flags["neutral"] = flags["neutral"].astype(int)
    flags["cross_conf"] = flags["cross_conf"].astype(int)
    return out.join(flags, how="outer")
