"""Country geography: centroids for travel distance + notable altitudes.

We approximate a match venue by the centroid of the *country* the match was
played in (the martj42 results carry a ``country`` column). Travel distance for
a team is the great-circle distance from its own country's centroid to the
venue centroid — a robust proxy without needing per-city geocoding. Unknown
countries return ``None`` so downstream features become NaN (the GBM handles
missingness natively).
"""
from __future__ import annotations

import math

from .team_names import normalize_team

# canonical country -> (latitude, longitude) approximate centroid
CENTROIDS: dict[str, tuple[float, float]] = {
    "Argentina": (-38.4, -63.6), "Australia": (-25.3, 133.8),
    "Austria": (47.5, 14.6), "Belgium": (50.5, 4.5), "Bolivia": (-16.3, -63.6),
    "Bosnia and Herzegovina": (43.9, 17.7), "Brazil": (-14.2, -51.9),
    "Bulgaria": (42.7, 25.5), "Cameroon": (7.4, 12.4), "Canada": (56.1, -106.3),
    "Chile": (-35.7, -71.5), "China": (35.9, 104.2), "Colombia": (4.6, -74.3),
    "Costa Rica": (9.7, -83.8), "Croatia": (45.1, 15.2),
    "Czech Republic": (49.8, 15.5), "Denmark": (56.3, 9.5), "DR Congo": (-4.0, 21.8),
    "Ecuador": (-1.8, -78.2), "Egypt": (26.8, 30.8), "England": (52.4, -1.5),
    "Finland": (61.9, 25.7), "France": (46.6, 2.2), "Germany": (51.2, 10.4),
    "Ghana": (7.9, -1.0), "Greece": (39.1, 21.8), "Honduras": (15.2, -86.2),
    "Hungary": (47.2, 19.5), "Iceland": (64.9, -19.0), "Iran": (32.4, 53.7),
    "Iraq": (33.2, 43.7), "Republic of Ireland": (53.4, -8.2), "Italy": (41.9, 12.6),
    "Ivory Coast": (7.5, -5.5), "Jamaica": (18.1, -77.3), "Japan": (36.2, 138.3),
    "Mexico": (23.6, -102.6), "Morocco": (31.8, -7.1), "Netherlands": (52.1, 5.3),
    "New Zealand": (-40.9, 174.9), "Nigeria": (9.1, 8.7), "North Korea": (40.3, 127.5),
    "North Macedonia": (41.6, 21.7), "Northern Ireland": (54.6, -6.7),
    "Norway": (60.5, 8.5), "Panama": (8.5, -80.8), "Paraguay": (-23.4, -58.4),
    "Peru": (-9.2, -75.0), "Poland": (51.9, 19.1), "Portugal": (39.4, -8.2),
    "Qatar": (25.4, 51.2), "Romania": (45.9, 25.0), "Russia": (61.5, 105.3),
    "Saudi Arabia": (23.9, 45.1), "Scotland": (56.5, -4.2), "Senegal": (14.5, -14.5),
    "Serbia": (44.0, 21.0), "Slovakia": (48.7, 19.7), "Slovenia": (46.2, 14.8),
    "South Africa": (-30.6, 22.9), "South Korea": (35.9, 127.8),
    "Spain": (40.5, -3.7), "Sweden": (60.1, 18.6), "Switzerland": (46.8, 8.2),
    "Tunisia": (33.9, 9.5), "Turkey": (39.0, 35.2), "Ukraine": (48.4, 31.2),
    "United Arab Emirates": (23.4, 53.8), "United States": (39.8, -98.6),
    "Uruguay": (-32.5, -55.8), "Venezuela": (6.4, -66.6), "Wales": (52.3, -3.8),
    "Algeria": (28.0, 1.7), "Cape Verde": (16.0, -24.0), "Uzbekistan": (41.4, 64.6),
    "Jordan": (30.6, 36.2),
}

# Country -> typical playing altitude (m) where it materially affects play.
# Default (sea-level-ish) is 0 for anything not listed.
ALTITUDE: dict[str, float] = {
    "Bolivia": 3640.0,   # La Paz
    "Ecuador": 2850.0,   # Quito
    "Mexico": 2240.0,    # Mexico City
    "Colombia": 2640.0,  # Bogotá
    "Peru": 3400.0,      # frequent highland venues
}


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def centroid_of(country: str | None) -> tuple[float, float] | None:
    canon = normalize_team(country)
    if canon is None:
        return None
    return CENTROIDS.get(canon)


def travel_distance_km(team_country: str | None, venue_country: str | None) -> float | None:
    """Distance a team travels from home to the venue. None if unknown."""
    a = centroid_of(team_country)
    b = centroid_of(venue_country)
    if a is None or b is None:
        return None
    return haversine_km(a, b)


def altitude_of(country: str | None) -> float:
    canon = normalize_team(country)
    if canon is None:
        return 0.0
    return ALTITUDE.get(canon, 0.0)
