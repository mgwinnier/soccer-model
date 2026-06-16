"""Country flags for national teams — flagcdn.com image URLs.

Maps each (normalized) team name to an ISO 3166-1 alpha-2 code (with flagcdn's
UK-subdivision codes for the home nations) and builds the flag image URL. Unknown
teams simply get no flag — never a wrong one (the project's no-fabrication rule).
"""
from __future__ import annotations

# Normalized team name -> flagcdn code. Covers all 48 of the 2026 bracket plus the
# common national teams that show up in the Team explorer / historical views.
TEAM_TO_ISO2: dict[str, str] = {
    # 2026 World Cup field
    "Mexico": "mx", "South Africa": "za", "South Korea": "kr", "Czech Republic": "cz",
    "Canada": "ca", "Bosnia and Herzegovina": "ba", "Qatar": "qa", "Switzerland": "ch",
    "Brazil": "br", "Morocco": "ma", "Haiti": "ht", "Scotland": "gb-sct",
    "United States": "us", "Paraguay": "py", "Australia": "au", "Turkey": "tr",
    "Germany": "de", "Ivory Coast": "ci", "Ecuador": "ec", "Curaçao": "cw",
    "Sweden": "se", "Japan": "jp", "Netherlands": "nl", "Tunisia": "tn",
    "New Zealand": "nz", "Iran": "ir", "Belgium": "be", "Egypt": "eg",
    "Uruguay": "uy", "Saudi Arabia": "sa", "Spain": "es", "Cape Verde": "cv",
    "France": "fr", "Senegal": "sn", "Iraq": "iq", "Norway": "no",
    "Argentina": "ar", "Algeria": "dz", "Austria": "at", "Jordan": "jo",
    "Portugal": "pt", "DR Congo": "cd", "Uzbekistan": "uz", "Colombia": "co",
    "England": "gb-eng", "Croatia": "hr", "Ghana": "gh", "Panama": "pa",
    # Other common national teams (historical / Team explorer)
    "Italy": "it", "Wales": "gb-wls", "Poland": "pl", "Denmark": "dk",
    "Russia": "ru", "Ukraine": "ua", "Serbia": "rs", "Nigeria": "ng",
    "Cameroon": "cm", "Chile": "cl", "Peru": "pe", "Costa Rica": "cr",
    "Honduras": "hn", "Greece": "gr", "Romania": "ro", "Hungary": "hu",
    "Slovenia": "si", "Slovakia": "sk", "Republic of Ireland": "ie",
    "Northern Ireland": "gb-nir", "Iceland": "is", "Finland": "fi",
    "North Macedonia": "mk", "Albania": "al", "Georgia": "ge",
    "North Korea": "kp", "China PR": "cn", "China": "cn", "India": "in",
    "Mali": "ml", "Burkina Faso": "bf", "Cameroon ": "cm", "Zambia": "zm",
    "Venezuela": "ve", "Bolivia": "bo", "Israel": "il", "Bulgaria": "bg",
    "Russia ": "ru", "Montenegro": "me", "Armenia": "am", "Azerbaijan": "az",
    "Kazakhstan": "kz", "United Arab Emirates": "ae", "Bahrain": "bh",
    "Oman": "om", "Kuwait": "kw", "Jamaica": "jm", "Trinidad and Tobago": "tt",
    "Guatemala": "gt", "El Salvador": "sv", "Honduras ": "hn",
}


def flag_code(team: str | None) -> str | None:
    if not isinstance(team, str):
        return None
    return TEAM_TO_ISO2.get(team.strip())


def flag_url(team: str | None, w: int = 40) -> str | None:
    """flagcdn image URL (``w`` = pixel width: 20/40/80/160), or None if unknown."""
    code = flag_code(team)
    return f"https://flagcdn.com/w{w}/{code}.png" if code else None


def flag_html(team: str | None, height: int = 16) -> str:
    """An <img> flag (rounded, subtle border) or an empty string if unknown.
    Renders inside any unsafe_allow_html markdown."""
    url = flag_url(team, w=40)
    if not url:
        return ""
    return (f'<img src="{url}" alt="" loading="lazy" '
            f'style="height:{height}px;width:auto;border-radius:2px;'
            f'box-shadow:0 0 0 1px rgba(255,255,255,.12);vertical-align:middle;'
            f'margin-right:6px;" onerror="this.style.display=\'none\'">')


def team_with_flag(team: str, height: int = 16, bold: bool = False) -> str:
    """'🇧🇷 Brazil' as inline HTML (flag image + name)."""
    weight = "700" if bold else "500"
    return (f'<span style="white-space:nowrap;">{flag_html(team, height)}'
            f'<span style="font-weight:{weight};vertical-align:middle;">{team}</span></span>')
