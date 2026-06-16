"""Canonical team-name normalization and confederation lookup.

Different public sources spell national teams differently ("USA" vs
"United States", "Korea Republic" vs "South Korea"). We pick the *martj42*
spelling as canonical (it is the most consistent long-run results dataset) and
map every known alias onto it. ``normalize_team`` is deliberately forgiving:
unknown names pass through unchanged (case/whitespace-normalized) so the
pipeline never crashes on a new team — it just won't get an alias merge.
"""
from __future__ import annotations

# alias (lowercased) -> canonical martj42 name
_ALIASES: dict[str, str] = {
    "usa": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "korea dpr": "North Korea",
    "ir iran": "Iran",
    "iran islamic republic of": "Iran",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "ivory coast (côte d'ivoire)": "Ivory Coast",
    "china pr": "China",
    "chinese taipei": "Taiwan",
    "czechia": "Czech Republic",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "north macedonia": "North Macedonia",
    "fyr macedonia": "North Macedonia",
    "macedonia": "North Macedonia",
    "cape verde islands": "Cape Verde",
    "cabo verde": "Cape Verde",
    "curacao": "Curaçao",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "congo": "Congo",
    "republic of ireland": "Republic of Ireland",
    "ireland": "Republic of Ireland",
    "uae": "United Arab Emirates",
    "trinidad & tobago": "Trinidad and Tobago",
    "st kitts and nevis": "Saint Kitts and Nevis",
    "st lucia": "Saint Lucia",
    "st vincent and the grenadines": "Saint Vincent and the Grenadines",
    "antigua & barbuda": "Antigua and Barbuda",
    "the gambia": "Gambia",
    "kyrgyz republic": "Kyrgyzstan",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "brasil": "Brazil",
}

# canonical name -> confederation
CONFEDERATION: dict[str, str] = {}


def _seed_confederations() -> None:
    uefa = [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
        "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
        "Czech Republic", "Denmark", "England", "Estonia", "Faroe Islands",
        "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece",
        "Hungary", "Iceland", "Israel", "Italy", "Kazakhstan", "Kosovo",
        "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta",
        "Moldova", "Montenegro", "Netherlands", "North Macedonia",
        "Northern Ireland", "Norway", "Poland", "Portugal",
        "Republic of Ireland", "Romania", "Russia", "San Marino", "Scotland",
        "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
        "Turkey", "Ukraine", "Wales",
    ]
    conmebol = [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Paraguay", "Peru", "Uruguay", "Venezuela",
    ]
    concacaf = [
        "Antigua and Barbuda", "Barbados", "Belize", "Canada", "Costa Rica",
        "Cuba", "Curaçao", "Dominican Republic", "El Salvador", "Grenada",
        "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Martinique",
        "Mexico", "Nicaragua", "Panama", "Puerto Rico",
        "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Suriname",
        "Trinidad and Tobago", "United States",
    ]
    conmebol_set = conmebol
    caf = [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cameroon", "Cape Verde", "Central African Republic", "Chad",
        "Comoros", "Congo", "DR Congo", "Djibouti", "Egypt",
        "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon",
        "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya",
        "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali",
        "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia",
        "Niger", "Nigeria", "Rwanda", "Senegal", "Sierra Leone", "Somalia",
        "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia",
        "Uganda", "Zambia", "Zimbabwe",
    ]
    afc = [
        "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan",
        "Brunei", "Cambodia", "China", "Chinese Taipei", "Guam", "Hong Kong",
        "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan", "Kuwait",
        "Kyrgyzstan", "Laos", "Lebanon", "Macau", "Malaysia", "Maldives",
        "Mongolia", "Myanmar", "Nepal", "North Korea", "Oman", "Pakistan",
        "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore",
        "South Korea", "Sri Lanka", "Syria", "Tajikistan", "Thailand",
        "Timor-Leste", "Turkmenistan", "United Arab Emirates", "Uzbekistan",
        "Vietnam", "Yemen",
    ]
    ofc = [
        "American Samoa", "Cook Islands", "Fiji", "New Caledonia",
        "New Zealand", "Papua New Guinea", "Samoa", "Solomon Islands",
        "Tahiti", "Tonga", "Vanuatu",
    ]
    for group, conf in [
        (uefa, "UEFA"), (conmebol_set, "CONMEBOL"), (concacaf, "CONCACAF"),
        (caf, "CAF"), (afc, "AFC"), (ofc, "OFC"),
    ]:
        for team in group:
            CONFEDERATION[team] = conf


_seed_confederations()


def normalize_team(name: str | None) -> str | None:
    """Map a raw team string onto its canonical name. None/empty -> None."""
    if name is None:
        return None
    cleaned = " ".join(str(name).strip().split())
    if not cleaned:
        return None
    return _ALIASES.get(cleaned.lower(), cleaned)


def confederation_of(name: str | None) -> str:
    """Return the confederation for a (raw or canonical) team, or 'OTHER'."""
    canon = normalize_team(name)
    if canon is None:
        return "OTHER"
    return CONFEDERATION.get(canon, "OTHER")
