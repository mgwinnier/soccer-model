"""Tests for v10 UI: flags coverage + theme component output."""
import pytest

from app.flags import flag_url, flag_code, flag_html, team_with_flag
from app import theme
from src.simulate.bracket_2026 import all_teams


def test_every_2026_team_has_a_flag():
    missing = [t for t in all_teams() if flag_url(t) is None]
    assert missing == [], f"teams without a flag code: {missing}"
    assert len(all_teams()) == 48


def test_uk_subdivision_overrides():
    assert flag_code("England") == "gb-eng"
    assert flag_code("Scotland") == "gb-sct"
    assert flag_code("Wales") == "gb-wls"


def test_flag_url_format_and_unknown():
    assert flag_url("Brazil") == "https://flagcdn.com/w40/br.png"
    assert flag_url("Brazil", w=80) == "https://flagcdn.com/w80/br.png"
    assert flag_url("Atlantis") is None      # unknown -> None, never a wrong flag
    assert flag_url(None) is None
    assert flag_html("Atlantis") == ""        # no img for unknown
    assert "<img" in flag_html("France")
    assert "France" in team_with_flag("France")


def test_theme_components_return_html():
    assert "pill" in theme.pill("hi", "green")
    bar = theme.prob_bar(0.5, 0.3, 0.2, "A", "B")
    assert "pbar" in bar and "A 50%" in bar
    assert theme.GREEN.startswith("#") and theme.GOLD.startswith("#")
    cfg = theme._stadium_theme()
    assert "config" in cfg and cfg["config"]["background"] == "transparent"
