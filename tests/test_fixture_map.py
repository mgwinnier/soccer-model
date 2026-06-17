"""Tests for the ESPN-fixture -> TheStatsAPI match_id mapper.

The real ``/matches`` item shape isn't captured to a fixture yet (firewall-blocked locally), so
the matcher is deliberately shape-tolerant. These lock that tolerance to the shapes it claims to
handle AND lock the honesty guarantee: an unknown/garbled name pair returns None, never a guess.
"""
from src.data import fixture_map as fm
from src.data import thestatsapi as ts


def test_team_names_flat_string():
    m = {"match_id": "mt_1", "home": "Argentina", "away": "France"}
    assert fm._team_names(m) == ("Argentina", "France")


def test_team_names_dict_and_home_team_keys():
    m = {"home_team": {"name": "Brazil"}, "away_team": {"short_name": "ARG"}}
    assert fm._team_names(m) == ("Brazil", "ARG")


def test_team_names_nested_teams_object():
    m = {"teams": {"home": {"name": "Spain"}, "away": "Germany"}}
    assert fm._team_names(m) == ("Spain", "Germany")


def test_team_names_participants_flag():
    m = {"participants": [{"is_home": False, "name": "Mexico"},
                          {"is_home": True, "name": "USA"}]}
    assert fm._team_names(m) == ("USA", "Mexico")


def test_find_match_id_unordered_pair_and_date_window():
    cands = [
        {"match_id": "mt_a", "home": "France", "away": "Argentina", "date": "2022-12-18"},
        {"match_id": "mt_b", "home": "Croatia", "away": "Morocco", "date": "2022-12-17"},
    ]
    # order flipped vs the candidate (neutral-site home/away ambiguity) still matches
    assert fm.find_match_id("Argentina", "France", "2022-12-18", cands) == "mt_a"
    # one-day tolerance
    assert fm.find_match_id("Morocco", "Croatia", "2022-12-18", cands, day_tol=1) == "mt_b"


def test_find_match_id_normalizes_aliases():
    cands = [{"match_id": "mt_k", "home": "Korea Republic", "away": "United States",
              "date": "2026-06-20"}]
    # "South Korea"/"USA" normalize onto the API's "Korea Republic"/"United States"
    assert fm.find_match_id("South Korea", "USA", "2026-06-20", cands) == "mt_k"


def test_find_match_id_returns_none_when_absent_or_out_of_window():
    cands = [{"match_id": "mt_a", "home": "France", "away": "Argentina", "date": "2022-12-18"}]
    assert fm.find_match_id("Brazil", "Spain", "2022-12-18", cands) is None      # no such pair
    assert fm.find_match_id("Argentina", "France", "2022-12-25", cands) is None  # outside window


def test_find_match_id_refuses_to_guess_on_unknown_name():
    cands = [{"match_id": "mt_a", "home": "France", "away": "Argentina", "date": "2022-12-18"}]
    # a None/blank side must never resolve to an id
    assert fm.find_match_id("France", "", "2022-12-18", cands) is None


def test_real_2026_matches_item_shape():
    # the EXACT shape captured live from /matches (comp_6107): id=mt_..., home_team/away_team are
    # {id,name} dicts, date is utc_date. Locks the matcher to reality so a shape change is caught.
    item = {"id": "mt_465851784", "competition_id": "comp_6107", "season_id": "sn_118868",
            "matchday": 2, "group_label": "E", "status": "scheduled",
            "utc_date": "2026-06-20T20:00:00.000Z",
            "home_team": {"id": "tm_28696", "name": "Germany"},
            "away_team": {"id": "tm_86577", "name": "Côte d'Ivoire"},
            "score": {"home": None, "away": None, "final_score": None},
            "odds_available": False, "xg_available": False}
    assert fm.match_id_of(item) == "mt_465851784"
    assert fm._team_names(item) == ("Germany", "Côte d'Ivoire")
    assert fm._date_of(item) == "2026-06-20"
    # ESPN's "Ivory Coast" normalizes onto the API's "Côte d'Ivoire"
    assert fm.find_match_id("Germany", "Ivory Coast", "2026-06-20", [item]) == "mt_465851784"


def test_xg_for_fixture_resolves_then_pulls(monkeypatch):
    monkeypatch.setattr(ts, "matches", lambda **k: [
        {"match_id": "mt_final", "home": "Argentina", "away": "France", "date": "2022-12-18"}])
    monkeypatch.setattr(ts, "match_xg", lambda mid, **k: (3.23, 2.27) if mid == "mt_final" else None)
    assert ts.xg_for_fixture("Argentina", "France", "2022-12-18") == (3.23, 2.27)


def test_xg_for_fixture_none_when_unmatched(monkeypatch):
    monkeypatch.setattr(ts, "matches", lambda **k: [
        {"match_id": "mt_x", "home": "Brazil", "away": "Spain", "date": "2022-12-18"}])
    monkeypatch.setattr(ts, "match_xg", lambda mid, **k: (1.0, 1.0))
    assert ts.xg_for_fixture("Argentina", "France", "2022-12-18") is None
