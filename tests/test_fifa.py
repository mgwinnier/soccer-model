"""Tests for the FIFA official-lineup client (network mocked with real captured shapes)."""
from src.data import fifa


def _en(s):
    return [{"Locale": "en-GB", "Description": s}]


# Calendar entry shape (Home/Away carry TeamName + ids).
CAL = [
    {"IdCompetition": "17", "IdSeason": "285023", "IdStage": "289273", "IdMatch": "400021510",
     "Date": "2026-06-17T23:00:00Z",
     "Home": {"TeamName": _en("Ghana"), "IdCountry": "GHA"},
     "Away": {"TeamName": _en("Panama"), "IdCountry": "PAN"}},
    {"IdCompetition": "17", "IdSeason": "285023", "IdStage": "289273", "IdMatch": "400021500",
     "Date": "2026-06-27T19:00:00Z",
     "Home": {"TeamName": _en("Congo DR"), "IdCountry": "COD"},   # FIFA alias -> DR Congo
     "Away": {"TeamName": _en("Uzbekistan"), "IdCountry": "UZB"}},
]


def _player(name, num, status, pos, cap=False):
    return {"PlayerName": _en(name), "ShirtNumber": num, "Status": status,
            "Position": pos, "Captain": cap}


def _team(name, tactics, n_start=11, n_bench=3):
    players = [_player(f"{name} S{i}", i + 1, 1, i % 4, cap=(i == 1)) for i in range(n_start)]
    players += [_player(f"{name} B{i}", 30 + i, 2, 2) for i in range(n_bench)]
    return {"TeamName": _en(name), "Tactics": tactics, "Players": players}


def _live(officiality):
    return {"Date": "2026-06-17T23:00:00Z", "OfficialityStatus": officiality,
            "HomeTeam": _team("Ghana", "4-2-3-1"), "AwayTeam": _team("Panama", "3-4-3")}


def test_match_ref_resolves_by_pair():
    r = fifa.match_ref("Ghana", "Panama", cal=CAL)
    assert r and r["match"] == "400021510" and r["stage"] == "289273"
    # orientation-agnostic + FIFA name alias (Congo DR -> DR Congo)
    assert fifa.match_ref("Panama", "Ghana", cal=CAL)["match"] == "400021510"
    assert fifa.match_ref("DR Congo", "Uzbekistan", cal=CAL)["match"] == "400021500"
    assert fifa.match_ref("Brazil", "Spain", cal=CAL) is None


def test_parse_team_takes_only_starters():
    t = fifa._parse_team(_team("Ghana", "4-2-3-1", n_start=11, n_bench=5))
    assert len(t["xi"]) == 11 and t["formation"] == "4-2-3-1"
    assert all("name" in p and "pos" in p for p in t["xi"])


def test_lineups_projected_vs_confirmed(monkeypatch):
    # officiality 0 -> projected; >=1 -> confirmed
    monkeypatch.setattr(fifa, "match_ref", lambda h, a, cal=None: {
        "comp": "17", "season": "285023", "stage": "289273", "match": "400021510"})
    monkeypatch.setattr(fifa, "_get", lambda path, ttl=0, **k: _live(0))
    ln = fifa.lineups("Ghana", "Panama")
    assert ln["source"] == "FIFA" and ln["confirmed"] is False
    assert len(ln["home"]["xi"]) == 11 and ln["away"]["formation"] == "3-4-3"

    monkeypatch.setattr(fifa, "_get", lambda path, ttl=0, **k: _live(1))
    assert fifa.lineups("Ghana", "Panama")["confirmed"] is True


def test_lineups_orientation_by_name(monkeypatch):
    monkeypatch.setattr(fifa, "match_ref", lambda h, a, cal=None: {
        "comp": "17", "season": "285023", "stage": "289273", "match": "x"})
    monkeypatch.setattr(fifa, "_get", lambda path, ttl=0, **k: _live(1))
    # caller asks Panama as home -> home block must be Panama regardless of FIFA's Home/Away
    ln = fifa.lineups("Panama", "Ghana")
    assert ln["home"]["team"] == "Panama" and ln["away"]["team"] == "Ghana"


def test_lineups_none_without_ref(monkeypatch):
    monkeypatch.setattr(fifa, "match_ref", lambda h, a, cal=None: None)
    assert fifa.lineups("Ghana", "Panama") is None
