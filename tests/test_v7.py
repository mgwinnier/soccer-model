"""Tests for v7: off-consensus filter, motivation features, CLV kill-switches."""
import numpy as np
import pandas as pd
import pytest

from src.data.odds_consensus import (extract_book_prices, consensus,
                                      consensus_prob_for_code, code_edge_vs_consensus,
                                      _norm_provider)
from src.data.clean import load_matches
from src.features.motivation import compute_motivation_features, FEATURE_COLS
from src.models.segment_gate import (SegmentGate, segment_from_code, DEFAULT_DISABLED,
                                      disabled_set)
from src.predict.value import best_bets


# ----------------------------------------------------- Component 1: consensus
def _book(provider, h, d, a, ou_line=2.5, over=-110, under=-110):
    return {"provider": {"name": provider},
            "homeTeamOdds": {"moneyLine": h}, "awayTeamOdds": {"moneyLine": a},
            "drawOdds": {"moneyLine": d}, "overUnder": ou_line,
            "overOdds": over, "underOdds": under}


def test_consensus_sums_to_one_and_excludes_own_book():
    data = {"odds": [
        _book("Bet365", -200, 350, 500),
        _book("MGM", -190, 340, 470),
        _book("Caesars (NJ)", -210, 360, 520),
        _book("Unibet", -205, 350, 500),
    ]}
    books = extract_book_prices(data)
    assert len(books) == 4
    cons = consensus(books, exclude="bet365")
    ml = cons["moneyline"]
    assert ml["n"] == 3                                   # own book excluded
    assert ml["H"] + ml["D"] + ml["A"] == pytest.approx(1.0)
    assert ml["H"] > ml["A"] > 0                          # home favourite


def test_off_consensus_sign_on_a_stale_line():
    # three sharp books agree home is a strong favourite; our book offers a much
    # LONGER home price (a stale line) -> off-consensus edge must be positive.
    data = {"odds": [
        _book("SharpA", -250, 380, 650),
        _book("SharpB", -240, 370, 600),
        _book("SharpC", -260, 390, 700),
    ]}
    cons = consensus(extract_book_prices(data), exclude="mybook")
    from src.data.odds import american_to_decimal
    stale = american_to_decimal(+120)        # our book pays +120 on the favourite
    edge = code_edge_vs_consensus(cons, "H", stale)
    assert edge is not None and edge > 0     # consensus fair_p * 2.2 - 1 > 0

    # a short price (worse than consensus) is unfavourable
    short = american_to_decimal(-400)
    assert code_edge_vs_consensus(cons, "H", short) < 0


def test_consensus_dedups_provider_variants_and_skips_live():
    data = {"odds": [
        _book("Caesars (New Jersey)", -150, 280, 400),
        _book("Caesars Sportsbook (NJ)", -160, 290, 420),
        _book("Caesars (New Jersey) - Live", -120, 250, 380),
        _book("Bet365", -155, 285, 410),
    ]}
    books = extract_book_prices(data)
    provs = {_norm_provider(b["provider"]) for b in books}
    assert provs == {"caesars", "bet365"}     # 3 Caesars variants collapse, Live dropped


def test_norm_provider_handles_non_string():
    import numpy as np
    # a NaN-float / None provider must not crash (.lower() on a float) — returns no key
    assert _norm_provider(np.nan) == ""
    assert _norm_provider(None) == ""
    assert _norm_provider(123) == ""
    # and excluding a NaN provider just excludes nothing
    data = {"odds": [_book("A", -150, 280, 400), _book("B", -150, 280, 400)]}
    cons = consensus(extract_book_prices(data), exclude=np.nan)
    assert cons["moneyline"]["n"] == 2


def test_totals_consensus_only_when_line_matches():
    data = {"odds": [
        _book("A", -150, 280, 400, ou_line=2.5),
        _book("B", -150, 280, 400, ou_line=2.5),
        _book("C", -150, 280, 400, ou_line=3.0),
    ]}
    cons = consensus(extract_book_prices(data))
    assert cons["totals"]["line"] == 2.5
    # over@2.5 is priced; over@3.0 has no consensus support (only one book)
    assert consensus_prob_for_code(cons, ("over", 2.5)) is not None
    assert consensus_prob_for_code(cons, ("over", 3.5)) is None


# -------------------------------------------- Component 2: motivation features
@pytest.fixture(scope="module")
def motivation():
    m = load_matches().sort_values("date").reset_index(drop=True)
    return m.set_index("match_id"), compute_motivation_features(m)


def test_final_round_count_in_expected_range(motivation):
    _, mot = motivation
    n = int(mot["is_final_group_match"].sum())
    # 4 WCs x 8 groups x 2 + Euro2012 x 4 groups x 2 = 72 (allow for data gaps)
    assert 60 <= n <= 72


def test_features_nan_outside_final_group_rounds(motivation):
    _, mot = motivation
    non_final = mot[mot["is_final_group_match"].isna()]
    # every motivation column is NaN where it's not a covered final group match
    assert non_final[FEATURE_COLS].isna().all().all()


def test_2018_group_a_dead_rubbers(motivation):
    # 2018 WC Group A final round: Russia & Uruguay both already through (both on 6
    # pts), Egypt & Saudi both out (both on 0) -> both final matches are dead rubbers.
    meta, mot = motivation
    flagged = mot[mot["is_final_group_match"] == 1].index
    rows = meta.loc[flagged]
    wc18 = rows[(rows["tournament"] == "FIFA World Cup") & (rows["date"].dt.year == 2018)]

    def find(t1, t2):
        hit = wc18[((wc18.home_team == t1) & (wc18.away_team == t2)) |
                   ((wc18.home_team == t2) & (wc18.away_team == t1))]
        assert len(hit) == 1, f"{t1} v {t2} final-round match not found"
        return mot.loc[hit.index[0]]

    ur = find("Uruguay", "Russia")
    assert ur["dead_rubber"] == 1
    assert ur["home_already_q"] == 1 and ur["away_already_q"] == 1

    se = find("Saudi Arabia", "Egypt")
    assert se["dead_rubber"] == 1
    assert se["home_eliminated"] == 1 and se["away_eliminated"] == 1


def test_2018_group_a_needs_are_leak_free(motivation):
    # The dead-rubber verdict must come only from matchdays 1-2: if standings had
    # leaked the final results, the flags could differ. Spot-check that the two
    # already-through teams are NOT also marked needs_win/eliminated (contradiction).
    meta, mot = motivation
    flagged = mot[mot["is_final_group_match"] == 1].index
    rows = meta.loc[flagged]
    wc18 = rows[(rows["tournament"] == "FIFA World Cup") & (rows["date"].dt.year == 2018)]
    hit = wc18[((wc18.home_team == "Uruguay") & (wc18.away_team == "Russia")) |
              ((wc18.home_team == "Russia") & (wc18.away_team == "Uruguay"))]
    f = mot.loc[hit.index[0]]
    for side in ("home", "away"):
        assert f[f"{side}_already_q"] == 1
        assert f[f"{side}_eliminated"] == 0
        assert f[f"{side}_needs_win"] == 0


# ------------------------------------------ Component 3: CLV kill-switches
def test_spreads_disabled_by_default():
    gate = SegmentGate({})
    assert gate.is_disabled("SP:home") and gate.is_disabled("SP:away")
    assert not gate.is_disabled("MR:H")
    assert {"SP:home", "SP:away"} <= gate.disabled_set()


def test_segment_from_code():
    assert segment_from_code("H") == "MR:H"
    assert segment_from_code("over@2.5") == "TG:over"
    assert segment_from_code("under@2.5") == "TG:under"
    assert segment_from_code("cover_home@-1.5") == "SP:home"


def test_kill_switch_disables_negative_clv_segment(tmp_path, monkeypatch):
    import src.models.segment_gate as sg
    monkeypatch.setattr(sg, "path_for", lambda key, cfg=None: tmp_path)
    # 30 TG:over tickets with negative CLV, 30 TG:under with positive CLV
    rows = []
    for i in range(30):
        rows.append({"code": "over@2.5", "clv": -0.02, "result": "loss", "pnl": -1.0})
        rows.append({"code": "under@2.5", "clv": 0.02, "result": "win", "pnl": 0.9})
    pd.DataFrame(rows).to_csv(tmp_path / "clv_ledger.csv", index=False)

    gate = sg.evaluate_kill_switches(min_bets=30, now="2026-06-16")
    assert gate.is_disabled("TG:over")          # negative CLV over the threshold
    assert not gate.is_disabled("TG:under")     # positive CLV stays enabled
    assert (tmp_path / "disabled_segments.json").exists()


def test_kill_switch_respects_min_bets(tmp_path, monkeypatch):
    import src.models.segment_gate as sg
    monkeypatch.setattr(sg, "path_for", lambda key, cfg=None: tmp_path)
    rows = [{"code": "over@2.5", "clv": -0.05, "result": "loss", "pnl": -1.0}
            for _ in range(10)]                 # below min_bets
    pd.DataFrame(rows).to_csv(tmp_path / "clv_ledger.csv", index=False)
    gate = sg.evaluate_kill_switches(min_bets=30)
    assert not gate.is_disabled("TG:over")      # too few bets to judge


# ------------------------------------------ Live 2026 group state (format-correct)
def _group_a_fully_played():
    from src.data.team_names import normalize_team
    from src.simulate.bracket_2026 import GROUPS
    mex, rsa, kor, cze = (normalize_team(t) for t in GROUPS["A"])
    # Mexico wins all (9), Korea 6, Czechia 3, South Africa 0
    return {
        frozenset((mex, rsa)): (mex, 2, 0),
        frozenset((mex, kor)): (mex, 1, 0),
        frozenset((mex, cze)): (mex, 3, 0),
        frozenset((kor, rsa)): (kor, 2, 0),
        frozenset((kor, cze)): (kor, 1, 0),
        frozenset((cze, rsa)): (cze, 1, 0),
    }, (mex, rsa, kor, cze)


def test_standings_counts_points_and_orders():
    from src.simulate import live_state as ls
    played, (mex, rsa, kor, cze) = _group_a_fully_played()
    st = ls.standings(played=played)
    a = st["A"]
    assert list(a["team"]) == [mex, kor, cze, rsa]      # ordered by pts/GD
    assert int(a[a["team"] == mex]["Pts"].iloc[0]) == 9
    assert int(a[a["team"] == rsa]["Pts"].iloc[0]) == 0


def test_clinch_flags_exact_when_group_complete():
    from src.simulate import live_state as ls
    played, (mex, rsa, kor, cze) = _group_a_fully_played()
    fl = ls.clinch_flags(played=played)
    assert fl[mex]["clinched_top2"] and fl[kor]["clinched_top2"]
    assert not fl[cze]["clinched_top2"] and not fl[rsa]["clinched_top2"]


def test_team_summary_status_labels():
    from src.simulate import live_state as ls
    played, (mex, rsa, kor, cze) = _group_a_fully_played()
    state = {"standings": ls.standings(played=played),
             "clinch": ls.clinch_flags(played=played),
             "qual": pd.DataFrame({"team": [mex, rsa, kor, cze],
                                   "advance": [1.0, 0.0, 0.9, 0.3]})}
    assert ls.team_summary(state, mex)["status"] == "✓ qualified (top 2)"
    assert ls.team_summary(state, rsa)["status"] == "eliminated"        # 4th, 3 GP
    assert ls.team_summary(state, cze)["status"].startswith("3rd")      # best-third pending
    assert ls.team_summary(state, "Italy") is None                       # not a 2026 team


def test_best_bets_drops_disabled_segments():
    df = pd.DataFrame([
        {"ev": 0.10, "segment": "MR:H", "disabled": False, "match": "A v B",
         "market": "Match Result", "selection": "A", "american": 150},
        {"ev": 0.20, "segment": "SP:home", "disabled": True, "match": "A v B",
         "market": "Spread", "selection": "A -1.5", "american": 120},
    ])
    bb = best_bets(df, min_ev=0.0)
    assert list(bb["segment"]) == ["MR:H"]       # disabled spread dropped despite higher EV
    bb_all = best_bets(df, min_ev=0.0, include_disabled=True)
    assert set(bb_all["segment"]) == {"MR:H", "SP:home"}
