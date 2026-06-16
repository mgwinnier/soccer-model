"""Model vs market across all priced markets → EV, Kelly, and a best-bets board.

For every upcoming fixture we run the rich model analysis and pair each priced
selection (Match Result, Total Goals, Spread) with its **offered** odds to compute
Expected Value and a Kelly stake. BTTS is included as a model probability only
(ESPN prices no BTTS market). `build_bets()` returns both a per-match structure
(for the dashboard cards) and a flat ledger of evaluated bets; `best_bets()`
ranks the +EV ones.

Honest framing: a positive EV means the model disagrees with the price, not that
profit is guaranteed — the numbers are only as good as the model's probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config
from ..data.odds import fetch_fixtures
from .predict_match import MatchPredictor
from .betting import evaluate_bet

VALUE_THRESHOLD = 0.05    # legacy edge flag (model prob − fair prob)


def _host_neutral(home: str) -> bool:
    from ..simulate.bracket_2026 import HOST_TEAMS
    return home not in HOST_TEAMS


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def build_bets(start: str, days: int = 3, bankroll: float = 100.0,
               kelly_fraction: float = 0.5, cfg: dict | None = None,
               predictor: MatchPredictor | None = None,
               use_cache: bool = True, anchor_w: float | None = None,
               recenter: bool = False, recenter_shrink: float = 0.8,
               league: str = "fifa.world", consensus_ref: bool = True) -> dict:
    """Return {'matches': [rich per-match dicts], 'bets': DataFrame of all bets}.

    The model's probabilities are **pure** — the DC+Elo blend calibrated to historical
    match *outcomes* (the isotonic maps in ``analyze``), and NOT influenced by the
    betting market. EV/Kelly compare that honest probability to the offered price; the
    de-vigged market is shown only as a reference ("Fair"). ``anchor_w``/``recenter``
    are retained for backward compatibility but default to OFF — the deployed model is
    market-independent by design. A positive EV means the model genuinely disagrees with
    the price (which historically has *not* been a proven profit edge — see CLV)."""
    cfg = cfg or load_config()
    fixtures = fetch_fixtures(start, days=days, league=league, cfg=cfg, use_cache=use_cache)
    if fixtures.empty:
        return {"matches": [], "bets": pd.DataFrame()}
    predictor = predictor or MatchPredictor(cfg)

    from ..data import lineups
    from ..features import availability
    use_injuries = lineups.is_available()

    matches, ledger = [], []
    for r in fixtures.itertuples(index=False):
        home, away = r.home_team, r.away_team
        # injury-driven availability multipliers (1.0 without API-Football)
        home_out = away_out = []
        h_avail = a_avail = 1.0
        if use_injuries:
            home_out = lineups.injured_players(home)
            away_out = lineups.injured_players(away)
            h_avail = availability.availability_multiplier(home, home_out)
            a_avail = availability.availability_multiplier(away, away_out)

        total_line = _f(r.total_line) or 2.5
        spread_line = _f(r.spread_home_line)
        try:
            neutral = _host_neutral(home)
            a = predictor.analyze(home, away, neutral=neutral, market_total=total_line,
                                  spread_home_line=spread_line,
                                  home_avail=h_avail, away_avail=a_avail)
        except ValueError:
            continue  # unknown team — skip, never fabricate

        bets = []
        # pure model probability vs the offered price (no market anchoring)
        def ev_bet(market, sel, am, model_p, fair):
            return evaluate_bet(market, sel, am, model_p,
                                fair, bankroll, kelly_fraction)
        # --- Match Result (1X2) ---
        for sel, code, am, fair in [
            (home, "H", r.ml_home, _f(r.mkt_home)),
            ("Draw", "D", r.ml_draw, _f(r.mkt_draw)),
            (away, "A", r.ml_away, _f(r.mkt_away)),
        ]:
            if am is None or pd.isna(am):       # no price (e.g. odds nulled post-kickoff)
                continue
            bets.append(ev_bet("Match Result", sel, am, a["probs"][code], fair))
        # --- Total Goals (over/under) ---
        if _f(r.ou_over_odds) is not None:
            bets.append(ev_bet("Total Goals", f"Over {total_line}", r.ou_over_odds,
                               a["p_over_market"], _f(r.mkt_over)))
            bets.append(ev_bet("Total Goals", f"Under {total_line}", r.ou_under_odds,
                               a["p_under_market"], _f(r.mkt_under)))
        # --- Spread / handicap ---
        if spread_line is not None and _f(r.spread_home_odds) is not None and "spread" in a:
            sl = a["spread"]
            bets.append(ev_bet("Spread", f"{home} {spread_line:+g}", r.spread_home_odds,
                               sl["p_home_cover"], _f(r.mkt_spread_home)))
            bets.append(ev_bet("Spread", f"{away} {-spread_line:+g}", r.spread_away_odds,
                               sl["p_away_cover"], _f(r.mkt_spread_away)))

        key_out_home = [p for p in home_out
                        if p.lower() in availability.team_key_players(home)]
        key_out_away = [p for p in away_out
                        if p.lower() in availability.team_key_players(away)]
        match = {
            "game_id": getattr(r, "game_id", None),
            "date": r.date, "home": home, "away": away, "provider": r.provider,
            "neutral": neutral, "analysis": a, "bets": bets,
            "key_out_home": key_out_home, "key_out_away": key_out_away,
            "home_line_move": _f(getattr(r, "home_line_move", None)),
        }
        matches.append(match)

    if recenter:
        _recenter_matches(matches, bankroll, kelly_fraction, recenter_shrink)

    if consensus_ref:
        _attach_consensus(matches, league, cfg)

    from ..models.segment_gate import disabled_set
    disabled = disabled_set(cfg)
    ledger = []
    for m in matches:
        for b in m["bets"]:
            d = b.as_dict()
            d["match"] = f"{m['home']} v {m['away']}"
            d["date"] = m["date"]
            d["cons_p"] = m.get("cons_p", {}).get(b.selection)
            d["cons_edge"] = m.get("cons_edge", {}).get(b.selection)
            seg = _type_key(b.market, b.selection, m["home"], m["away"])
            d["segment"] = seg
            d["disabled"] = seg in disabled
            ledger.append(d)
    return {"matches": matches, "bets": pd.DataFrame(ledger)}


def _cons_code(market: str, selection: str, home: str, away: str):
    """Map a bet (market, selection) to a consensus grade code, or None if the
    market has no usable cross-book consensus (spread lines vary too much)."""
    if market == "Match Result":
        return "H" if selection == home else ("A" if selection == away else "D")
    if market == "Total Goals":
        import re
        m = re.search(r"[-+]?\d*\.?\d+", selection)
        if not m:
            return None
        line = float(m.group())
        return ("over", line) if selection.startswith("Over") else ("under", line)
    return None


def _attach_consensus(matches: list, league: str, cfg: dict | None) -> None:
    """Per match, compute the read-only multi-book consensus and tag each bet with
    the consensus fair prob and our-book edge-vs-consensus. Best-effort: a missing
    summary just leaves the fields None (no consensus, no crash)."""
    from ..data.odds_consensus import (match_consensus, consensus_prob_for_code,
                                        code_edge_vs_consensus)
    from ..data.odds import american_to_decimal
    for m in matches:
        gid = m.get("game_id")
        m["cons_p"], m["cons_edge"] = {}, {}
        if gid is None:
            continue
        prov = m.get("provider")
        exclude = prov if isinstance(prov, str) and prov else "bet365"   # never NaN
        cons = match_consensus(gid, league, exclude=exclude, cfg=cfg, allow_fetch=True)
        if not cons:
            continue
        for b in m["bets"]:
            code = _cons_code(b.market, b.selection, m["home"], m["away"])
            if code is None:
                continue
            dec = american_to_decimal(b.american)
            m["cons_p"][b.selection] = consensus_prob_for_code(cons, code)
            m["cons_edge"][b.selection] = code_edge_vs_consensus(cons, code, dec)


def _type_key(market: str, selection: str, home: str, away: str) -> str:
    """Group selections by directional role so each role is de-biased separately."""
    if market == "Match Result":
        return "MR:H" if selection == home else ("MR:A" if selection == away else "MR:D")
    if market == "Total Goals":
        return "TG:over" if selection.startswith("Over") else "TG:under"
    if market == "Spread":
        return "SP:home" if selection.startswith(home) else "SP:away"
    return market


def _recenter_matches(matches: list, bankroll: float, kelly_fraction: float,
                      shrink: float) -> None:
    """Remove the model's slate-wide systematic tilt vs the market, per role.

    For each role (home/draw/away, over/under, spread sides) we subtract a shrunk
    estimate of the model's average (model − market) edge. This preserves each
    market's sum-to-one and leaves only *game-specific* disagreement as value."""
    # Prefer the stable bias learned from thousands of historical prices; fall back
    # to a noisy per-slate estimate only if it hasn't been fit yet.
    from ..models.market_bias import load_default
    persisted = load_default()
    if persisted.bias:
        bias = {k: v * persisted.shrink / shrink for k, v in persisted.bias.items()}
    else:
        from collections import defaultdict
        edges = defaultdict(list)
        for m in matches:
            for b in m["bets"]:
                if b.fair_p is not None:
                    edges[_type_key(b.market, b.selection, m["home"], m["away"])].append(
                        b.model_p - b.fair_p)
        bias = {k: float(np.mean(v)) for k, v in edges.items() if v}
    for m in matches:
        rebuilt = []
        for b in m["bets"]:
            tk = _type_key(b.market, b.selection, m["home"], m["away"])
            adj = b.model_p - shrink * bias.get(tk, 0.0)
            adj = min(max(adj, 1e-4), 1 - 1e-4)
            rebuilt.append(evaluate_bet(b.market, b.selection, b.american, adj,
                                        b.fair_p, bankroll, kelly_fraction))
        m["bets"] = rebuilt


def best_bets(bets_df: pd.DataFrame, min_ev: float = 0.0,
              include_disabled: bool = False, min_prob_edge: float = 0.02,
              max_decimal: float | None = 6.0) -> pd.DataFrame:
    """Filter to recommended bets, ranked by EV descending.

    Beyond EV ≥ min_ev, a bet must clear the **probability-edge gate**
    (``betting.qualifies``): the model must beat the de-vigged price by at least
    ``min_prob_edge`` (a *real* disagreement, not EV leverage on long odds) and not
    be a longshot past ``max_decimal``. This rebalances the flags away from the
    leverage-driven underdog junk. Disabled segments (spreads etc.) are dropped
    unless ``include_disabled``."""
    if bets_df.empty:
        return bets_df
    keep = bets_df.copy()
    if not include_disabled and "disabled" in keep.columns:
        keep = keep[~keep["disabled"].astype(bool)]
    if {"model_p", "decimal"}.issubset(keep.columns):
        try:
            from .betting import qualifies
        except ImportError:  # resilient to a stale deploy of betting.py
            return keep[keep["ev"] >= min_ev].sort_values("ev", ascending=False) \
                .reset_index(drop=True)
        mask = keep.apply(lambda r: qualifies(
            r["model_p"], r.get("fair_p"), r.get("decimal"),
            min_ev, min_prob_edge, max_decimal), axis=1)
        keep = keep[mask]
    else:
        keep = keep[keep["ev"] >= min_ev]
    cols = ["date", "match", "market", "selection", "segment", "american", "model_p",
            "fair_p", "cons_edge", "edge", "ev", "kelly_used", "stake"]
    keep = keep[[c for c in cols if c in keep.columns]].copy()
    return keep.sort_values("ev", ascending=False).reset_index(drop=True)


def cap_exposure(bets_df: pd.DataFrame, bankroll: float,
                 max_fraction: float = 1.0) -> pd.DataFrame:
    """Scale stakes down so total exposure ≤ bankroll·max_fraction.

    Independent per-bet Kelly stakes can sum to more than the bankroll when many
    bets clear the bar at once; this proportionally rescales them so you never
    risk more than intended across simultaneous bets."""
    if bets_df.empty or "stake" not in bets_df.columns:
        return bets_df
    total = bets_df["stake"].sum()
    cap = bankroll * max_fraction
    out = bets_df.copy()
    if total > cap and total > 0:
        out["stake"] = (out["stake"] * cap / total).round(2)
    return out


def compare(start: str, days: int = 3, cfg: dict | None = None,
            predictor: MatchPredictor | None = None,
            use_cache: bool = True) -> pd.DataFrame:
    """Back-compat flat 1X2 table (used by the simple CLI)."""
    res = build_bets(start, days, cfg=cfg, predictor=predictor, use_cache=use_cache)
    rows = []
    for m in res["matches"]:
        a = m["analysis"]
        rows.append({
            "date": m["date"], "home_team": m["home"], "away_team": m["away"],
            "model_home": a["probs"]["H"], "model_draw": a["probs"]["D"],
            "model_away": a["probs"]["A"], "provider": m["provider"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from datetime import datetime
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    start = sys.argv[1] if len(sys.argv) > 1 else datetime.utcnow().strftime("%Y-%m-%d")
    res = build_bets(start, days=2, bankroll=1000, kelly_fraction=0.5, use_cache=False)
    bb = best_bets(res["bets"], min_ev=0.0)
    if bb.empty:
        print("no fixtures/odds found")
    else:
        show = bb.copy()
        show["model%"] = (show["model_p"] * 100).round(0)
        show["ev%"] = (show["ev"] * 100).round(1)
        # off-consensus edge: + means our book beats the market consensus (a soft line)
        if "cons_edge" in show.columns:
            show["cons%"] = (show["cons_edge"] * 100).round(1)
        cols = [c for c in ["match", "market", "selection", "segment", "american",
                            "model%", "ev%", "cons%", "stake"] if c in show.columns]
        with pd.option_context("display.width", 240, "display.max_rows", 40):
            print(f"\n+EV bets since {start} (bankroll $1000, half-Kelly; "
                  f"disabled segments suppressed — spreads off by default):\n")
            print(show[cols].head(25).to_string(index=False))
        print(f"\n{len(bb)} +EV selections · total recommended stake "
              f"${bb['stake'].sum():.0f}")
        print("cons% = our-book edge vs market consensus (+ = our book offers a "
              "longer-than-market price; blank = no multi-book consensus available)")
