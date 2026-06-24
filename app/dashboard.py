"""Streamlit dashboard for the Badass Soccer Model.

Run:  streamlit run app/dashboard.py

Pages
  • Matches     — rich multi-market cards: model vs de-vigged Vegas for Match
                  Result / Total Goals / Spread (+ model BTTS), each with EV and a
                  Kelly stake, plus team context (Elo/form/xG/H2H/injuries) and a
                  scoreline heatmap. Live odds from ESPN.
  • Value Board — every +EV bet across all matches ranked by EV, with Kelly stakes
                  capped to your bankroll.
  • Tournament  — 2026 championship & advancement odds (Monte-Carlo).
  • Performance — backtest RPS, calibration, ablation.
  • Team        — strength + any head-to-head predictor.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# All displayed times are Central (America/Chicago, auto CST/CDT). ESPN datetimes are UTC.
try:
    from zoneinfo import ZoneInfo
    _CT = ZoneInfo("America/Chicago")
except Exception:  # pragma: no cover - fallback if tzdata missing
    _CT = None


def _ct(dt):
    """Convert a (UTC) datetime to a Central-time pandas Timestamp."""
    ts = pd.to_datetime(dt, errors="coerce")
    if ts is None or pd.isna(ts):
        return ts
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(_CT) if _CT is not None else ts


def _ct_str(dt, fmt="%a %d %b %I:%M %p CT"):
    ts = _ct(dt)
    return ts.strftime(fmt) if ts is not None and not pd.isna(ts) else ""


def _today_ct():
    """Today's date in Central time — the cloud server clock is UTC, which rolls over
    to tomorrow while it's still today in the US."""
    base = datetime.now(_CT) if _CT is not None else datetime.utcnow()
    return base.date()

from src.config import load_config, path_for  # noqa: E402
from src.predict.predict_match import MatchPredictor  # noqa: E402
from src.predict import value as value_mod  # noqa: E402
from src.models.base import OUTCOMES  # noqa: E402
from app import theme  # noqa: E402
from app.flags import flag_url, flag_html, team_with_flag  # noqa: E402
from src.predict.betting import expected_value  # noqa: E402
try:                                              # resilient to a stale deploy
    from src.predict.betting import qualifies  # noqa: E402
except Exception:  # noqa: BLE001
    def qualifies(model_p, fair_p, decimal, min_ev=0.03, min_prob_edge=0.02,
                  max_decimal=6.0):
        if model_p is None or decimal is None or not (decimal > 1):
            return False
        if expected_value(model_p, decimal) < min_ev:
            return False
        if max_decimal is not None and decimal > max_decimal:
            return False
        if fair_p is not None and (model_p - fair_p) < min_prob_edge:
            return False
        return True

st.set_page_config(page_title="FIFA World Cup 2026 · Soccer Model",
                   page_icon="🏆", layout="wide")
theme.inject_css()
theme.enable_altair()
CFG = load_config()

GREEN, GREY, RED, GOLD = theme.GREEN, theme.MUTED, theme.RED, theme.GOLD


def _bet_units(b) -> float:
    """Recommended stake (units, 1u = 1% bankroll) for a BetEval — disciplined sizing that leaves the
    well-calibrated bulk as capped quarter-Kelly and down-weights longshots (src/predict/staking.py).
    The Kelly fraction is recovered from the bet so we don't thread it through every call site."""
    from src.predict import staking
    kfull = getattr(b, "kelly_full", 0.0) or 0.0
    frac = (b.kelly_used / kfull) if kfull else 0.25
    return staking.recommended_units(b.model_p, getattr(b, "fair_p", None), b.decimal, frac=frac)


# ----------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Fitting models (one-time)…")
def get_predictor() -> MatchPredictor:
    return MatchPredictor(CFG)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    p = path_for("reports", CFG) / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=180, show_spinner="Pulling live odds + analysing…")
def get_bets(day: str, days: int, bankroll: float, kelly: float,
             upset_temp: float = 1.0) -> dict:
    return value_mod.build_bets(day, days=days, bankroll=bankroll,
                                kelly_fraction=kelly, cfg=CFG,
                                predictor=get_predictor(), use_cache=False,
                                upset_temp=upset_temp)


@st.cache_data(ttl=900, show_spinner=False)
def get_wc_cands() -> list:
    """The full current-season WC match list, pulled ONCE and shared by every per-card lookup
    (xG / stats / lineups) so we don't re-pull /matches for each card. [] without a key."""
    try:
        from src.data import thestatsapi as _ts
        if not _ts.is_available():
            return []
        return _ts.matches(competition_id=_ts.WC_COMP,
                           season_id=_ts.current_season_id(cfg=CFG), cfg=CFG)
    except Exception:  # noqa: BLE001
        return []


def _resolve_mid(home: str, away: str, date: str):
    """Resolve a fixture to a TheStatsAPI match_id via the shared cached match list (no extra
    /matches pull)."""
    from src.data import fixture_map as _fm
    from src.data import thestatsapi as _ts
    cands = get_wc_cands()
    if cands:
        return _fm.find_match_id(home, away, str(date)[:10], cands)
    # shared list empty (transient) -> fall back to a direct per-fixture resolve
    return _ts.match_id_for_fixture(home, away, str(date)[:10], cfg=CFG)


@st.cache_data(ttl=3600, show_spinner=False)
def get_fixture_xg(home: str, away: str, date: str):
    """Real (home_xg, away_xg) for a played fixture, or None. Honest reality-check only."""
    try:
        from src.data import thestatsapi as _ts
        mid = _resolve_mid(home, away, date)
        return _ts.match_xg(mid, cfg=CFG) if mid else None
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_match_stats(home: str, away: str, date: str):
    """Full match-stats overview (possession, shots, xG, …) for a played fixture, or {}."""
    try:
        from src.data import thestatsapi as _ts
        mid = _resolve_mid(home, away, date)
        return _ts.match_stats(mid, cfg=CFG) if mid else {}
    except Exception:  # noqa: BLE001
        return {}


def match_stats_block(m: dict):
    """Broadcast-style stat comparison for a played match (real fields only)."""
    s = get_match_stats(m["home"], m["away"], str(m["date"])[:10])
    if not s:
        return
    def _pa(side):  # pass accuracy %
        p, ap = (s.get("passes") or {}).get(side), (s.get("accurate_passes") or {}).get(side)
        return round(100 * ap / p) if p and ap else None
    rows = []
    order = [("ball_possession", "Possession", "{}%"), ("expected_goals", "Expected goals", "{}"),
             ("total_shots", "Shots", "{}"), ("shots_on_target", "On target", "{}"),
             ("big_chances", "Big chances", "{}"), ("corner_kicks", "Corners", "{}")]
    for key, label, fmt in order:
        v = s.get(key)
        if v:
            rows.append({"label": label, "home": v["home"], "away": v["away"],
                         "disp_home": fmt.format(v["home"]), "disp_away": fmt.format(v["away"])})
    pah, paa = _pa("home"), _pa("away")
    if pah is not None:
        rows.append({"label": "Pass accuracy", "home": pah, "away": paa,
                     "disp_home": f"{pah}%", "disp_away": f"{paa}%"})
    yc = s.get("yellow_cards"); rc = s.get("red_cards")
    if yc:
        def _cards(side):
            r = (rc or {}).get(side, 0)
            return f"{yc[side]}Y" + (f" {r}R" if r else "")
        rows.append({"label": "Cards", "home": yc["home"], "away": yc["away"],
                     "disp_home": _cards("home"), "disp_away": _cards("away")})
    if rows:
        st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:11px;'
                    f'color:{GREY};margin-bottom:2px"><span style="color:{GREEN}">{m["home"]}</span>'
                    f'<span style="color:{GOLD}">{m["away"]}</span></div>'
                    + theme.stat_bars(rows), unsafe_allow_html=True)


@st.cache_data(ttl=240, show_spinner=False)
def get_book_odds(home: str, away: str, date: str = ""):
    """The user's PPH book line (DST, open — no auth) for a fixture: moneyline/totals/BTTS, or
    None. This is the price you'd actually bet, shown alongside the model."""
    try:
        from src.data import dst
        return dst.book_odds(home, away)
    except Exception:  # noqa: BLE001
        return None


def _book_dec(b, m: dict, book: dict | None):
    """The user's book decimal price for one bet selection, or None."""
    if not book:
        return None
    if b.market == "Match Result":
        code = "H" if b.selection == m["home"] else ("A" if b.selection == m["away"] else "D")
        return (book.get("moneyline") or {}).get(code)
    if b.market == "Total Goals":
        import re
        mt = re.search(r"[\d.]+", b.selection)
        if not mt:
            return None
        t = (book.get("totals") or {}).get(float(mt.group())) or {}
        return t.get("over" if b.selection.startswith("Over") else "under")
    if b.market == "BTTS":
        return (book.get("btts") or {}).get("yes" if "Yes" in b.selection else "no")
    return None


def _match_not_started(m: dict) -> bool:
    """True only for fixtures that haven't kicked off — excludes finished AND in-play games.
    Kalshi signals/prices are shown only for these (settled/live games are hidden)."""
    if m.get("played"):
        return False
    stt = m.get("status")
    if stt in ("post", "in"):
        return False
    if stt == "pre":
        return True
    ko = pd.to_datetime(m.get("date"), errors="coerce", utc=True)
    return ko is not None and not pd.isna(ko) and ko > pd.Timestamp.now(tz="UTC")


@st.cache_data(ttl=30, show_spinner=False)
def get_kalshi_book(home: str, away: str):
    """Full live Kalshi book (Match Result + BTTS + Totals + Spread) for a fixture, or None."""
    try:
        from src.data import kalshi
        return kalshi.match_book(home, away)
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=120, show_spinner=False)
def get_kalshi_futures():
    """Kalshi tournament-winner Yes prices by normalized team, or {} if unavailable."""
    try:
        from src.data import kalshi
        return kalshi.winner_futures()
    except Exception:  # noqa: BLE001
        return {}


def _kalshi_dec(b, m: dict, kal: dict | None):
    """Kalshi decimal price (1/ask) for a bet selection across all four markets (Match Result,
    Total Goals, Spread, BTTS), or None when Kalshi doesn't list that selection."""
    if not kal:
        return None
    from src.data import kalshi
    px = kalshi.price_for(kal, b.market, b.selection, m["home"], m["away"])
    ask = (px or {}).get("ask")
    return (1.0 / ask) if (ask and ask > 0) else None


@st.cache_data(ttl=600, show_spinner=False)
def get_lineup_status(home: str, away: str, date: str):
    """Confirmed-XI status + 'regular starter missing today' flag, or None.

    Honest no-op without a key / before the team sheet posts (~75 min pre-KO). Cached so the
    shared app makes few calls."""
    try:
        from src.data import lineup_status as _ls
        return _ls.lineup_status(home, away, date, cfg=CFG, cands=get_wc_cands())
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_espn_lineups(game_id):
    """Confirmed XIs from ESPN, or None."""
    if not game_id:
        return None
    try:
        from src.data.odds import fetch_lineups
        return fetch_lineups(str(game_id))
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=60, show_spinner=False)
def get_fast_lineups(home: str, away: str, date: str = "", game_id=None):
    """The earliest XI we can get: **FIFA official** first (its own match centre is the source of
    the team sheet — a *projected* XI hours early, the *confirmed* one the moment it's official),
    falling back to ESPN. Returns ``{home, away, source, confirmed}`` or None. 60s cache so it
    refreshes near kickoff instead of waiting on a stale read."""
    try:
        from src.data import fifa
        fl = fifa.lineups(home, away)
        if fl and fl.get("home") and fl.get("away"):
            return {"home": fl["home"], "away": fl["away"], "source": "FIFA",
                    "confirmed": bool(fl.get("confirmed"))}
    except Exception:  # noqa: BLE001
        pass
    el = get_espn_lineups(game_id)
    if el:                                        # ESPN only posts the confirmed sheet
        return {"home": el.get("home"), "away": el.get("away"), "source": "ESPN", "confirmed": True}
    return None


def _hours_to_ko(date) -> float | None:
    try:
        ko = pd.to_datetime(date, utc=True)
        now = pd.Timestamp.now(tz="UTC")
        return (ko - now).total_seconds() / 3600.0
    except Exception:  # noqa: BLE001
        return None


def _value_str(v) -> str:
    if not v:
        return "—"
    return f"€{v/1e6:.0f}M" if v >= 1e6 else f"€{v/1e3:.0f}K"


def _key_out_flag(team: str, xi_names: list):
    """Value-based 'key player out' flag — squad's top players by market value missing from the
    confirmed XI. Works from match 1 (no prior game needed). Uses the committed squad snapshot."""
    try:
        from src.data import squad_values as sv
        miss = sv.key_absentees(team, [n for n in xi_names if n], top_n=6)
    except Exception:  # noqa: BLE001
        miss = []
    if miss:
        names = ", ".join(f"{x['name']} ({_value_str(x['market_value'])})" for x in miss)
        st.markdown(f"<span style='color:{GOLD};font-weight:600'>Key player out:</span> "
                    f"<span style='color:{GOLD}'>{names}</span>", unsafe_allow_html=True)


def _xi_strength_line(team: str, names: list):
    """Per-team 'how loaded is the XI' caption: share of squad market value on the pitch."""
    try:
        from src.data import squad_values as sv
        s, share = sv.xi_value(team, [n for n in names if n])
        if share is not None:
            st.caption(f"XI strength: **{share*100:.0f}%** · {_value_str(s)} of "
                       f"{_value_str(sv.total_value(team))}")
    except Exception:  # noqa: BLE001
        pass


def _value_gap_and_upset(m: dict, home_names: list, away_names: list):
    """Matchup value gap + a 'weakened favorite' upset-watch flag (insight only, no model change)."""
    try:
        from src.data import squad_values as sv
    except Exception:  # noqa: BLE001
        return
    hv, hs = sv.xi_value(m["home"], [n for n in home_names if n])
    av, as_ = sv.xi_value(m["away"], [n for n in away_names if n])
    if hv is None or av is None:
        return
    gap = hv - av
    if abs(gap) >= 50e6:
        stronger = m["home"] if gap > 0 else m["away"]
        st.caption(f"XI value gap: **{stronger}** by {_value_str(abs(gap))}")
    # upset watch: the model's FAVORITE is missing a top-value player (a rested/injured star) —
    # the classic upset setup. Heuristic context only; validated forward on the Performance page.
    probs = _display_probs(m)
    fav_home = probs["H"] >= probs["A"]
    fav, fav_names, fav_share = ((m["home"], home_names, hs) if fav_home
                                 else (m["away"], away_names, as_))
    ko = sv.key_absentees(fav, [n for n in fav_names if n], top_n=3)
    if ko:
        outs = ", ".join(x["name"] for x in ko)
        share_txt = f" ({fav_share*100:.0f}% of value starting)" if fav_share is not None else ""
        st.markdown(f"<span style='color:{GOLD};font-weight:600'>Upset watch:</span> "
                    f"{fav} (model favorite) without top player(s) {outs}{share_txt}.",
                    unsafe_allow_html=True)


def _espn_lineup_fallback(m: dict) -> bool:
    """Render the XI from the fastest source (FIFA official, else ESPN) when TheStatsAPI hasn't
    posted its richer sheet. Labels projected vs confirmed; adds market values + a value-based
    'key player out' flag. Returns True if it rendered something."""
    from src.data import squad_values as sv
    el = get_fast_lineups(m["home"], m["away"], str(m.get("date"))[:10], m.get("game_id"))
    if not el:
        return False
    status = "Confirmed" if el.get("confirmed") else "Projected"
    src = el.get("source") or "—"
    tone = GREEN if el.get("confirmed") else GOLD
    st.markdown(f"**{status} lineups** · <span style='color:{tone}'>{src}"
                f"{'' if el.get('confirmed') else ' · projected XI'}</span>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    xi_names = {}
    for col, side, team in ((c1, "home", m["home"]), (c2, "away", m["away"])):
        s = el.get(side) or {}
        names = [p.get("name") for p in s.get("xi", [])]
        xi_names[side] = names
        tot = sv.total_value(team)
        with col:
            head = f"**{team_with_flag(team, 16, True)}** · {s.get('formation') or '?'}"
            if tot:
                head += f" · squad {_value_str(tot)}"
            st.markdown(head, unsafe_allow_html=True)
            rows = []
            for p in s.get("xi", []):
                rows.append({"#": p.get("jersey") or "", "Starter": p.get("name"),
                             "Pos": p.get("pos") or "",
                             "Value": _value_str(sv.player_value(team, p.get("name")))})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            _xi_strength_line(team, names)
            _key_out_flag(team, names)
    _value_gap_and_upset(m, xi_names.get("home", []), xi_names.get("away", []))
    st.caption(f"XI from {src} ({status.lower()}) · **Value** = player market value · **XI strength** "
               "= share of squad value starting · **Key player out** = a top-value player benched. "
               "Form ratings appear once TheStatsAPI posts its sheet.")
    return True


def lineup_block(m: dict):
    """Confirmed XI + flagged absences — only when the sheet could be posted (played / near KO).
    TheStatsAPI is primary (it adds ratings + the missing-starter flag); ESPN is the faster
    fallback so the XI shows as soon as it's announced."""
    hrs = _hours_to_ko(m.get("date"))
    if not (m.get("played") or (hrs is not None and hrs <= 6)):
        return
    ls = get_lineup_status(m["home"], m["away"], str(m["date"])[:10])
    if not (ls and ls.get("posted")):
        # TheStatsAPI not ready -> show ESPN's confirmed XI (usually earlier)
        if not _espn_lineup_fallback(m) and not m.get("played"):
            st.caption("Confirmed XI not posted yet (appears here as soon as it's announced).")
        return
    played = ls.get("played")
    st.markdown("**Confirmed lineups**")
    c1, c2 = st.columns(2)
    flagged = False
    xi_names = {}
    for col, side, team in ((c1, "home", m["home"]), (c2, "away", m["away"])):
        s = ls.get(side) or {}
        form = s.get("formation") or "?"
        from src.data import squad_values as sv
        with col:
            head = f"**{team_with_flag(team, 16, True)}** · {form}"
            tot = sv.total_value(team)
            if tot:
                head += f" · squad {_value_str(tot)}"
            st.markdown(head, unsafe_allow_html=True)
            rows = []
            for p in s.get("xi", []):
                recent = " · ".join(f"{r:.1f}" for r in (p.get("recent") or [])[:3]) or "—"
                row = {"Starter": p.get("name"), "Pos": p.get("position") or "",
                       "Value": _value_str(sv.player_value(team, p.get("name"))),
                       "Form (last 3)": recent, "Avg": (f"{p['avg']:.2f}" if p.get("avg") else "—")}
                if played:
                    row["Today"] = f"{p['today']:.1f}" if p.get("today") is not None else "—"
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            xi_nm = [p.get("name") for p in s.get("xi", [])]
            xi_names[side] = xi_nm
            _xi_strength_line(team, xi_nm)
            # value-based key-player-out flag (works from match 1)
            _key_out_flag(team, xi_nm)
            # secondary: rotation vs the previous match (ratings-based, matchday 2+)
            miss = s.get("missing_starters") or []
            if miss:
                flagged = True
                names = ", ".join(f"{x['name']}" + (f" ({x['avg']:.1f})" if x.get("avg") else "")
                                  for x in miss)
                st.caption(f"Out vs last XI: {names}")
    _value_gap_and_upset(m, xi_names.get("home", []), xi_names.get("away", []))
    # before → after: re-run the model with a bounded availability penalty for missing regulars
    if flagged and not played:
        from src.data.lineup_status import lineup_availability
        hm, am = lineup_availability(m["home"], m["away"], str(m["date"])[:10], cfg=CFG, status=ls)
        if hm < 1.0 or am < 1.0:
            base = m["analysis"]["probs"]
            try:
                adj = get_predictor().analyze(m["home"], m["away"], neutral=m["neutral"],
                                              home_avail=hm, away_avail=am)["probs"]
                st.markdown(
                    f"**Lineup-adjusted model** (missing regulars, capped 10%): "
                    f"{m['home']} {base['H']*100:.0f}% → **{adj['H']*100:.0f}%** · "
                    f"Draw {base['D']*100:.0f}% → **{adj['D']*100:.0f}%** · "
                    f"{m['away']} {base['A']*100:.0f}% → **{adj['A']*100:.0f}%**")
            except Exception:  # noqa: BLE001
                pass
    cap = ("Ratings are each starter's recent **previous-match** form (today's game isn't played "
           "yet); **Avg** = mean of the last 3. " if not played else
           "**Today** = this match's player rating; **Form** = the prior 3 matches. ")
    cap += ("“Out vs last XI” = started the **previous** match but isn't in today's XI "
            "(injury/suspension/rotation)." if flagged else "")
    st.caption(cap)


@st.cache_data(ttl=300, show_spinner="Grading the 2026 World Cup so far…")
def get_2026_played() -> list:
    """All played 2026 WC matches with the model's pre-match call (for Performance)."""
    res = value_mod.build_bets("2026-06-09", days=40, bankroll=1000, kelly_fraction=0.25,
                               cfg=CFG, predictor=get_predictor(), use_cache=False)
    return [m for m in res["matches"] if m.get("played")]


@st.cache_data(ttl=300, show_spinner="Computing live group state + qualification odds…")
def get_live_state(n_iter: int = 20000) -> dict:
    """Live 2026 standings + clinch flags + Monte-Carlo qualification odds with the
    latest results locked in AND team strength (Elo/DC) recomputed from 2026 form.
    Builds a fresh live simulator each run so a hot team is rated stronger for its
    remaining matches. Cached 5 min; the Tournament page exposes a refresh."""
    from src.simulate import live_state as ls
    return ls.live_state(CFG, n_iter=n_iter)   # builds a live=True simulator internally


# ----------------------------------------------------------------- helpers
def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x*100:.0f}%"


def _am(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    x = int(x)
    return f"+{x}" if x > 0 else str(x)


def _ev_color(ev: float) -> str:
    return GREEN if ev > 0.02 else (RED if ev < -0.02 else GREY)


def _implied_pct(american) -> str:
    """Break-even % implied by the OFFERED (vigged) price — what the model must beat."""
    from src.data.odds import american_to_decimal, decimal_to_prob
    p = decimal_to_prob(american_to_decimal(american))
    return _pct(p)


def _grade_bet(b, m: dict):
    """'win'/'loss'/'push' for a played match, else None. Reuses the backtest grader."""
    if not m.get("played"):
        return None
    import re
    from src.predict.bet_grade import _grade
    hs, as_ = m["home_score"], m["away_score"]
    mk, sel = b.market, b.selection
    if mk == "Match Result":
        code = "H" if sel == m["home"] else ("A" if sel == m["away"] else "D")
    elif mk == "Total Goals":
        mt = re.search(r"[-+]?\d*\.?\d+", sel)
        if not mt:
            return None
        code = ("over" if sel.startswith("Over") else "under", float(mt.group()))
    elif mk == "BTTS":
        both = hs > 0 and as_ > 0
        won = both if "Yes" in sel else (not both)
        return "win" if won else "loss"
    else:
        return None                      # spreads off by default — skip grading
    try:
        return _grade(code, hs, as_)
    except Exception:  # noqa: BLE001
        return None


_GRADE_MARK = {"win": "✅ Win", "loss": "❌ Loss", "push": "➖ Push"}


def market_table(title: str, bets: list, key_note: str | None = None, m: dict | None = None,
                 angle: dict | None = None):
    """Render one market's selections: model%, fair%, the ESPN price + EV, and — when available —
    **your book** (DST) price + the model's EV at *that* price. Played matches get a Result col.
    ``angle`` (optional) is the AI's grounded read for THIS market, shown as a one-line note."""
    from src.data.odds import decimal_to_american
    from src.predict.betting import expected_value
    st.markdown(f"**{title}**")
    if angle:
        tone = {"support": GREEN, "undercut": RED, "neutral": GREY}.get(angle.get("read"), GREY)
        st.markdown(f'<div class="angle-note">AI · <b style="color:{tone}">'
                    f'{angle.get("read", "neutral")}</b>: {angle.get("why", "")}</div>',
                    unsafe_allow_html=True)
    played = bool(m and m.get("played"))
    # Suggestion-only markets (BTTS) are shown for information — model %, fair, EV vs the
    # available prices — but never staked, so they carry no Stake column.
    suggestion_only = bool(bets) and getattr(bets[0], "market", None) in value_mod.SUGGESTION_ONLY_MARKETS
    book = get_book_odds(m["home"], m["away"], str(m.get("date"))[:10]) if m else None
    # Kalshi book (Match Result / Total Goals / Spread / BTTS), only for not-yet-started games.
    kal = (get_kalshi_book(m["home"], m["away"])
           if (m and _match_not_started(m)) else None)
    rows = []
    for b in bets:
        units = _bet_units(b)
        row = {
            "Selection": b.selection,
            "Model": _pct(b.model_p),
            "Fair (no-vig)": _pct(b.fair_p),
            "Price": _am(b.american),
            "EV": f"{b.ev*100:+.0f}%",
        }
        if book:
            bdec = _book_dec(b, m, book)
            row["Your book"] = _am(decimal_to_american(bdec)) if bdec else "—"
            row["Book EV"] = f"{expected_value(b.model_p, bdec)*100:+.0f}%" if bdec else "—"
        if kal:
            kdec = _kalshi_dec(b, m, kal)
            row["Kalshi"] = (f"{1/kdec*100:.0f}¢" if kdec else "—")     # ask in cents ≈ implied %
            row["Kalshi EV"] = f"{expected_value(b.model_p, kdec)*100:+.0f}%" if kdec else "—"
        if not suggestion_only:
            row["Stake"] = f"{units:.1f}u" if units > 0.05 else "—"
        if played:
            row["Result"] = _GRADE_MARK.get(_grade_bet(b, m), "—")
        rows.append(row)
    df = pd.DataFrame(rows)

    ev_cols = {"EV", "Book EV", "Kalshi EV"}

    def _style(row):
        out = []
        for c in row.index:
            if c in ev_cols:
                try:
                    ev = float(str(row[c]).rstrip("%")) / 100
                except ValueError:
                    ev = 0.0
                out.append(f"color:{_ev_color(ev)};font-weight:600")
            else:
                out.append("")
        return out

    st.dataframe(df.style.apply(_style, axis=1), hide_index=True,
                 use_container_width=True)
    if kal:
        st.caption("**Kalshi** = the exchange ask (cents ≈ the implied % you pay) — a price you can "
                   "actually trade; **Kalshi EV** is the model's edge at that ask.")
    if key_note:
        st.caption(key_note)


def heatmap(mat: np.ndarray, home: str, away: str, max_goals: int = 5):
    sub = mat[:max_goals + 1, :max_goals + 1]
    data = [{"home": i, "away": j, "p": float(sub[i, j])}
            for i in range(sub.shape[0]) for j in range(sub.shape[1])]
    d = pd.DataFrame(data)
    chart = alt.Chart(d).mark_rect().encode(
        x=alt.X("away:O", title=f"{away} goals"),
        y=alt.Y("home:O", title=f"{home} goals", sort="descending"),
        color=alt.Color("p:Q", scale=alt.Scale(scheme="greens"), legend=None),
        tooltip=[alt.Tooltip("p:Q", format=".1%")],
    ).properties(height=240)
    text = chart.mark_text(baseline="middle", fontSize=10).encode(
        text=alt.Text("p:Q", format=".0%"),
        color=alt.condition(alt.datum.p > 0.06, alt.value("white"), alt.value("#555")))
    st.altair_chart(chart + text, use_container_width=True)


def context_strip(m: dict):
    a = m["analysis"]
    hc, ac, h2h = a["home_context"], a["away_context"], a["h2h"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption(f"**{m['home']}**")
        st.write(f"Elo {int(hc['elo'])} · form `{hc['form']}` · "
                 f"GF/GA(5) {hc['gf5']}/{hc['ga5']}"
                 + (f" · xG {hc['xg_rating']}" if hc['xg_rating'] else ""))
        if m["key_out_home"]:
            st.caption(f"🩹 out: {', '.join(m['key_out_home'])}")
    with c2:
        st.caption(f"**{m['away']}**")
        st.write(f"Elo {int(ac['elo'])} · form `{ac['form']}` · "
                 f"GF/GA(5) {ac['gf5']}/{ac['ga5']}"
                 + (f" · xG {ac['xg_rating']}" if ac['xg_rating'] else ""))
        if m["key_out_away"]:
            st.caption(f"🩹 out: {', '.join(m['key_out_away'])}")
    with c3:
        st.caption("**Head-to-head**")
        if h2h["n"]:
            st.write(f"{h2h['home_wins']}-{h2h['draws']}-{h2h['away_wins']} "
                     f"(last {h2h['n']})")
            for line in h2h["recent"][-3:]:
                st.caption(line)
        else:
            st.caption("no prior meetings")


def motivation_block(m: dict, live: dict | None):
    """For a 2026 group match, show each side's live standing + P(reach knockouts)
    — the format-correct 'what's at stake' signal (handles the best-third rule)."""
    if not live:
        return
    from src.simulate import live_state as ls
    hs, aw = ls.team_summary(live, m["home"]), ls.team_summary(live, m["away"])
    if not hs or not aw:
        return  # not a 2026 group-stage match (knockout / non-WC fixture)
    st.markdown("**Group state & stakes** · live, updates as results come in")
    c1, c2 = st.columns(2)
    for col, team, sm in ((c1, m["home"], hs), (c2, m["away"], aw)):
        with col:
            st.markdown(f"{team_with_flag(team, bold=True)} · Group {sm['group']}",
                        unsafe_allow_html=True)
            st.write(f"{sm['pos_str']}, **{sm['pts']} pts** · GD {sm['gd']:+d} "
                     f"· {sm['played']} GP")
            adv = sm["p_advance"]
            tone = ("green" if adv >= 0.85 else "gold" if adv >= 0.4 else "red")
            st.markdown(theme.pill(f"{adv*100:.0f}% to reach knockouts", tone),
                        unsafe_allow_html=True)
            st.caption(sm["status"])
    st.divider()


def _qualifying_bets(m: dict, min_ev: float, min_prob_edge: float):
    """Bets that clear the probability-edge gate AND aren't in a disabled segment."""
    from src.models.segment_gate import disabled_set
    from src.predict.value import _type_key
    disabled = disabled_set(CFG)
    return [b for b in m["bets"]
            if qualifies(b.model_p, b.fair_p, b.decimal, min_ev, min_prob_edge, 6.0)
            and _type_key(b.market, b.selection, m["home"], m["away"]) not in disabled]


def best_bet_block(m: dict, min_ev: float = 0.03, min_prob_edge: float = 0.02):
    """One clear 'best bet' per card: the highest-EV selection that clears the
    probability-edge gate in a non-disabled segment, or an explicit 'pass'."""
    cands = _qualifying_bets(m, min_ev, min_prob_edge)
    if not cands:
        st.markdown('<div class="bbet"><span class="h">Best bet</span> &nbsp; '
                    '<span style="color:#969aa6">no edge here — pass</span></div>',
                    unsafe_allow_html=True)
        return
    # Pick by EDGE (model − market), not raw EV: EV on a long price explodes from a tiny prob gap and
    # keeps surfacing longshots (39% hit-rate). Edge = the real disagreement → near-even, ~57% picks.
    b = max(cands, key=lambda x: ((x.edge if x.edge is not None else (x.model_p - (x.fair_p or x.model_p))),
                                  x.ev))
    units = _bet_units(b)
    cons = (m.get("cons_edge") or {}).get(b.selection)
    cons_txt = ""
    if cons is not None:
        cons_txt = (f' &nbsp;·&nbsp; {"beats consensus" if cons > 0 else "≤ consensus"} '
                    f'({cons*100:+.0f}%)')
    st.markdown(
        f'<div class="bbet"><span class="h">Best bet</span> &nbsp; '
        f'<b style="font-size:15px">{b.market}: {b.selection}</b> &nbsp; '
        f'{theme.pill(_am(b.american), "grey")} &nbsp;·&nbsp; '
        f'model {_pct(b.model_p)} vs market {_pct(b.fair_p)} &nbsp;·&nbsp; '
        f'{theme.pill(f"EV {b.ev*100:+.0f}%", "green" if b.ev > 0 else "red")} &nbsp;·&nbsp; '
        f'stake <b style="color:{GOLD}">{units:.1f}u</b>{cons_txt}</div>',
        unsafe_allow_html=True)
    st.caption("Highest-edge selection here (spreads & disabled segments excluded). An "
               "edge = the model disagrees with the price, **not** a guaranteed win — "
               "the Tracker page shows how these picks are actually doing.")


def _display_probs(m: dict) -> dict:
    """The model's W/D/L probabilities for display — the SAME pure (market-independent)
    `model_p` used in the Match Result table, so the header bar, the table, the expected
    goals and the scoreline heatmap all tell one consistent story. Falls back to the raw
    forecast if there are no Match Result bets (no odds)."""
    a = m["analysis"]
    mr = {b.selection: b.model_p for b in m["bets"] if b.market == "Match Result"}
    p = {"H": mr.get(m["home"]), "D": mr.get("Draw"), "A": mr.get(m["away"])}
    if any(v is None for v in p.values()):
        return dict(a["probs"])
    s = sum(p.values()) or 1.0
    return {k: v / s for k, v in p.items()}


def _brief_facts(m: dict) -> dict:
    """Assemble the REAL facts for the AI brief (model, form, H2H, lineups+formations+values,
    result/stats when played, best bet) — nothing invented; the brief may use only these."""
    from src.data import squad_values as sv
    a = m["analysis"]
    probs = _display_probs(m)
    pick = OUTCOMES[int(np.argmax([probs["H"], probs["D"], probs["A"]]))]
    pick_name = {"H": m["home"], "D": "Draw", "A": m["away"]}[pick]
    eg = a["expected_goals"]
    top = a["top_scorelines"][0][0] if a.get("top_scorelines") else "?"
    f = {"Match": f"{m['home']} vs {m['away']} "
                  f"({'neutral site' if m['neutral'] else m['home'] + ' at home'})",
         "Model": (f"{m['home']} {probs['H']*100:.0f}% / Draw {probs['D']*100:.0f}% / "
                   f"{m['away']} {probs['A']*100:.0f}%, leans {pick_name}; expected goals "
                   f"{m['home']} {eg[0]:.1f}, {m['away']} {eg[1]:.1f} ({eg[0]+eg[1]:.1f} total); "
                   f"BTTS {a['btts']*100:.0f}%; most likely score {top}")}
    hc, ac, h2h = a.get("home_context") or {}, a.get("away_context") or {}, a.get("h2h") or {}
    if hc.get("form") or ac.get("form"):
        f["Recent form"] = f"{m['home']} {hc.get('form', '?')}, {m['away']} {ac.get('form', '?')}"
    if h2h.get("n"):
        f["Head-to-head"] = (f"last {h2h['n']} meetings: {m['home']} {h2h.get('home_wins', 0)}-"
                             f"{h2h.get('draws', 0)}-{h2h.get('away_wins', 0)} {m['away']}")
    el = get_fast_lineups(m["home"], m["away"], str(m.get("date"))[:10], m.get("game_id"))
    if el:
        parts = []
        for side, team in (("home", m["home"]), ("away", m["away"])):
            s = el.get(side) or {}
            names = [p.get("name") for p in s.get("xi", [])]
            val, share = sv.xi_value(team, names)
            ko = sv.key_absentees(team, names, top_n=3)
            seg = f"{team} {s.get('formation') or '?'}"
            if share is not None:
                seg += f" (XI {_value_str(val)}, {share*100:.0f}% of squad value)"
            if ko:
                seg += f", missing {', '.join(x['name'] + ' ' + _value_str(x['market_value']) for x in ko)}"
            parts.append(seg)
        _lbl = (f"Lineups ({el.get('source', '')} "
                f"{'confirmed' if el.get('confirmed') else 'projected'})").strip()
        f[_lbl] = " | ".join(parts)
    if m.get("played"):
        f["Result"] = (f"FINAL {m['home_score']}-{m['away_score']} "
                       f"({'model called it' if m['result'] == pick else 'upset vs the model'})")
        sx = get_match_stats(m["home"], m["away"], str(m["date"])[:10])
        if sx.get("ball_possession"):
            def _ha(k):
                v = sx.get(k) or {}
                return f"{v.get('home', '?')}-{v.get('away', '?')}"
            f["Match stats"] = (f"possession {_ha('ball_possession')}, xG "
                                f"{_ha('expected_goals')}, shots {_ha('total_shots')}")
    sig = a.get("signals") or {}
    if sig:
        f["Variance"] = (f"upset risk {sig.get('upset_risk', 0)*100:.0f}%, shootout "
                         f"{sig.get('shootout_potential', 0)*100:.0f}%, expected total "
                         f"{sig.get('expected_total', eg[0]+eg[1]):.1f}"
                         + (f", draw likely {sig.get('draw_risk', 0)*100:.0f}% "
                            "(these draw ~32% historically)" if sig.get("high_draw") else ""))
    cands = _qualifying_bets(m, min_ev=0.03, min_prob_edge=0.02)
    if cands:
        cands = sorted(cands, key=lambda x: x.ev, reverse=True)
        b = cands[0]
        book_txt = ""
        try:
            from src.data.odds import decimal_to_american
            bdec = _book_dec(b, m, get_book_odds(m["home"], m["away"], str(m["date"])[:10]))
            if bdec:
                book_txt = f"; your book {_am(decimal_to_american(bdec))}"
        except Exception:  # noqa: BLE001
            pass
        f["Best bet (model)"] = (f"{b.market}: {b.selection} at {_am(b.american)} — model "
                                 f"{b.model_p*100:.0f}% vs market {(b.fair_p or 0)*100:.0f}%, "
                                 f"EV {b.ev*100:+.0f}%{book_txt}")
        # All flagged +EV selections — the list the AI judges support/undercut against.
        f["Flagged bets"] = " | ".join(
            f"{x.market}: {x.selection} at {_am(x.american)} (model {x.model_p*100:.0f}% vs "
            f"market {(x.fair_p or 0)*100:.0f}%, EV {x.ev*100:+.0f}%)" for x in cands[:6])
    return f


class _BriefError(Exception):
    """Carries a failed/empty brief result out of the cached call WITHOUT letting
    ``st.cache_data`` store it — raising means the failure isn't cached and we retry next time."""
    def __init__(self, res):
        self.res = res


@st.cache_data(show_spinner=False)
def _brief_cached(cache_key: str, _facts: dict):
    """Run the Gemini brief ONCE per match, cached across ALL sessions/reruns (survives page
    reloads and other viewers — recomputed only on app reboot). Keyed on ``cache_key`` (game_id)
    only; ``_facts`` is underscore-prefixed so Streamlit doesn't hash it, so live drift in
    EV/odds/lineups never busts the cache. Raises ``_BriefError`` on a no-text result so transient
    errors / missing-key are NOT cached and get retried."""
    from src.ai import match_brief as mb
    res = mb.brief(_facts)
    if not (res and res.get("text")):
        raise _BriefError(res)
    return res


def get_match_brief(facts: dict, cache_key: str):
    """Grounded brief for a match, cached cross-session on SUCCESS only (see ``_brief_cached``).
    Returns {"summary"/"text","angles","sources"} | {"error"} | None."""
    try:
        return _brief_cached(str(cache_key), facts)
    except _BriefError as e:                     # transient error/no-key — not cached, retried
        return e.res


def _kalshi_alert_block(m: dict):
    """🔔 Fire a Kalshi-value callout when the model's edge at the Kalshi ask clears the threshold
    on a Match Result outcome (the price you could trade on Kalshi right now)."""
    if not _match_not_started(m):
        return
    thr = float(st.session_state.get("kalshi_alert_ev", 0.08))
    kal = get_kalshi_book(m["home"], m["away"])
    if not kal:
        return
    from src.data import kalshi
    best = None                                  # best Kalshi value across ALL markets
    for b in m.get("bets", []):
        px = kalshi.price_for(kal, b.market, b.selection, m["home"], m["away"])
        ask = (px or {}).get("ask")
        if ask and ask > 0:
            ev = b.model_p / ask - 1.0
            if ev >= thr and (best is None or ev > best[0]):
                best = (ev, f"{b.market}: {b.selection}", ask, b.model_p)
    if best:
        ev, name, ask, p = best
        theme.callout(
            f"🔔 <b>Kalshi value</b> — buy <b>{name}</b> at <b>{ask*100:.0f}¢</b> "
            f"(model {p*100:.0f}% vs {ask*100:.0f}% implied) → <b>EV {ev*100:+.0f}%</b>. "
            f"Tradeable on Kalshi now; an edge ≠ a guaranteed win.", "good")


def render_match(m: dict, live: dict | None = None, min_ev: float = 0.03,
                 min_prob_edge: float = 0.02):
    a = m["analysis"]
    t = _ct_str(m["date"])
    probs = _display_probs(m)
    pick = OUTCOMES[int(np.argmax([probs["H"], probs["D"], probs["A"]]))]
    pick_name = {"H": m["home"], "D": "Draw", "A": m["away"]}[pick]
    n_value = len(_qualifying_bets(m, min_ev, min_prob_edge))
    played = bool(m.get("played"))
    if played:
        hit = (m["result"] == pick)
        title = (f"{m['home']} {m['home_score']}–{m['away_score']} {m['away']}    ·    {t}    ·    "
                 f"model {'✓' if hit else '✗'}")
    else:
        title = (f"{m['home']}   vs   {m['away']}    ·    {t}    ·    leans {pick_name} "
                 f"{_pct(probs[pick])}" + (f"    ·    {n_value} value" if n_value else ""))
    with st.expander(title, expanded=False):
        # ---- header: crests + score/vs + kickoff ----
        home_h = f'{flag_html(m["home"], 30)} {m["home"]}'
        away_h = f'{m["away"]} {flag_html(m["away"], 30)}'
        center = f'{m["home_score"]}–{m["away_score"]}' if played else "vs"
        sub = (f'FT · {t}' if played else t)
        st.markdown(theme.match_header(home_h, away_h, center, sub), unsafe_allow_html=True)
        st.markdown(theme.prob_bar(probs["H"], probs["D"], probs["A"], m["home"], m["away"]),
                    unsafe_allow_html=True)
        # ---- key numbers strip ----
        eg = a["expected_goals"]
        top_s = a["top_scorelines"][0][0] if a.get("top_scorelines") else "—"
        kn = [{"label": "Model lean", "value": f"{pick_name.split()[0]} {_pct(probs[pick])}",
               "color": theme.GREEN},
              {"label": "Exp. goals", "value": f"{eg[0] + eg[1]:.1f}"},
              {"label": "Likely score", "value": top_s}]
        if played:
            kn.append({"label": "Result", "value": "model ✓" if hit else "upset ✗",
                       "color": theme.GREEN if hit else theme.RED})
        else:
            kn.append({"label": "Value bets", "value": str(n_value),
                       "color": theme.GOLD if n_value else theme.MUTED})
        st.markdown(theme.key_numbers(kn), unsafe_allow_html=True)
        # variance signals — meaningful, no decorative emoji
        sig = a.get("signals") or {}
        chips = []
        ur = sig.get("upset_risk")
        if ur is not None:
            chips.append(theme.pill(f"Upset risk {ur * 100:.0f}%",
                                    "gold" if sig.get("high_upset") else "grey"))
        sp, et = sig.get("shootout_potential"), sig.get("expected_total")
        if sp is not None:
            chips.append(theme.pill(f"High-scoring {sp * 100:.0f}%",
                                    "gold" if sig.get("high_scoring") else "grey"))
        dr = sig.get("draw_risk")
        if dr is not None:
            chips.append(theme.pill(f"Draw likely {dr * 100:.0f}%",
                                    "gold" if sig.get("high_draw") else "grey"))
        if chips:
            st.markdown('<div style="margin:2px 0 8px">' + " ".join(chips) + '</div>',
                        unsafe_allow_html=True)
            if sig.get("high_draw"):
                st.caption("Draws are never the favourite, but this isn't noise: matches the model flags "
                           "≥28% draw historically drew ~32% (vs a ~23% base). Descriptive, not a bet.")
        best_bet_block(m, min_ev, min_prob_edge)
        _kalshi_alert_block(m)

        by_market: dict[str, list] = {}
        for b in m["bets"]:
            by_market.setdefault(b.market, []).append(b)
        brief_angles: list[dict] = []           # AI angles, surfaced per-market in tab_markets
        tab_insights, tab_markets, tab_scores = st.tabs(["Insights", "Markets", "Scorelines"])

        with tab_insights:
            from src.ai import match_brief as _mb
            if _mb.is_available():
                bkey = f"brief_{m.get('game_id') or (m['home'] + m['away'])}"
                if st.button("✨ AI betting angles", key=bkey + "_btn",
                             help="Grounded team-news + per-bet read (Gemini + Google Search)."):
                    st.session_state[bkey] = True
                if st.session_state.get(bkey):
                    with st.spinner("Searching the news + reading the bets…"):
                        res = get_match_brief(_brief_facts(m),
                                              m.get("game_id") or (m["home"] + m["away"]))
                    if res and res.get("text"):
                        theme.callout(res.get("summary") or res["text"], "info")
                        brief_angles = res.get("angles") or []
                        for ang in brief_angles:
                            st.markdown(theme.angle_chip(ang.get("market", ""), ang.get("lean", ""),
                                                         ang.get("read", "neutral"), ang.get("why", "")),
                                        unsafe_allow_html=True)
                        tail = []
                        if res.get("confidence"):
                            tail.append(f"confidence: {res['confidence']}")
                        if res.get("watch"):
                            tail.append(f"watch: {res['watch']}")
                        if tail:
                            st.caption(" · ".join(tail))
                        srcs = res.get("sources") or []
                        if srcs:
                            st.caption("Sources: " + " · ".join(
                                f"[{s['title'][:38]}]({s['uri']})" for s in srcs))
                        st.caption("AI adds *sourced context*; it never invents an edge — the model's "
                                   "EV is unchanged.")
                    elif res and res.get("error"):
                        st.caption(f"Gemini: {res['error']}")
                    else:
                        st.caption("Brief unavailable (no key detected).")
            if played:
                axg = get_fixture_xg(m["home"], m["away"], str(m["date"])[:10])
                if axg is not None:
                    st.caption(f"**xG reality-check** — model projected {eg[0]:.1f}–{eg[1]:.1f} · "
                               f"actual xG {axg[0]:.2f}–{axg[1]:.2f} · final "
                               f"{m['home_score']}–{m['away_score']}. (xG is a reality-check, not "
                               "a model input.)")
                match_stats_block(m)
            else:
                move = m.get("home_line_move")
                vtxt = ("neutral site" if m["neutral"] else f"{m['home']} at home")
                if move is not None and not pd.isna(move) and abs(move) >= 0.015:
                    who = m["home"] if move > 0 else m["away"]
                    vtxt += f" · line moving toward {who} ({abs(move) * 100:.0f}%)"
                st.caption(f"{vtxt} · model expected goals {m['home']} {eg[0]:.1f}, "
                           f"{m['away']} {eg[1]:.1f}")
            motivation_block(m, live)
            context_strip(m)
            lineup_block(m)

        with tab_markets:
            _ang = {a.get("market"): a for a in brief_angles if a.get("market")}
            for mk in ["Match Result", "Total Goals", "Spread"]:
                if mk in by_market:
                    market_table(mk, by_market[mk], m=m, angle=_ang.get(mk))
            if "BTTS" in by_market:
                book = m.get("btts_book") or "book"
                src = m.get("btts_source")
                tag = ("pre-match · best of US books" if src == "actionnetwork"
                       else "settled/closing" if src == "thestatsapi" else "line")
                market_table("Both Teams To Score", by_market["BTTS"], m=m,
                             key_note=f"Line: {book} · {tag}.", angle=_ang.get("BTTS"))
            elif "Match Result" in by_market or "Total Goals" in by_market:
                st.markdown(f"**Both Teams To Score** — model **{_pct(a['btts'])}** "
                            f"(no line available — info only)")
            st.caption("**Model** = our pure probability, calibrated to historical results, "
                       "**independent of the market** · **Fair** = de-vigged market · "
                       "**Price/EV** = ESPN line · **Your book** = your buckeye (DST) price and "
                       "**Book EV** = the model's EV at *that* price — the number that matters for "
                       "what you actually bet.")

        with tab_scores:
            heatmap(a["scoreline_matrix"], m["home"], m["away"])
            tops = " · ".join(f"{s} ({p*100:.0f}%)" for s, p in a["top_scorelines"][:5])
            st.caption("Most likely scorelines: " + tops)


# --------------------------------------------------------------------- pages
def page_matches(bankroll, kelly, min_ev=0.03, min_prob_edge=0.02, upset_temp=1.0):
    theme.hero("Matches", "Model vs market across every priced market — flags, probabilities, "
               "and the single best bet per game.")
    c1, c2, c3 = st.columns([1, 1, 1])
    day = c1.date_input("From date", value=_today_ct())
    days = c2.slider("Days ahead", 1, 7, 3)
    lookback = c3.slider("Days back (show results)", 0, 21, 5,
                         help="Include recently-played matches so you can see how the model did "
                              "vs the result and the closing Vegas line.")
    if st.button("🔄 Refresh odds & results"):
        get_bets.clear()
        get_live_state.clear()
    start = (day - timedelta(days=lookback)).strftime("%Y-%m-%d")
    res = get_bets(start, days + lookback, bankroll, kelly, upset_temp)
    matches = res["matches"]
    if not matches:
        theme.callout("No fixtures with odds in this window — try the current World Cup dates.",
                      "info")
        theme.footer()
        return
    played = [m for m in matches if m.get("played")]
    upcoming = [m for m in matches if not m.get("played")]
    bets = res["bets"]
    n_val = int((bets["ev"] > 0.02).sum()) if not bets.empty else 0
    theme.kpi_row([
        {"label": "Upcoming", "value": len(upcoming), "accent": theme.GREEN},
        {"label": "Results shown", "value": len(played), "accent": theme.BLUE},
        {"label": "+EV selections", "value": n_val, "accent": theme.GOLD,
         "value_color": theme.GOLD if n_val else theme.TEXT},
    ])
    # live 2026 group state for the "stakes" block on group-stage cards (best-effort)
    live = None
    try:
        live = get_live_state()
    except Exception:  # noqa: BLE001 — cards still render without the stakes block
        live = None
    if upcoming:
        theme.section("Upcoming")
        for m in sorted(upcoming, key=lambda x: pd.to_datetime(x["date"])):
            render_match(m, live, min_ev, min_prob_edge)
    if played:
        hits = sum(1 for m in played
                   if m["result"] == OUTCOMES[int(np.argmax(list(_display_probs(m).values())))])
        theme.section("Recent results",
                      right=f"model called {hits}/{len(played)} ({hits/len(played)*100:.0f}%)")
        for m in sorted(played, key=lambda x: pd.to_datetime(x["date"]), reverse=True):
            render_match(m, live, min_ev, min_prob_edge)
    theme.footer()


def page_value_board(bankroll, kelly, min_ev, max_exposure, min_prob_edge=0.02, upset_temp=1.0):
    theme.hero("Value Board", "Every +EV bet across the slate, ranked — staked by fractional "
               "Kelly and capped to your max exposure.")
    day = st.date_input("From date", value=_today_ct(), key="vb_date")
    days = st.slider("Days ahead", 1, 7, 3, key="vb_days")
    res = get_bets(day.strftime("%Y-%m-%d"), days, bankroll, kelly, upset_temp)
    bb = value_mod.best_bets(res["bets"], min_ev=min_ev, min_prob_edge=min_prob_edge)
    if bb.empty:
        theme.callout("No bets clear the EV threshold for this window.", "info")
        theme.footer()
        return
    # disciplined sizing (longshots down-weighted), then scale to the max-exposure cap
    from src.predict import staking as _stk
    from src.data.odds import american_to_decimal as _a2d
    bb["units"] = bb.apply(lambda r: _stk.recommended_units(
        r.get("model_p"), r.get("fair_p"), _a2d(r.get("american")), frac=kelly), axis=1)
    _tot, _cap = bb["units"].sum(), max_exposure * 100.0
    if _tot > _cap and _tot > 0:
        bb["units"] = (bb["units"] * _cap / _tot).round(2)
    show = bb.copy()
    show["model"] = (show["model_p"] * 100).round(0).astype(int).astype(str) + "%"
    show["vegas"] = (show["fair_p"] * 100).round(0).astype(int).astype(str) + "%"
    show["EV"] = (show["ev"] * 100).round(0).astype(int).astype(str) + "%"
    show["price"] = show["american"].map(_am)
    show["stake"] = show["units"].round(2).astype(str) + "u"
    theme.kpi_row([
        {"label": "+EV bets", "value": len(bb), "accent": theme.GREEN},
        {"label": "Total stake", "value": f"{bb['units'].sum():.1f}u", "accent": theme.GOLD},
        {"label": "Avg edge", "value": f"{bb['edge'].mean()*100:.1f}%", "accent": theme.BLUE},
    ])
    st.dataframe(
        show[["match", "market", "selection", "price", "model", "vegas", "EV", "stake"]],
        hide_index=True, use_container_width=True)
    st.download_button("⬇ Download CSV", bb.to_csv(index=False), "value_bets.csv", "text/csv")
    theme.callout("⚠ <b>Reality check:</b> many of these are unders / draws / underdogs — "
                  "markets the model's calibration was <b>not</b> backtested against (only 1X2 "
                  "RPS was). Treat large EVs on longshots with extra skepticism.", "warn")
    theme.footer()


def _group_color(row):
    # Color the TEXT by qualification status (readable on dark or light themes;
    # a light background fill would hide the dark theme's white text).
    pos = row["Pos"]
    if pos <= 2:
        css = "color:#43a047;font-weight:700"      # green = top-2, auto-qualify
    elif pos == 3:
        css = "color:#fb8c00;font-weight:600"      # amber = 3rd, best-third bubble
    else:
        css = "color:#9e9e9e"                        # grey = bottom, likely out
    return [css] * len(row)


def page_tournament():
    theme.hero("Tournament", "Live group standings + Monte-Carlo qualification and title odds — "
               "results locked in, team strength updating from 2026 form.")
    cc1, cc2 = st.columns([1, 4])
    if cc1.button("🔄 Refresh results"):
        get_live_state.clear()
    try:
        live = get_live_state()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not compute live state: {e}")
        return
    qual = live["qual"]
    cc2.caption(f"{live['n_played']} group games played and locked in · top-2 of each group "
                "**plus the 8 best third-placed** teams reach the knockouts.")

    # --- title odds: KPI for the favourite + chart ---
    top = qual.sort_values("champion", ascending=False).head(20)
    fav = top.iloc[0]
    theme.kpi_row([
        {"label": "Title favourite", "value": fav["team"],
         "sub": f"{fav['champion']*100:.1f}% to win it all", "accent": theme.GOLD},
        {"label": "Games locked in", "value": live["n_played"], "accent": theme.GREEN},
        {"label": "Teams alive", "value": int((qual["advance"] > 0.001).sum()),
         "sub": "still able to reach the knockouts", "accent": theme.BLUE},
    ])
    theme.section("Championship probability")
    chart = alt.Chart(top).mark_bar(color=GREEN, cornerRadiusEnd=4).encode(
        x=alt.X("champion:Q", axis=alt.Axis(format="%"), title="Championship probability"),
        y=alt.Y("team:N", sort="-x", title=None),
        tooltip=["team", "group", alt.Tooltip("champion:Q", format=".1%"),
                 alt.Tooltip("advance:Q", format=".1%")]).properties(height=460)
    st.altair_chart(chart, use_container_width=True)

    # --- model vs Kalshi tournament-winner futures (tradeable) ---
    fut = get_kalshi_futures()
    if fut:
        from src.predict.betting import expected_value
        rows = []
        for _, r in qual.iterrows():
            px = fut.get(r["team"])
            ask = (px or {}).get("ask")
            if not ask or ask <= 0:
                continue
            ev = expected_value(float(r["champion"]), 1.0 / ask)
            rows.append({"Team": r["team"], "Model": float(r["champion"]),
                         "Kalshi ask": ask, "Edge": float(r["champion"]) - ask, "EV": ev})
        if rows:
            kdf = pd.DataFrame(rows).sort_values("EV", ascending=False)
            theme.section("Model vs Kalshi — tournament winner", sub="tradeable on Kalshi")
            thr = float(st.session_state.get("kalshi_alert_ev", 0.08))
            n_val = int((kdf["EV"] >= thr).sum())
            disp = pd.DataFrame({
                "Team": kdf["Team"],
                "Model": (kdf["Model"] * 100).map(lambda x: f"{x:.1f}%"),
                "Kalshi ask": kdf["Kalshi ask"].map(lambda x: f"{x*100:.0f}¢"),
                "Edge": (kdf["Edge"] * 100).map(lambda x: f"{x:+.1f}%"),
                "EV": (kdf["EV"] * 100).map(lambda x: f"{x:+.0f}%")})
            st.dataframe(disp.head(12), hide_index=True, use_container_width=True)
            st.caption(f"Our simulator's champion probability vs the Kalshi winner **ask** (≈ implied %). "
                       f"EV = model ÷ ask − 1. {n_val} team(s) clear your {thr*100:.0f}% alert bar. "
                       "Tradeable on Kalshi; an edge ≠ a guaranteed win.")

    # --- live group leaderboards with flags + P(advance) ---
    theme.section("Group standings & qualification odds")
    st.caption("🟢 top-2 (auto-qualify) · 🟡 3rd (best-third bubble) · ⚪ bottom. "
               "**Adv%** = model probability of reaching the knockouts.")
    adv = qual[["team", "advance", "win_group"]]
    groups = live["standings"]
    letters = list(groups)
    for r in range(0, len(letters), 3):
        cols = st.columns(3)
        for col, g in zip(cols, letters[r:r + 3]):
            with col:
                show = groups[g].merge(adv, on="team", how="left")
                show["Adv%"] = (show["advance"] * 100).round(0).astype("Int64")
                show["S"] = show["Pos"].map(lambda p: "🟢" if p <= 2 else ("🟡" if p == 3 else "⚪"))
                show = show[["S", "team", "P", "Pts", "GD", "Adv%"]]
                st.markdown(f"**Group {g}**")
                st.dataframe(show, hide_index=True, use_container_width=True, column_config={
                    "S": st.column_config.TextColumn("", width="small"),
                    "team": st.column_config.TextColumn("Team"),
                    "Adv%": st.column_config.NumberColumn("Adv%", format="%d%%")})

    with st.expander("Full qualification & advancement table"):
        show = qual.sort_values("champion", ascending=False).copy()
        for c in ["win_group", "advance", "reach_r16", "reach_qf", "reach_sf",
                  "reach_final", "champion"]:
            if c in show:
                show[c] = (show[c] * 100).round(1)
        st.dataframe(show, use_container_width=True, hide_index=True)
        theme.note("Stakes are quarter-Kelly capped at 2u — but <b>longshots (≈4.5+ odds) are "
                   "down-weighted</b> because the model overrates the tail (5.0+ picks win ~7.8% vs the "
                   "~11.8% it expects, so they size toward 0). Favourites & pickems are unchanged. "
                   "Suggestions, not advice.")
    theme.footer()


def page_performance():
    theme.hero("Performance", "How the model actually scores — its 2026 record so far, "
               "walk-forward accuracy on 7 past World Cups (1998–2022), and an honest betting "
               "backtest of the 2022 tournament.")

    # --- 2026 World Cup so far: the model's live record on already-played matches ---
    try:
        played26 = get_2026_played()
    except Exception:  # noqa: BLE001
        played26 = []
    if played26:
        picks = [OUTCOMES[int(np.argmax(list(_display_probs(m).values())))] for m in played26]
        hits = sum(1 for m, p in zip(played26, picks) if m["result"] == p)
        n = len(played26)
        # average goal error (model expected total vs actual)
        gerr = np.mean([abs(sum(m["analysis"]["expected_goals"])
                            - (m["home_score"] + m["away_score"])) for m in played26])
        theme.section("2026 World Cup — live so far")
        theme.kpi_row([
            {"label": "Matches played", "value": n, "accent": theme.GREEN},
            {"label": "Result called", "value": f"{hits}/{n} ({hits/n*100:.0f}%)",
             "accent": theme.GOLD},
            {"label": "Goal-total error", "value": f"{gerr:.2f}", "sub": "avg |model − actual|",
             "accent": theme.BLUE},
        ])
        st.caption("The model's pre-match pick vs what actually happened in 2026, updated live as "
                   "games finish. Small sample — one tournament is noise, not proof.")

    # --- XI value-share forward check (the honest test of the squad-value signal) ---
    xiv = load_csv("xi_value_2026.csv")
    if not xiv.empty:
        from src.backtest.xi_value_2026 import summarize
        s = summarize(xiv)
        theme.section("Does the stronger (by market value) starting XI win?")
        if "value_fav_winrate" in s:
            theme.kpi_row([
                {"label": "Higher-value XI won", "value": f"{s['value_fav_winrate']*100:.0f}%",
                 "sub": f"of {s['n_decisive']} decisive games", "accent": theme.GREEN},
                {"label": "Model favorite won", "value": f"{s['model_fav_winrate']*100:.0f}%",
                 "sub": "for reference", "accent": theme.BLUE},
                {"label": "Games tracked", "value": s["n"], "accent": theme.GOLD},
            ])
        buckets = s.get("fav_share_buckets") or []
        if buckets:
            bdf = pd.DataFrame([{"Favorite's XI value-share": b["band"],
                                 "Upset rate": f"{b['upset_rate']*100:.0f}%", "N": b["n"]}
                                for b in buckets])
            st.caption("Do **weakened favorites** get upset more? (favorite = model's pick; "
                       "upset = it didn't win)")
            st.dataframe(bdf, hide_index=True, use_container_width=True)
        theme.callout("Market-value XI strength can't be backtested (no historical lineups), so "
                      "this is a <b>forward, directional</b> check that accrues as the WC plays — "
                      "<b>not a validated edge</b>, and the model's probabilities don't use it. "
                      "Team-level squad value already adds 0.0 RPS to the model.", "warn")

    # Deployed-model accuracy, walk-forward over 7 World Cups (1998–2022). This is the
    # EXACT live pipeline (market-independent DC+Elo blend + WC goals correction).
    acc = load_csv("wc_accuracy_backtest.csv")
    if not acc.empty:
        pooled = acc[acc["world_cup"].astype(str) == "POOLED"].set_index("model")
        _LABEL = {"deployed": "Deployed model (DC+Elo + WC correction)",
                  "no_wc_corr": "…without the WC goals correction",
                  "dixon_coles": "Dixon-Coles only", "elo": "Elo only",
                  "home_prior": "Home-prior baseline", "climatology": "Base-rate baseline"}
        if "deployed" in pooled.index:
            dep = pooled.loc["deployed"]
            elo_rps = pooled.loc["elo"]["rps"] if "elo" in pooled.index else None
            theme.kpi_row([
                {"label": "Deployed RPS", "value": f"{dep['rps']:.4f}",
                 "sub": "7 World Cups 1998–2022 · ≈0.20 is bookmaker-grade", "accent": theme.GREEN},
                {"label": "Pick accuracy", "value": f"{dep['accuracy']*100:.0f}%",
                 "sub": f"correct result · {int(dep['n'])} matches", "accent": theme.GOLD},
                {"label": "vs Elo-only", "value": (f"{elo_rps:.4f}" if elo_rps else "—"),
                 "sub": "the model beats it", "accent": theme.BLUE},
            ])
        theme.section("Accuracy", sub="walk-forward over 7 World Cups, 1998–2022 · lower RPS is better")
        st.caption("The exact live pipeline, market-independent. Leak-free: for each World "
                   "Cup the model is trained only on matches before it.")
        comp = pooled.reset_index()
        comp["model"] = comp["model"].map(_LABEL).fillna(comp["model"])
        comp["accuracy"] = (comp["accuracy"] * 100).round(1)
        comp = comp.rename(columns={"model": "Model", "rps": "RPS", "log_loss": "Log loss",
                                    "brier": "Brier", "accuracy": "Accuracy %", "n": "N"})
        st.dataframe(comp[["Model", "N", "RPS", "Log loss", "Brier", "Accuracy %"]].round(4),
                     use_container_width=True, hide_index=True)
        per = acc[(acc["model"] == "deployed") & (acc["world_cup"].astype(str) != "POOLED")].copy()
        if not per.empty:
            per["accuracy"] = (per["accuracy"] * 100).round(0)
            per = per.rename(columns={"world_cup": "World Cup", "rps": "RPS",
                                      "accuracy": "Accuracy %", "n": "N"})
            with st.expander("Per–World Cup breakdown", expanded=False):
                st.dataframe(per[["World Cup", "N", "RPS", "Accuracy %"]].round(4),
                             use_container_width=True, hide_index=True)
    else:
        bt = load_csv("backtest_pooled.csv")
        if not bt.empty:
            theme.section("Accuracy", sub="pooled over 2010–2022 World Cups · lower RPS is better")
            st.dataframe(bt.round(4), use_container_width=True, hide_index=True)
    cal = load_csv("calibration.csv")
    if not cal.empty:
        theme.section("Calibration", sub="predicted vs observed")
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeDash=[4, 4], color=GREY).encode(x="x", y="y")
        pts = alt.Chart(cal).mark_circle(size=90, color=GREEN).encode(
            x=alt.X("mean_predicted:Q", scale=alt.Scale(domain=[0, 1]), title="Predicted"),
            y=alt.Y("observed_freq:Q", scale=alt.Scale(domain=[0, 1]), title="Observed"),
            size=alt.Size("n:Q", title="N"),
            tooltip=["bin", "mean_predicted", "observed_freq", "n"])
        st.altair_chart(diag + pts, use_container_width=True)
    abl = load_csv("ablation.csv")
    if not abl.empty:
        theme.section("Ablation", sub="does each block lower RPS?")
        st.dataframe(abl.round(4), use_container_width=True, hide_index=True)

    wc = load_csv("wc2022_backtest.csv")
    if not wc.empty:
        theme.section("2022 World Cup", sub="how our betting model would have done")
        st.caption("Exactly how the model bets: its own market-independent probabilities, the "
                   "quality gate (≥3% EV, ≥2% edge, odds ≤ +500), staked at **quarter-Kelly** "
                   "(1 unit = 1% of bankroll). Out-of-sample — the model was trained only on data "
                   "before the tournament and priced at the Bet365 close.")
        ov = wc[wc["segment"] == "OVERALL"]
        has_kelly = "kelly_units" in wc.columns and not ov.empty
        if has_kelly:
            ku = float(ov["kelly_units"].iloc[0])           # net units = % of bankroll
            kroi = float(ov["kelly_roi"].iloc[0]) * 100
            froi = float(ov["roi"].iloc[0]) * 100
            bets = int(ov["bets"].iloc[0]); wins = int(ov["wins"].iloc[0])
            kcol = theme.GREEN if ku > 0 else (theme.RED if ku < 0 else theme.TEXT)
            theme.kpi_row([
                {"label": "Bankroll result", "value": f"{ku:+.1f}%",
                 "sub": "quarter-Kelly · 1u = 1% bankroll", "accent": kcol, "value_color": kcol},
                {"label": "Record", "value": f"{wins}/{bets}", "sub": "bets won", "accent": theme.BLUE},
                {"label": "Kelly ROI", "value": f"{kroi:+.1f}%", "sub": "net ÷ staked",
                 "accent": kcol, "value_color": kcol},
                {"label": "Flat ROI", "value": f"{froi:+.1f}%", "sub": "1u/bet, for reference",
                 "accent": theme.GOLD},
            ])
        # by-market table (Kelly units + ROI, with the flat ROI + CI alongside)
        show = wc.copy()
        show["record"] = show["wins"].astype(int).astype(str) + "/" + show["bets"].astype(int).astype(str)
        if "kelly_units" in show:
            show["Net units (Kelly)"] = show["kelly_units"].map(lambda x: f"{x:+.1f}u")
            show["Kelly ROI"] = (show["kelly_roi"] * 100).round(1).astype(str) + "%"
        show["Net units (flat)"] = show["units"].map(lambda x: f"{x:+.1f}u")
        show["flat ROI"] = (show["roi"] * 100).round(1).astype(str) + "%"
        show["flat 95% CI"] = ("[" + (show["roi_lo"] * 100).round(0).astype(int).astype(str)
                               + "%, " + (show["roi_hi"] * 100).round(0).astype(int).astype(str) + "%]")
        cols = [c for c in ["segment", "record", "Net units (Kelly)", "Kelly ROI",
                            "Net units (flat)", "flat ROI", "flat 95% CI"] if c in show.columns]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)
        st.caption("Net units = profit/loss in units (1u = 1% of bankroll). "
                   "Kelly = how the model actually stakes; flat = 1u per bet, for reference.")
        theme.callout(
            "<b>Read this honestly:</b> quarter-Kelly turns the full-2022 slate slightly positive, "
            "but the 95% CI <b>includes 0</b> and the result flips sign if you change the staking "
            "method or slice by stage — i.e. it's <b>one tiny, variance-heavy tournament, not a "
            "proven edge</b>. The large all-internationals backtest still shows no reliable edge "
            "against the closing line. Bet responsibly.", "warn")
    theme.footer()


def page_team():
    theme.hero("Team Explorer", "Pick any two nations and get the model's full read — "
               "win/draw/win, expected goals, scoreline heatmap, form and head-to-head.")
    pred = get_predictor()
    teams = sorted(pred.known_teams)
    c1, c2, c3 = st.columns([2, 2, 1])
    home = c1.selectbox("Home / Team A", teams,
                        index=teams.index("Brazil") if "Brazil" in teams else 0)
    away = c2.selectbox("Away / Team B", teams,
                        index=teams.index("Argentina") if "Argentina" in teams else 1)
    neutral = c3.checkbox("Neutral venue", value=True)
    if home == away:
        theme.callout("Pick two different teams.", "info")
        theme.footer()
        return
    a = pred.analyze(home, away, neutral=neutral)
    st.markdown(
        f'<div class="mcard-head" style="font-size:22px;justify-content:center;gap:14px;'
        f'margin:6px 0">{team_with_flag(home, 24, True)}'
        f'<span style="color:{GREY};font-size:14px">vs</span>'
        f'{team_with_flag(away, 24, True)}</div>', unsafe_allow_html=True)
    st.markdown(theme.prob_bar(a["probs"]["H"], a["probs"]["D"], a["probs"]["A"], home, away),
                unsafe_allow_html=True)
    theme.kpi_row([
        {"label": f"{home} win", "value": _pct(a["probs"]["H"]), "accent": theme.GREEN},
        {"label": "Draw", "value": _pct(a["probs"]["D"]), "accent": theme.MUTED},
        {"label": f"{away} win", "value": _pct(a["probs"]["A"]), "accent": theme.GOLD},
        {"label": "Exp. goals", "value": f"{a['expected_goals'][0]:.1f}–{a['expected_goals'][1]:.1f}",
         "sub": f"BTTS {_pct(a['btts'])}", "accent": theme.BLUE},
    ])
    st.write("**Over/Under ladder:** " + " · ".join(
        f"O{ln} {p*100:.0f}%" for ln, p in a["ou_ladder"].items()))
    left, right = st.columns([2, 3])
    with left:
        context_strip({"analysis": a, "home": home, "away": away,
                       "key_out_home": [], "key_out_away": []})
    with right:
        st.markdown("**Scoreline heatmap**")
        heatmap(a["scoreline_matrix"], home, away)
    theme.footer()


@st.cache_data(ttl=300, show_spinner="Syncing tracker (recording picks + settling results)…")
def clv_sync(day: str, min_ev: float) -> dict:
    """Record today's model picks as open tickets and settle any finished ones.
    Cached 5 min so the tracker self-updates without hammering ESPN; the page has a
    manual refresh too. Snapshot dedupes, so repeated calls are safe."""
    from src.predict import clv
    added = graded = 0
    errs = []
    try:
        added = clv.snapshot(day, days=3, min_ev=min_ev, cfg=CFG)
    except Exception as e:  # noqa: BLE001
        errs.append(f"snapshot: {type(e).__name__}: {e}")
    try:
        graded = clv.grade(CFG)
    except Exception as e:  # noqa: BLE001 — grade() is per-ticket-guarded; surface anything that escapes
        errs.append(f"grade: {type(e).__name__}: {e}")
    return {"added": added, "graded": graded, "error": " · ".join(errs)}


def _read_fresh(path) -> pd.DataFrame:
    """Read a tracker CSV without Streamlit's cache (it mutates during a session)."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _kelly_units(df: pd.DataFrame, frac: float, cap_u: float = 2.0):
    """Per-bet stake and P&L in units (1u = 1% bankroll) at the given Kelly fraction.
    Stake = min(Kelly_fraction(model_p, decimal) · frac · 100, ``cap_u``). The cap keeps any
    single high-edge/longshot pick from demanding a wild stake — disciplined bankroll management,
    not raw Kelly (which can ask for 8–10% on a big-edge dog)."""
    from src.predict import staking
    mp = pd.to_numeric(df["model_p"], errors="coerce").to_numpy()
    dec = pd.to_numeric(df["decimal"], errors="coerce").to_numpy()
    fp = pd.to_numeric(df["fair_p"], errors="coerce").to_numpy() if "fair_p" in df else [None] * len(mp)
    # disciplined sizing: capped quarter-Kelly, longshots down-weighted (src/predict/staking.py)
    stake = np.array([staking.recommended_units(
        p if pd.notna(p) else None, (f if (f is not None and pd.notna(f)) else None),
        d if pd.notna(d) else None, frac=frac, cap_u=cap_u) for p, f, d in zip(mp, fp, dec)])
    dec_safe = np.nan_to_num(dec, nan=1.0)        # un-priceable rows get stake 0 anyway
    res = df["result"].to_numpy()
    pnl = np.where(res == "push", 0.0,
                   np.where(res == "win", stake * (dec_safe - 1), -stake))
    return stake, pnl


def _equity_curve(df, kelly: float, height: int = 240):
    """Clean daily cumulative-units curve for any settled-bet frame (needs ``pnl_u`` + a date col).
    One point per day so the line steps smoothly; seeded at 0; green up / red down."""
    if not (len(df) and "pnl_u" in df.columns):
        return
    d2 = df.copy()
    when = d2["match_date"] if "match_date" in d2 else d2.get("graded_time")
    d2["day"] = pd.to_datetime(when, errors="coerce").dt.normalize()
    d2 = d2.dropna(subset=["day"])
    if d2.empty:
        return
    daily = (d2.groupby("day", as_index=False)
             .agg(net=("pnl_u", "sum"), bets=("pnl_u", "size")).sort_values("day"))
    daily["cum_units"] = daily["net"].cumsum()
    seed = pd.DataFrame({"day": [daily["day"].iloc[0] - pd.Timedelta(days=1)],
                         "net": [0.0], "bets": [0], "cum_units": [0.0]})
    daily = pd.concat([seed, daily], ignore_index=True)
    col = GREEN if daily["cum_units"].iloc[-1] >= 0 else RED
    base = alt.Chart(daily).encode(
        x=alt.X("day:T", title=None, axis=alt.Axis(format="%b %d", grid=False)),
        y=alt.Y("cum_units:Q", title=f"Cumulative units ({kelly:.2f}× Kelly)",
                axis=alt.Axis(grid=True)))
    area = base.mark_area(interpolate="monotone", line=False, opacity=0.16, color=col).encode(
        y="cum_units:Q")
    ln = base.mark_line(interpolate="monotone", strokeWidth=2.5, color=col).encode(
        tooltip=[alt.Tooltip("day:T", title="Date", format="%b %d"),
                 alt.Tooltip("cum_units:Q", title="Cumulative", format="+.1f"),
                 alt.Tooltip("net:Q", title="Day net", format="+.1f"),
                 alt.Tooltip("bets:Q", title="Bets")])
    pts = base.mark_point(size=34, filled=True, color=col, opacity=0.9).encode(y="cum_units:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color=GREY, strokeDash=[4, 4]).encode(y="y")
    st.altair_chart((zero + area + ln + pts).properties(height=height), use_container_width=True)


def page_clv(min_ev=0.03, kelly=0.25):
    theme.hero("Live Tracker", f"Every +EV pick recorded at the offered price, then settled "
               f"vs the result and the closing line. Staked at {kelly:.2f}× Kelly, capped at 2u "
               f"(2% of bankroll) per bet.")
    from src.predict import clv
    today = _today_ct().strftime("%Y-%m-%d")

    cc = st.columns([1, 1, 3])
    if cc[0].button("🔄 Sync now"):
        clv_sync.clear()
    auto = cc[1].checkbox("Auto-sync", value=True)
    if auto:
        s = clv_sync(today, min_ev)
        cc[2].caption(f"Synced · +{s['added']} new picks recorded · {s['graded']} just settled")
        if s.get("error"):
            cc[2].caption(f"⚠️ sync error — {s['error']}")

    # read FRESH (not via cached load_csv — the tracker writes to these during the session)
    led = _read_fresh(clv._ledger_path(CFG))
    op = _read_fresh(clv._open_path(CFG))
    # BTTS is suggestion-only — drop any historic/backfilled BTTS rows so it has no
    # record, equity, best-bets, or pending presence on the tracker (it's shown as a
    # model angle on the match cards instead).
    suggest_only = value_mod.SUGGESTION_ONLY_MARKETS
    if not led.empty and "market" in led.columns:
        led = led[~led["market"].isin(suggest_only)]
    if not op.empty and "market" in op.columns:
        op = op[~op["market"].isin(suggest_only)]
    if not led.empty:                       # only real, priced bets count as tracked
        led = led[pd.to_numeric(led["decimal"], errors="coerce").notna()]
        dedup_keys = [c for c in ["game_id", "market", "selection"] if c in led.columns]
        if dedup_keys:                      # one row per distinct bet (guard re-snapshots)
            led = led.drop_duplicates(subset=dedup_keys, keep="first")
    settled = led[led["result"].isin(["win", "loss", "push"])].copy() if not led.empty else led

    if not settled.empty:
        stake, pnl = _kelly_units(settled, kelly)
        net, staked = float(pnl.sum()), float(stake.sum())
        roi = net / staked if staked else 0.0
        wins = int((settled["result"] == "win").sum())
        losses = int((settled["result"] == "loss").sum())
        with_clv = settled.dropna(subset=["clv"]) if "clv" in settled else settled.iloc[:0]
        beat = float((with_clv["clv"] > 0).mean()) if len(with_clv) else float("nan")
        settled = settled.assign(stake_u=stake.round(2), pnl_u=pnl.round(2))
    else:
        net = staked = roi = 0.0
        wins = losses = 0
        beat = float("nan")

    net_col = theme.GREEN if net > 0 else (theme.RED if net < 0 else theme.TEXT)
    roi_col = theme.GREEN if roi > 0 else (theme.RED if roi < 0 else theme.TEXT)
    theme.kpi_row([
        {"label": "Settled bets", "value": len(settled),
         "sub": f"{wins}-{losses}" if len(settled) else "none yet", "accent": theme.BLUE},
        {"label": "Net (units)", "value": f"{net:+.1f}u" if len(settled) else "—",
         "sub": f"{staked:.1f}u staked @ {kelly:.2f}× Kelly", "accent": net_col,
         "value_color": net_col},
        {"label": "ROI", "value": f"{roi*100:+.1f}%" if staked else "—",
         "sub": "net ÷ staked", "accent": roi_col, "value_color": roi_col},
        {"label": "Beat the close", "value": f"{beat*100:.0f}%" if beat == beat else "—",
         "sub": "forward bets only", "accent": theme.GOLD},
    ])

    # cumulative Kelly P&L over time — a clean daily equity curve ("how it's doing" at a glance).
    _equity_curve(settled, kelly)

    # tracked systems (e.g. the v8 pick'em candidate — forward, observational)
    if not settled.empty and "system" in settled.columns:
        sys_rows = []
        for sysname, g in settled[settled["system"].fillna("") != ""].groupby("system"):
            gs, gp = _kelly_units(g, kelly)
            gc = g.dropna(subset=["clv"]) if "clv" in g else g.iloc[:0]
            sys_rows.append({"system": sysname, "bets": len(g),
                             "units": round(float(gp.sum()), 1),
                             "ROI %": round(float(gp.sum() / gs.sum() * 100), 1) if gs.sum() else 0.0,
                             "avg CLV %": round(float(gc["clv"].mean()) * 100, 2) if len(gc) else float("nan")})
        if sys_rows:
            theme.section("Tracked systems", sub="forward, observational")
            st.dataframe(pd.DataFrame(sys_rows), hide_index=True, use_container_width=True)
            st.caption("`pickem_ml_2_3` = the even-money moneyline candidate from v8 — tracked, "
                       "**not** a deployed bet (it failed the pre-registered backtest bar).")

    # ⭐ Best bets — the single strongest qualifying pick per match, settled
    if not settled.empty:
        from src.predict.betting import qualifies
        s = settled.copy()
        for c in ("model_p", "fair_p", "decimal", "ev"):
            s[c] = pd.to_numeric(s.get(c), errors="coerce")
        s["_q"] = s.apply(lambda r: qualifies(r["model_p"], r["fair_p"], r["decimal"],
                                              0.03, 0.02, 6.0), axis=1)
        bb = s[s["_q"]].copy()
        bb["_edge"] = bb["model_p"] - bb["fair_p"]       # rank by edge, not raw EV (avoids longshots)
        keycol = "game_id" if "game_id" in bb.columns else "match"
        if not bb.empty:
            bb = (bb.sort_values(["_edge", "ev"], ascending=False)
                  .drop_duplicates(subset=[keycol], keep="first"))
            bs, bp = _kelly_units(bb, kelly)
            bb = bb.assign(stake_u=bs.round(2), pnl_u=bp.round(2))
            bnet, bst = float(bp.sum()), float(bs.sum())
            broi = bnet / bst if bst else 0.0
            bw = int((bb["result"] == "win").sum())
            bl = int((bb["result"] == "loss").sum())
            bcol = theme.GREEN if bnet > 0 else (theme.RED if bnet < 0 else theme.TEXT)
            theme.section("Best bets", sub="the model's strongest call per match")
            theme.kpi_row([
                {"label": "Best bets settled", "value": len(bb),
                 "sub": f"{bw}-{bl}", "accent": theme.BLUE},
                {"label": "Record", "value": (f"{bw/(bw+bl)*100:.0f}% won" if (bw + bl) else "—"),
                 "accent": theme.GOLD},
                {"label": "Net (units)", "value": f"{bnet:+.1f}u",
                 "sub": f"{bst:.1f}u staked", "accent": bcol, "value_color": bcol},
                {"label": "ROI", "value": f"{broi*100:+.1f}%", "accent": bcol,
                 "value_color": bcol},
            ])
            _equity_curve(bb, kelly, height=200)
            cols = [c for c in ["match_date", "match", "market", "selection", "american",
                                "model_p", "result", "stake_u", "pnl_u"] if c in bb.columns]
            st.dataframe(bb[cols].iloc[::-1], hide_index=True, use_container_width=True)
            st.caption("One pick per match: the highest-EV selection that clears the quality gate "
                       "(≥3% EV, ≥2% edge vs the price, odds ≤ +500). Honest read — small sample; "
                       "the model has **no proven betting edge**, this is a track record, not a promise.")

    if not op.empty:
        theme.section(f"Pending ({len(op)})", sub="awaiting results")
        cols = [c for c in ["match", "market", "selection", "american", "ev", "system"]
                if c in op.columns]
        st.dataframe(op[cols], hide_index=True, use_container_width=True)

    if not settled.empty:
        # units by category — same layout as the Performance 2022 table
        if "stake_u" in settled.columns:
            theme.section("Units by category")

            def _boot(pnl: np.ndarray, n: int = 1000):
                if len(pnl) == 0:
                    return (0.0, 0.0, 0.0)
                rng = np.random.default_rng(0)
                idx = rng.integers(0, len(pnl), size=(n, len(pnl)))
                r = pnl[idx].mean(axis=1)
                return float(pnl.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

            def _row(name, g):
                dec = pd.to_numeric(g["decimal"], errors="coerce").to_numpy()
                res = g["result"].to_numpy()
                settled_m = res != "push"
                flat = np.where(res == "push", 0.0, np.where(res == "win", dec - 1, -1.0))
                m, lo, hi = _boot(flat[settled_m])
                kstaked = float(g["stake_u"].sum()); knet = float(g["pnl_u"].sum())
                wins = int((res == "win").sum())
                return {"segment": name, "record": f"{wins}/{len(g)}",
                        "Net units (Kelly)": f"{knet:+.1f}u",
                        "Kelly ROI": f"{(knet/kstaked*100 if kstaked else 0):+.1f}%",
                        "Net units (flat)": f"{flat.sum():+.1f}u",
                        "flat ROI": f"{m*100:+.1f}%",
                        "flat 95% CI": f"[{lo*100:+.0f}%, {hi*100:+.0f}%]", "_net": knet}

            rows = [_row("OVERALL", settled)]
            rows += sorted((_row(mk, g) for mk, g in settled.groupby("market")),
                           key=lambda r: -r["_net"])
            cat_df = pd.DataFrame(rows).drop(columns="_net")
            st.dataframe(cat_df, hide_index=True, use_container_width=True)
            st.caption("Kelly = how the model actually stakes (current fraction, 1u = 1% of "
                       "bankroll); flat = 1u per bet, for reference.")

        theme.section("Settled")
        cols = [c for c in ["match_date", "match", "market", "selection", "american",
                            "result", "stake_u", "pnl_u", "clv", "system"]
                if c in settled.columns]
        st.dataframe(settled[cols].iloc[::-1], hide_index=True, use_container_width=True)
    elif op.empty:
        theme.callout("No picks tracked yet. Hit <b>Sync now</b> (or wait for auto-sync) on a "
                      "day with upcoming fixtures + odds to start recording the model's bets.",
                      "info")
    theme.footer()


def _kalshi_signals(matches: list, mk: list, buy_edge: float, sell_edge: float, kelly: float):
    """Scan every upcoming fixture's model bets (Match Result / Total / Spread / BTTS) against the
    live Kalshi book → BUY rows (model beats the ask) and SELL/fade rows (bid richer than model)."""
    from src.data import kalshi as kal
    from src.predict import staking
    buys, sells = [], []
    for m in matches:
        book = kal.match_book(m["home"], m["away"], markets=mk)
        if not book:
            continue
        match = f"{m['home']} v {m['away']}"
        ko = _ct(m.get("date"))                      # kickoff in Central time (sortable datetime)
        try:
            ko = ko.tz_localize(None) if (ko is not None and ko.tzinfo) else ko
        except Exception:  # noqa: BLE001
            pass
        for b in m.get("bets", []):
            px = kal.price_for(book, b.market, b.selection, m["home"], m["away"])
            if not px:
                continue
            ask, bid, prev = px.get("ask"), px.get("bid"), px.get("prev_ask")
            p = b.model_p
            sig = kal.signal(p, bid, ask, buy_edge, sell_edge)
            delta = (ask - prev) if (ask is not None and prev is not None) else None
            if sig["action"] == "BUY" and ask:
                # disciplined sizing on the exchange decimal (1/ask); longshots down-weighted
                stake = staking.recommended_units(p, None, 1.0 / ask, frac=kelly)
                buys.append({"Kickoff": ko, "Match": match, "Market": b.market, "Pick": b.selection,
                             "Model": p, "Ask": ask, "Edge": p - ask, "EV": sig["ev_buy"],
                             "Δ": delta, "Stake": stake})
            elif sig["action"] == "SELL" and bid is not None:
                sells.append({"Kickoff": ko, "Match": match, "Market": b.market, "Pick": b.selection,
                              "Model": p, "Bid": bid, "Gap": bid - p, "Fade EV": sig["ev_sell"],
                              "Δ": delta})
    return buys, sells


def _fmt_cent(x):
    return f"{x*100:.0f}¢" if x is not None else "—"


def _fmt_pct(x):
    return f"{x*100:.0f}%" if x is not None else "—"


def _fmt_signed_cent(x):
    return f"{x*100:+.0f}¢" if x is not None else "·"


def page_kalshi(bankroll=1000, kelly=0.25, upset_temp=1.0):
    theme.hero("Kalshi", "Live exchange prices vs the model — when to <b>buy</b> (model beats the ask) "
               "and when to <b>sell</b> (the bid runs richer than the model). Prices refresh on their "
               "own; the exchange spread is ~3%, so only net-of-spread edges count.")
    from src.data import kalshi as kal

    c = st.columns([1, 1, 1, 1])
    buy_edge = c[0].slider("Buy edge", 0.0, 0.20, 0.05, 0.01,
                           help="BUY when model% − Kalshi ask ≥ this.")
    sell_edge = c[1].slider("Sell edge", 0.0, 0.20, 0.05, 0.01,
                            help="SELL/fade when Kalshi bid − model% ≥ this.")
    sort_by = c[2].radio("Sort by", ["Best EV", "Kickoff"], horizontal=True,
                         help="Order the boards by edge, or by soonest kickoff. "
                              "(You can also click any column header to sort.)")
    auto = c[3].checkbox("Auto-refresh (30s)", value=True)

    today = _today_ct().strftime("%Y-%m-%d")
    try:
        res = get_bets(today, 10, bankroll, kelly, upset_temp)
    except Exception:  # noqa: BLE001
        res = {"matches": []}

    matches = [m for m in res.get("matches", []) if _match_not_started(m)]
    if not matches:
        st.info("No upcoming (not-yet-started) fixtures right now — settled and in-play games are hidden.")
        theme.footer()
        return
    st.caption("Markets: Match Result · Total Goals · Spread · Both Teams To Score — every "
               "model bet priced against the live Kalshi order book.")

    @st.fragment(run_every=(30 if auto else None))
    def _live():
        mk = kal.all_match_markets(ttl=20.0)
        now_ct = (datetime.now(_CT) if _CT is not None else datetime.utcnow()).strftime("%I:%M:%S %p")
        if not mk:
            st.warning("Kalshi prices unavailable right now (the exchange API may be blocking this "
                       "server). Try again shortly.")
            return
        buys, sells = _kalshi_signals(matches, mk, buy_edge, sell_edge, kelly)
        _far = pd.Timestamp.max

        def _order(rows, ev_key):
            if sort_by == "Kickoff":
                rows.sort(key=lambda r: r.get("Kickoff") or _far)
            else:
                rows.sort(key=lambda r: (r[ev_key] is None, -(r[ev_key] or 0)))
            return rows

        _kocol = {"Kickoff": st.column_config.DatetimeColumn("Kickoff", format="ddd h:mm a")}
        st.caption(f"Kalshi prices as of {now_ct} CT · {len(mk)} live contracts · "
                   f"{len(buys)} buy / {len(sells)} sell signals")

        theme.section("🟢 BUY — model probability beats the Kalshi ask")
        if buys:
            df = pd.DataFrame(_order(buys, "EV"))
            disp = pd.DataFrame({
                "Kickoff": df["Kickoff"], "Match": df["Match"], "Market": df["Market"],
                "Pick": df["Pick"], "Model %": df["Model"].map(_fmt_pct),
                "Buy @": df["Ask"].map(_fmt_cent),
                "Edge": df["Edge"].map(_fmt_pct), "EV": df["EV"].map(_fmt_pct),
                "Δ": df["Δ"].map(_fmt_signed_cent),
                "Stake": df["Stake"].map(lambda u: f"{u:.1f}u" if u > 0.05 else "—")})
            st.dataframe(disp, hide_index=True, use_container_width=True, column_config=_kocol)
        else:
            st.caption("No buy signal clears your edge right now.")

        theme.section("🔴 SELL / fade — the Kalshi bid is richer than the model")
        if sells:
            df = pd.DataFrame(_order(sells, "Fade EV"))
            disp = pd.DataFrame({
                "Kickoff": df["Kickoff"], "Match": df["Match"], "Market": df["Market"],
                "Pick": df["Pick"], "Model %": df["Model"].map(_fmt_pct),
                "Sell @": df["Bid"].map(_fmt_cent),
                "Over-priced by": df["Gap"].map(_fmt_pct),
                "Fade EV": df["Fade EV"].map(_fmt_pct), "Δ": df["Δ"].map(_fmt_signed_cent)})
            st.dataframe(disp, hide_index=True, use_container_width=True, column_config=_kocol)
            st.caption("**Sell @** is the bid you'd receive — if you hold this pick, sell to lock value. "
                       "**Fade EV** is the model's edge on the opposite side at its current price.")
        else:
            st.caption("Nothing is trading rich enough to fade right now.")

    _live()
    st.caption("Kalshi is an exchange order book — the **ask ≈ the implied % you pay**, so EV = "
               "model ÷ ask − 1 with no de-vig. **Buy @** = ask, **Sell @** = bid; the ~3% spread "
               "means only net-of-spread edges count. **Δ** = ask move since the last tick. An edge = "
               "the model disagrees with a tradeable price, **not** a guaranteed win. Stakes capped at "
               "2u (2% of bankroll). **Not financial advice.**")
    theme.footer()


NAV = [
    ("⚽", "Matches", "matches"),
    ("💰", "Value Board", "value"),
    ("🎯", "Kalshi", "kalshi"),
    ("📈", "Tracker", "clv"),
    ("🏆", "Tournament", "tournament"),
    ("📊", "Performance", "performance"),
    ("🔎", "Team Explorer", "team"),
]
_NAV_SLUGS = {slug for _, _, slug in NAV}


def _nav_links(active: str) -> str:
    out = ""
    for icon, name, slug in NAV:
        cls = "navitem active" if slug == active else "navitem"
        out += f'<a class="{cls}" href="?page={slug}" target="_self">{icon} {name}</a>'
    return out


def render_topnav(active: str) -> None:
    """A real top navigation bar: brand + pill links on desktop, a full-screen
    hamburger drawer on mobile. Pure HTML/CSS (a CSS ``:target`` toggle — no JS, no
    <details>, which Streamlit's sanitizer mangles). Links are ``?page=…`` anchors."""
    links = _nav_links(active)
    st.markdown(
        f'<div class="topnav">'
        f'<a class="navbrand" href="?page=matches" target="_self">🏆 FIFA&nbsp;WC&nbsp;<b>2026</b></a>'
        f'<nav class="navlinks">{links}</nav>'
        f'<a class="navham-btn" href="#wcnav" target="_self">☰</a>'
        f'</div>'
        f'<div id="wcnav" class="navdrawer">'
        f'<div class="navdrawer-head">'
        f'<span class="navbrand">🏆 FIFA&nbsp;WC&nbsp;<b>2026</b></span>'
        f'<a class="navdrawer-close" href="#" target="_self">✕</a></div>'
        f'{links}'
        f'</div>', unsafe_allow_html=True)


def main():
    # Query-param routing drives a real top nav bar (mobile hamburger included).
    page = st.query_params.get("page", "matches")
    if page not in _NAV_SLUGS:
        page = "matches"
    render_topnav(page)
    # staking/filters only matter on the betting pages
    show_filters = page in ("matches", "value", "clv", "kalshi")
    bankroll, kelly, min_ev, min_edge, max_exp, upset_temp = 1000, 0.25, 0.05, 0.02, 1.0, 1.0
    if show_filters:
        with st.expander("⚙️  Staking & filters", expanded=False):
            st.markdown(
                '<div style="padding:6px 2px 10px 2px">'
                '<div style="font-family:Oswald;font-size:22px;font-weight:700;line-height:1.05;'
                'text-transform:uppercase;letter-spacing:.5px">🏆 FIFA World&nbsp;Cup'
                f'<span style="color:{GREEN}"> 2026</span></div>'
                '<div style="font-family:inherit;font-size:15px;color:#969aa6;'
                'letter-spacing:2px;text-transform:uppercase">Soccer Model</div></div>',
                unsafe_allow_html=True)
            st.caption("Dixon-Coles · Elo · LightGBM ensemble · live ESPN data")
            labels = [f"{icon} {name}" for icon, name, _ in NAV]
            st.divider()
            st.markdown("**⚙️ Staking** &nbsp; <span style='color:#969aa6;font-size:12px'>"
                        "1 unit = 1% of bankroll</span>", unsafe_allow_html=True)
            bankroll = st.number_input("Bankroll ($)", 10, 1_000_000, 1000, step=50)
            kelly = st.slider("Kelly fraction", 0.0, 1.0, 0.25, 0.05,
                              help="0.25 = quarter Kelly (default, conservative). 0.5 = half. "
                                   "1.0 = full Kelly (aggressive). Stakes + tracker units scale "
                                   "with this.")
            min_ev = st.slider("Min EV", 0.0, 0.30, 0.05, 0.01)
            min_edge = st.slider("Min edge (model − market)", 0.0, 0.10, 0.02, 0.005,
                                 help="The model must beat the de-vigged price by at least this "
                                      "much — a REAL disagreement, not EV leverage on long odds. "
                                      "This is what stops the underdog/longshot junk.")
            max_exp = st.slider("Max total exposure (× bankroll)", 0.1, 2.0, 1.0, 0.1)
            st.session_state["kalshi_alert_ev"] = st.slider(
                "Kalshi alert EV", 0.0, 0.30, 0.08, 0.01,
                help="Fire a 🔔 on a match card when the model's edge at the Kalshi ask clears this.")
            st.divider()
            st.markdown("**Upset sensitivity**", help=None)
            upset_temp = st.slider("Upset sensitivity (τ)", 1.0, 2.0, 1.0, 0.05,
                                   label_visibility="collapsed",
                                   help="1.0 = the model's calibrated, most-accurate forecast. "
                                        "Higher spreads probability toward underdogs/draws to surface "
                                        "more upset value — it does NOT change the model's actual pick, "
                                        "only how it shares probability.")
            if upset_temp > 1.0:
                st.caption(f"τ={upset_temp:.2f}: more upset value, lower precision. "
                           "Backtest cost — pooled WC RPS 0.196→~0.201 at 1.5 (worse); "
                           "upset-recall 30%→37%. Pick is unchanged.")
            else:
                st.caption("τ=1.0 — calibrated forecast (deployed accuracy).")
            st.caption("⚠ EV/Kelly are only as good as the model's probabilities. The **min-edge "
                       "gate** drops leverage-driven longshot flags. **Not betting advice** · "
                       "independent model, not affiliated with FIFA.")

    if page == "matches":
        page_matches(bankroll, kelly, min_ev, min_edge, upset_temp)
    elif page == "value":
        page_value_board(bankroll, kelly, min_ev, max_exp, min_edge, upset_temp)
    elif page == "kalshi":
        page_kalshi(bankroll, kelly, upset_temp)
    elif page == "clv":
        page_clv(min_ev, kelly)
    elif page == "tournament":
        page_tournament()
    elif page == "performance":
        page_performance()
    else:
        page_team()


main()
