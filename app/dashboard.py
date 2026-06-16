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
from datetime import date, datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


# ----------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Fitting models (one-time)…")
def get_predictor() -> MatchPredictor:
    return MatchPredictor(CFG)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    p = path_for("reports", CFG) / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600, show_spinner="Pulling live odds + analysing…")
def get_bets(day: str, days: int, bankroll: float, kelly: float) -> dict:
    return value_mod.build_bets(day, days=days, bankroll=bankroll,
                                kelly_fraction=kelly, cfg=CFG,
                                predictor=get_predictor(), use_cache=False)


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


def market_table(title: str, bets: list, key_note: str | None = None):
    """Render one market's selections with model%, fair%, price, break-even, EV, stake."""
    st.markdown(f"**{title}**")
    rows = []
    for b in bets:
        units = b.kelly_used * 100  # 1 unit = 1% of bankroll
        rows.append({
            "Selection": b.selection,
            "Model": _pct(b.model_p),
            "Fair (no-vig)": _pct(b.fair_p),
            "Price": _am(b.american),
            "Break-even": _implied_pct(b.american),
            "EV": f"{b.ev*100:+.0f}%",
            "Stake": f"{units:.1f}u" if units > 0.05 else "—",
        })
    df = pd.DataFrame(rows)

    def _style(row):
        # color EV cell
        ev = float(row["EV"].rstrip("%")) / 100
        return ["" if c != "EV" else f"color:{_ev_color(ev)};font-weight:600"
                for c in row.index]

    st.dataframe(df.style.apply(_style, axis=1), hide_index=True,
                 use_container_width=True)
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
            badge = ("🟢" if adv >= 0.85 else "🟡" if adv >= 0.4 else "🔴")
            st.write(f"{badge} **{adv*100:.0f}%** to reach knockouts")
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
        st.markdown('<div class="bbet"><span class="h">💡 Best bet</span> &nbsp; '
                    '<span style="color:#8b93a7">no edge here — pass</span></div>',
                    unsafe_allow_html=True)
        return
    b = max(cands, key=lambda x: x.ev)
    units = b.kelly_used * 100
    cons = (m.get("cons_edge") or {}).get(b.selection)
    cons_txt = ""
    if cons is not None:
        cons_txt = (f' &nbsp;·&nbsp; {"✅ beats consensus" if cons > 0 else "⚠️ ≤ consensus"} '
                    f'({cons*100:+.0f}%)')
    st.markdown(
        f'<div class="bbet"><span class="h">💡 Best bet</span> &nbsp; '
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


def render_match(m: dict, live: dict | None = None, min_ev: float = 0.03,
                 min_prob_edge: float = 0.02):
    a = m["analysis"]
    t = pd.to_datetime(m["date"]).strftime("%a %d %b %H:%M UTC")
    probs = _display_probs(m)
    pick = OUTCOMES[int(np.argmax([probs["H"], probs["D"], probs["A"]]))]
    pick_name = {"H": m["home"], "D": "Draw", "A": m["away"]}[pick]
    n_value = len(_qualifying_bets(m, min_ev, min_prob_edge))
    title = (f"{m['home']}  vs  {m['away']}   ·   {t}   ·   "
             f"⚽ leans {pick_name}" + (f"   ·   💰 {n_value} value" if n_value else ""))
    with st.expander(title, expanded=False):
        # flag-vs-flag header + lean pill + value badge
        lean_pill = theme.pill(f"model leans {pick_name}", "green")
        val_pill = theme.pill(f"💰 {n_value} value bet" + ("s" if n_value != 1 else ""),
                              "gold") if n_value else ""
        st.markdown(
            f'<div class="mcard-head" style="font-size:20px;justify-content:center;'
            f'gap:14px;margin-bottom:2px">{team_with_flag(m["home"], 22, True)}'
            f'<span style="color:{GREY};font-size:14px">vs</span>'
            f'{team_with_flag(m["away"], 22, True)}</div>'
            f'<div style="text-align:center;margin-bottom:8px">{lean_pill} &nbsp; {val_pill}</div>',
            unsafe_allow_html=True)
        st.markdown(theme.prob_bar(probs["H"], probs["D"], probs["A"], m["home"], m["away"]),
                    unsafe_allow_html=True)
        move = m.get("home_line_move")
        move_txt = ""
        if move is not None and not pd.isna(move) and abs(move) >= 0.015:
            who = m["home"] if move > 0 else m["away"]
            move_txt = f" · 📈 line moving toward {who} ({abs(move)*100:.0f}%)"
        eg = a["expected_goals"]
        st.caption(f"Venue: {'neutral' if m['neutral'] else m['home'] + ' home'} · "
                   f"expected goals: {m['home']} {eg[0]:.1f}, {m['away']} {eg[1]:.1f} "
                   f"(**{eg[0] + eg[1]:.1f} total**) · odds: {m['provider'] or 'n/a'}{move_txt}")
        motivation_block(m, live)
        context_strip(m)
        st.divider()
        left, right = st.columns([3, 2])
        with left:
            by_market: dict[str, list] = {}
            for b in m["bets"]:
                by_market.setdefault(b.market, []).append(b)
            for mk in ["Match Result", "Total Goals", "Spread"]:
                if mk in by_market:
                    market_table(mk, by_market[mk])
            st.caption("**Model** = our pure probability, calibrated to historical results — "
                       "**independent of the betting market** (matches the bar, the expected "
                       "goals and the heatmap) · **Fair** = de-vigged market (shown only as a "
                       "reference) · **Break-even** = the offered price's implied %. EV is "
                       "positive only when **Model > Break-even**.")
            st.markdown(f"**Both Teams To Score** — model **{_pct(a['btts'])}** "
                        f"(no Vegas line — info only)")
        with right:
            st.markdown("**Scoreline heatmap**")
            heatmap(a["scoreline_matrix"], m["home"], m["away"])
            tops = " · ".join(f"{s} ({p*100:.0f}%)" for s, p in a["top_scorelines"][:5])
            st.caption("Most likely: " + tops)
        best_bet_block(m, min_ev, min_prob_edge)


# --------------------------------------------------------------------- pages
def page_matches(bankroll, kelly, min_ev=0.03, min_prob_edge=0.02):
    theme.hero("Matches", "Model vs market across every priced market — flags, probabilities, "
               "and the single best bet per game.", icon="⚽")
    c1, c2 = st.columns([1, 1])
    day = c1.date_input("From date", value=date.today())
    days = c2.slider("Days ahead", 1, 7, 3)
    res = get_bets(day.strftime("%Y-%m-%d"), days, bankroll, kelly)
    matches = res["matches"]
    if not matches:
        theme.callout("No upcoming fixtures with odds in this window — ESPN nulls odds once a "
                      "match kicks off. Try the current World Cup dates.", "info")
        theme.footer()
        return
    bets = res["bets"]
    n_val = int((bets["ev"] > 0.02).sum()) if not bets.empty else 0
    theme.kpi_row([
        {"label": "Fixtures", "value": len(matches), "accent": theme.GREEN},
        {"label": "+EV selections", "value": n_val, "accent": theme.GOLD,
         "value_color": theme.GOLD if n_val else theme.TEXT},
        {"label": "Markets / match", "value": "4", "sub": "Result · Totals · Spread · BTTS",
         "accent": theme.BLUE},
    ])
    # live 2026 group state for the "stakes" block on group-stage cards (best-effort)
    live = None
    try:
        live = get_live_state()
    except Exception:  # noqa: BLE001 — cards still render without the stakes block
        live = None
    for m in matches:
        render_match(m, live, min_ev, min_prob_edge)
    theme.footer()


def page_value_board(bankroll, kelly, min_ev, max_exposure, min_prob_edge=0.02):
    theme.hero("Value Board", "Every +EV bet across the slate, ranked — staked by fractional "
               "Kelly and capped to your max exposure.", icon="💰")
    day = st.date_input("From date", value=date.today(), key="vb_date")
    days = st.slider("Days ahead", 1, 7, 3, key="vb_days")
    res = get_bets(day.strftime("%Y-%m-%d"), days, bankroll, kelly)
    bb = value_mod.best_bets(res["bets"], min_ev=min_ev, min_prob_edge=min_prob_edge)
    if bb.empty:
        theme.callout("No bets clear the EV threshold for this window.", "info")
        theme.footer()
        return
    bb = value_mod.cap_exposure(bb, bankroll, max_fraction=max_exposure)
    bb["units"] = bb["stake"] / (bankroll / 100.0)   # 1 unit = 1% of bankroll
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
               "results locked in, team strength updating from 2026 form.", icon="🏆")
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
    st.markdown("#### 🏆 Championship probability")
    chart = alt.Chart(top).mark_bar(color=GREEN, cornerRadiusEnd=4).encode(
        x=alt.X("champion:Q", axis=alt.Axis(format="%"), title="Championship probability"),
        y=alt.Y("team:N", sort="-x", title=None),
        tooltip=["team", "group", alt.Tooltip("champion:Q", format=".1%"),
                 alt.Tooltip("advance:Q", format=".1%")]).properties(height=460)
    st.altair_chart(chart, use_container_width=True)

    # --- live group leaderboards with flags + P(advance) ---
    st.markdown("#### Group standings & qualification odds")
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
                show["flag"] = show["team"].map(lambda t: flag_url(t, 40))
                show = show[["S", "flag", "team", "P", "Pts", "GD", "Adv%"]]
                st.markdown(f"**Group {g}**")
                st.dataframe(show, hide_index=True, use_container_width=True, column_config={
                    "S": st.column_config.TextColumn("", width="small"),
                    "flag": st.column_config.ImageColumn("", width="small"),
                    "team": st.column_config.TextColumn("Team"),
                    "Adv%": st.column_config.NumberColumn("Adv%", format="%d%%")})

    with st.expander("Full qualification & advancement table"):
        show = qual.sort_values("champion", ascending=False).copy()
        show.insert(0, "flag", show["team"].map(lambda t: flag_url(t, 40)))
        for c in ["win_group", "advance", "reach_r16", "reach_qf", "reach_sf",
                  "reach_final", "champion"]:
            if c in show:
                show[c] = (show[c] * 100).round(1)
        st.dataframe(show, use_container_width=True, hide_index=True, column_config={
            "flag": st.column_config.ImageColumn("", width="small")})
    theme.footer()


def page_performance():
    theme.hero("Performance", "How the model actually scores — accuracy on past World Cups and "
               "an honest betting backtest of the 2022 tournament.", icon="📊")
    bt = load_csv("backtest_pooled.csv")
    if not bt.empty:
        # headline RPS KPI (ensemble row if present)
        try:
            rps_col = next(c for c in bt.columns if "rps" in c.lower())
            best = bt[rps_col].min()
            theme.kpi_row([
                {"label": "Best pooled RPS", "value": f"{best:.4f}",
                 "sub": "2010–2022 World Cups · ≈0.20 is bookmaker-grade", "accent": theme.GREEN},
                {"label": "Benchmark", "value": "≈ 0.20", "sub": "good football models",
                 "accent": theme.BLUE},
            ])
        except Exception:  # noqa: BLE001
            pass
        st.markdown("#### Accuracy — pooled over 2010–2022 World Cups (lower RPS is better)")
        st.dataframe(bt.round(4), use_container_width=True, hide_index=True)
    cal = load_csv("calibration.csv")
    if not cal.empty:
        st.markdown("#### Calibration — predicted vs observed")
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
        st.markdown("#### Ablation — does each block lower RPS?")
        st.dataframe(abl.round(4), use_container_width=True, hide_index=True)

    wc = load_csv("wc2022_backtest.csv")
    if not wc.empty:
        st.markdown("#### 🏆 2022 World Cup betting backtest (Bet365 close, out-of-sample)")
        show = wc.copy()
        for c in ["roi", "roi_lo", "roi_hi"]:
            if c in show:
                show[c] = (show[c] * 100).round(1).astype(str) + "%"
        st.dataframe(show, use_container_width=True, hide_index=True)
        ov = wc[wc["segment"] == "OVERALL"]
        roi = float(ov["roi"].iloc[0]) * 100 if not ov.empty else 0
        n = int(ov["bets"].iloc[0]) if not ov.empty else 0
        theme.callout(
            f"<b>Read this honestly:</b> the model's 2022 WC bets returned <b>{roi:+.1f}% ROI</b> "
            f"over {n} bets — but the 95% CI <b>includes 0</b> (one tournament is a tiny, "
            f"variance-heavy sample). This is encouraging, <b>not a proven edge</b>. The large "
            f"all-internationals backtest still shows no reliable edge against the closing line.",
            "warn")
    theme.footer()


def page_team():
    theme.hero("Team Explorer", "Pick any two nations and get the model's full read — "
               "win/draw/win, expected goals, scoreline heatmap, form and head-to-head.",
               icon="🔎")
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
    try:
        added = clv.snapshot(day, days=3, min_ev=min_ev, cfg=CFG)
    except Exception:  # noqa: BLE001
        pass
    try:
        graded = clv.grade(CFG)
    except Exception:  # noqa: BLE001
        pass
    return {"added": added, "graded": graded}


def _read_fresh(path) -> pd.DataFrame:
    """Read a tracker CSV without Streamlit's cache (it mutates during a session)."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _kelly_units(df: pd.DataFrame, frac: float):
    """Per-bet stake and P&L in units (1u = 1% bankroll) at the given Kelly fraction.
    Stake = Kelly_fraction(model_p, decimal) · frac · 100."""
    from src.predict.betting import kelly_fraction
    mp = pd.to_numeric(df["model_p"], errors="coerce").to_numpy()
    dec = pd.to_numeric(df["decimal"], errors="coerce").to_numpy()
    kf = np.array([kelly_fraction(p, d) if (pd.notna(p) and pd.notna(d) and d > 1) else 0.0
                   for p, d in zip(mp, dec)])
    stake = kf * frac * 100.0
    dec_safe = np.nan_to_num(dec, nan=1.0)        # un-priceable rows get stake 0 anyway
    res = df["result"].to_numpy()
    pnl = np.where(res == "push", 0.0,
                   np.where(res == "win", stake * (dec_safe - 1), -stake))
    return stake, pnl


def page_clv(min_ev=0.03, kelly=0.25):
    theme.hero("Live Tracker", f"Every +EV pick recorded at the offered price, then settled "
               f"vs the result and the closing line. Units staked at {kelly:.2f}× Kelly.",
               icon="📈")
    from src.predict import clv
    today = date.today().strftime("%Y-%m-%d")

    cc = st.columns([1, 1, 3])
    if cc[0].button("🔄 Sync now"):
        clv_sync.clear()
    auto = cc[1].checkbox("Auto-sync", value=True)
    if auto:
        s = clv_sync(today, min_ev)
        cc[2].caption(f"Synced · +{s['added']} new picks recorded · {s['graded']} just settled")

    # read FRESH (not via cached load_csv — the tracker writes to these during the session)
    led = _read_fresh(clv._ledger_path(CFG))
    op = _read_fresh(clv._open_path(CFG))
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

    # cumulative Kelly P&L over time — "how it's doing" at a glance
    if len(settled) and "pnl_u" in settled.columns:
        d2 = settled.copy()
        when = d2["match_date"] if "match_date" in d2 else d2.get("graded_time")
        d2["when"] = pd.to_datetime(when, errors="coerce")
        d2 = d2.dropna(subset=["when"]).sort_values("when")
        d2["cum_units"] = d2["pnl_u"].cumsum()
        line = alt.Chart(d2).mark_area(line=True, opacity=0.2, color=GREEN).encode(
            x=alt.X("when:T", title=None),
            y=alt.Y("cum_units:Q", title=f"Cumulative units ({kelly:.2f}× Kelly)"),
            tooltip=[alt.Tooltip("when:T"), alt.Tooltip("cum_units:Q", format="+.1f"),
                     "match", "selection", "result"]).properties(height=220)
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=GREY).encode(y="y")
        st.altair_chart(zero + line, use_container_width=True)

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
            st.subheader("Tracked systems (forward, observational)")
            st.dataframe(pd.DataFrame(sys_rows), hide_index=True, use_container_width=True)
            st.caption("`pickem_ml_2_3` = the even-money moneyline candidate from v8 — tracked, "
                       "**not** a deployed bet (it failed the pre-registered backtest bar).")

    if not op.empty:
        st.subheader(f"⏳ Pending ({len(op)}) — awaiting results")
        cols = [c for c in ["match", "market", "selection", "american", "ev", "system"]
                if c in op.columns]
        st.dataframe(op[cols], hide_index=True, use_container_width=True)

    if not settled.empty:
        st.subheader("✅ Settled")
        cols = [c for c in ["match_date", "match", "market", "selection", "american",
                            "result", "stake_u", "pnl_u", "clv", "system"]
                if c in settled.columns]
        st.dataframe(settled[cols].iloc[::-1], hide_index=True, use_container_width=True)
    elif op.empty:
        theme.callout("No picks tracked yet. Hit <b>Sync now</b> (or wait for auto-sync) on a "
                      "day with upcoming fixtures + odds to start recording the model's bets.",
                      "info")
    theme.footer()


NAV = [
    ("⚽", "Matches", "matches"),
    ("💰", "Value Board", "value"),
    ("📈", "Tracker", "clv"),
    ("🏆", "Tournament", "tournament"),
    ("📊", "Performance", "performance"),
    ("🔎", "Team Explorer", "team"),
]


def main():
    with st.sidebar:
        st.markdown(
            '<div style="padding:6px 2px 10px 2px">'
            '<div style="font-family:Oswald;font-size:22px;font-weight:700;line-height:1.05;'
            'text-transform:uppercase;letter-spacing:.5px">🏆 FIFA World&nbsp;Cup'
            '<span style="color:#1ec773"> 2026</span></div>'
            '<div style="font-family:Oswald;font-size:15px;color:#8b93a7;'
            'letter-spacing:2px;text-transform:uppercase">Soccer Model</div></div>',
            unsafe_allow_html=True)
        st.caption("Dixon-Coles · Elo · LightGBM ensemble · live ESPN data")
        labels = [f"{icon} {name}" for icon, name, _ in NAV]
        choice = st.radio("Navigate", labels, label_visibility="collapsed")
        page = NAV[labels.index(choice)][2]
        st.divider()
        st.markdown("**⚙️ Staking** &nbsp; <span style='color:#8b93a7;font-size:12px'>"
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
        st.divider()
        st.caption("⚠ EV/Kelly are only as good as the model's probabilities. The **min-edge "
                   "gate** drops leverage-driven longshot flags. **Not betting advice** · "
                   "independent model, not affiliated with FIFA.")

    if page == "matches":
        page_matches(bankroll, kelly, min_ev, min_edge)
    elif page == "value":
        page_value_board(bankroll, kelly, min_ev, max_exp, min_edge)
    elif page == "clv":
        page_clv(min_ev, kelly)
    elif page == "tournament":
        page_tournament()
    elif page == "performance":
        page_performance()
    else:
        page_team()


main()
