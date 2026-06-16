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

st.set_page_config(page_title="Badass Soccer Model", page_icon="⚽", layout="wide")
CFG = load_config()

GREEN, GREY, RED = "#2e7d32", "#9e9e9e", "#c62828"


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


def market_table(title: str, bets: list, key_note: str | None = None):
    """Render one market's selections with model%, Vegas%, price, EV, stake."""
    st.markdown(f"**{title}**")
    rows = []
    for b in bets:
        units = b.kelly_used * 100  # 1 unit = 1% of bankroll
        rows.append({
            "Selection": b.selection,
            "Model": _pct(b.model_p),
            "Vegas": _pct(b.fair_p),
            "Price": _am(b.american),
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
            st.caption(f"**{team}** · Group {sm['group']}")
            st.write(f"{sm['pos_str']}, **{sm['pts']} pts** · GD {sm['gd']:+d} "
                     f"· {sm['played']} GP")
            adv = sm["p_advance"]
            badge = ("🟢" if adv >= 0.85 else "🟡" if adv >= 0.4 else "🔴")
            st.write(f"{badge} **{adv*100:.0f}%** to reach knockouts")
            st.caption(sm["status"])
    st.divider()


def best_bet_block(m: dict, min_ev: float = 0.03):
    """One clear 'best bet' per card: the highest-EV selection in a non-disabled
    segment (spreads off by default), or an explicit 'pass' when there's no edge."""
    from src.models.segment_gate import disabled_set
    from src.predict.value import _type_key
    disabled = disabled_set(CFG)
    cands = [b for b in m["bets"]
             if b.ev is not None and b.ev >= min_ev
             and _type_key(b.market, b.selection, m["home"], m["away"]) not in disabled]
    st.divider()
    if not cands:
        st.markdown("💡 **Best bet:** _no edge — pass_")
        return
    b = max(cands, key=lambda x: x.ev)
    units = b.kelly_used * 100
    cons = (m.get("cons_edge") or {}).get(b.selection)
    cons_txt = ""
    if cons is not None:
        cons_txt = (f" · {'✅ beats consensus' if cons > 0 else '⚠️ ≤ consensus'} "
                    f"({cons*100:+.0f}%)")
    st.markdown(
        f"💡 **Best bet — {b.market}: {b.selection}** &nbsp; {_am(b.american)} &nbsp;·&nbsp; "
        f"model {_pct(b.model_p)} vs market {_pct(b.fair_p)} &nbsp;·&nbsp; "
        f"**EV {b.ev*100:+.0f}%** &nbsp;·&nbsp; stake **{units:.1f}u**{cons_txt}")
    st.caption("Highest-edge selection here (spreads & disabled segments excluded). An "
               "edge = the model disagrees with the price, **not** a guaranteed win — "
               "the Tracker page shows how these picks are actually doing.")


def render_match(m: dict, live: dict | None = None, min_ev: float = 0.03):
    a = m["analysis"]
    t = pd.to_datetime(m["date"]).strftime("%a %d %b %H:%M UTC")
    probs = a["probs"]
    pick = OUTCOMES[int(np.argmax([probs["H"], probs["D"], probs["A"]]))]
    pick_name = {"H": m["home"], "D": "Draw", "A": m["away"]}[pick]
    n_value = sum(1 for b in m["bets"] if b.ev > 0.02)
    title = (f"{m['home']}  vs  {m['away']}   ·   {t}   ·   "
             f"model leans {pick_name}" + (f"   ·   💰{n_value} value" if n_value else ""))
    with st.expander(title, expanded=False):
        move = m.get("home_line_move")
        move_txt = ""
        if move is not None and not pd.isna(move) and abs(move) >= 0.015:
            who = m["home"] if move > 0 else m["away"]
            move_txt = f" · 📈 line moving toward {who} ({abs(move)*100:.0f}%)"
        st.caption(f"Venue: {'neutral' if m['neutral'] else m['home'] + ' home'} · "
                   f"proj {a['expected_goals'][0]:.1f}–{a['expected_goals'][1]:.1f} goals · "
                   f"odds: {m['provider'] or 'n/a'}{move_txt}")
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
            st.markdown(f"**Both Teams To Score** — model **{_pct(a['btts'])}** "
                        f"(no Vegas line — info only)")
        with right:
            st.markdown("**Scoreline heatmap**")
            heatmap(a["scoreline_matrix"], m["home"], m["away"])
            tops = " · ".join(f"{s} ({p*100:.0f}%)" for s, p in a["top_scorelines"][:5])
            st.caption("Most likely: " + tops)
        best_bet_block(m, min_ev)


# --------------------------------------------------------------------- pages
def page_matches(bankroll, kelly, min_ev=0.03):
    st.header("⚽ Matches — model vs Vegas, all markets")
    c1, c2 = st.columns([1, 1])
    day = c1.date_input("From date", value=date.today())
    days = c2.slider("Days ahead", 1, 7, 3)
    res = get_bets(day.strftime("%Y-%m-%d"), days, bankroll, kelly)
    matches = res["matches"]
    if not matches:
        st.info("No upcoming fixtures with odds in this window (ESPN nulls odds once "
                "a match kicks off). Try the current World Cup dates.")
        return
    bets = res["bets"]
    n_val = int((bets["ev"] > 0.02).sum()) if not bets.empty else 0
    a, b, c = st.columns(3)
    a.metric("Fixtures", len(matches))
    b.metric("+EV selections", n_val)
    c.metric("Markets / match", "Result · Totals · Spread · BTTS")
    st.divider()
    # live 2026 group state for the "stakes" block on group-stage cards (best-effort)
    live = None
    try:
        live = get_live_state()
    except Exception:  # noqa: BLE001 — cards still render without the stakes block
        live = None
    for m in matches:
        render_match(m, live, min_ev)


def page_value_board(bankroll, kelly, min_ev, max_exposure):
    st.header("💰 Value Board — every +EV bet, ranked")
    st.caption("Stakes shown in **units (1 unit = 1% of bankroll)**, sized by fractional "
               "Kelly and **capped to your max exposure** so simultaneous bets never "
               "exceed it. EV uses the offered (vigged) price; 'Vegas' is the de-vigged "
               "fair probability.")
    day = st.date_input("From date", value=date.today(), key="vb_date")
    days = st.slider("Days ahead", 1, 7, 3, key="vb_days")
    res = get_bets(day.strftime("%Y-%m-%d"), days, bankroll, kelly)
    bb = value_mod.best_bets(res["bets"], min_ev=min_ev)
    if bb.empty:
        st.info("No bets clear the EV threshold for this window.")
        return
    bb = value_mod.cap_exposure(bb, bankroll, max_fraction=max_exposure)
    bb["units"] = bb["stake"] / (bankroll / 100.0)   # 1 unit = 1% of bankroll
    show = bb.copy()
    show["model"] = (show["model_p"] * 100).round(0).astype(int).astype(str) + "%"
    show["vegas"] = (show["fair_p"] * 100).round(0).astype(int).astype(str) + "%"
    show["EV"] = (show["ev"] * 100).round(0).astype(int).astype(str) + "%"
    show["price"] = show["american"].map(_am)
    show["stake"] = show["units"].round(2).astype(str) + "u"
    a, b, c = st.columns(3)
    a.metric("+EV bets", len(bb))
    b.metric("Total stake", f"{bb['units'].sum():.1f}u")
    c.metric("Avg edge", f"{bb['edge'].mean()*100:.1f}%")
    st.dataframe(
        show[["match", "market", "selection", "price", "model", "vegas", "EV", "stake"]],
        hide_index=True, use_container_width=True)
    st.download_button("Download CSV", bb.to_csv(index=False),
                       "value_bets.csv", "text/csv")
    st.warning("⚠ Reality check: many of these are unders/draws/underdogs — markets "
               "the model's calibration was **not** backtested against (only 1X2 RPS "
               "was). Treat large EVs on longshots with extra skepticism.")


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
    st.header("🏆 2026 World Cup — live groups, qualification & title odds")
    st.caption("Standings from results played so far; **P(advance)** and championship "
               "odds are Monte-Carlo with those results **locked in** and the rest "
               "simulated. **Team strength (Elo + Dixon-Coles) updates from 2026 form** "
               "too — a team that beats strong opponents is rated a little stronger for "
               "its remaining matches (standard Elo, no momentum-chasing). Format: top-2 "
               "of each group **plus the 8 best third-placed** teams reach the knockouts.")
    cc1, cc2 = st.columns([1, 4])
    if cc1.button("🔄 Refresh results"):
        get_live_state.clear()
    try:
        live = get_live_state()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not compute live state: {e}")
        return
    qual = live["qual"]
    cc2.caption(f"{live['n_played']} group games played and locked in.")

    # --- championship odds ---
    top = qual.sort_values("champion", ascending=False).head(20)
    chart = alt.Chart(top).mark_bar(color=GREEN).encode(
        x=alt.X("champion:Q", axis=alt.Axis(format="%"), title="Championship probability"),
        y=alt.Y("team:N", sort="-x", title=None),
        tooltip=["team", "group", alt.Tooltip("champion:Q", format=".1%"),
                 alt.Tooltip("advance:Q", format=".1%")]).properties(height=460)
    st.altair_chart(chart, use_container_width=True)

    # --- live group leaderboards with P(advance) ---
    st.subheader("Group standings & qualification odds")
    st.caption("🟢 green = top-2 (auto-qualify) · 🟠 amber = 3rd (best-third bubble) · "
               "grey = bottom. **Adv%** = model probability of reaching the knockouts.")
    adv = qual[["team", "advance", "win_group"]]
    groups = live["standings"]
    letters = list(groups)
    for r in range(0, len(letters), 3):
        cols = st.columns(3)
        for col, g in zip(cols, letters[r:r + 3]):
            with col:
                show = groups[g].merge(adv, on="team", how="left")
                show["Adv%"] = (show["advance"] * 100).round(0).astype("Int64")
                show = show[["Pos", "team", "P", "Pts", "GD", "Adv%"]]
                st.markdown(f"**Group {g}**")
                st.dataframe(show.style.apply(_group_color, axis=1),
                             hide_index=True, use_container_width=True)

    with st.expander("Full qualification & advancement table"):
        show = qual.sort_values("champion", ascending=False).copy()
        for c in ["win_group", "advance", "reach_r16", "reach_qf", "reach_sf",
                  "reach_final", "champion"]:
            if c in show:
                show[c] = (show[c] * 100).round(1)
        st.dataframe(show, use_container_width=True, hide_index=True)


def page_performance():
    st.header("📊 Model performance (backtest)")
    bt = load_csv("backtest_pooled.csv")
    if not bt.empty:
        st.subheader("Pooled over 2010–2022 World Cups (lower RPS is better)")
        st.dataframe(bt.round(4), use_container_width=True, hide_index=True)
    cal = load_csv("calibration.csv")
    if not cal.empty:
        st.subheader("Calibration — predicted vs observed")
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeDash=[4, 4], color="gray").encode(x="x", y="y")
        pts = alt.Chart(cal).mark_circle(size=90, color="#1565c0").encode(
            x=alt.X("mean_predicted:Q", scale=alt.Scale(domain=[0, 1]), title="Predicted"),
            y=alt.Y("observed_freq:Q", scale=alt.Scale(domain=[0, 1]), title="Observed"),
            size=alt.Size("n:Q", title="N"),
            tooltip=["bin", "mean_predicted", "observed_freq", "n"])
        st.altair_chart(diag + pts, use_container_width=True)
    abl = load_csv("ablation.csv")
    if not abl.empty:
        st.subheader("Ablation — does each block lower RPS?")
        st.dataframe(abl.round(4), use_container_width=True, hide_index=True)
    hist = load_csv("odds_history_backtest.csv")
    if not hist.empty:
        st.subheader("Historical betting backtest (Bet365 close, out-of-sample)")
        st.caption("Thousands of international bets graded at the closing line. An edge "
                   "is real only if a row's ROI 95% CI (roi_lo…roi_hi) is **clearly "
                   "above 0** — otherwise it's variance.")
        show = hist.copy()
        for c in ["roi", "roi_lo", "roi_hi"]:
            if c in show:
                show[c] = (show[c] * 100).round(1).astype(str) + "%"
        st.dataframe(show, use_container_width=True, hide_index=True)


def page_team():
    st.header("🔎 Team explorer & match predictor")
    pred = get_predictor()
    teams = sorted(pred.known_teams)
    c1, c2, c3 = st.columns([2, 2, 1])
    home = c1.selectbox("Home / Team A", teams,
                        index=teams.index("Brazil") if "Brazil" in teams else 0)
    away = c2.selectbox("Away / Team B", teams,
                        index=teams.index("Argentina") if "Argentina" in teams else 1)
    neutral = c3.checkbox("Neutral venue", value=True)
    if home == away:
        st.warning("Pick two different teams.")
        return
    a = pred.analyze(home, away, neutral=neutral)
    x, y, z = st.columns(3)
    x.metric(f"{home} win", _pct(a["probs"]["H"]))
    y.metric("Draw", _pct(a["probs"]["D"]))
    z.metric(f"{away} win", _pct(a["probs"]["A"]))
    st.caption(f"Expected goals {a['expected_goals'][0]:.2f}–{a['expected_goals'][1]:.2f} · "
               f"BTTS {_pct(a['btts'])}")
    st.write("**Over/Under ladder:** " + " · ".join(
        f"O{ln} {p*100:.0f}%" for ln, p in a["ou_ladder"].items()))
    left, right = st.columns([2, 3])
    with left:
        context_strip({"analysis": a, "home": home, "away": away,
                       "key_out_home": [], "key_out_away": []})
    with right:
        heatmap(a["scoreline_matrix"], home, away)


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
    st.header("📈 Live tracker — how the model's suggested bets are doing")
    st.caption(f"Every +EV pick is recorded at the offered price, then settled against the "
               f"result and the **closing line** once the match finishes. Units are staked "
               f"at **{kelly:.2f}× Kelly** (set by the sidebar slider). Auto-syncs every few "
               f"minutes; a daily scheduled job keeps it current too.")
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

    a, b, c, d = st.columns(4)
    a.metric("Settled bets", len(settled))
    b.metric("Net (units)", f"{net:+.1f}u" if len(settled) else "—",
             delta=f"{wins}-{losses}" if len(settled) else None, delta_color="off",
             help=f"At {kelly:.2f}× Kelly · {staked:.1f}u staked")
    c.metric("ROI", f"{roi*100:+.1f}%" if staked else "—", help="net ÷ staked")
    d.metric("Beat the close", f"{beat*100:.0f}%" if beat == beat else "—")

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
        st.info("No picks tracked yet. Hit **Sync now** (or wait for auto-sync) on a day "
                "with upcoming fixtures + odds to start recording the model's bets.")


PAGES = {
    "Matches": "matches", "Value Board": "value", "Tracker": "clv",
    "Tournament": "tournament", "Performance": "performance", "Team explorer": "team",
}


def main():
    st.sidebar.title("⚽ Badass Soccer Model")
    st.sidebar.caption("Ensemble (Dixon-Coles + Elo + GBM) · live odds from ESPN")
    choice = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.divider()
    st.sidebar.subheader("Staking")
    st.sidebar.caption("Stakes shown in **units** · 1 unit = 1% of bankroll")
    bankroll = st.sidebar.number_input("Bankroll ($)", 10, 1_000_000, 1000, step=50)
    kelly = st.sidebar.slider("Kelly fraction", 0.0, 1.0, 0.25, 0.05,
                              help="0.25 = quarter Kelly (default, conservative). "
                                   "0.5 = half Kelly. 1.0 = full Kelly (aggressive). "
                                   "Stakes and tracker units scale with this.")
    min_ev = st.sidebar.slider("Min EV for Value Board", 0.0, 0.30, 0.05, 0.01)
    max_exp = st.sidebar.slider("Max total exposure (× bankroll)", 0.1, 2.0, 1.0, 0.1)
    st.sidebar.divider()
    st.sidebar.caption("⚠ EV/Kelly are only as good as the model's probabilities. "
                       "Fractional Kelly + capped exposure by default. Not betting "
                       "advice. Odds shown for upcoming matches only.")

    page = PAGES[choice]
    if page == "matches":
        page_matches(bankroll, kelly, min_ev)
    elif page == "value":
        page_value_board(bankroll, kelly, min_ev, max_exp)
    elif page == "clv":
        page_clv(min_ev, kelly)
    elif page == "tournament":
        page_tournament()
    elif page == "performance":
        page_performance()
    else:
        page_team()


main()
