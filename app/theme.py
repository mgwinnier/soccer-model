"""Modern, minimal design system for the FIFA World Cup 2026 Soccer Model dashboard.

A single ``inject_css()`` themes the whole app (fonts, palette, sidebar, nav, cards, tables, charts),
plus reusable HTML component builders (page header, KPI cards, pills, probability bars, section
headers, notes, callouts, footer) and a matching Altair theme.

Flat graphite surfaces, one emerald accent, Space Grotesk + Inter type. Pure presentation — no
data/model logic here. Constant NAMES and component signatures are stable so the dashboard inherits
the new skin automatically.
"""
from __future__ import annotations

import altair as alt
import streamlit as st

# ----------------------------------------------------------------- palette (graphite + one accent)
BG = "#0b0c0e"          # near-black base
ELEV = "#0e1014"        # sidebar / elevated chrome
CARD = "#15171c"        # card surface
CARD2 = "#1b1e25"       # secondary surface (key-number boxes)
BORDER = "rgba(255,255,255,0.08)"
TEXT = "#e8eaed"
MUTED = "#969aa6"
GREEN = "#34d39a"       # emerald accent (positive EV / model)
GREEN_DIM = "#10b981"
GOLD = "#f5a524"        # restrained amber — warnings / away / value (no gold gradients)
RED = "#f04438"         # negative
BLUE = "#5b8def"        # info
GREY = MUTED            # back-compat alias

_DISPLAY = "'Space Grotesk','Inter',sans-serif"


def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
  --bg:{BG}; --elev:{ELEV}; --card:{CARD}; --card2:{CARD2}; --border:{BORDER};
  --text:{TEXT}; --muted:{MUTED}; --green:{GREEN}; --gold:{GOLD}; --red:{RED}; --blue:{BLUE};
}}

/* base — flat, one very faint top accent */
.stApp {{ background:
    radial-gradient(900px 360px at 50% -120px, rgba(52,211,154,.05), transparent 70%),
    var(--bg); color: var(--text); }}
html, body, [class*="css"], .stMarkdown, p, span, div, label, input, select, textarea {{
    font-family: 'Inter', system-ui, sans-serif; }}
h1, h2, h3, h4 {{ font-family:{_DISPLAY}; letter-spacing:-.01em; color:var(--text); font-weight:600; }}
a {{ color: var(--green); }}
hr {{ border-color: var(--border); }}
.block-container {{ padding-top: 1.4rem; max-width: 1280px; }}
#MainMenu, header [data-testid="stToolbar"], footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: var(--elev); border-right: 1px solid var(--border); }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 3px; display:flex; flex-direction:column; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background: transparent; border: 1px solid transparent; border-radius: 9px;
    padding: 8px 12px; margin: 0; cursor: pointer; transition: all .14s ease;
    font-weight: 500; color: var(--muted); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background: rgba(255,255,255,.04); color: var(--text); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background: rgba(52,211,154,.12); border-color: rgba(52,211,154,.35); color: #fff; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display:none; }}

/* metric cards (native st.metric) */
[data-testid="stMetric"] {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }}
[data-testid="stMetricLabel"] {{ color: var(--muted); font-weight: 600; }}
[data-testid="stMetricValue"] {{ font-family:{_DISPLAY}; font-weight: 600; letter-spacing:-.01em; }}

/* dataframes, expanders, inputs, buttons */
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
[data-testid="stExpander"] {{ border: 1px solid var(--border) !important; border-radius: 12px !important;
    background: var(--card); overflow: hidden; }}
[data-testid="stExpander"] summary {{ font-weight: 600; }}
.stButton > button {{ border-radius: 9px; border: 1px solid var(--border);
    background: var(--card2); color: var(--text); font-weight: 600; transition: all .14s ease; }}
.stButton > button:hover {{ background: rgba(52,211,154,.12); border-color: rgba(52,211,154,.4);
    color: #fff; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--green); }}

/* scrollbar */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: #2a2e38; border-radius: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* ---- custom components ---- */
/* page header — flat, hairline rule (no gradient banner) */
.hero {{ padding: 4px 2px 14px; margin: 2px 0 18px; border-bottom: 1px solid var(--border); }}
.hero .kicker {{ font-family:{_DISPLAY}; text-transform: uppercase; letter-spacing: 1.6px;
    font-size: 11px; color: var(--green); font-weight: 600; }}
.hero h1 {{ font-size: 26px; margin: 4px 0 3px; line-height: 1.1; letter-spacing:-.02em; }}
.hero p {{ color: var(--muted); margin: 0; font-size: 13.5px; max-width: 760px; }}

/* section header — the single section style across the app */
.sec {{ display:flex; align-items:flex-end; justify-content:space-between; gap:12px;
    margin: 18px 0 10px; }}
.sec-t {{ font-family:{_DISPLAY}; font-size: 16px; font-weight: 600; letter-spacing:-.01em;
    color: var(--text); display:flex; align-items:center; gap:8px; }}
.sec-t::before {{ content:""; width:3px; height:14px; border-radius:2px; background:var(--green);
    display:inline-block; }}
.sec-sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
.sec-right {{ color: var(--muted); font-size: 12px; white-space:nowrap; }}

/* note — tidy explanatory block (replaces overloaded captions) */
.note {{ border-left: 2px solid var(--border); padding: 2px 0 2px 11px; margin: 6px 0 10px;
    color: var(--muted); font-size: 12.5px; line-height: 1.55; }}
.note.good {{ border-left-color: rgba(52,211,154,.5); }}
.note.warn {{ border-left-color: rgba(245,165,36,.5); }}

.kpis {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 14px 0; }}
.kpi {{ flex: 1; min-width: 150px; background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 16px; position: relative; overflow: hidden; }}
.kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:2px; background: var(--accent, var(--green)); }}
.kpi .l {{ color: var(--muted); font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing:.5px; }}
.kpi .v {{ font-family:{_DISPLAY}; font-size: 25px; font-weight: 600; line-height: 1.1; margin-top: 3px; letter-spacing:-.01em; }}
.kpi .s {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}

.pill {{ display: inline-block; padding: 3px 10px; border-radius: 7px; font-size: 12px;
    font-weight: 600; border: 1px solid var(--border); }}
.pill.green {{ background: rgba(52,211,154,.13); color: #7ff0c6; border-color: rgba(52,211,154,.35); }}
.pill.gold  {{ background: rgba(245,165,36,.13); color: #ffce7a; border-color: rgba(245,165,36,.35); }}
.pill.red   {{ background: rgba(240,68,56,.13); color: #ffa9a2; border-color: rgba(240,68,56,.35); }}
.pill.grey  {{ background: rgba(255,255,255,.05); color: var(--muted); }}

.pbar {{ display:flex; height: 8px; border-radius: 6px; overflow: hidden; margin: 8px 0 4px 0;
    border: 1px solid var(--border); }}
.pbar > span {{ display:block; height: 100%; }}
.pbar-leg {{ display:flex; justify-content: space-between; font-size: 12px; color: var(--muted); }}

.callout {{ border-radius: 10px; padding: 10px 14px; margin: 8px 0; font-size: 13px;
    border: 1px solid var(--border); background: var(--card); }}
.callout.warn {{ border-color: rgba(245,165,36,.4); background: rgba(245,165,36,.06); }}
.callout.good {{ border-color: rgba(52,211,154,.4); background: rgba(52,211,154,.06); }}
.callout.info {{ border-color: rgba(91,141,239,.35); background: rgba(91,141,239,.06); }}

.bbet {{ border: 1px solid rgba(52,211,154,.4); background: rgba(52,211,154,.06);
    border-radius: 10px; padding: 12px 14px; margin-top: 8px; }}
.bbet .h {{ font-family:{_DISPLAY}; font-size: 12px; letter-spacing: .8px; text-transform: uppercase; color: var(--green); }}

.angle {{ border: 1px solid var(--border); border-radius: 10px; padding: 8px 11px; margin: 6px 0;
    background: rgba(255,255,255,.015); }}
.angle-h {{ font-size: 13px; }}
.angle-why {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
.angle-note {{ font-size: 12px; color: var(--muted); margin: 2px 0 8px; }}

.mcard-head {{ display:flex; align-items:center; gap: 10px; font-family:{_DISPLAY};
    flex-wrap: wrap; justify-content: center; }}

/* ---- match card ---- */
.mc2 {{ display:flex; align-items:center; justify-content:space-between; gap:10px; padding:6px 2px 12px; }}
.mc2-team {{ display:flex; align-items:center; gap:9px; font-family:{_DISPLAY};
    font-size:18px; font-weight:600; color:var(--text); min-width:0; flex:1; letter-spacing:-.01em; }}
.mc2-team.away {{ justify-content:flex-end; text-align:right; }}
.mc2-team img {{ width:28px; height:auto; border-radius:3px; }}
.mc2-mid {{ display:flex; flex-direction:column; align-items:center; min-width:96px; gap:3px; }}
.mc2-score {{ font-family:{_DISPLAY}; font-weight:600; font-size:25px; line-height:1; color:#fff; letter-spacing:-.01em; }}
.mc2-vs {{ font-family:{_DISPLAY}; font-size:12px; color:var(--muted); letter-spacing:1px; text-transform:uppercase; }}
.mc2-time {{ font-size:11.5px; color:var(--muted); white-space:nowrap; }}

.keynum {{ display:flex; gap:8px; flex-wrap:wrap; margin:4px 0 10px; }}
.keynum .kn {{ flex:1; min-width:84px; background:var(--card2); border:1px solid var(--border);
    border-radius:10px; padding:8px 10px; text-align:center; }}
.keynum .kn .l {{ color:var(--muted); font-size:10.5px; font-weight:600; text-transform:uppercase; letter-spacing:.5px; }}
.keynum .kn .v {{ font-family:{_DISPLAY}; font-size:18px; font-weight:600; margin-top:2px; line-height:1.05; letter-spacing:-.01em; }}

.sbrow {{ margin:7px 0; }}
.sbtop {{ display:flex; justify-content:space-between; align-items:baseline; font-size:12.5px; }}
.sbtop .sbv {{ font-weight:700; color:var(--text); font-variant-numeric:tabular-nums; }}
.sbtop .sblab {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.4px; }}
.sbbar {{ display:flex; height:6px; border-radius:999px; overflow:hidden; margin-top:4px; background:rgba(255,255,255,.05); }}
.sbbar > span {{ display:block; height:100%; }}

/* ============ top navigation bar ============ */
.topnav {{ display:flex; align-items:center; justify-content:space-between; gap:14px;
    padding:9px 14px; margin:-6px 0 18px; position:sticky; top:0; z-index:1000;
    background:rgba(11,12,14,.82); backdrop-filter:saturate(140%) blur(10px);
    border:1px solid var(--border); border-radius:12px; }}
.navbrand {{ font-family:{_DISPLAY}; font-weight:700; font-size:17px; letter-spacing:-.01em;
    color:var(--text); text-decoration:none; white-space:nowrap; display:flex; align-items:center; gap:7px; }}
.navbrand b {{ color:var(--green); }}
.navlinks {{ display:flex; flex-wrap:wrap; gap:4px; }}
.navitem {{ font-family:{_DISPLAY}; font-size:13px; font-weight:500;
    color:var(--muted); text-decoration:none; padding:7px 13px; border-radius:8px;
    border:1px solid transparent; transition:all .14s ease; white-space:nowrap; }}
.navitem:hover {{ color:var(--text); background:rgba(255,255,255,.05); }}
.navitem.active {{ color:#08130d; font-weight:600; background:var(--green); }}
/* hamburger button (CSS :target toggle — no JS) */
.navham-btn {{ display:none; cursor:pointer; font-size:22px; line-height:1; color:var(--text);
    padding:7px 13px; border:1px solid var(--border); border-radius:9px; background:var(--card);
    text-decoration:none; user-select:none; }}
.navham-btn:hover {{ color:var(--green); border-color:var(--green); }}
.navdrawer {{ display:none; }}
.navdrawer-head {{ display:flex; align-items:center; justify-content:space-between;
    width:100%; max-width:440px; margin-bottom:10px; }}
.navdrawer-close {{ font-size:22px; color:var(--muted); text-decoration:none; padding:6px 12px;
    border:1px solid var(--border); border-radius:9px; }}
.navdrawer .navitem {{ font-size:17px; padding:13px 18px; width:100%; max-width:440px;
    text-align:center; background:var(--card); border:1px solid var(--border); }}
@media (max-width:820px) {{
    .navlinks {{ display:none; }}
    .navham-btn {{ display:inline-flex; align-items:center; }}
    .navbrand {{ font-size:15px; }}
    .topnav {{ padding:8px 12px; }}
    .navdrawer:target {{ display:flex; flex-direction:column; align-items:center;
        position:fixed; inset:0; z-index:100000; padding:18px;
        gap:10px; background:rgba(8,9,11,.985); backdrop-filter:blur(6px); overflow:auto; }}
}}
.foot {{ color: var(--muted); font-size: 12px; text-align: center; margin: 26px 0 6px 0;
    padding-top: 14px; border-top: 1px solid var(--border); line-height: 1.7; }}

/* ---------- mobile / narrow viewports ---------- */
@media (max-width: 640px) {{
    .block-container {{ padding-left: .6rem !important; padding-right: .6rem !important; padding-top: .6rem; }}
    [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: .4rem !important; }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important; }}
    .hero h1 {{ font-size: 21px; }}
    .hero p {{ font-size: 12.5px; }}
    .kpi .v {{ font-size: 21px; }}
    .mcard-head {{ font-size: 16px !important; gap: 8px; }}
    .pill {{ font-size: 11px; padding: 2px 8px; }}
    h1, h2, h3 {{ font-size: 1.15rem !important; }}
    [data-testid="stDataFrame"] {{ overflow-x: auto; }}
}}

[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {{
    display: flex !important; visibility: visible !important; opacity: 1 !important;
    z-index: 2147483000 !important; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------- components
def hero(title: str, subtitle: str = "", kicker: str = "FIFA World Cup 2026") -> None:
    st.markdown(
        f'<div class="hero"><div class="kicker">{kicker}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(title: str, sub: str | None = None, right: str | None = None) -> None:
    """One consistent section header: accent tick + title, optional subtitle + right-aligned meta."""
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    right_html = f'<div class="sec-right">{right}</div>' if right else ""
    st.markdown(f'<div class="sec"><div><div class="sec-t">{title}</div>{sub_html}</div>'
                f'{right_html}</div>', unsafe_allow_html=True)


def note(text: str, tone: str = "muted") -> None:
    """A tidy muted explanatory block (replaces overloaded st.caption for multi-line text)."""
    st.markdown(f'<div class="note {tone}">{text}</div>', unsafe_allow_html=True)


def kpi_row(cards: list[dict]) -> None:
    """cards: list of {label, value, sub?, accent?}."""
    html = '<div class="kpis">'
    for c in cards:
        accent = c.get("accent", GREEN)
        sub = f'<div class="s">{c["sub"]}</div>' if c.get("sub") else ""
        html += (f'<div class="kpi" style="--accent:{accent}">'
                 f'<div class="l">{c["label"]}</div>'
                 f'<div class="v" style="color:{c.get("value_color", TEXT)}">{c["value"]}</div>'
                 f'{sub}</div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def pill(text: str, tone: str = "grey") -> str:
    return f'<span class="pill {tone}">{text}</span>'


def prob_bar(p_h: float, p_d: float, p_a: float, home: str, away: str) -> str:
    h, d, a = (max(0.0, float(x)) for x in (p_h, p_d, p_a))
    tot = (h + d + a) or 1.0
    h, d, a = 100 * h / tot, 100 * d / tot, 100 * a / tot
    draw_col = "#3a3f4b"
    return (
        f'<div class="pbar">'
        f'<span style="width:{h:.1f}%;background:{GREEN}"></span>'
        f'<span style="width:{d:.1f}%;background:{draw_col}"></span>'
        f'<span style="width:{a:.1f}%;background:{GOLD}"></span>'
        f'</div>'
        f'<div class="pbar-leg"><span>{home} {h:.0f}%</span>'
        f'<span>Draw {d:.0f}%</span><span>{a:.0f}% {away}</span></div>')


def match_header(home_html: str, away_html: str, center: str, sub: str = "") -> str:
    """Card header: home (crest+name) · center score/vs · away. ``center`` is the score or 'vs';
    ``sub`` the kickoff time / status."""
    return (f'<div class="mc2"><div class="mc2-team home">{home_html}</div>'
            f'<div class="mc2-mid"><div class="mc2-score">{center}</div>'
            f'<div class="mc2-time">{sub}</div></div>'
            f'<div class="mc2-team away">{away_html}</div></div>')


def key_numbers(items: list[dict]) -> str:
    """Strip of KPI boxes. items: [{label, value, color?}]."""
    cells = "".join(
        f'<div class="kn"><div class="l">{c["label"]}</div>'
        f'<div class="v" style="color:{c.get("color", TEXT)}">{c["value"]}</div></div>'
        for c in items)
    return f'<div class="keynum">{cells}</div>'


def stat_bars(rows: list[dict], home_color: str = GREEN, away_color: str = GOLD) -> str:
    """Comparison bars. rows: [{label, home, away, disp_home?, disp_away?}]."""
    out = []
    for r in rows:
        h, a = float(r.get("home") or 0), float(r.get("away") or 0)
        tot = (h + a) or 1.0
        hp = 100 * h / tot
        dh = r.get("disp_home", r["home"])
        da = r.get("disp_away", r["away"])
        out.append(
            f'<div class="sbrow"><div class="sbtop"><span class="sbv">{dh}</span>'
            f'<span class="sblab">{r["label"]}</span><span class="sbv">{da}</span></div>'
            f'<div class="sbbar"><span style="width:{hp:.0f}%;background:{home_color}"></span>'
            f'<span style="width:{100 - hp:.0f}%;background:{away_color}"></span></div></div>')
    return "".join(out)


_READ_TONE = {"support": "green", "undercut": "red", "neutral": "grey"}
_READ_LABEL = {"support": "supports", "undercut": "undercuts", "neutral": "neutral"}


def angle_chip(market: str, lean: str, read: str, why: str) -> str:
    """One AI betting angle as a compact row: market · lean · support/undercut pill · grounded why."""
    tone = _READ_TONE.get(read, "grey")
    label = _READ_LABEL.get(read, read)
    head = f"<b>{market}</b>" + (f" · {lean}" if lean else "")
    return (f'<div class="angle"><div class="angle-h">{head} &nbsp;{pill(label, tone)}</div>'
            f'<div class="angle-why">{why}</div></div>')


def callout(text: str, tone: str = "info") -> None:
    st.markdown(f'<div class="callout {tone}">{text}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="foot">'
        '<b>FIFA World Cup 2026 · Soccer Model</b> — Dixon-Coles · Elo · LightGBM ensemble · '
        'live data from ESPN<br>'
        'Independent project · <b>not affiliated with or endorsed by FIFA</b> · '
        'for analysis &amp; entertainment, <b>not betting advice</b>.'
        '</div>', unsafe_allow_html=True)


# ------------------------------------------------------------- altair theme
def _modern_theme():
    base = {"labelFont": "Inter", "titleFont": "Inter", "labelColor": MUTED, "titleColor": MUTED}
    return {"config": {
        "background": "transparent",
        "view": {"stroke": "transparent"},
        "title": {"color": TEXT, "font": "Space Grotesk", "fontSize": 15, "anchor": "start"},
        "axis": {**base, "gridColor": "rgba(255,255,255,0.05)", "domainColor": BORDER,
                 "tickColor": BORDER},
        "legend": base,
        "range": {"category": [GREEN, GOLD, BLUE, RED, "#a78bfa", "#f472b6"],
                  "heatmap": ["#11131a", "#123329", "#15703a", GREEN]},
    }}


def enable_altair() -> None:
    try:
        alt.themes.register("modern", _modern_theme)
        alt.themes.enable("modern")
    except Exception:  # noqa: BLE001 — charts still render with the default theme
        pass
