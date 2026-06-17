"""'Stadium Night' design system for the FIFA World Cup 2026 Soccer Model dashboard.

A single ``inject_css()`` call themes the whole app (fonts, palette, sidebar, nav,
cards, tables, charts), plus a set of reusable HTML component builders (hero banner,
KPI cards, pills, probability bars, callouts, footer) and a matching Altair theme.

Pure presentation — no data/model logic here.
"""
from __future__ import annotations

import altair as alt
import streamlit as st

# ----------------------------------------------------------------- palette
BG = "#0b0f19"
ELEV = "#0d1322"
CARD = "#151b2b"
CARD2 = "#1a2236"
BORDER = "rgba(255,255,255,0.08)"
TEXT = "#e6e9f0"
MUTED = "#8b93a7"
GREEN = "#1ec773"
GREEN_DIM = "#15a35e"
GOLD = "#f5b50a"
RED = "#ef4444"
BLUE = "#38bdf8"
# back-compat aliases used by the existing dashboard helpers
GREY = MUTED


def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Oswald:wght@500;600;700&display=swap');

:root {{
  --bg:{BG}; --elev:{ELEV}; --card:{CARD}; --card2:{CARD2}; --border:{BORDER};
  --text:{TEXT}; --muted:{MUTED}; --green:{GREEN}; --gold:{GOLD}; --red:{RED}; --blue:{BLUE};
}}

/* base */
.stApp {{ background:
    radial-gradient(1200px 600px at 80% -10%, rgba(30,199,115,.07), transparent 60%),
    radial-gradient(900px 500px at 0% 0%, rgba(245,181,10,.05), transparent 55%),
    var(--bg); color: var(--text); }}
html, body, [class*="css"], .stMarkdown, p, span, div, label, input, select, textarea {{
    font-family: 'Inter', system-ui, sans-serif; }}
h1, h2, h3, h4 {{ font-family:'Oswald','Inter',sans-serif; letter-spacing:.4px; color:var(--text); }}
a {{ color: var(--green); }}
hr {{ border-color: var(--border); }}
.block-container {{ padding-top: 1.2rem; max-width: 1300px; }}
#MainMenu, header [data-testid="stToolbar"], footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: var(--elev); border-right: 1px solid var(--border); }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}

/* pill nav (radio in the sidebar) */
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 4px; display:flex; flex-direction:column; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background: transparent; border: 1px solid transparent; border-radius: 10px;
    padding: 8px 12px; margin: 0; cursor: pointer; transition: all .15s ease;
    font-weight: 600; color: var(--muted); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background: var(--card); color: var(--text); }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background: linear-gradient(90deg, rgba(30,199,115,.18), rgba(30,199,115,.04));
    border-color: rgba(30,199,115,.45); color: #fff; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display:none; }}

/* metric cards (native st.metric) */
[data-testid="stMetric"] {{
    background: var(--card); border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 16px; }}
[data-testid="stMetricLabel"] {{ color: var(--muted); font-weight: 600; }}
[data-testid="stMetricValue"] {{ font-family:'Oswald'; font-weight: 700; }}

/* dataframes, expanders, inputs, buttons */
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
[data-testid="stExpander"] {{ border: 1px solid var(--border) !important; border-radius: 14px !important;
    background: var(--card); overflow: hidden; }}
[data-testid="stExpander"] summary {{ font-weight: 600; }}
.stButton > button {{ border-radius: 10px; border: 1px solid rgba(30,199,115,.4);
    background: rgba(30,199,115,.12); color: #eafff4; font-weight: 600; }}
.stButton > button:hover {{ background: rgba(30,199,115,.22); border-color: var(--green); }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--green); }}

/* scrollbar */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: #243049; border-radius: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* ---- custom components ---- */
.hero {{ position: relative; border-radius: 18px; padding: 22px 26px; margin: 2px 0 18px 0;
    background: linear-gradient(125deg, rgba(30,199,115,.16), rgba(13,19,34,.2) 42%, rgba(245,181,10,.12));
    border: 1px solid var(--border); box-shadow: 0 10px 40px rgba(0,0,0,.35); overflow: hidden; }}
.hero::after {{ content:""; position:absolute; right:-40px; top:-60px; width:220px; height:220px;
    background: radial-gradient(circle, rgba(30,199,115,.25), transparent 70%); }}
.hero .kicker {{ font-family:'Oswald'; text-transform: uppercase; letter-spacing: 2px;
    font-size: 12px; color: var(--green); font-weight: 600; }}
.hero h1 {{ font-size: 30px; margin: 2px 0 2px 0; line-height: 1.05; text-transform: uppercase; }}
.hero p {{ color: var(--muted); margin: 0; font-size: 14px; }}

.kpis {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 14px 0; }}
.kpi {{ flex: 1; min-width: 150px; background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 14px 16px; position: relative; overflow: hidden; }}
.kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--accent, var(--green)); }}
.kpi .l {{ color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing:.5px; }}
.kpi .v {{ font-family:'Oswald'; font-size: 26px; font-weight: 700; line-height: 1.1; margin-top: 2px; }}
.kpi .s {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}

.pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px;
    font-weight: 600; border: 1px solid var(--border); }}
.pill.green {{ background: rgba(30,199,115,.14); color: #8ff0c0; border-color: rgba(30,199,115,.4); }}
.pill.gold  {{ background: rgba(245,181,10,.14); color: #ffd874; border-color: rgba(245,181,10,.4); }}
.pill.red   {{ background: rgba(239,68,68,.14); color: #fca5a5; border-color: rgba(239,68,68,.4); }}
.pill.grey  {{ background: rgba(255,255,255,.06); color: var(--muted); }}

.pbar {{ display:flex; height: 10px; border-radius: 999px; overflow: hidden; margin: 8px 0 4px 0;
    border: 1px solid var(--border); }}
.pbar > span {{ display:block; height: 100%; }}
.pbar-leg {{ display:flex; justify-content: space-between; font-size: 12px; color: var(--muted); }}

.callout {{ border-radius: 12px; padding: 10px 14px; margin: 8px 0; font-size: 13px;
    border: 1px solid var(--border); background: var(--card); }}
.callout.warn {{ border-color: rgba(245,181,10,.4); background: rgba(245,181,10,.07); }}
.callout.good {{ border-color: rgba(30,199,115,.4); background: rgba(30,199,115,.07); }}
.callout.info {{ border-color: rgba(56,189,248,.35); background: rgba(56,189,248,.06); }}

.bbet {{ border: 1px solid rgba(245,181,10,.5); background: linear-gradient(90deg, rgba(245,181,10,.10), transparent);
    border-radius: 12px; padding: 12px 14px; margin-top: 8px; }}
.bbet .h {{ font-family:'Oswald'; font-size: 14px; letter-spacing: 1px; text-transform: uppercase; color: var(--gold); }}

.mcard-head {{ display:flex; align-items:center; gap: 10px; font-family:'Oswald';
    flex-wrap: wrap; justify-content: center; }}
/* top-of-page navigation radio — wrap into tidy rows, touch-friendly */
div[role="radiogroup"] {{ flex-wrap: wrap; gap: 6px 14px; }}
div[role="radiogroup"] label {{ font-family:'Oswald'; letter-spacing:.3px; }}
.foot {{ color: var(--muted); font-size: 12px; text-align: center; margin: 26px 0 6px 0;
    padding-top: 14px; border-top: 1px solid var(--border); line-height: 1.7; }}

/* ---------- mobile / narrow viewports ---------- */
@media (max-width: 640px) {{
    .block-container {{ padding-left: .6rem !important; padding-right: .6rem !important;
        padding-top: .6rem; }}
    /* stack side-by-side columns instead of cramming them */
    [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: .4rem !important; }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        flex: 1 1 100% !important; width: 100% !important; min-width: 100% !important; }}
    .hero {{ padding: 16px 16px; border-radius: 14px; }}
    .hero h1 {{ font-size: 22px; }}
    .hero p {{ font-size: 12.5px; }}
    .hero::after {{ display: none; }}              /* drop the decorative blob */
    .kpi .v {{ font-size: 21px; }}
    .mcard-head {{ font-size: 16px !important; gap: 8px; }}
    .pill {{ font-size: 11px; padding: 2px 8px; }}
    h1, h2, h3 {{ font-size: 1.15rem !important; }}
    /* let wide dataframes scroll instead of overflow */
    [data-testid="stDataFrame"] {{ overflow-x: auto; }}
    /* keep the sidebar usable when expanded */
    section[data-testid="stSidebar"] {{ min-width: 16rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------- components
def hero(title: str, subtitle: str = "", kicker: str = "FIFA World Cup 2026", icon: str = "") -> None:
    st.markdown(
        f'<div class="hero"><div class="kicker">{icon} {kicker}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


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
    draw_col = "#3a4258"
    return (
        f'<div class="pbar">'
        f'<span style="width:{h:.1f}%;background:{GREEN}"></span>'
        f'<span style="width:{d:.1f}%;background:{draw_col}"></span>'
        f'<span style="width:{a:.1f}%;background:{GOLD}"></span>'
        f'</div>'
        f'<div class="pbar-leg"><span>{home} {h:.0f}%</span>'
        f'<span>Draw {d:.0f}%</span><span>{a:.0f}% {away}</span></div>')


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
def _stadium_theme():
    base = {"labelFont": "Inter", "titleFont": "Inter",
            "labelColor": MUTED, "titleColor": MUTED}
    return {"config": {
        "background": "transparent",
        "view": {"stroke": "transparent"},
        "title": {"color": TEXT, "font": "Oswald", "fontSize": 15, "anchor": "start"},
        "axis": {**base, "gridColor": "rgba(255,255,255,0.06)", "domainColor": BORDER,
                 "tickColor": BORDER},
        "legend": base,
        "range": {"category": [GREEN, GOLD, BLUE, RED, "#a78bfa", "#f472b6"],
                  "heatmap": ["#0e1626", "#12351f", "#15703a", "#1ec773"]},
    }}


def enable_altair() -> None:
    try:
        alt.themes.register("stadium", _stadium_theme)
        alt.themes.enable("stadium")
    except Exception:  # noqa: BLE001 — charts still render with the default theme
        pass
