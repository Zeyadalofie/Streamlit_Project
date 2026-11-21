"""Shared look for both pages: CSS, chart style and the sidebar header.

Optional: app.py runs without this file (for example in the notebook), just unstyled.
"""
import matplotlib as mpl
import streamlit as st

NAVY, PURPLE, RED, MUTED, LINE = "#1B2A41", "#4F29B7", "#C0392B", "#5B6577", "#E2E6EE"

CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}
h1 {{ letter-spacing: -0.02em; }}

/* indicator cards */
div[data-testid="stMetric"] {{
    background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px;
    padding: 16px 18px 14px; box-shadow: 0 1px 3px rgba(27, 42, 65, 0.06);
}}
div[data-testid="stMetricLabel"] p {{
    color: {MUTED}; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
}}
div[data-testid="stMetricValue"] {{ color: {NAVY}; }}

/* tabs */
div[data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
button[data-baseweb="tab"] {{ padding: 8px 16px; border-radius: 10px 10px 0 0; }}
button[data-baseweb="tab"] p {{ font-weight: 600; font-size: 0.95rem; }}
button[data-baseweb="tab"][aria-selected="true"] {{ background: #F4F6FA; }}

/* sidebar header */
.brand {{ display: flex; align-items: center; gap: 10px; margin: 0 0 6px; }}
.brand .logo {{
    width: 38px; height: 38px; border-radius: 10px; background: {NAVY}; color: #FFFFFF;
    display: flex; align-items: center; justify-content: center; font-size: 20px;
}}
.brand .name {{ font-weight: 700; color: {NAVY}; line-height: 1.15; }}
.brand .sub {{ font-size: 0.78rem; color: {MUTED}; }}

/* small coloured label, used for the evidence level */
.pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }}
</style>
"""


def apply_style():
    """Inject the CSS and set a clean Matplotlib style for every chart drawn after this call."""
    st.html(CSS)
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#CBD2DC", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#EEF1F6", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.titlecolor": NAVY, "axes.labelcolor": MUTED, "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "font.size": 10,
    })


def sidebar_brand(section=None):
    """Logo, named links to both pages and (optionally) a section heading, at the top of the sidebar."""
    with st.sidebar:
        st.html('<div class="brand"><div class="logo">🚓</div><div><div class="name">Police Requests</div>'
                '<div class="sub">Operations office · Q1 2025</div></div></div>')
        try:
            st.page_link("app.py", label="Dashboard", icon=":material/dashboard:")
            st.page_link("pages/ai_predictor.py", label="AI Wait Predictor", icon=":material/smart_toy:")
        except Exception:   # page links need the multipage setup; skip them if it is missing
            pass
        if section:
            st.markdown(f"**{section}**")


def evidence_pill(level):
    """Coloured HTML label for weak / moderate / strong."""
    colors = {"strong": ("#E6F4EC", "#1E7B4F"), "moderate": ("#FFF4DE", "#9A6200"), "weak": ("#FDECEA", RED)}
    bg, fg = colors.get(level, ("#EEF1F6", NAVY))
    return f'<span class="pill" style="background:{bg};color:{fg}">{level.upper()}</span>'
