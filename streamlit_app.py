"""
AB Audit — Statistical Validity Engine
Dark dashboard UI matching the Analytycs reference design.
Run: streamlit run app/streamlit_app.py  (from ab_audit/ root)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from engine import ExperimentConfig, MetricType, Severity
from engine.checks import (
    run_full_audit, _compute_power_proportion, _required_n_proportion,
)
from engine.simulation import (
    run_peeking_simulation, simulate_null_trajectories,
)
from engine.cuped import run_cuped_analysis
from engine.data_generator import load_scenario

# ══════════════════════════════════════════════════════════════
# TOKENS  — match reference screenshot exactly
# ══════════════════════════════════════════════════════════════
BG       = "#1C1C1E"
CARD     = "#2A2A2D"
CARD2    = "#333336"
BORDER   = "#3A3A3E"
TXT      = "#FFFFFF"
TXT2     = "#A0A0A8"
TXT3     = "#5A5A62"

MINT     = "#4ECDC4"
PURPLE   = "#A78BFA"
YELLOW   = "#FCD34D"
CORAL    = "#FB7185"
BLUE     = "#60A5FA"
ORANGE   = "#FB923C"

PASS_C   = MINT
WARN_C   = YELLOW
FAIL_C   = CORAL

st.set_page_config(
    page_title="AB Audit",
    page_icon="AB",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {{
    background: {BG} !important;
    font-family: 'Inter', sans-serif !important;
    color: {TXT} !important;
}}

/* ── Hide ALL streamlit chrome completely ─────────────── */
#MainMenu,
footer,
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
[data-testid="manage-app-button"],
.stDeployButton {{
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
}}

/* ── Hide sidebar collapse/expand toggle permanently ─── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[aria-label="Close sidebar"],
button[kind="header"] {{
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}}

/* ── Force sidebar always visible and non-collapsible ── */
[data-testid="stSidebar"] {{
    transform: none !important;
    visibility: visible !important;
    display: flex !important;
    width: 16rem !important;
    min-width: 16rem !important;
    max-width: 16rem !important;
}}

/* ── Remove top padding gap that header normally creates ─ */
[data-testid="stAppViewContainer"] {{
    padding-top: 0 !important;
}}
[data-testid="stMain"] {{
    padding-top: 0 !important;
}}

/* ── SIDEBAR ─────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {CARD} !important;
    border-right: 1px solid {BORDER} !important;
    padding-top: 0 !important;
}}
[data-testid="stSidebar"] * {{ color: {TXT} !important; }}
section[data-testid="stSidebar"] > div {{
    padding-top: 1.2rem !important;
}}

/* ── RADIO (nav) — hide radio circles, style as nav rows ── */
[data-testid="stRadio"] > div {{
    gap: 2px !important;
    flex-direction: column !important;
}}
/* hide the radio button circle */
[data-testid="stRadio"] [data-testid="stWidgetLabel"],
[data-testid="stRadio"] > div > label > div:first-child {{
    display: none !important;
}}
[data-testid="stRadio"] label {{
    background: transparent !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: {TXT2} !important;
    cursor: pointer !important;
    transition: background .15s, color .15s !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
    margin: 0 !important;
}}
[data-testid="stRadio"] label > div:last-child {{
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}}
[data-testid="stRadio"] label:hover {{
    background: {CARD2} !important;
    color: {TXT} !important;
}}
[data-testid="stRadio"] label[data-checked="true"],
[data-testid="stRadio"] input:checked ~ div {{
    background: {CARD2} !important;
    color: {TXT} !important;
}}

/* ── INPUTS ───────────────────────────────────────────── */
input, textarea,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {{
    background: {CARD2} !important;
    color: {TXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}}
[data-testid="stSelectbox"] > div > div {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TXT} !important;
}}


/* ── MULTISELECT — remove blob, clean flat tags ──────────── */
[data-testid="stMultiSelect"] {{
    background: transparent !important;
}}
[data-testid="stMultiSelect"] > div {{
    background: #333336 !important;
    border: 1px solid #3A3A3E !important;
    border-radius: 10px !important;
}}
/* individual selected tag */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
    background: #2A2A2D !important;
    border: 1px solid #3A3A3E !important;
    border-radius: 6px !important;
    color: #FFFFFF !important;
    font-size: 12px !important;
    padding: 2px 8px !important;
    margin: 2px !important;
    height: auto !important;
    line-height: 1.5 !important;
}}
/* remove the large oval/blob shape */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] > span:first-child {{
    background: none !important;
    padding: 0 !important;
    border-radius: 0 !important;
}}
/* the X button inside the tag */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] svg {{
    width: 12px !important;
    height: 12px !important;
    color: #A0A0A8 !important;
}}
/* the clear-all X button */
[data-testid="stMultiSelect"] div[role="combobox"] + div svg {{
    color: #A0A0A8 !important;
}}

/* ── SLIDERS ─────────────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stTickBar"] span {{
    color: {TXT3} !important;
    font-size: 11px !important;
}}
div[class*="StyledThumb"] {{
    background: {MINT} !important;
    border: 2px solid {MINT} !important;
}}

/* ── BUTTONS ──────────────────────────────────────────── */
.stButton > button {{
    background: {CARD2} !important;
    color: {TXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.6rem !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: .01em !important;
    transition: all .18s ease !important;
}}
.stButton > button:hover {{
    background: {MINT}22 !important;
    border-color: {MINT}66 !important;
    color: {MINT} !important;
    transform: translateY(-1px) !important;
}}

/* ── METRICS ──────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {CARD} !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.4rem !important;
    border: 1px solid {BORDER} !important;
}}
[data-testid="stMetricLabel"] > div {{
    color: {TXT2} !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: .03em !important;
}}
[data-testid="stMetricValue"] > div {{
    color: {TXT} !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: -.5px !important;
}}
[data-testid="stMetricDelta"] > div {{
    font-size: 12px !important;
    font-weight: 500 !important;
}}

/* ── EXPANDERS ────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    margin-bottom: 6px !important;
    overflow: hidden !important;
}}
[data-testid="stExpander"] summary {{
    padding: 0.9rem 1.2rem !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}}

/* ── TABS ────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {{
    background: {CARD} !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid {BORDER} !important;
    width: fit-content !important;
}}
[data-testid="stTabs"] [role="tab"] {{
    border-radius: 8px !important;
    padding: 6px 16px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: {TXT2} !important;
    border: none !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    background: {CARD2} !important;
    color: {TXT} !important;
}}

/* ── FILE UPLOADER ───────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background: {CARD} !important;
    border: 1.5px dashed {BORDER} !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}}
[data-testid="stFileUploader"] * {{ color: {TXT2} !important; }}

/* ── DATAFRAME ───────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid {BORDER} !important;
}}

/* ── SCROLLBAR ───────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

/* ── MAIN CONTENT PADDING ────────────────────────────── */
.main .block-container {{
    padding: 1.8rem 2.2rem 2rem !important;
    max-width: 1400px !important;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """Strip em dashes from engine-generated text, replace with cleaner punctuation."""
    if not text:
        return text
    import re
    # em dash with spaces around it -> comma or colon depending on context
    text = re.sub(r'\s*—\s*', ', ', text)
    return text

def _chart(fig, height=300):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=TXT2, size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=TXT2, size=11)),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=TXT2, size=11)),
    )
    return fig


def kpi_card(label, value, delta_text="", delta_up=True, icon="", icon_bg=MINT):
    arrow = "↑" if delta_up else "↓"
    delta_color = MINT if delta_up else CORAL
    delta_html = (
        f'<div style="margin-top:10px;font-size:12px;font-weight:500;">'
        f'<span style="color:{delta_color};">{arrow} {delta_text}</span>'
        f'</div>'
    ) if delta_text else ""
    icon_html = (
        f'<div style="background:{icon_bg}22;border-radius:12px;'
        f'padding:10px;width:44px;height:44px;display:flex;'
        f'align-items:center;justify-content:center;'
        f'flex-shrink:0;">{icon}</div>'
    ) if icon else ""
    st.markdown(f"""
    <div style="background:{CARD};border-radius:16px;padding:1.4rem 1.6rem;
                border:1px solid {BORDER};height:100%;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <div style="font-size:12px;color:{TXT2};font-weight:500;
                            letter-spacing:.04em;margin-bottom:10px;">{label}</div>
                <div style="font-size:32px;font-weight:800;color:{TXT};
                            letter-spacing:-.8px;line-height:1;">{value}</div>
                {delta_html}
            </div>
            {icon_html}
        </div>
    </div>""", unsafe_allow_html=True)


def section_label(text, color=MINT):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:9px;margin:1.8rem 0 1rem;">
        <div style="width:3px;height:18px;background:{color};border-radius:2px;flex-shrink:0;"></div>
        <span style="font-size:15px;font-weight:700;color:{TXT};letter-spacing:-.2px;">{text}</span>
    </div>""", unsafe_allow_html=True)


def badge(sev):
    c = {Severity.PASS: MINT, Severity.WARN: YELLOW, Severity.FAIL: CORAL}[sev]
    icons = {Severity.PASS: "✓", Severity.WARN: "⚠", Severity.FAIL: "✗"}
    labels = {Severity.PASS: "PASS", Severity.WARN: "WARN", Severity.FAIL: "FAIL"}
    return (f'<span style="background:{c}1a;color:{c};border:1px solid {c}44;'
            f'border-radius:6px;padding:2px 9px;font-size:10px;font-weight:700;'
            f'letter-spacing:.06em;">{icons[sev]} {labels[sev]}</span>')


def page_header(title, subtitle, crumb="Home"):
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;
                align-items:flex-start;margin-bottom:1.8rem;">
        <div>
            <div style="font-size:11px;color:{TXT3};font-weight:500;
                        letter-spacing:.06em;text-transform:uppercase;
                        margin-bottom:6px;">
                Dashboard &nbsp;/&nbsp; <span style="color:{TXT2};">{crumb}</span>
            </div>
            <h1 style="font-size:26px;font-weight:800;color:{TXT};
                       letter-spacing:-.6px;margin:0 0 4px;">{title}</h1>
            <p style="font-size:13px;color:{TXT2};margin:0;">{subtitle}</p>
        </div>
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;
                    padding:7px 14px;font-size:12px;color:{TXT2};
                    display:flex;align-items:center;gap:6px;cursor:pointer;">
            <span style="color:{TXT3};">●</span>
            <span>AB Audit <span style="color:{MINT};font-weight:600;">v1.0</span></span>
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:1rem 1rem 1.4rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <div style="background:{MINT}22;border-radius:10px;width:36px;height:36px;
                        display:flex;align-items:center;justify-content:center;">
                <svg width='18' height='18' viewBox='0 0 24 24' fill='none'
                     stroke='{MINT}' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'>
                    <path d='M9 3v8L4.5 18.5A2 2 0 0 0 6.3 21h11.4a2 2 0 0 0 1.8-2.5L15 11V3'/>
                    <line x1='9' y1='3' x2='15' y2='3'/>
                    <line x1='8' y1='7' x2='16' y2='7'/>
                </svg>
            </div>
            <div>
                <div style="font-size:15px;font-weight:700;color:{TXT};letter-spacing:-.2px;">
                    AB Audit
                </div>
                <div style="font-size:11px;color:{TXT3};">Validity Engine</div>
            </div>
        </div>
    </div>
    <div style="height:1px;background:{BORDER};margin:0 0 .8rem;"></div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="padding:0 .4rem .4rem;font-size:11px;color:#5A5A62;
                font-weight:600;letter-spacing:.06em;text-transform:uppercase;
                margin-bottom:.4rem;">
        Navigation
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("nav", [
        "Overview",
        "Experiment Designer",
        "Validity Audit",
        "Peeking Simulator",
        "CUPED Engine",
    ], label_visibility="collapsed")

    _status_rows = "".join([
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:.35rem 0;border-bottom:1px solid {BORDER}22;">'
        f'<span style="font-size:11px;color:{TXT2};">{k}</span>'
        f'<span style="font-size:11px;color:{MINT};font-weight:600;">{v}</span></div>'
        for k, v in [("Checks", "8 / 8"), ("Unit tests", "53 / 53"),
                     ("Scenarios", "3 loaded"), ("CUPED", "Active")]
    ])
    st.markdown(
        f'<div style="height:1px;background:{BORDER};margin:1rem 0;"></div>'
        f'<div style="padding:0 .6rem;">'
        f'<div style="font-size:11px;color:{TXT3};font-weight:600;'
        f'letter-spacing:.06em;text-transform:uppercase;margin-bottom:.8rem;">'
        f'Engine Status</div>'
        f'{_status_rows}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "Overview":
    page_header("AB Audit", "Statistical validity engine for A/B experiments", "Overview")

    # KPI row
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: kpi_card("Validity Checks", "8", "All implemented", True, "<svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='9 11 12 14 22 4'/><path d='M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11'/></svg>", MINT)
    with c2: kpi_card("Unit Tests", "53/53", "100% passing", True, "<svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='9'/><polyline points='8.5 12.5 11 15 15.5 9.5'/></svg>", BLUE)
    with c3: kpi_card("Monte Carlo Sims", "10k", "Per run", True, "<svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='22 12 18 12 15 21 9 3 6 12 2 12'/></svg>", PURPLE)
    with c4: kpi_card("Demo Scenarios", "3", "Pre-loaded", True, "<svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><rect x='3' y='3' width='7' height='7'/><rect x='14' y='3' width='7' height='7'/><rect x='14' y='14' width='7' height='7'/><path d='M3 17l3 3 5-5'/></svg>", YELLOW)

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.6, 1], gap="large")

    with col_left:
        section_label("The 8 Validity Checks", MINT)

        checks = [
            ("01", "Sample Ratio Mismatch", "Chi-square on assignment counts", MINT),
            ("02", "Statistical Power",     "Non-central normal power formula", BLUE),
            ("03", "Variance Homogeneity",  "Levene + Welch auto-correction",   PURPLE),
            ("04", "Normality Assessment",  "Shapiro-Wilk + Mann-Whitney",      YELLOW),
            ("05", "Multiple Testing",      "Bonferroni + Benjamini-Hochberg",  CORAL),
            ("06", "Peeking Detection",     "10,000 Monte Carlo null sims",     ORANGE),
            ("07", "Novelty Effect",        "OLS regression on daily lift",     MINT),
            ("08", "SUTVA Violations",      "Network / social contamination",   BLUE),
        ]

        # 2-col grid
        left_checks  = checks[:4]
        right_checks = checks[4:]
        gc1, gc2 = st.columns(2, gap="small")

        for col, items in [(gc1, left_checks), (gc2, right_checks)]:
            with col:
                for num, name, desc, color in items:
                    st.markdown(f"""
                    <div style="background:{CARD};border-radius:13px;
                                padding:1rem 1.1rem;border:1px solid {BORDER};
                                border-left:3px solid {color};margin-bottom:6px;">
                        <div style="font-size:10px;color:{color};font-weight:700;
                                    letter-spacing:.1em;margin-bottom:4px;">{num}</div>
                        <div style="font-size:13px;font-weight:600;color:{TXT};
                                    margin-bottom:3px;">{name}</div>
                        <div style="font-size:11px;color:{TXT2};">{desc}</div>
                    </div>""", unsafe_allow_html=True)

    with col_right:
        section_label("Checks by Category", PURPLE)

        labels  = ["Randomisation", "Sample Size", "Distribution", "Inference", "Design"]
        values  = [1, 2, 2, 2, 1]
        colors  = [MINT, BLUE, PURPLE, YELLOW, CORAL]

        fig_donut = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=.65,
            marker=dict(colors=colors, line=dict(width=0)),
            textinfo="none",
            hovertemplate="%{label}: %{value} check<extra></extra>",
        ))
        fig_donut.add_annotation(
            text="<b>8</b><br><span style='font-size:10px'>Checks</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=TXT, family="Inter"),
        )
        _chart(fig_donut, height=220)
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        # legend
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        for label, color in zip(labels, colors):
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;
                        padding:.3rem 0;border-bottom:1px solid {BORDER}33;">
                <div style="width:8px;height:8px;border-radius:50%;
                            background:{color};flex-shrink:0;"></div>
                <span style="font-size:12px;color:{TXT2};">{label}</span>
            </div>""", unsafe_allow_html=True)

    # ── QUICK LOAD — full-width row below the two-column block ──
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    section_label("Quick Load", CORAL)
    scenarios = [
        ("<svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='9' cy='21' r='1'/><circle cx='20' cy='21' r='1'/><path d='M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6'/></svg>",
         "Zepto", "Scarcity Badge", "Scarcity badge test with minor SRM issues. Good baseline for learning the audit flow.", MINT, "zepto_scarcity_badge"),
        ("<svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><rect x='1' y='4' width='22' height='16' rx='2'/><line x1='1' y1='10' x2='23' y2='10'/></svg>",
         "Fintech", "Cashback Offer", "Cashback offer test with a Sample Ratio Mismatch bug. Tests your SRM detection skills.", CORAL, "fintech_cashback_srm"),
        ("<svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><path d='M4 19.5A2.5 2.5 0 0 1 6.5 17H20'/><path d='M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z'/></svg>",
         "EdTech", "Notification Test", "Push notification experiment with a clean, well-powered result. All checks pass.", BLUE, "edtech_notification"),
    ]
    qs_cols = st.columns(3, gap="medium")
    for col, (icon_svg, short, tag, desc, color, key) in zip(qs_cols, scenarios):
        with col:
            loaded = st.session_state.get("qs_loaded") == short
            border_style = f"border:1.5px solid {color};" if loaded else f"border:1px solid {BORDER};"
            st.markdown(f"""
            <div style="background:{CARD};border-radius:14px;padding:1.2rem 1.4rem;
                        {border_style}border-top:3px solid {color};
                        margin-bottom:8px;min-height:110px;">
                <div style="display:flex;align-items:center;gap:9px;margin-bottom:6px;">
                    <span style="color:{color};">{icon_svg}</span>
                    <div>
                        <div style="font-size:14px;font-weight:700;color:{TXT};line-height:1.2;">{short}</div>
                        <div style="font-size:10px;color:{color};font-weight:600;
                                    text-transform:uppercase;letter-spacing:.07em;">{tag}</div>
                    </div>
                </div>
                <div style="font-size:12px;color:{TXT2};line-height:1.55;">{desc}</div>
            </div>""", unsafe_allow_html=True)
            btn_label = "Loaded" if loaded else f"Load {short}"
            if st.button(btn_label, key=f"qs_{key}", use_container_width=True):
                with st.spinner("Loading..."):
                    st.session_state["audit_df"] = load_scenario(key)
                    st.session_state["qs_loaded"] = short
                st.success("Loaded. Head to Validity Audit to run checks.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — EXPERIMENT DESIGNER
# ══════════════════════════════════════════════════════════════
elif page == "Experiment Designer":
    page_header("Experiment Designer", "Calculate sample size, power, and duration before running your experiment.", "Designer")

    col_form, col_results = st.columns([1, 1.3], gap="large")

    with col_form:
        section_label("Parameters", BLUE)

        st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:-6px 0 10px;">Baseline rate (%)</p>',
                    unsafe_allow_html=True)
        p_control = st.slider("Baseline rate", 0.5, 35.0, 9.4, 0.1,
                              format="%.1f%%", label_visibility="collapsed") / 100

        st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:6px 0 10px;">Min. Detectable Effect (pp)</p>',
                    unsafe_allow_html=True)
        mde_pp = st.slider("MDE", 0.2, 15.0, 2.0, 0.1,
                           format="+%.1fpp", label_visibility="collapsed")
        mde = mde_pp / 100

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:6px 0 8px;">Alpha (α)</p>',
                        unsafe_allow_html=True)
            alpha = st.select_slider("alpha", [0.01, 0.05, 0.10], 0.05,
                                     label_visibility="collapsed")
        with c2:
            st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:6px 0 8px;">Target power</p>',
                        unsafe_allow_html=True)
            tgt_power = st.select_slider("power", [0.70, 0.80, 0.90, 0.95], 0.80,
                                         label_visibility="collapsed")

        st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:6px 0 8px;">Daily users per arm</p>',
                    unsafe_allow_html=True)
        daily = st.number_input("daily", 100, 500_000, 4200, 100,
                                label_visibility="collapsed")

        st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:6px 0 8px;">Number of variants</p>',
                    unsafe_allow_html=True)
        n_var = st.slider("variants", 2, 5, 2, label_visibility="collapsed")

    with col_results:
        section_label("Results", MINT)

        req_n    = _required_n_proportion(p_control, mde, alpha, tgt_power)
        min_days = int(np.ceil(req_n / daily))
        p_trt    = p_control + mde
        achieved = _compute_power_proportion(req_n, p_control, p_trt, alpha)

        alpha_adj = alpha / n_var if n_var > 2 else alpha
        req_n_adj = _required_n_proportion(p_control, mde, alpha_adj, tgt_power)
        days_adj  = int(np.ceil(req_n_adj / daily))

        days_col = MINT if min_days <= 14 else YELLOW if min_days <= 28 else CORAL

        # KPI strip
        k1, k2, k3 = st.columns(3, gap="small")
        with k1:
            st.markdown(f"""
            <div style="background:{CARD};border-radius:14px;padding:1.1rem;
                        border:1px solid {BORDER};text-align:center;">
                <div style="font-size:11px;color:{TXT2};margin-bottom:8px;">Users / arm</div>
                <div style="font-size:28px;font-weight:800;color:{MINT};
                            letter-spacing:-.5px;">{req_n:,}</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div style="background:{CARD};border-radius:14px;padding:1.1rem;
                        border:1px solid {BORDER};text-align:center;">
                <div style="font-size:11px;color:{TXT2};margin-bottom:8px;">Min. days</div>
                <div style="font-size:28px;font-weight:800;color:{days_col};
                            letter-spacing:-.5px;">{min_days}</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div style="background:{CARD};border-radius:14px;padding:1.1rem;
                        border:1px solid {BORDER};text-align:center;">
                <div style="font-size:11px;color:{TXT2};margin-bottom:8px;">Power</div>
                <div style="font-size:28px;font-weight:800;color:{PURPLE};
                            letter-spacing:-.5px;">{achieved:.0%}</div>
            </div>""", unsafe_allow_html=True)

        if n_var > 2:
            st.markdown(f"""
            <div style="background:{YELLOW}12;border:1px solid {YELLOW}33;border-radius:11px;
                        padding:.7rem 1rem;margin:.8rem 0;font-size:12px;color:{YELLOW};">
                ⚠ {n_var} variants: Bonferroni-adjusted α = {alpha_adj:.4f}.
                Need <b style="color:{TXT};">{req_n_adj:,} users/arm</b> ({days_adj} days).
            </div>""", unsafe_allow_html=True)

        section_label("Power Curve", PURPLE)

        n_range  = np.linspace(100, req_n * 2.2, 55).astype(int)
        pwr_vals = [_compute_power_proportion(int(n), p_control, p_trt, alpha) * 100
                    for n in n_range]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=n_range, y=pwr_vals,
            mode="lines", line=dict(color=MINT, width=2.5),
            fill="tozeroy", fillcolor="rgba(78,205,196,0.08)",
        ))
        fig.add_hline(y=80, line=dict(color=YELLOW, dash="dot", width=1.5),
                      annotation_text="80%", annotation_position="right",
                      annotation_font=dict(color=YELLOW, size=10))
        fig.add_vline(x=req_n, line=dict(color=CORAL, dash="dot", width=1.5),
                      annotation_text=f"n={req_n:,}",
                      annotation_position="top right",
                      annotation_font=dict(color=CORAL, size=10))
        fig.update_layout(yaxis=dict(range=[0, 105]),
                          xaxis_title="Users per arm",
                          yaxis_title="Power (%)")
        _chart(fig, 240)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Flags
        flags = []
        if min_days > 28: flags.append((CORAL, "⚠ Exceeds 4 weeks. Increase MDE or traffic."))
        if mde < 0.005:   flags.append((YELLOW, "⚠ MDE < 0.5 pp: sample sizes will be very large."))
        if not flags:     flags.append((MINT, "✓ Experiment plan looks valid."))
        for col, msg in flags:
            st.markdown(f"""
            <div style="background:{col}0f;border:1px solid {col}30;border-radius:10px;
                        padding:.6rem .9rem;margin-top:.4rem;font-size:12px;color:{col};">
                {msg}
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — VALIDITY AUDIT
# ══════════════════════════════════════════════════════════════
elif page == "Validity Audit":
    page_header("Validity Audit", "Run all 8 statistical checks on your experiment data.", "Audit")

    tab_demo, tab_upload = st.tabs(["  Demo Scenarios  ", "  Upload CSV  "])
    df = st.session_state.get("audit_df")

    with tab_demo:
        st.markdown(f'<p style="font-size:12px;color:{TXT2};padding:.5rem 0;">Choose a pre-built experiment scenario.</p>',
                    unsafe_allow_html=True)
        chosen = st.selectbox("Scenario", [
            "Zepto - Scarcity Badge (minor issues)",
            "Fintech - Cashback (SRM bug)",
            "EdTech - Notification (clean)",
        ], label_visibility="collapsed")
        smap = {
            "Zepto": "zepto_scarcity_badge",
            "Fintech": "fintech_cashback_srm",
            "EdTech": "edtech_notification",
        }
        key = next((v for k, v in smap.items() if k in chosen), "zepto_scarcity_badge")
        if st.button("Load Scenario", key="audit_load"):
            with st.spinner("Generating..."):
                df = load_scenario(key)
                st.session_state["audit_df"] = df
            st.success(f"Loaded {len(df):,} rows")

    with tab_upload:
        up = st.file_uploader("CSV with columns: user_id, arm, converted, metric_value, day, pre_metric",
                              type=["csv"], label_visibility="collapsed")
        if up:
            df = pd.read_csv(up)
            st.session_state["audit_df"] = df
            st.success(f"Uploaded {len(df):,} rows")

    if df is not None:
        section_label("Configuration", BLUE)

        cc1, cc2, cc3, cc4 = st.columns(4, gap="small")
        with cc1:
            st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:0 0 6px;">Experiment name</p>',
                        unsafe_allow_html=True)
            exp_name = st.text_input("name", "My Experiment", label_visibility="collapsed")
        with cc2:
            st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:0 0 6px;">Alpha (α)</p>',
                        unsafe_allow_html=True)
            exp_alpha = st.select_slider("ea", [0.01, 0.05, 0.10], 0.05,
                                         label_visibility="collapsed")
        with cc3:
            st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:0 0 6px;">Target power</p>',
                        unsafe_allow_html=True)
            exp_power = st.select_slider("ep", [0.70, 0.80, 0.90], 0.80,
                                         label_visibility="collapsed")
        with cc4:
            st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:0 0 6px;">Peeking days</p>',
                        unsafe_allow_html=True)
            peek_raw = st.text_input("pd", value="", placeholder="e.g. 3, 7, 10",
                                     label_visibility="collapsed")
            peek_days = sorted(set(
                int(x.strip()) for x in peek_raw.split(",")
                if x.strip().isdigit() and 1 <= int(x.strip()) <= 14
            )) if peek_raw.strip() else []

        fc1, fc2, fc3 = st.columns(3)
        with fc1: is_social   = st.checkbox("Social feature")
        with fc2: is_referral = st.checkbox("Referral feature")
        with fc3: is_comms    = st.checkbox("Comms feature")

        if st.button("Run Full Audit", use_container_width=False):
            with st.spinner("Checking..."):
                cfg = ExperimentConfig(
                    name=exp_name, metric_type=MetricType.PROPORTION,
                    alpha=exp_alpha, target_power=exp_power,
                    peeking_days=peek_days, is_social_feature=is_social,
                )
                audit = run_full_audit(df, cfg, sutva_flags={
                    "is_social_feature": is_social,
                    "is_referral_feature": is_referral,
                    "is_comms_feature": is_comms,
                })
                st.session_state["last_audit"] = audit

        if "last_audit" in st.session_state:
            audit  = st.session_state["last_audit"]
            sev    = audit.overall_severity
            sc     = {Severity.PASS: MINT, Severity.WARN: YELLOW, Severity.FAIL: CORAL}[sev]
            meta   = audit.experiment_meta

            # Overall verdict
            st.markdown(f"""
            <div style="background:{sc}12;border:1.5px solid {sc}33;border-radius:16px;
                        padding:1.3rem 1.6rem;margin:1rem 0 1.4rem;">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                    {badge(sev)}
                    <span style="font-size:16px;font-weight:700;color:{TXT};">
                        {_clean(audit.score_summary)}
                    </span>
                </div>
                <p style="font-size:13px;color:{TXT2};margin:0;line-height:1.7;">
                    {_clean(audit.overall_verdict)}
                </p>
            </div>""", unsafe_allow_html=True)

            # Meta strip
            m1, m2, m3, m4, m5 = st.columns(5, gap="small")
            for col, lbl, val in zip(
                [m1, m2, m3, m4, m5],
                ["Control n","Treatment n","Control rate","Treat rate","Lift"],
                [f"{meta.get('n_control',0):,}", f"{meta.get('n_treatment',0):,}",
                 f"{meta.get('p_control',0)*100:.2f}%",
                 f"{meta.get('p_treatment',0)*100:.2f}%",
                 f"{meta.get('lift_pp',0):+.2f} pp"],
            ):
                col.metric(lbl, val)

            section_label("Check Breakdown", PURPLE)

            # Summary bar chart
            sev_counts = [audit.n_passed, audit.n_warned, audit.n_failed]
            fig_bar = go.Figure(go.Bar(
                x=["Passed", "Warnings", "Failed"],
                y=sev_counts,
                marker=dict(color=[MINT, YELLOW, CORAL], line=dict(width=0)),
                text=sev_counts, textfont=dict(color=TXT, size=12),
                textposition="outside",
                width=0.45,
            ))
            fig_bar.update_layout(yaxis=dict(range=[0, 9]))
            _chart(fig_bar, 180)
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            icons = ["01","02","03","04","05","06","07","08"]
            for i, check in enumerate(audit.checks):
                sc2 = {Severity.PASS: MINT, Severity.WARN: YELLOW,
                       Severity.FAIL: CORAL}[check.severity]
                with st.expander(
                    f"{icons[i]}  {check.name}",
                    expanded=(check.severity != Severity.PASS),
                ):
                    ec1, ec2 = st.columns([3, 1])
                    with ec1:
                        st.markdown(
                            f"{badge(check.severity)}"
                            f"<p style='font-size:13px;color:{TXT};margin:8px 0 6px;'>"
                            f"{_clean(check.verdict)}</p>",
                            unsafe_allow_html=True,
                        )
                        if check.severity != Severity.PASS and check.cost_of_violation != "N/A":
                            st.markdown(f"""
                            <div style="background:{CORAL}0f;border-left:3px solid {CORAL};
                                        padding:.6rem .9rem;border-radius:0 8px 8px 0;
                                        font-size:12px;color:{TXT2};margin-bottom:6px;">
                                <b style="color:{CORAL};">Cost:</b> {_clean(check.cost_of_violation)}
                            </div>""", unsafe_allow_html=True)
                        if check.recommendation and check.recommendation != "N/A":
                            st.markdown(f"""
                            <div style="background:{MINT}0f;border-left:3px solid {MINT};
                                        padding:.6rem .9rem;border-radius:0 8px 8px 0;
                                        font-size:12px;color:{TXT2};">
                                <b style="color:{MINT};">Action:</b> {_clean(check.recommendation)}
                            </div>""", unsafe_allow_html=True)
                    with ec2:
                        if check.p_value is not None:
                            st.metric("p-value", f"{check.p_value:.4f}")
                        st.metric("statistic", f"{check.statistic:.4f}")


# ══════════════════════════════════════════════════════════════
# PAGE 4 — PEEKING SIMULATOR
# ══════════════════════════════════════════════════════════════
elif page == "Peeking Simulator":
    page_header("Peeking Simulator", "Watch false positive rates inflate when you check results early.", "Peeking")

    col_ctrl, col_res = st.columns([1, 1.6], gap="large")

    with col_ctrl:
        section_label("Controls", CORAL)

        st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:0 0 8px;">Users per arm</p>',
                    unsafe_allow_html=True)
        n_arm = st.slider("n_arm", 500, 10_000, 5_000, 500, label_visibility="collapsed")

        st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:6px 0 8px;">Baseline rate (%)</p>',
                    unsafe_allow_html=True)
        p_c = st.slider("p_ctrl", 1.0, 30.0, 9.4, 0.5, format="%.1f%%",
                        label_visibility="collapsed") / 100

        st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:6px 0 8px;">Peeking strategy</p>',
                    unsafe_allow_html=True)
        strategy = st.select_slider("strat", options=[
            "No peeking", "Once (day 7)", "Twice (days 5, 10)",
            "Every 3 days", "Daily",
        ], value="Every 3 days", label_visibility="collapsed")

        st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:6px 0 8px;">Simulations</p>',
                    unsafe_allow_html=True)
        n_sims = st.select_slider("nsims", [500, 1_000, 2_000, 5_000], 2_000,
                                  label_visibility="collapsed")

        st.markdown(f'<p style="font-size:12px;color:{TXT2};margin:6px 0 8px;">Alpha (α)</p>',
                    unsafe_allow_html=True)
        alpha_p = st.select_slider("ap", [0.01, 0.05, 0.10], 0.05,
                                   label_visibility="collapsed")

        peek_map = {
            "No peeking": [], "Once (day 7)": [7],
            "Twice (days 5, 10)": [5, 10],
            "Every 3 days": [3, 6, 9, 12], "Daily": list(range(1, 15)),
        }
        p_days = peek_map[strategy]

        run_btn = st.button("Run Simulation", use_container_width=True)

    with col_res:
        if run_btn or "peek_res" in st.session_state:
            if run_btn:
                with st.spinner(f"Running {n_sims:,} null experiments..."):
                    res  = run_peeking_simulation(n_arm, p_c, p_days, n_simulations=n_sims, alpha=alpha_p, seed=42)
                    traj = simulate_null_trajectories(n_arm, p_c, n_days=14, n_trajectories=50, alpha=alpha_p, seed=42)
                    st.session_state.update({"peek_res": res, "peek_traj": traj})
            else:
                res  = st.session_state["peek_res"]
                traj = st.session_state.get("peek_traj")

            inflated = res.inflated_alpha
            factor   = res.inflation_factor
            ic = MINT if inflated <= alpha_p*1.2 else YELLOW if inflated <= alpha_p*2 else CORAL

            section_label("False Positive Rate", MINT)

            k1, k2, k3 = st.columns(3, gap="small")
            with k1:
                st.markdown(f"""
                <div style="background:{CARD};border-radius:14px;padding:1rem;
                            border:1px solid {BORDER};text-align:center;">
                    <div style="font-size:11px;color:{TXT2};margin-bottom:6px;">Nominal α</div>
                    <div style="font-size:24px;font-weight:800;color:{MINT};">{alpha_p:.0%}</div>
                </div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div style="background:{CARD};border-radius:14px;padding:1rem;
                            border:1px solid {BORDER};text-align:center;">
                    <div style="font-size:11px;color:{TXT2};margin-bottom:6px;">Actual α</div>
                    <div style="font-size:24px;font-weight:800;color:{ic};">{inflated:.1%}</div>
                </div>""", unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div style="background:{CARD};border-radius:14px;padding:1rem;
                            border:1px solid {BORDER};text-align:center;">
                    <div style="font-size:11px;color:{TXT2};margin-bottom:6px;">Inflation</div>
                    <div style="font-size:24px;font-weight:800;color:{PURPLE};">{factor:.1f}×</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:{ic}0f;border:1px solid {ic}33;border-radius:11px;
                        padding:.75rem 1rem;margin:.8rem 0;font-size:12px;color:{ic};">
                {'No peeking: false positive rate equals α exactly.' if not p_days
                 else f'Peeking {len(p_days)} time(s) inflated your actual false positive rate from {alpha_p:.0%} to {inflated:.1%}.'}
            </div>""", unsafe_allow_html=True)

            if traj:
                section_label("P-value Trajectories (50 null experiments)", PURPLE)
                fig = go.Figure()
                for i, vals in enumerate(traj["trajectories"]):
                    is_fp = any(v < alpha_p for v in vals)
                    fig.add_trace(go.Scatter(
                        x=traj["days"], y=vals,
                        mode="lines",
                        line=dict(color=CORAL if is_fp else "rgba(96,165,250,0.25)",
                                  width=1.8 if is_fp else 0.7),
                        showlegend=False, hoverinfo="skip",
                    ))
                fig.add_hline(y=alpha_p, line=dict(color=YELLOW, dash="dot", width=2),
                              annotation_text=f"α = {alpha_p}", annotation_position="right",
                              annotation_font=dict(color=YELLOW, size=10))
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                         line=dict(color=CORAL, width=2),
                                         name=f"False positive ({traj['fp_rate']:.1%})",
                                         showlegend=True))
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                         line=dict(color=BLUE, width=1),
                                         name="True negative", showlegend=True))
                fig.update_layout(
                    showlegend=True,
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TXT2, size=11)),
                    xaxis_title="Day", yaxis_title="p-value",
                    yaxis=dict(range=[0, 1.05]),
                )
                _chart(fig, 300)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            if p_days and res.alpha_by_n_peeks:
                section_label("Inflation by Number of Peeks", YELLOW)
                x_lbl = [f"{i+1}" for i in range(len(res.alpha_by_n_peeks))]
                y_vals = [v * 100 for v in res.alpha_by_n_peeks]
                bar_colors = [MINT if v <= alpha_p*120 else YELLOW if v <= alpha_p*200 else CORAL
                              for v in y_vals]
                fig2 = go.Figure(go.Bar(
                    x=x_lbl, y=y_vals,
                    marker=dict(color=bar_colors, line=dict(width=0)),
                    text=[f"{v:.1f}%" for v in y_vals],
                    textfont=dict(color=TXT, size=11), textposition="outside",
                    width=0.5,
                ))
                fig2.add_hline(y=alpha_p*100, line=dict(color=YELLOW, dash="dot", width=1.5),
                               annotation_text=f"nominal {alpha_p:.0%}",
                               annotation_font=dict(color=YELLOW, size=10))
                fig2.update_layout(xaxis_title="Peek number", yaxis_title="FP rate (%)",
                                   yaxis=dict(range=[0, max(y_vals)*1.35]))
                _chart(fig2, 220)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            section_label("Results", MINT)
            st.markdown(f"""
            <div style="background:{CARD};border-radius:16px;padding:3.5rem 2rem;
                        text-align:center;border:1px dashed {BORDER};">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#5A5A62" stroke-width="1.5" stroke-linecap="round" style="margin-bottom:14px;"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                <div style="font-size:14px;font-weight:600;color:{TXT};margin-bottom:6px;">
                    Configure and run a simulation
                </div>
                <div style="font-size:12px;color:{TXT2};">
                    Set your peeking strategy on the left, then click Run.
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — CUPED ENGINE
# ══════════════════════════════════════════════════════════════
elif page == "CUPED Engine":
    page_header("CUPED Engine", "Reduce required sample size 20–40% using pre-experiment data.", "CUPED")

    # Theory strip
    st.markdown(f"""
    <div style="background:{CARD};border-radius:14px;padding:1.1rem 1.5rem;
                border:1px solid {BORDER};border-left:3px solid {PURPLE};
                margin-bottom:1.5rem;">
        <div style="font-size:11px;color:{PURPLE};font-weight:700;
                    letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">
            Deng, Xu, Kohavi &amp; Walker · Microsoft Research KDD 2013
        </div>
        <code style="font-size:13px;color:{TXT};background:none;">
            Ỹᵢ = Yᵢ &minus; θ·(Xᵢ &minus; X̄) &nbsp;&nbsp;
            θ = Cov(Y,X)/Var(X) &nbsp;&nbsp;
            Var(Ỹ) = Var(Y)·(1 &minus; ρ²)
        </code>
        <div style="font-size:11px;color:{TXT3};margin-top:6px;">
            If ρ = 0.5 → 25% variance reduction → equivalent to 33% more users for free.
        </div>
    </div>""", unsafe_allow_html=True)

    # Load data
    tab_c1, tab_c2 = st.tabs(["  Demo  ", "  Upload  "])
    c_df = st.session_state.get("cuped_df")

    with tab_c1:
        chosen_c = st.selectbox("Scenario", [
            "EdTech Notification (low correlation)",
            "High-correlation synthetic (rho ~ 0.7)",
            "Zepto Scarcity Badge",
        ], label_visibility="collapsed")
        if st.button("Load", key="lc"):
            with st.spinner("Loading..."):
                if "High-corr" in chosen_c or "high" in chosen_c.lower():
                    rng = np.random.default_rng(42)
                    n = 5000
                    pre = rng.normal(100, 30, n * 2)
                    out = 0.7 * pre + rng.normal(0, 15, n * 2)
                    out[n:] += 8
                    c_df = pd.DataFrame({
                        "arm": ["control"]*n + ["treatment"]*n,
                        "metric_value": out, "pre_metric": pre,
                        "converted": (out > out.mean()).astype(int),
                        "day": rng.integers(1, 15, n*2),
                        "user_id": range(n*2),
                    })
                elif "EdTech" in chosen_c or "edtech" in chosen_c.lower() or "low" in chosen_c.lower():
                    c_df = load_scenario("edtech_notification")
                else:
                    c_df = load_scenario("zepto_scarcity_badge")
                st.session_state["cuped_df"] = c_df
            st.success(f"Loaded {len(c_df):,} rows")

    with tab_c2:
        up_c = st.file_uploader("CSV with metric_value + pre_metric", type=["csv"],
                                label_visibility="collapsed")
        if up_c:
            c_df = pd.read_csv(up_c)
            st.session_state["cuped_df"] = c_df

    if c_df is None:
        c_df = st.session_state.get("cuped_df")

    if c_df is not None:
        run_c = st.button("Run CUPED Analysis", use_container_width=False)

        if run_c or "cuped_res" in st.session_state:
            if run_c:
                with st.spinner("Computing adjustment..."):
                    try:
                        r = run_cuped_analysis(c_df)
                        st.session_state["cuped_res"] = r
                    except Exception as e:
                        st.error(str(e)); r = None
            else:
                r = st.session_state.get("cuped_res")

            if r:
                section_label("CUPED Results", MINT)

                k1, k2, k3, k4 = st.columns(4, gap="small")
                for col, lbl, val, ic in [
                    (k1, "Correlation ρ",       f"{r.rho:+.3f}",           PURPLE),
                    (k2, "Variance reduction",  f"{r.variance_reduction_pct:.1f}%", MINT),
                    (k3, "CI width reduction",  f"{r.ci_width_reduction_pct:.1f}%", BLUE),
                    (k4, "Equiv. sample size",  f"{r.sample_size_equivalent:,.0f}",  YELLOW),
                ]:
                    with col:
                        st.markdown(f"""
                        <div style="background:{CARD};border-radius:14px;padding:1.1rem;
                                    border:1px solid {BORDER};text-align:center;">
                            <div style="font-size:11px;color:{TXT2};margin-bottom:6px;">{lbl}</div>
                            <div style="font-size:26px;font-weight:800;color:{ic};
                                        letter-spacing:-.4px;">{val}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
                col_tbl, col_ci = st.columns([1, 1.2], gap="large")

                with col_tbl:
                    section_label("Before vs After", BLUE)
                    rows = [
                        ("Lift",     f"{r.lift_unadjusted*100:+.3f} pp",   f"{r.lift_adjusted*100:+.3f} pp"),
                        ("Std Error",f"{r.se_unadjusted*100:.4f} pp",       f"{r.se_adjusted*100:.4f} pp"),
                        ("CI lower", f"{r.ci_unadjusted[0]*100:+.3f} pp",   f"{r.ci_adjusted[0]*100:+.3f} pp"),
                        ("CI upper", f"{r.ci_unadjusted[1]*100:+.3f} pp",   f"{r.ci_adjusted[1]*100:+.3f} pp"),
                        ("p-value",  f"{r.p_unadjusted:.4f}",               f"{r.p_adjusted:.4f}"),
                        ("Result",
                         f"{'✓ Sig' if r.significant_before else '✗ Not sig'}",
                         f"{'✓ Sig' if r.significant_after  else '✗ Not sig'}"),
                    ]
                    for metric, before, after in rows:
                        st.markdown(f"""
                        <div style="display:grid;grid-template-columns:100px 1fr 1fr;
                                    gap:8px;padding:.5rem .2rem;
                                    border-bottom:1px solid {BORDER}33;
                                    font-size:12px;align-items:center;">
                            <span style="color:{TXT3};font-weight:500;">{metric}</span>
                            <span style="color:{CORAL};">{before}</span>
                            <span style="color:{MINT};font-weight:600;">{after}</span>
                        </div>""", unsafe_allow_html=True)

                with col_ci:
                    section_label("Confidence Interval Comparison", PURPLE)
                    fig_ci = go.Figure()
                    for lbl, lift, lo, hi, color in [
                        ("Without CUPED", r.lift_unadjusted, r.ci_unadjusted[0], r.ci_unadjusted[1], CORAL),
                        ("With CUPED",    r.lift_adjusted,   r.ci_adjusted[0],   r.ci_adjusted[1],   MINT),
                    ]:
                        fig_ci.add_trace(go.Scatter(
                            x=[lo*100, hi*100], y=[lbl, lbl],
                            mode="lines",
                            line=dict(color=color, width=5),
                            showlegend=True, name=lbl,
                        ))
                        fig_ci.add_trace(go.Scatter(
                            x=[lift*100], y=[lbl],
                            mode="markers",
                            marker=dict(color=color, size=13, symbol="circle"),
                            showlegend=False,
                        ))
                    fig_ci.add_vline(x=0, line=dict(color=TXT3, dash="dot", width=1))
                    fig_ci.update_layout(
                        showlegend=True,
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TXT2, size=11)),
                        xaxis_title="Lift (pp)",
                        yaxis=dict(autorange="reversed"),
                        height=190,
                    )
                    _chart(fig_ci, 190)
                    st.plotly_chart(fig_ci, use_container_width=True, config={"displayModeBar": False})

                    _flipped = 'Result flipped to significant after CUPED adjustment.' if not r.significant_before and r.significant_after else ''
                    _interp_html = (
                        f'<div style="background:rgba(167,139,250,0.06);border:1px solid rgba(167,139,250,0.19);'
                        f'border-radius:11px;padding:.8rem 1rem;margin-top:.5rem;">'
                        f'<div style="font-size:13px;font-weight:600;color:{PURPLE};margin-bottom:5px;">'
                        f'Interpretation</div>'
                        f'<div style="font-size:12px;color:{TXT2};line-height:1.7;">'
                        f'&rho; = {r.rho:.3f} &rarr; {r.variance_reduction_pct:.1f}% variance removed. '
                        f'CI is {r.ci_width_reduction_pct:.1f}% narrower. '
                        f'Equivalent to running on '
                        f'<strong style="color:{TXT};">{r.sample_size_equivalent:,.0f} users/arm</strong> '
                        f'at no extra cost. {_flipped}'
                        f'</div></div>'
                    )
                    st.markdown(_interp_html, unsafe_allow_html=True)
