"""
Shared design system for the healthcare analytics dashboards.
Palette validated with the data-viz skill's validator (six-checks: lightness
band, chroma floor, CVD separation, contrast) — see palette.md for the source
instance. Both dashboards import this module so they read as one system
rather than two one-off Streamlit apps.
"""

# ---------------- Categorical (identity: infection types, measures) ----------------
# Fixed hue order -- never cycle, never reassign per-filter.
CATEGORICAL_LIGHT = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                     "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
CATEGORICAL_DARK = ["#3987e5", "#199e70", "#c98500", "#008300",
                    "#9085e9", "#e66767", "#d55181", "#d95926"]

# ---------------- Sequential (magnitude: choropleth, heatmaps) ----------------
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

# ---------------- Diverging (polarity: risk-adjusted deltas) ----------------
DIVERGING_BLUE_RED = ["#0d366b", "#256abf", "#5598e7", "#9ec5f4",
                      "#f0efec",
                      "#f0a19f", "#e34948", "#b93534", "#7a1f1f"]

# ---------------- Status (fixed meaning -- never themed, always icon+label) ----------------
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    "neutral": "#898781",
}

# Benchmark-flag -> status color mapping used across both dashboards
BENCHMARK_STATUS_COLORS = {
    "Worse than National Benchmark": STATUS["critical"],
    "Worse Than the National Rate": STATUS["critical"],
    "No Different than National Benchmark": STATUS["neutral"],
    "No Different Than the National Rate": STATUS["neutral"],
    "Better than National Benchmark": STATUS["good"],
    "Better Than the National Rate": STATUS["good"],
    "Not Available": "#c3c2b7",
}

# ---------------- Chrome & ink ----------------
CHROME = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink_primary": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
    "border": "rgba(11,11,11,0.10)",
}

FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def plotly_layout_defaults(title: str | None = None) -> dict:
    """Common Plotly layout overrides applied to every chart in both dashboards."""
    layout = dict(
        paper_bgcolor=CHROME["surface"],
        plot_bgcolor=CHROME["surface"],
        font=dict(family=FONT_FAMILY, color=CHROME["ink_primary"], size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=60, b=40, l=10, r=10),
        xaxis=dict(gridcolor=CHROME["gridline"], zerolinecolor=CHROME["baseline"], linecolor=CHROME["baseline"]),
        yaxis=dict(gridcolor=CHROME["gridline"], zerolinecolor=CHROME["baseline"], linecolor=CHROME["baseline"]),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=16, color=CHROME["ink_primary"]))
    return layout


def inject_css():
    """Return the shared CSS block (call via st.markdown(theme.inject_css(), unsafe_allow_html=True))."""
    return f"""
    <style>
    .kpi-card {{
        background: {CHROME['surface']};
        border: 1px solid {CHROME['border']};
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(11,11,11,0.06);
    }}
    .kpi-label {{
        font-family: {FONT_FAMILY};
        font-size: 0.80rem;
        color: {CHROME['ink_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-family: {FONT_FAMILY};
        font-size: 1.9rem;
        font-weight: 700;
        color: {CHROME['ink_primary']};
        line-height: 1.1;
    }}
    .kpi-delta-good {{ color: {STATUS['good']}; font-size: 0.85rem; font-weight: 600; }}
    .kpi-delta-bad {{ color: {STATUS['critical']}; font-size: 0.85rem; font-weight: 600; }}
    .status-pill {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: white;
    }}
    .data-badge-real {{
        display: inline-block;
        background: {STATUS['good']};
        color: white;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}
    .data-badge-synthetic {{
        display: inline-block;
        background: {STATUS['warning']};
        color: {CHROME['ink_primary']};
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}
    section[data-testid="stSidebar"] {{
        border-right: 1px solid {CHROME['border']};
    }}
    </style>
    """


def kpi_card(label: str, value: str, sub: str | None = None, status: str | None = None) -> str:
    """Render one KPI card as an HTML string for st.markdown(..., unsafe_allow_html=True)."""
    sub_html = ""
    if sub:
        css_class = "kpi-delta-good" if status == "good" else ("kpi-delta-bad" if status == "critical" else "kpi-label")
        sub_html = f'<div class="{css_class}">{sub}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """


def status_pill(text: str, color: str) -> str:
    return f'<span class="status-pill" style="background:{color};">{text}</span>'
