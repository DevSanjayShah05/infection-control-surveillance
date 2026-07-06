import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme
from data_loader import load_enriched

RNG = np.random.default_rng(7)


def simulate_monthly_series(annual_predicted: float, true_sir: float, n_months: int = 24, drift_start_month: int | None = None):
    """SYNTHETIC monthly reconstruction: splits a facility's real annual predicted-infection
    volume into 24 months and simulates monthly observed infections as a Poisson process
    around the real SIR, optionally injecting a gradual drift partway through to demonstrate
    how CUSUM/EWMA detect a shift a simple month-to-month look would miss."""
    monthly_expected = max(annual_predicted / 12, 0.5)
    sir_series = np.full(n_months, true_sir, dtype=float)
    if drift_start_month is not None:
        ramp = np.linspace(0, 1, n_months - drift_start_month)
        sir_series[drift_start_month:] += 0.6 * ramp  # simulated gradual quality decline
    observed = RNG.poisson(monthly_expected * sir_series)
    monthly_sir = observed / monthly_expected
    return monthly_sir


def cusum(series: np.ndarray, target: float = 1.0, k: float = 0.5, h: float = 4.0):
    """Standard two-sided CUSUM in units of the series' own std dev."""
    std = series.std() if series.std() > 0 else 1.0
    dev = (series - target) / std
    c_pos, c_neg = [0.0], [0.0]
    for d in dev:
        c_pos.append(max(0, c_pos[-1] + d - k))
        c_neg.append(min(0, c_neg[-1] + d + k))
    return np.array(c_pos[1:]), np.array(c_neg[1:]), h


def ewma(series: np.ndarray, lam: float = 0.2):
    out = [series[0]]
    for x in series[1:]:
        out.append(lam * x + (1 - lam) * out[-1])
    return np.array(out)


def render():
    df, _ = load_enriched()

    st.title("Control Chart Demo")
    st.markdown(
        '<span class="data-badge-synthetic">SYNTHETIC DEMONSTRATION</span> &nbsp; '
        "CMS publishes only a single 12-month snapshot per facility — no real monthly history "
        "is public. This page **simulates** a monthly series (anchored to one real facility's "
        "real annual SIR) purely to demonstrate CUSUM and EWMA control-chart technique.",
        unsafe_allow_html=True,
    )

    facility_options = df[["facility_id", "facility_name", "state", "infection_type", "sir", "predicted_infections"]].dropna()
    facility_options = facility_options[facility_options["predicted_infections"] >= 20].drop_duplicates("facility_id")
    label_map = {
        row.facility_id: f"{row.facility_name} ({row.state}) — {row.infection_type}"
        for row in facility_options.itertuples()
    }
    selected_id = st.selectbox("Anchor facility (real SIR used as the simulation's baseline)",
                                options=list(label_map.keys()), format_func=lambda i: label_map[i])
    row = facility_options[facility_options["facility_id"] == selected_id].iloc[0]

    inject_drift = st.checkbox("Inject a simulated gradual quality decline at month 15", value=True)
    drift_month = 15 if inject_drift else None

    monthly_sir = simulate_monthly_series(row["predicted_infections"], row["sir"], drift_start_month=drift_month)
    months = pd.date_range("2024-01-01", periods=24, freq="MS")

    c_pos, c_neg, h = cusum(monthly_sir)
    ewma_series = ewma(monthly_sir)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=monthly_sir, mode="lines+markers",
                                  line=dict(color=theme.CATEGORICAL_LIGHT[0]), name="Simulated monthly SIR"))
        fig.add_trace(go.Scatter(x=months, y=ewma_series, mode="lines",
                                  line=dict(color=theme.CATEGORICAL_LIGHT[5], width=3), name="EWMA (λ=0.2)"))
        fig.add_hline(y=1.0, line_dash="dash", line_color=theme.CHROME["ink_primary"], annotation_text="benchmark")
        fig.update_layout(**theme.plotly_layout_defaults("Simulated monthly SIR with EWMA smoothing"))
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=months, y=c_pos, mode="lines+markers",
                                   line=dict(color=theme.STATUS["critical"]), name="CUSUM+ (detects upward shift)"))
        fig2.add_trace(go.Scatter(x=months, y=c_neg, mode="lines+markers",
                                   line=dict(color=theme.STATUS["good"]), name="CUSUM- (detects downward shift)"))
        fig2.add_hline(y=h, line_dash="dot", line_color=theme.STATUS["critical"], annotation_text="upper decision limit h")
        fig2.add_hline(y=-h, line_dash="dot", line_color=theme.STATUS["good"], annotation_text="lower decision limit -h")
        fig2.update_layout(**theme.plotly_layout_defaults("CUSUM chart"))
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, width="stretch")

    breach_month = np.argmax(c_pos > h) if (c_pos > h).any() else None
    if breach_month is not None:
        st.error(
            f"🔴 CUSUM crosses the upper decision limit at month {breach_month + 1} "
            f"({months[breach_month].strftime('%b %Y')}) — in a real monitoring system this "
            "would trigger an infection-control review, well before 12 months of aggregate "
            "data would have made the shift visible in the standard annual report."
        )
    else:
        st.success("🟢 CUSUM stays within control limits across the simulated window — no sustained shift detected.")

    st.caption(
        "Why this matters even as a demo: CUSUM/EWMA detect a *sustained gradual* shift in mean much "
        "faster than eyeballing raw monthly points, because they accumulate small deviations over time "
        "rather than resetting each month. This is the standard technique hospital infection-control "
        "teams use for real internal monthly surveillance (which is not public data)."
    )
