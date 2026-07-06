import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import theme
from data_loader import load_enriched


def render():
    df, control_df = load_enriched()

    st.title("Facility Benchmarking")
    st.caption(
        "🟢 Real data. SIR = Standardized Infection Ratio (observed / predicted infections; "
        "national baseline = 1.0)."
    )

    st.sidebar.header("Filters")
    infection_options = sorted(df["infection_type"].unique())
    selected_infections = st.sidebar.multiselect("Infection type", infection_options, default=infection_options)
    state_options = sorted(df["state"].dropna().unique())
    selected_states = st.sidebar.multiselect("State", state_options, default=[])

    filtered = df[df["infection_type"].isin(selected_infections)]
    if selected_states:
        filtered = filtered[filtered["state"].isin(selected_states)]

    st.subheader("Geographic benchmarking")
    state_sir = (
        filtered.groupby("state")
        .agg(numerator=("numerator", "sum"), predicted=("predicted_infections", "sum"),
             facilities=("facility_id", "nunique"))
        .reset_index()
    )
    state_sir = state_sir[state_sir["predicted"] >= 5]
    state_sir["state_sir"] = state_sir["numerator"] / state_sir["predicted"]

    fig_map = px.choropleth(
        state_sir, locations="state", locationmode="USA-states", scope="usa",
        color="state_sir", color_continuous_scale=theme.SEQUENTIAL_BLUE,
        hover_data={"state": True, "facilities": True, "state_sir": ":.2f"},
        labels={"state_sir": "Volume-weighted SIR"},
        title="State-level volume-weighted SIR (selected infection types)",
    )
    fig_map.update_layout(**theme.plotly_layout_defaults())
    fig_map.update_layout(geo=dict(bgcolor=theme.CHROME["surface"], lakecolor=theme.CHROME["surface"]))
    st.plotly_chart(fig_map, width="stretch")

    st.divider()
    st.subheader("Funnel plots (statistical control limits)")
    st.caption(
        "Each point is one facility's SIR for one infection type, plotted against its predicted "
        "infection volume. Control bands are the expected statistical spread under a Poisson "
        "process (variance = 1 / predicted infections)."
    )

    for infection in selected_infections:
        sub = filtered[filtered["infection_type"] == infection]
        cl = control_df[control_df["infection_type"] == infection].sort_values("predicted_infections")
        if sub.empty:
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cl["predicted_infections"], y=cl["upper_3sigma"],
                                  line=dict(color="rgba(208,59,59,0.20)", width=1), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cl["predicted_infections"], y=cl["upper_2sigma"],
                                  line=dict(color="rgba(208,59,59,0.35)", width=1), name="2σ control limit", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cl["predicted_infections"], y=[1.0] * len(cl),
                                  line=dict(color=theme.CHROME["ink_primary"], width=1, dash="dash"),
                                  name="National benchmark (SIR=1)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cl["predicted_infections"], y=cl["lower_2sigma"],
                                  line=dict(color="rgba(12,163,12,0.35)", width=1), name="2σ control limit (lower)", hoverinfo="skip"))

        for flag, color in theme.BENCHMARK_STATUS_COLORS.items():
            pts = sub[sub["benchmark_flag"] == flag]
            if pts.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pts["predicted_infections"], y=pts["sir"], mode="markers", name=flag,
                marker=dict(color=color, size=7, opacity=0.8, line=dict(width=1, color=theme.CHROME["surface"])),
                customdata=pts[["facility_name", "state", "numerator", "p_value"]],
                hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>SIR: %{y:.2f}<br>"
                              "Predicted infections: %{x:.1f}<br>Observed: %{customdata[2]:.0f}<br>"
                              "p-value: %{customdata[3]:.4f}<extra></extra>",
            ))

        fig.update_layout(**theme.plotly_layout_defaults(infection))
        fig.update_layout(xaxis_title="Predicted infections (log scale)", yaxis_title="SIR", xaxis_type="log", height=420)
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Facilities flagged worse than national benchmark")
    outliers = filtered[filtered["benchmark_flag"] == "Worse than National Benchmark"].sort_values("ci_lower", ascending=False)
    st.dataframe(
        outliers[["facility_name", "state", "infection_type", "sir", "ci_lower", "ci_upper", "p_value", "q_value_fdr"]]
        .rename(columns={"facility_name": "Facility", "state": "State", "infection_type": "Infection Type",
                          "sir": "SIR", "ci_lower": "CI Lower", "ci_upper": "CI Upper",
                          "p_value": "p-value", "q_value_fdr": "FDR q-value"}),
        width="stretch", height=350,
    )
