import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import theme
from data_loader import load_enriched


def render():
    df, _ = load_enriched()

    st.title("Statistical Testing")
    st.caption("🟢 Real data. Exact Poisson test per facility-measure, with Benjamini-Hochberg FDR correction.")

    st.markdown("""
CMS's published "Compared to National" flag is confidence-interval based. Here we compute an
**exact two-sided Poisson test** p-value directly (`H0: observed infections ~ Poisson(predicted infections)`,
i.e. true SIR = 1) for every facility-measure, then apply a **Benjamini-Hochberg false discovery rate (FDR)
correction** across all tests run simultaneously — because testing ~11,300 hypotheses at once with an
uncorrected alpha=0.05 would produce hundreds of false positives by chance alone.
""")

    alpha = st.slider("Significance threshold (alpha)", 0.001, 0.10, 0.05, step=0.001)

    naive_sig = (df["p_value"] < alpha).sum()
    fdr_sig_worse = ((df["q_value_fdr"] < alpha) & (df["sir"] > 1)).sum()
    fdr_sig_better = ((df["q_value_fdr"] < alpha) & (df["sir"] < 1)).sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(theme.kpi_card("Naive p < α (both directions)", f"{naive_sig:,}",
                                    f"{100*naive_sig/len(df):.1f}% of all {len(df):,} tests", "critical"), unsafe_allow_html=True)
    with c2:
        st.markdown(theme.kpi_card("FDR-corrected, worse (SIR>1)", f"{fdr_sig_worse:,}", "true outlier candidates", "critical"), unsafe_allow_html=True)
    with c3:
        st.markdown(theme.kpi_card("FDR-corrected, better (SIR<1)", f"{fdr_sig_better:,}", "true high-performer candidates", "good"), unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x="p_value", nbins=50, color_discrete_sequence=[theme.CATEGORICAL_LIGHT[0]],
            title="Distribution of raw p-values across all facility-measure tests",
            labels={"p_value": "p-value"},
        )
        fig.add_hline(y=len(df) / 50, line_dash="dash", line_color=theme.STATUS["neutral"],
                      annotation_text="expected under a uniform null")
        fig.update_layout(**theme.plotly_layout_defaults())
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "A sharp spike near p=0 (far above the flat 'expected under the null' line) shows there is "
            "real signal in this data, not just noise — but it also means an uncorrected threshold will "
            "over-flag facilities, which is exactly why the FDR correction matters."
        )
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["p_value"], y=df["q_value_fdr"], mode="markers",
            marker=dict(size=5, color=theme.CATEGORICAL_LIGHT[0], opacity=0.4),
            name="Facility-measure tests",
        ))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                   line=dict(color=theme.STATUS["neutral"], dash="dash"), name="q = p (no correction)"))
        fig2.update_layout(**theme.plotly_layout_defaults("Raw p-value vs. FDR-adjusted q-value"))
        fig2.update_layout(xaxis_title="Raw p-value", yaxis_title="FDR q-value")
        st.plotly_chart(fig2, width="stretch")
        st.caption("Points below the diagonal show where the raw p-value understated true multiple-testing risk.")

    st.divider()
    st.subheader(f"Facilities significant at FDR q < {alpha}, worse than benchmark")
    sig_table = (
        df[(df["q_value_fdr"] < alpha) & (df["sir"] > 1)]
        .sort_values("q_value_fdr")
        [["facility_name", "state", "infection_type", "sir", "p_value", "q_value_fdr", "hospital_type", "overall_rating"]]
    )
    st.dataframe(
        sig_table.rename(columns={
            "facility_name": "Facility", "state": "State", "infection_type": "Infection Type",
            "sir": "SIR", "p_value": "p-value", "q_value_fdr": "FDR q-value",
            "hospital_type": "Hospital Type", "overall_rating": "CMS Star Rating",
        }),
        width="stretch", height=350,
    )
