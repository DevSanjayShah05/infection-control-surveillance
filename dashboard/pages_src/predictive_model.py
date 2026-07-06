import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme
from data_loader import load_model_results


def render():
    model_results = load_model_results()
    results = model_results["results"]

    st.title("Predictive Model")
    st.caption(
        "🟢 Real data throughout. Target: is this facility-measure flagged 'Worse than National "
        "Benchmark' (the real CMS CI-based flag)? Features: infection type, hospital type/ownership/"
        "ER services (real CMS Hospital General Information), CMS star rating, predicted infection volume."
    )
    st.warning(
        f"⚠️ Class imbalance: only **{model_results['test_positive_rate']*100:.1f}%** of test-set rows "
        "are positive. Plain accuracy would be misleading here (a model predicting 'never flagged' "
        "scores ~98% accuracy while catching nothing useful) — evaluated on ROC-AUC and PR-AUC instead, "
        "and trained with class-balanced weighting."
    )

    model_name = st.selectbox("Model", list(results.keys()), format_func=lambda m: m.replace("_", " ").title())
    r = results[model_name]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(theme.kpi_card("ROC-AUC", f"{r['roc_auc']:.3f}"), unsafe_allow_html=True)
    with c2:
        st.markdown(theme.kpi_card("PR-AUC (avg. precision)", f"{r['pr_auc']:.3f}",
                                    f"vs. {model_results['test_positive_rate']:.3f} baseline"), unsafe_allow_html=True)
    with c3:
        cm = np.array(r["confusion_matrix"])
        recall = cm[1, 1] / max(cm[1].sum(), 1)
        st.markdown(theme.kpi_card("Recall @ 0.5 threshold", f"{recall*100:.0f}%", "of true positives caught"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fpr, tpr = r["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=theme.CATEGORICAL_LIGHT[0], width=3), name="ROC curve"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=theme.STATUS["neutral"], dash="dash"), name="Random baseline"))
        fig.update_layout(**theme.plotly_layout_defaults(f"ROC curve (AUC={r['roc_auc']:.3f})"))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)
        st.plotly_chart(fig, width="stretch")
    with col2:
        prec, rec = r["pr_curve"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line=dict(color=theme.CATEGORICAL_LIGHT[1], width=3), name="PR curve"))
        fig2.add_hline(y=model_results["test_positive_rate"], line_dash="dash", line_color=theme.STATUS["neutral"],
                       annotation_text="random baseline")
        fig2.update_layout(**theme.plotly_layout_defaults(f"Precision-Recall curve (AP={r['pr_auc']:.3f})"))
        fig2.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=400)
        st.plotly_chart(fig2, width="stretch")

    st.subheader("Confusion matrix @ 0.5 probability threshold")
    cm = np.array(r["confusion_matrix"])
    fig3 = go.Figure(data=go.Heatmap(
        z=cm, x=["Predicted: Not flagged", "Predicted: Worse"], y=["Actual: Not flagged", "Actual: Worse"],
        colorscale=[[0, theme.CHROME["surface"]], [1, theme.CATEGORICAL_LIGHT[0]]],
        text=cm, texttemplate="%{text}", showscale=False,
    ))
    fig3.update_layout(**theme.plotly_layout_defaults())
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, width="stretch")

    st.divider()
    st.subheader("Feature importance / interpretability")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Logistic regression coefficients (odds ratios)**")
        coefs = model_results["logreg_coefs"].head(12)
        fig4 = go.Figure(go.Bar(
            x=coefs["coefficient"], y=coefs["feature"], orientation="h",
            marker_color=[theme.STATUS["critical"] if c > 0 else theme.STATUS["good"] for c in coefs["coefficient"]],
        ))
        fig4.update_layout(**theme.plotly_layout_defaults())
        fig4.update_layout(height=420, xaxis_title="Coefficient (log-odds)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig4, width="stretch")
        st.caption("Positive = higher odds of being flagged worse-than-national; negative = lower odds.")
    with col4:
        st.markdown("**Gradient boosting permutation importance**")
        perm = model_results["hgb_perm_importance"]
        fig5 = go.Figure(go.Bar(
            x=perm["importance_mean"], y=perm["feature"], orientation="h",
            error_x=dict(array=perm["importance_std"]),
            marker_color=theme.CATEGORICAL_LIGHT[2],
        ))
        fig5.update_layout(**theme.plotly_layout_defaults())
        fig5.update_layout(height=420, xaxis_title="Mean decrease in PR-AUC when shuffled", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig5, width="stretch")

    st.caption(
        "Model comparison note: logistic regression outperforms gradient boosting here on both ROC-AUC "
        "and PR-AUC — a sign the relationship is dominated by a few strong main effects (infection type, "
        "star rating) rather than complex nonlinear interactions the boosted model would otherwise exploit."
    )
