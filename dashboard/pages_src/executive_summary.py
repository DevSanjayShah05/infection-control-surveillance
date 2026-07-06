import streamlit as st
import theme
from data_loader import load_enriched, load_model_results


def render():
    df, _ = load_enriched()
    model_results = load_model_results()

    st.title("Infection Control Command Center")
    st.markdown(
        f'<span class="data-badge-real">100% REAL DATA</span> &nbsp; '
        f'CMS Provider Data Catalog / CDC NHSN &mdash; reporting period '
        f'{df["start_date"].iloc[0]} to {df["end_date"].iloc[0]}',
        unsafe_allow_html=True,
    )
    st.write("")

    n_facilities = df["facility_id"].nunique()
    n_worse = df[df["benchmark_flag"] == "Worse than National Benchmark"]["facility_id"].nunique()
    n_fdr_worse = df[df["significant_fdr_05"]]["facility_id"].nunique()
    n_systemic = (
        df[df["benchmark_flag"] == "Worse than National Benchmark"]
        .groupby("facility_id")["infection_type"].nunique()
    )
    n_systemic = (n_systemic >= 2).sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(theme.kpi_card("Facilities Surveilled", f"{n_facilities:,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(theme.kpi_card("Flagged Worse (CI-based)", f"{n_worse:,}", "of 6 infection types", "critical"), unsafe_allow_html=True)
    with c3:
        st.markdown(theme.kpi_card("Flagged Worse (FDR-corrected)", f"{n_fdr_worse:,}", "after multiple-testing correction", "critical"), unsafe_allow_html=True)
    with c4:
        st.markdown(theme.kpi_card("Systemic Outliers", f"{n_systemic:,}", "2+ infection types flagged", "critical"), unsafe_allow_html=True)

    st.write("")
    st.subheader("Headline findings")
    best_auc = max(model_results["results"], key=lambda k: model_results["results"][k]["roc_auc"])
    st.markdown(f"""
- **{n_worse} facility-measures** are statistically significantly worse than the national infection benchmark (SIR confidence interval excludes 1.0); **{n_systemic} facilities** are flagged on 2 or more distinct infection types simultaneously — a much stronger signal of a systemic infection-control issue than a single-measure statistical fluke.
- Running ~11,300 simultaneous statistical tests (one per facility-measure) inflates false positives: an uncorrected significance threshold flags roughly **28% of all tests**, while Benjamini-Hochberg FDR correction narrows genuine worse-than-benchmark outliers to **{n_fdr_worse}** — see the *Statistical Testing* page for the full comparison.
- A predictive model trained on real hospital characteristics (type, ownership, ER services, CMS star rating) and infection type reaches **ROC-AUC {model_results['results'][best_auc]['roc_auc']:.2f}** ({best_auc.replace('_', ' ')}) at predicting which facility-measures get flagged — see the *Predictive Model* page for the full evaluation, including how this imbalanced classification problem (~1.7% positive rate) was handled.
- Hospital **CMS star rating is inversely associated with being flagged** (higher rating -> lower odds of a worse-than-benchmark flag), a sanity-check result consistent with star ratings partly reflecting infection control quality.
""")

    st.divider()
    st.subheader("Methodology summary")
    st.markdown("""
1. **Data**: CMS Provider Data Catalog, *Healthcare Associated Infections - Hospital* (`77hc-ibv8`) merged with *Hospital General Information* (`xubh-q36u`) on Facility ID — both real, public datasets sourced ultimately from CDC NHSN and CMS Care Compare.
2. **Benchmarking**: funnel-plot control limits (Poisson variance around predicted infection volume) separate genuine outlier facilities from small-sample noise at a single reporting-period snapshot.
3. **Statistical testing**: exact Poisson test per facility-measure (not a normal approximation), with Benjamini-Hochberg FDR correction across all simultaneous tests.
4. **Predictive modeling**: logistic regression and gradient boosting (`HistGradientBoostingClassifier`), stratified train/test split, evaluated on ROC-AUC *and* PR-AUC given severe class imbalance.
5. **Control chart demo**: a supplementary page reconstructs a synthetic monthly SIR series (clearly labeled) purely to demonstrate CUSUM/EWMA technique, since CMS does not publish per-facility monthly history.
""")

    report_html = _build_report_html(df, n_facilities, n_worse, n_fdr_worse, n_systemic, model_results, best_auc)
    st.download_button(
        "Download executive summary (HTML report)",
        data=report_html,
        file_name="infection_control_executive_summary.html",
        mime="text/html",
    )


def _build_report_html(df, n_facilities, n_worse, n_fdr_worse, n_systemic, model_results, best_auc):
    top_outliers = (
        df[df["benchmark_flag"] == "Worse than National Benchmark"]
        .sort_values("ci_lower", ascending=False)
        .head(15)[["facility_name", "state", "infection_type", "sir", "p_value", "q_value_fdr"]]
    )
    rows = "".join(
        f"<tr><td>{r.facility_name}</td><td>{r.state}</td><td>{r.infection_type}</td>"
        f"<td>{r.sir:.2f}</td><td>{r.p_value:.4f}</td><td>{r.q_value_fdr:.4f}</td></tr>"
        for r in top_outliers.itertuples()
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Infection Control Command Center — Executive Summary</title>
<style>
body{{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;max-width:900px;margin:40px auto;color:#0b0b0b;}}
h1{{color:#2a78d6;}} table{{border-collapse:collapse;width:100%;margin-top:12px;}}
th,td{{border:1px solid #e1e0d9;padding:6px 10px;text-align:left;font-size:0.9rem;}}
th{{background:#f9f9f7;}} .kpi{{display:inline-block;margin-right:24px;}}
.kpi b{{font-size:1.4rem;display:block;color:#2a78d6;}}
</style></head><body>
<h1>Infection Control Command Center — Executive Summary</h1>
<p>Real data: CMS Provider Data Catalog (Healthcare Associated Infections - Hospital, Hospital General Information)</p>
<div class="kpi"><b>{n_facilities:,}</b>Facilities surveilled</div>
<div class="kpi"><b>{n_worse:,}</b>Flagged worse (CI-based)</div>
<div class="kpi"><b>{n_fdr_worse:,}</b>Flagged worse (FDR-corrected)</div>
<div class="kpi"><b>{n_systemic:,}</b>Systemic outliers (2+ types)</div>
<h2>Top facilities flagged worse than national benchmark</h2>
<table><tr><th>Facility</th><th>State</th><th>Infection Type</th><th>SIR</th><th>p-value</th><th>FDR q-value</th></tr>{rows}</table>
<h2>Predictive model</h2>
<p>Best model: {best_auc.replace('_',' ')}, ROC-AUC {model_results['results'][best_auc]['roc_auc']:.3f}, PR-AUC {model_results['results'][best_auc]['pr_auc']:.3f} (baseline {model_results['test_positive_rate']:.3f}).</p>
</body></html>"""
