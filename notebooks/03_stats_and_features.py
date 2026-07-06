"""
Statistical testing + real feature engineering for the HAI dashboard.

1. Exact Poisson test (the "E-test" standard in infection-control epidemiology)
   for each facility-measure: H0: observed infections ~ Poisson(predicted
   infections), i.e. true SIR = 1. Two-sided exact p-value, no normal
   approximation -- appropriate here since many facilities have small
   predicted-infection counts where a CI-based approximation is shaky.
2. Benjamini-Hochberg FDR correction across all ~11K simultaneous tests --
   with this many tests run at once, an uncorrected alpha=0.05 would produce
   hundreds of false positives by chance alone; this is the standard fix.
3. Merges REAL hospital metadata (CMS Hospital General Information: type,
   ownership, ER services, star rating) onto the facility-measure table, to
   give the downstream predictive model real, non-tautological features.
"""
import numpy as np
import pandas as pd
import duckdb
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Standard US Census Bureau 4-region grouping, used to keep the categorical
# feature space small (52 states/territories -> 4 regions) for the ML model.
REGION_MAP = {
    **{s: "Northeast" for s in ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"]},
    **{s: "Midwest" for s in ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"]},
    **{s: "South" for s in ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV",
                              "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"]},
    **{s: "West" for s in ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY",
                             "AK", "CA", "HI", "OR", "WA"]},
}


def exact_poisson_test(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Two-sided exact Poisson test p-value for each (O, E) pair."""
    observed = np.asarray(observed)
    expected = np.clip(np.asarray(expected), 1e-6, None)
    p_lower = stats.poisson.cdf(observed, expected)
    p_upper = stats.poisson.sf(observed - 1, expected)
    return np.minimum(1.0, 2 * np.minimum(p_lower, p_upper))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values)."""
    p = np.asarray(p_values)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank down
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0, 1)
    return q


def load_hospital_metadata() -> pd.DataFrame:
    meta = pd.read_csv(DATA_DIR / "hospital_general_info_raw.csv", dtype={"Facility ID": str})
    meta.columns = [c.strip() for c in meta.columns]
    meta = meta.rename(columns={
        "Facility ID": "facility_id",
        "Hospital Type": "hospital_type",
        "Hospital Ownership": "hospital_ownership",
        "Emergency Services": "emergency_services",
        "Hospital overall rating": "overall_rating",
    })
    keep = ["facility_id", "hospital_type", "hospital_ownership", "emergency_services", "overall_rating"]
    meta = meta[keep].drop_duplicates(subset="facility_id")
    meta["overall_rating"] = pd.to_numeric(meta["overall_rating"], errors="coerce")
    return meta


def main():
    con = duckdb.connect(str(DATA_DIR / "hai.duckdb"))
    facility_df = con.execute("SELECT * FROM hai_facility_measures").df()
    con.close()

    facility_df["p_value"] = exact_poisson_test(
        facility_df["numerator"].values, facility_df["predicted_infections"].values
    )
    facility_df["q_value_fdr"] = benjamini_hochberg(facility_df["p_value"].values)
    facility_df["significant_fdr_05"] = (facility_df["q_value_fdr"] < 0.05) & (facility_df["sir"] > 1)

    meta = load_hospital_metadata()
    facility_df = facility_df.merge(meta, on="facility_id", how="left")
    facility_df["region"] = facility_df["state"].map(REGION_MAP).fillna("Other/Territory")

    facility_df.to_parquet(DATA_DIR / "hai_long_enriched.parquet", index=False)
    con = duckdb.connect(str(DATA_DIR / "hai.duckdb"))
    con.execute("CREATE OR REPLACE TABLE hai_facility_measures_enriched AS SELECT * FROM facility_df")
    con.close()

    n_naive_sig = (facility_df["p_value"] < 0.05).sum()
    n_fdr_sig = facility_df["significant_fdr_05"].sum()
    print(f"Facility-measure tests: {len(facility_df):,}")
    print(f"Uncorrected p<0.05 (naive, inflated by multiple testing): {n_naive_sig:,}")
    print(f"FDR-corrected q<0.05 AND SIR>1 (true outlier candidates): {n_fdr_sig:,}")
    print(f"\nHospital metadata matched: {facility_df['hospital_type'].notna().mean()*100:.1f}% of rows")


if __name__ == "__main__":
    main()
