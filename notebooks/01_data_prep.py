"""
Data prep: Healthcare-Associated Infections (HAI) - Hospital
Source: CMS Provider Data Catalog, dataset 77hc-ibv8 (real data, CDC NHSN measures)
Reshapes the wide CMS export (one row per Facility x Measure) into a tidy
long table: one row per Facility x Infection Type, with SIR, CI bounds,
numerator, eligible cases, and days/opportunities of patient care.

Note on scope: CMS publishes a single most-recent 12-month rolling reporting
period per snapshot (no per-facility monthly history in this file). True
longitudinal (month-over-month) trending would require stitching together
sequential CMS archive snapshots, which are not available as a clean API.
Instead, we use the standard epidemiological approach for single-period
facility benchmarking: funnel-plot control limits (based on Poisson variance
around eligible/predicted cases), which is what real infection-control
epidemiologists use to flag outlier facilities from one reporting period.
"""
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "hai_hospital_raw.csv"

INFECTION_TYPES = {
    1: "CLABSI (Central Line Bloodstream Infection)",
    2: "CAUTI (Catheter-Associated UTI)",
    3: "SSI - Colon Surgery",
    4: "SSI - Abdominal Hysterectomy",
    5: "MRSA Bacteremia",
    6: "C. difficile (C.diff)",
}


def load_raw():
    df = pd.read_csv(RAW_PATH, dtype={"Facility ID": str, "ZIP Code": str})
    df.columns = [c.strip() for c in df.columns]
    return df


def reshape_long(df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "Facility ID", "Facility Name", "Address", "City/Town", "State",
        "ZIP Code", "County/Parish", "Start Date", "End Date",
    ]
    facility_meta = df[meta_cols].drop_duplicates(subset=["Facility ID"])

    rows = []
    for hai_num, label in INFECTION_TYPES.items():
        prefix = f"HAI_{hai_num}_"
        sub = df[df["Measure ID"].str.startswith(prefix)][
            ["Facility ID", "Measure ID", "Score"]
        ]
        pivot = sub.pivot(index="Facility ID", columns="Measure ID", values="Score")
        pivot = pivot.rename(columns=lambda c: c.replace(prefix, ""))
        pivot["infection_type"] = label
        pivot["hai_number"] = hai_num
        pivot = pivot.reset_index()
        rows.append(pivot)

    long_df = pd.concat(rows, ignore_index=True)
    # Per CMS's own data dictionary (Downloadable Database Dictionary, Appendix B,
    # "HAI" section): Numerator = observed number of infections; Denominator =
    # predicted number of infections. In this export that denominator is ELIGCASES.
    # DOPC ("device/patient days") is a separate exposure-volume metric, not the
    # SIR denominator, and is kept only for context (e.g. rate-per-1000-days views).
    long_df = long_df.rename(columns={
        "SIR": "sir",
        "CILOWER": "ci_lower",
        "CIUPPER": "ci_upper",
        "NUMERATOR": "numerator",
        "ELIGCASES": "predicted_infections",
        "DOPC": "device_days",
    })

    long_df = long_df.merge(facility_meta, on="Facility ID", how="left")

    for col in ["sir", "ci_lower", "ci_upper", "numerator", "predicted_infections", "device_days"]:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

    long_df = long_df.dropna(subset=["sir", "predicted_infections"])
    long_df = long_df[long_df["predicted_infections"] > 0]

    # Statistical significance: CI excludes 1.0 (the national benchmark ratio)
    def flag(row):
        if pd.isna(row["ci_lower"]) or pd.isna(row["ci_upper"]):
            return "Not Available"
        if row["ci_lower"] > 1.0:
            return "Worse than National Benchmark"
        if row["ci_upper"] < 1.0:
            return "Better than National Benchmark"
        return "No Different than National Benchmark"

    long_df["benchmark_flag"] = long_df.apply(flag, axis=1)

    keep = [
        "Facility ID", "Facility Name", "State", "County/Parish",
        "hai_number", "infection_type", "sir", "ci_lower", "ci_upper",
        "numerator", "predicted_infections", "device_days", "benchmark_flag",
        "Start Date", "End Date",
    ]
    long_df = long_df[keep].rename(columns={
        "Facility ID": "facility_id",
        "Facility Name": "facility_name",
        "State": "state",
        "County/Parish": "county",
        "Start Date": "start_date",
        "End Date": "end_date",
    })
    return long_df


def build_funnel_control_limits(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard SIR funnel-plot control limits: for a given number of predicted
    infections (E), the SIR (O/E) under a Poisson process has variance 1/E.
    2-sigma / 3-sigma bands narrow as E grows -> classic funnel shape.
    """
    frames = []
    for hai_type, grp in long_df.groupby("infection_type"):
        e_range = np.linspace(grp["predicted_infections"].min(),
                               grp["predicted_infections"].max(), 200)
        e_range = np.clip(e_range, 0.5, None)
        sigma = np.sqrt(1.0 / e_range)
        cl = pd.DataFrame({
            "infection_type": hai_type,
            "predicted_infections": e_range,
            "upper_2sigma": 1 + 2 * sigma,
            "lower_2sigma": np.clip(1 - 2 * sigma, 0, None),
            "upper_3sigma": 1 + 3 * sigma,
            "lower_3sigma": np.clip(1 - 3 * sigma, 0, None),
        })
        frames.append(cl)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_raw()
    long_df = reshape_long(df)
    control_limits = build_funnel_control_limits(long_df)

    long_df.to_parquet(DATA_DIR / "hai_long.parquet", index=False)
    control_limits.to_parquet(DATA_DIR / "hai_control_limits.parquet", index=False)

    con = duckdb.connect(str(DATA_DIR / "hai.duckdb"))
    con.execute("CREATE OR REPLACE TABLE hai_facility_measures AS SELECT * FROM long_df")
    con.execute("CREATE OR REPLACE TABLE hai_control_limits AS SELECT * FROM control_limits")
    con.close()

    print(f"Reshaped {len(long_df):,} facility-measure rows across "
          f"{long_df['facility_id'].nunique():,} facilities.")
    print(long_df["benchmark_flag"].value_counts())
    print(f"\nSaved: {DATA_DIR / 'hai_long.parquet'}")
    print(f"Saved: {DATA_DIR / 'hai_control_limits.parquet'}")
    print(f"Saved: {DATA_DIR / 'hai.duckdb'}")


if __name__ == "__main__":
    main()
