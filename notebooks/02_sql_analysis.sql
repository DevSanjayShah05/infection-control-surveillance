-- HAI Surveillance: SQL analysis queries (run against data/hai.duckdb)
-- Table: hai_facility_measures(facility_id, facility_name, state, county,
--   hai_number, infection_type, sir, ci_lower, ci_upper, numerator,
--   eligible_cases, predicted_infections, benchmark_flag, start_date, end_date)

-- 1. National benchmark summary per infection type: volume-weighted SIR
--    (sum of observed / sum of predicted, the epidemiologically correct
--    way to aggregate SIRs, vs a naive average which overweights small facilities)
SELECT
    infection_type,
    COUNT(DISTINCT facility_id)                              AS facilities_reporting,
    SUM(numerator)                                            AS total_observed_infections,
    SUM(predicted_infections)                                 AS total_predicted_infections,
    ROUND(SUM(numerator) / NULLIF(SUM(predicted_infections), 0), 3) AS national_sir,
    SUM(CASE WHEN benchmark_flag = 'Worse than National Benchmark' THEN 1 ELSE 0 END) AS n_worse,
    SUM(CASE WHEN benchmark_flag = 'Better than National Benchmark' THEN 1 ELSE 0 END) AS n_better
FROM hai_facility_measures
GROUP BY infection_type
ORDER BY national_sir DESC;

-- 2. State-level benchmarking: which states have the worst volume-weighted SIR
--    per infection type (surfaces geographic clusters worth investigating)
SELECT
    state,
    infection_type,
    COUNT(DISTINCT facility_id)                               AS facilities,
    ROUND(SUM(numerator) / NULLIF(SUM(predicted_infections), 0), 3) AS state_sir
FROM hai_facility_measures
GROUP BY state, infection_type
HAVING SUM(predicted_infections) >= 5   -- exclude tiny/unstable denominators
ORDER BY state_sir DESC
LIMIT 20;

-- 3. Outlier facilities: statistically significantly worse than national
--    benchmark (CI lower bound > 1.0), ranked by how far above 1.0
SELECT
    facility_name,
    state,
    infection_type,
    sir,
    ci_lower,
    ci_upper,
    numerator          AS observed_infections,
    predicted_infections
FROM hai_facility_measures
WHERE benchmark_flag = 'Worse than National Benchmark'
ORDER BY ci_lower DESC
LIMIT 25;

-- 4. Facilities flagged as outliers on 2+ distinct infection types
--    (a facility failing broadly across infection types signals a systemic
--    infection-control problem, not a single-measure statistical fluke)
SELECT
    facility_id,
    facility_name,
    state,
    COUNT(*) AS n_infection_types_flagged_worse,
    STRING_AGG(infection_type, '; ') AS flagged_types
FROM hai_facility_measures
WHERE benchmark_flag = 'Worse than National Benchmark'
GROUP BY facility_id, facility_name, state
HAVING COUNT(*) >= 2
ORDER BY n_infection_types_flagged_worse DESC;

-- 5. Reporting coverage / data completeness by state
--    (useful "state of the data" slide for a portfolio README)
SELECT
    state,
    COUNT(DISTINCT facility_id) AS facilities_with_any_hai_measure,
    ROUND(100.0 * SUM(CASE WHEN benchmark_flag != 'Not Available' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_measures_reportable
FROM hai_facility_measures
GROUP BY state
ORDER BY facilities_with_any_hai_measure DESC;
