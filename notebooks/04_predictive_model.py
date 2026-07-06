"""
Predictive model: which facility-measures get flagged as statistically worse
than the national HAI benchmark? Trained entirely on REAL data (CMS Care
Compare infection measures + CMS Hospital General Information metadata) --
no synthetic data anywhere in this project.

Target: benchmark_flag == "Worse than National Benchmark" (the real,
CI-derived CMS flag used throughout this dashboard). Positive rate is ~1.7%
(189 / 11,284), so this is a genuinely imbalanced classification problem --
handled explicitly below (class_weight='balanced', PR-AUC alongside ROC-AUC,
a confusion matrix at a chosen threshold) rather than reporting a
misleadingly high plain accuracy.

Two models for a methodological comparison:
  - Logistic regression: interpretable coefficients / odds ratios
  - HistGradientBoostingClassifier: handles missing values + categoricals
    natively, generally stronger nonlinear performance
"""
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
)
from sklearn.inspection import permutation_importance
import joblib

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FEATURES_NUMERIC = ["predicted_infections_log", "overall_rating"]
FEATURES_CATEGORICAL = ["infection_type", "hospital_type", "hospital_ownership", "emergency_services", "region"]


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["predicted_infections_log"] = np.log1p(df["predicted_infections"])
    df["overall_rating"] = df["overall_rating"].fillna(df["overall_rating"].median())
    for c in FEATURES_CATEGORICAL:
        df[c] = df[c].fillna("Unknown").astype("category")
    return df


def main():
    con = duckdb.connect(str(DATA_DIR / "hai.duckdb"), read_only=True)
    df = con.execute("SELECT * FROM hai_facility_measures_enriched").df()
    con.close()

    df = prep_features(df)
    y = (df["benchmark_flag"] == "Worse than National Benchmark").astype(int)
    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------- Logistic regression (one-hot + scaling) ----------------
    X_train_dummies = pd.get_dummies(X_train, columns=FEATURES_CATEGORICAL, drop_first=True)
    X_test_dummies = pd.get_dummies(X_test, columns=FEATURES_CATEGORICAL, drop_first=True)
    X_test_dummies = X_test_dummies.reindex(columns=X_train_dummies.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dummies)
    X_test_scaled = scaler.transform(X_test_dummies)

    logreg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    logreg.fit(X_train_scaled, y_train)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]

    # ---------------- Gradient boosting (native categorical + NaN support) ----------------
    hgb = HistGradientBoostingClassifier(
        categorical_features="from_dtype", class_weight="balanced", random_state=42, max_iter=300,
    )
    hgb.fit(X_train, y_train)
    hgb_proba = hgb.predict_proba(X_test)[:, 1]

    results = {}
    for name, proba in [("logistic_regression", logreg_proba), ("gradient_boosting", hgb_proba)]:
        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        preds_at_half = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, preds_at_half)
        results[name] = {
            "roc_auc": roc_auc, "pr_auc": pr_auc,
            "roc_curve": (fpr.tolist(), tpr.tolist()),
            "pr_curve": (prec.tolist(), rec.tolist()),
            "confusion_matrix": cm.tolist(),
            "y_test": y_test.tolist(), "proba": proba.tolist(),
        }
        print(f"\n=== {name} ===")
        print(f"ROC-AUC: {roc_auc:.3f}  PR-AUC: {pr_auc:.3f}  (baseline PR-AUC at random = {y_test.mean():.3f})")
        print(classification_report(y_test, preds_at_half, target_names=["Not flagged", "Worse than national"]))

    # Logistic regression coefficients (odds ratios) for interpretability
    coefs = pd.DataFrame({
        "feature": X_train_dummies.columns,
        "coefficient": logreg.coef_[0],
        "odds_ratio": np.exp(logreg.coef_[0]),
    }).sort_values("coefficient", key=abs, ascending=False)

    # Permutation importance for the gradient boosting model
    perm = permutation_importance(hgb, X_test, y_test, n_repeats=10, random_state=42, scoring="average_precision")
    perm_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    joblib.dump({
        "results": results,
        "logreg_coefs": coefs,
        "hgb_perm_importance": perm_importance,
        "feature_columns_logreg": list(X_train_dummies.columns),
        "test_positive_rate": float(y_test.mean()),
    }, DATA_DIR / "predictive_model_results.joblib")

    print("\nTop logistic regression coefficients (odds ratios):")
    print(coefs.head(10).to_string(index=False))
    print("\nTop gradient boosting permutation importances:")
    print(perm_importance.head(10).to_string(index=False))
    print(f"\nSaved: {DATA_DIR / 'predictive_model_results.joblib'}")


if __name__ == "__main__":
    main()
