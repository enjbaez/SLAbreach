"""
train_sla_breach_model.py

Train a predictive model to identify change requests that are at risk
of breaching their SLA, using dummy_change_request_data.csv in /data.
"""

import pathlib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import joblib


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
# ROOT_DIR = repo root (…/SLAbreach)
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "dummy_change_request_data.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "gb_sla_breach_model.pkl"


# -------------------------------------------------------------------
# DATA LOADER
# -------------------------------------------------------------------
def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load change request data from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    return df


# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------
def build_features(df: pd.DataFrame):
    """
    Build features and target for SLA breach modeling.

    Uses your actual column names:
        plan,
        RxCaseID,
        cr date created,
        cr date closed,
        category,
        TestingCaseID,
        test results received date,
        test results approved date,
        comments

    Steps:
    - Normalize column names (remove spaces, lower_snake_case)
    - Parse date columns
    - Create duration features
    - Auto-generate sla_breached label:
        here: SLA breach = case takes > 30 days to close
        (you can change this threshold if needed)
    """

    # 1) Normalize column names to snake_case
    df = df.rename(
        columns={
            "plan": "plan",
            "RxCaseID": "rx_case_id",
            "cr date created": "cr_date_created",
            "cr date closed": "cr_date_closed",
            "category": "category",
            "TestingCaseID": "testing_case_id",
            "test results received date": "test_results_received_date",
            "test results approved date": "test_results_approved_date",
            "comments": "comments",
        }
    )

    # 2) Convert YYYYMMDD-like columns to datetime
    date_cols = [
        "cr_date_created",
        "cr_date_closed",
        "test_results_received_date",
        "test_results_approved_date",
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col].astype(str), format="%Y%m%d", errors="coerce"
            )

    # 3) Duration features
    # How long the change request was open
    df["duration_days"] = (
        df["cr_date_closed"] - df["cr_date_created"]
    ).dt.days

    # Lab/Testing turnaround time
    df["turnaround_days"] = (
        df["test_results_approved_date"] - df["test_results_received_date"]
    ).dt.days

    # Days from creation to approval
    df["days_to_approval"] = (
        df["test_results_approved_date"] - df["cr_date_created"]
    ).dt.days

    # 4) Auto-generate SLA breach flag
    # Here we define "breach" = duration_days > 30 days.
    # You can change 30 to whatever SLA threshold you want.
    df["sla_breached"] = (df["duration_days"] > 30).astype(int)

    TARGET_COL = "sla_breached"

    # 5) Choose features
    # Numeric features: engineered durations
    numeric_features = ["duration_days", "turnaround_days", "days_to_approval"]

    # Fill missing numeric values (e.g., when dates are missing)
    df[numeric_features] = df[numeric_features].fillna(
        df[numeric_features].median()
    )

    # Categorical features: plan, category, comments
    categorical_features = ["plan", "category", "comments"]

    # 6) Build X, y
    X = df[numeric_features + categorical_features].copy()
    y = df[TARGET_COL].copy()

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Target:", TARGET_COL)
    print("Class balance:\n", y.value_counts(normalize=True))

    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features) -> Pipeline:
    """Create sklearn pipeline with preprocessing + model."""

    numeric_transformer = "passthrough"

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return clf


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print(f"📂 Loading data from: {DATA_PATH}")
    df = load_data(DATA_PATH)

    print("🧱 Building features…")
    X, y, num_cols, cat_cols = build_features(df)

    print("✂️ Train/test split…")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("🛠 Building pipeline…")
    clf = build_pipeline(num_cols, cat_cols)

    print("🏋🏽‍♀️ Training model…")
    clf.fit(X_train, y_train)

    print("📊 Evaluating model…")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.3f}")

    print(f"\n💾 Saving model to: {MODEL_PATH}")
    joblib.dump(clf, MODEL_PATH)
    print("✅ Done.")


if __name__ == "__main__":
    main()
