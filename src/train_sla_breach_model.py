
---

## 3️⃣ Training script: `src/train_sla_breach_model.py`

This script is a clean, production-ish pipeline you can run locally or in Azure.

```python
"""
train_sla_breach_model.py

Train a predictive model to identify change requests that are at risk
of breaching their SLA.
"""

import os
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

# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "dummy_change_request_data.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "gb_sla_breach_model.pkl"


def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load change request data from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    return df


def build_features(df: pd.DataFrame):
    """
    Basic feature engineering.

    NOTE: This assumes the dataset has example columns like:
      - 'priority'
      - 'team'
      - 'lead_time_hours'
      - 'sla_breached'
    Update the column names here to match your actual data.
    """
    # Drop rows with missing target
    df = df.dropna(subset=["sla_breached"])

    # Example: ensure binary target
    df["sla_breached"] = df["sla_breached"].astype(int)

    # Example numeric + categorical columns
    numeric_features = ["lead_time_hours"]
    categorical_features = ["priority", "team"]

    # Separate features/target
    X = df[numeric_features + categorical_features]
    y = df["sla_breached"]

    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features) -> Pipeline:
    """Create sklearn pipeline with preprocessing + model."""
    numeric_transformer = "passthrough"

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
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

def main():
    print(f"Loading data from: {DATA_PATH}")
    df = load_data(DATA_PATH)

    print("Building features...")
    X, y, num_cols, cat_cols = build_features(df)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("Building pipeline...")
    clf = build_pipeline(num_cols, cat_cols)

    print("Training model...")
    clf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.3f}")

    print(f"\nSaving model to: {MODEL_PATH}")
    joblib.dump(clf, MODEL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
