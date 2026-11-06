"""Shared helpers for the Telco churn project.

Both the training script (train_models.py) and the Streamlit app (app.py) import from here, so the
raw CSV is always cleaned and scored in exactly the same way.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"
POSITIVE_LABEL = "Yes"  # a customer who churned
CLASS_NAMES = ["No churn", "Churn"]

NUMERIC_FEATURES: List[str] = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES: List[str] = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS: List[str] = FEATURE_COLUMNS + [TARGET]

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input table (feature columns only, in a fixed order, numerics coerced).

    Raises ValueError if any required feature column is missing.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required feature column(s): " + ", ".join(missing))

    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str).str.strip()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # In the raw file TotalCharges is blank for customers with tenure == 0 (nothing billed yet).
    X.loc[X["TotalCharges"].isna() & (X["tenure"] == 0), "TotalCharges"] = 0.0
    return X


def encode_target(series: pd.Series) -> np.ndarray:
    """Map Yes/No (or 1/0) churn labels to 1/0. Raises ValueError for anything else."""
    values = series.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0, "true": 1, "false": 0}
    encoded = values.map(mapping)
    if encoded.isna().any():
        bad = sorted(series[encoded.isna()].astype(str).unique())[:5]
        raise ValueError(f"'{TARGET}' column must contain Yes/No (or 1/0); found: {bad}")
    return encoded.astype(int).to_numpy()


def build_preprocessor() -> ColumnTransformer:
    """Median-impute and standardise numerics; one-hot encode categoricals (dense output for GaussianNB)."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """The six metrics required by the assignment. Churn (1) is the positive class."""
    y_true = np.asarray(y_true)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:  # only one class present in y_true
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
