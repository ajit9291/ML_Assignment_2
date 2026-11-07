"""Streamlit front-end for the Telco churn classifiers.

Upload a test CSV (raw Telco columns + the Churn target), pick one of the five saved models, and the app
scores the file with that pre-trained model. Nothing is re-trained here.
"""
import json
import sys
from pathlib import Path

import joblib
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from churn_utils import (
    CLASS_NAMES, METRIC_NAMES, REQUIRED_COLUMNS, TARGET,
    compute_metrics, encode_target, prepare_features,
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
SAMPLE_FILE = BASE / "test_data.csv"
THRESHOLD_DEFAULT = 0.50

st.set_page_config(page_title="Telco Churn Model Lab", page_icon=":telephone_receiver:", layout="wide")


@st.cache_data
def load_metadata():
    return json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))


@st.cache_resource
def _load_model_file(file_name: str):
    return joblib.load(MODEL_DIR / file_name)


def load_model(file_name: str):
    """Load a saved pipeline; show a clear message (not a raw traceback) if the environment does not match."""
    try:
        return _load_model_file(file_name)
    except Exception as exc:  # unpickling fails when scikit-learn / numpy versions differ from training
        trained_with = load_metadata().get("scikit_learn_version", "1.9.0")
        st.error(
            f"Could not load `{file_name}` ({type(exc).__name__}: {exc}).\n\n"
            f"The models were trained with scikit-learn **{trained_with}**, but this environment has "
            f"scikit-learn **{sklearn.__version__}** on Python {sys.version.split()[0]}. "
            "Install the versions in requirements.txt (scikit-learn 1.9 needs Python 3.11 or newer)."
        )
        st.stop()


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_FILE)


def score(model, X: pd.DataFrame, threshold: float):
    proba = model.predict_proba(X)[:, 1]
    return proba, (proba >= threshold).astype(int)


def confusion_figure(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred, labels=[0, 1]), annot=True, fmt="d", cmap="Blues",
                cbar=False, ax=ax, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def roc_figure(y_true, proba, auc):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    fpr, tpr, _ = roc_curve(y_true, proba)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------
# Sidebar: data + model + threshold
# ------------------------------------------------------------------------------------------------
meta = load_metadata()
model_files = meta["model_files"]

st.title("Telco Customer Churn - Model Lab")
st.caption("Five pre-trained classifiers. Upload a test CSV, choose a model, and inspect how it performs. "
           "Models are loaded from disk and are never re-trained on your upload.")

with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV", type="csv",
                                help="Must contain the raw Telco feature columns and the 'Churn' (Yes/No) target.")
    use_sample = st.checkbox("Use the bundled sample test set", value=uploaded is None, disabled=uploaded is not None)
    st.download_button("Download sample test_data.csv", SAMPLE_FILE.read_bytes(), file_name="test_data.csv",
                       mime="text/csv")

    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(model_files))
    st.caption("Tuned hyper-parameters: " + ", ".join(f"{k}={v}" for k, v in meta["best_params"][model_name].items()))

    st.header("3. Decision threshold")
    threshold = st.slider("Predict 'Churn' when probability >=", 0.05, 0.95, THRESHOLD_DEFAULT, 0.05)
    st.caption("0.50 is what the reported README metrics use. AUC does not depend on this threshold.")

# ------------------------------------------------------------------------------------------------
# Load + validate the test file
# ------------------------------------------------------------------------------------------------
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:  # unreadable / not a CSV
        st.error(f"Could not read the uploaded file as CSV: {exc}")
        st.stop()
    source = f"uploaded file `{uploaded.name}`"
elif use_sample:
    df = load_sample()
    source = "bundled sample `test_data.csv`"
else:
    st.info("Upload a test CSV in the sidebar, or tick the sample option, to begin.")
    st.stop()

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error("The file is missing required column(s): " + ", ".join(missing))
    st.stop()

try:
    X = prepare_features(df)
    y_true = encode_target(df[TARGET])
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.success(f"Scoring {len(df):,} rows from the {source}. Actual churn rate in this file: {y_true.mean():.1%}.")
with st.expander("Preview the data"):
    st.dataframe(df.head(20), width="stretch")

# ------------------------------------------------------------------------------------------------
# Selected model
# ------------------------------------------------------------------------------------------------
model = load_model(model_files[model_name])
proba, y_pred = score(model, X, threshold)
metrics = compute_metrics(y_true, y_pred, proba)

st.subheader(f"{model_name} - evaluation metrics")
cols = st.columns(len(METRIC_NAMES))
for col, name in zip(cols, METRIC_NAMES):
    value = metrics[name]
    col.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}")
if len(np.unique(y_true)) < 2:
    st.warning("The file contains only one class, so AUC is undefined and some metrics are not meaningful.")

tab_cm, tab_report, tab_roc, tab_pred = st.tabs(["Confusion matrix", "Classification report", "ROC curve", "Predictions"])

with tab_cm:
    left, right = st.columns([1, 1])
    with left:
        st.pyplot(confusion_figure(y_true, y_pred))
    with right:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        st.markdown(
            f"- **True negatives:** {tn:,} loyal customers correctly left alone\n"
            f"- **False positives:** {fp:,} loyal customers wrongly flagged as churners\n"
            f"- **False negatives:** {fn:,} churners the model missed\n"
            f"- **True positives:** {tp:,} churners correctly caught"
        )

with tab_report:
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

with tab_roc:
    if np.isnan(metrics["AUC"]):
        st.info("ROC curve needs both classes in the test file.")
    else:
        st.pyplot(roc_figure(y_true, proba, metrics["AUC"]))

with tab_pred:
    out = df.copy()
    out["churn_probability"] = proba.round(4)
    out["predicted_churn"] = np.where(y_pred == 1, "Yes", "No")
    st.dataframe(out, width="stretch")
    st.download_button("Download predictions as CSV", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{model_files[model_name].replace('.joblib', '')}.csv", mime="text/csv")

# ------------------------------------------------------------------------------------------------
# Side-by-side comparison of all five models on the same file
# ------------------------------------------------------------------------------------------------
st.divider()
st.subheader("All five models on this file")
rows = []
for name, file_name in model_files.items():
    p, pred = score(load_model(file_name), X, threshold)
    rows.append({"ML Model Name": name, **compute_metrics(y_true, pred, p)})
comparison = pd.DataFrame(rows).set_index("ML Model Name")
st.dataframe(comparison.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"), width="stretch")
st.caption("Green = best model for that metric on this file.")
st.bar_chart(comparison[METRIC_NAMES])
