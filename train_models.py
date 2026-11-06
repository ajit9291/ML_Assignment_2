"""Train and evaluate the five classifiers on the Telco Customer Churn dataset.

Run from the project folder:   python train_models.py

Outputs
    model/<name>.joblib          fitted preprocessing + classifier pipeline for each model
    model/metadata.json          feature list, best hyper-parameters, split info
    test_data.csv                held-out test rows (raw columns + Churn) for the Streamlit app
    results/metrics_comparison.csv, results/*.png
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from churn_utils import (
    CLASS_NAMES, FEATURE_COLUMNS, METRIC_NAMES, TARGET,
    build_preprocessor, compute_metrics, encode_target, prepare_features,
)

SEED = 26
TEST_SIZE = 0.20
BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = BASE / "model"
RESULTS_DIR = BASE / "results"

# name -> (file stem, estimator, hyper-parameter grid searched with 5-fold CV on the training split)
MODELS = {
    "Logistic Regression": ("logistic_regression", LogisticRegression(max_iter=2000), {
        "clf__C": [0.01, 0.1, 1, 10, 100],
    }),
    "Decision Tree": ("decision_tree", DecisionTreeClassifier(random_state=SEED), {
        "clf__max_depth": [3, 5, 7, 10, 15],
        "clf__min_samples_leaf": [5, 20, 50],
        "clf__criterion": ["gini", "entropy"],
    }),
    "KNN": ("knn", KNeighborsClassifier(), {
        "clf__n_neighbors": [11, 21, 31, 41, 51, 71],
        "clf__weights": ["uniform", "distance"],
    }),
    "Naive Bayes": ("naive_bayes", GaussianNB(), {
        "clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1, 0.3, 1.0],
    }),
    "Support Vector Machine": ("support_vector_machine", SVC(kernel="rbf", probability=True, random_state=SEED), {
        "clf__C": [0.1, 0.5, 1, 2],
        "clf__gamma": ["scale", 0.01, 0.003],
    }),
}


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(DATA_FILE)
    print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns from {DATA_FILE.name}")

    y_all = encode_target(raw[TARGET])
    raw_train, raw_test, y_train, y_test = train_test_split(
        raw, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=SEED
    )
    X_train, X_test = prepare_features(raw_train), prepare_features(raw_test)
    print(f"Train: {len(raw_train)} rows | Test: {len(raw_test)} rows | churn rate "
          f"train {y_train.mean():.3f} / test {y_test.mean():.3f}")

    # The held-out rows are saved untouched (raw columns + target) so the app can read them as an upload.
    raw_test.to_csv(BASE / "test_data.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params, fitted, probas, preds = [], {}, {}, {}, {}

    for name, (stem, estimator, grid) in MODELS.items():
        started = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        joblib.dump(model, MODEL_DIR / f"{stem}.joblib", compress=3)
        fitted[name], probas[name], preds[name] = model, y_proba, y_pred
        best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        rows.append({"ML Model Name": name, **metrics})
        print(f"{name:<24} CV AUC {search.best_score_:.4f} | test "
              + " ".join(f"{m}={metrics[m]:.4f}" for m in METRIC_NAMES)
              + f" | {time.time() - started:.1f}s | {best_params[name]}")

    results = pd.DataFrame(rows).round(4)
    results.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False)

    metadata = {
        "dataset": "Telco Customer Churn (Kaggle / IBM sample data)",
        "target": TARGET,
        "positive_class": "Yes (churned)",
        "feature_columns": FEATURE_COLUMNS,
        "random_state": SEED,
        "test_size": TEST_SIZE,
        "n_train": int(len(raw_train)),
        "n_test": int(len(raw_test)),
        "model_files": {name: f"{stem}.joblib" for name, (stem, _, _) in MODELS.items()},
        "best_params": best_params,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    # --- figures -------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in MODELS:
        RocCurveDisplay.from_predictions(y_test, probas[name], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_title("ROC curves on the held-out test set")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, name in zip(axes.ravel(), MODELS):
        cm = confusion_matrix(y_test, preds[name])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Confusion matrices on the held-out test set", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    ax = results.set_index("ML Model Name")[METRIC_NAMES].plot.bar(figsize=(11, 5), width=0.8, rot=15)
    ax.set_ylim(0, 1)
    ax.set_title("Test-set metrics by model")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.figure.tight_layout()
    ax.figure.savefig(RESULTS_DIR / "metrics_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    print("\n" + results.to_string(index=False))
    print(f"\nSaved models to {MODEL_DIR}, test data to test_data.csv, results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
