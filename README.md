# Telco Customer Churn - Classification Models & Streamlit App

Machine Learning - Assignment 2 (M.Tech AIML, BITS WILP BITS-ID:2026AK05062)

## a. Problem statement

A telecom operator loses revenue every time a subscriber cancels. The goal is to predict, from a customer's
profile and billing information, whether the customer will **churn** (leave the service) so that retention
offers can be directed at the people most likely to go.

This is a **binary classification** problem (`Churn` = Yes / No). Five classifiers are trained on the same
data, compared on six metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and exposed through an interactive
Streamlit app that scores an uploaded test CSV with the saved models (no re-training in the app).

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Telco Customer Churn (IBM sample data, published on Kaggle) |
| File | `data/Telco-Customer-Churn.csv` |
| Instances | 7,043 customers (minimum required: 500) |
| Features | 20 input features after dropping the `customerID` identifier (minimum required: 12) |
| Target | `Churn` - Yes (1,869 customers, 26.5%) / No (5,174 customers, 73.5%) |
| Type | Binary classification, moderately imbalanced |

**Feature groups**

- *Demographics:* `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- *Account:* `tenure` (months), `Contract`, `PaperlessBilling`, `PaymentMethod`
- *Services:* `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
  `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- *Billing:* `MonthlyCharges`, `TotalCharges`

**Pre-processing** (identical in training and in the app, see `churn_utils.py`)

1. `customerID` is dropped. It is a unique identifier, not a predictor.
2. `TotalCharges` is stored as text in the raw file and is blank for 11 customers whose `tenure` is 0 (nothing
   billed yet). It is converted to a number and those blanks are set to 0.
3. Numeric columns (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) are median-imputed and standardised.
4. The 15 categorical columns are one-hot encoded (unknown categories are ignored at prediction time).
5. These steps live inside each saved scikit-learn `Pipeline`, so a model file accepts the *raw* CSV columns.

**Train / test split:** stratified 80 / 20 with `random_state=26` -> 5,634 training rows and 1,409 test rows
(churn rate 26.5% in both). The 1,409 held-out rows are saved as `test_data.csv` and all numbers below are
measured on them. Hyper-parameters were tuned **only** on the training split (5-fold stratified CV, scoring =
ROC-AUC); the test rows were not used for tuning.

## c. GitHub repository link

GitHub repository: `https://github.com/ajit9291/ML_Assignment_2`

Live Streamlit app: `https://mlassignment2-mscs9ivugkf4fy9ytbdyge.streamlit.app/`

```
ML_Assignment2_Submission/
|-- app.py                  Streamlit app (upload CSV, pick model, metrics, confusion matrix, report, ROC)
|-- churn_utils.py          shared cleaning, pre-processing and metric code (used by training and app)
|-- train_models.py         trains + tunes the 5 models, writes model/, test_data.csv and results/
|-- requirements.txt
|-- README.md
|-- test_data.csv           held-out test set (1,409 rows, raw columns + Churn)
|-- data/Telco-Customer-Churn.csv
|-- model/                  logistic_regression / decision_tree / knn / naive_bayes /
|                           support_vector_machine .joblib + metadata.json
`-- results/                metrics_comparison.csv, roc_curves.png, confusion_matrices.png, metrics_bar_chart.png
```

**Run locally**

```bash
pip install -r requirements.txt
python train_models.py      # optional - re-creates model/, test_data.csv and results/
streamlit run app.py
```

## d. Models used

All five models were trained on the same split with the same pre-processing. Positive class = Churn.
Predictions use a 0.5 probability threshold. Best hyper-parameters (from the CV grid search):

| Model | Tuned hyper-parameters |
|---|---|
| Logistic Regression | `C=10` |
| Decision Tree | `criterion=entropy`, `max_depth=10`, `min_samples_leaf=50` |
| KNN | `n_neighbors=51`, `weights=uniform` |
| Naive Bayes (Gaussian) | `var_smoothing=1.0` |
| Support Vector Machine | RBF kernel, `C=0.5`, `gamma=0.01` |

### Comparison table (1,409 held-out test rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8091 | 0.8367 | 0.6768 | 0.5374 | 0.5991 | 0.4814 |
| Decision Tree | 0.7906 | 0.8210 | 0.6223 | 0.5374 | 0.5768 | 0.4407 |
| KNN | 0.7828 | 0.8254 | 0.5929 | 0.5802 | 0.5865 | 0.4393 |
| Naive Bayes | 0.7126 | 0.8118 | 0.4755 | 0.8048 | 0.5978 | 0.4297 |
| Support Vector Machine | 0.8027 | 0.8237 | 0.6765 | 0.4920 | 0.5697 | 0.4553 |

(The same numbers are in `results/metrics_comparison.csv`. Plots: `results/roc_curves.png`,
`results/confusion_matrices.png`, `results/metrics_bar_chart.png`.)

Confusion-matrix counts on the test set (TN / FP / FN / TP): Logistic Regression 939 / 96 / 173 / 201,
Decision Tree 913 / 122 / 173 / 201, KNN 886 / 149 / 157 / 217, Naive Bayes 703 / 332 / 73 / 301,
SVM 947 / 88 / 190 / 184.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best all-rounder. It has the highest accuracy (0.809), AUC (0.837), F1 (0.599) and MCC (0.481). Churn here depends largely on additive effects, which a linear model on one-hot features captures well: its largest coefficients are long tenure and a two-year contract (both lower churn), and fibre-optic internet and a month-to-month contract (both higher churn). It is well-calibrated and the least prone to over-fitting. Recall is modest (0.537): it misses 173 of the 374 churners at the 0.5 threshold. |
| Decision Tree | Accuracy 0.791 and AUC 0.821, a little below logistic regression. Limiting depth to 10 and requiring 50 samples per leaf was necessary: an unrestricted tree scores AUC 1.00 on the training rows but only 0.66 on the test rows (the tuned tree: 0.87 / 0.82). Its recall equals logistic regression's (0.537) but with more false alarms (122 vs 96), so precision is lower (0.622). Its main advantage is interpretability: the most important features are month-to-month contract (importance 0.46), tenure (0.14) and fibre-optic internet (0.10). |
| KNN | Accuracy 0.783 and AUC 0.825. It needed a large neighbourhood (k=51) to smooth out noisy labels: test AUC is 0.65 at k=1, 0.77 at k=5 and 0.83 at k=51. It finds more churners than the linear model (recall 0.580) but flags more loyal customers (149 false positives), so its MCC (0.439) is lower. It is also the slowest at prediction time, since it keeps all training data. |
| Naive Bayes | Lowest accuracy (0.713) and precision (0.476), but the highest recall (0.805): it catches 301 of 374 churners, at the cost of 332 false alarms. The independence assumption is violated because the one-hot service columns are strongly correlated, which inflates the churn probabilities, and the Gaussian assumption is a poor fit for binary columns. Its F1 (0.598) is nonetheless close to the best because of that recall. It would suit a campaign where missing a churner costs far more than a wasted retention offer. |
| Support Vector Machine | Accuracy 0.803 and the second-highest MCC (0.455), the highest precision together with logistic regression (0.677), but the lowest recall (0.492): it is the most conservative model and misses 190 churners. AUC is 0.824. It was by far the slowest to train (about 75 s for the grid search, against 3 s for logistic regression) for no gain over the linear model. |
| **Overall winner for this dataset** | **Logistic Regression.** It ranks first on accuracy, AUC, F1 and MCC, is the cheapest to train and explain, and is the most robust choice. MCC is the headline metric because the classes are imbalanced (accuracy alone would reward predicting "no churn"). If the business priority were to catch as many churners as possible, Naive Bayes (recall 0.805) or any model with a lowered threshold would be preferable. The app's threshold slider lets you see that trade-off. |

**Caveat:** the AUC gaps between models are small (0.81 to 0.84). With only 374 churners in the test set, the
standard error of an AUC is roughly 0.013, so the ordering of the four closest models on AUC is not statistically
conclusive. Logistic regression's lead on MCC and accuracy is clearer.

## Streamlit app features

- **Dataset upload (CSV):** upload a test file with the raw Telco columns plus `Churn`. A bundled sample
  (`test_data.csv`) can be used instead, and downloaded from the sidebar. Missing columns or invalid target values
  produce a clear error message.
- **Model selection dropdown:** Logistic Regression, Decision Tree, KNN, Naive Bayes, Support Vector Machine.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model, plus a decision-threshold
  slider.
- **Confusion matrix and classification report**, an ROC curve, and the per-row predictions (downloadable).
- **Side-by-side comparison** of all five models on the uploaded file.

The app loads the saved `.joblib` pipelines from `model/` and only calls `predict` / `predict_proba`, so nothing is
re-trained on the uploaded data.
