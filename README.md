# Telco Customer Churn - Classification Model Comparison

## a. Problem Statement
Telecom companies lose revenue when customers leave. The goal here is to predict, from a customer's account, service and billing details, whether they will **churn**. Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM) are trained and tuned on the same data and compared with Accuracy, AUC, Precision, Recall, F1 and MCC. A Streamlit app lets anyone upload a test CSV and evaluate the saved models without retraining.

## b. Dataset Description
- **Source:** Telco Customer Churn (IBM sample data, available on Kaggle)
- **Task:** binary classification - `Churn` Yes (1) / No (0)
- **Instances:** 7,043 customers (26.5% churned)
- **Features:** 19 inputs after removing `customerID`
  - numeric (3): `tenure`, `MonthlyCharges`, `TotalCharges`
  - categorical (16): `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Cleaning:** `TotalCharges` is stored as text; its 11 blank rows (customers with tenure 0) were set to 0.
- **Preprocessing:** numeric columns standardised, categorical columns one-hot encoded; both steps are stored inside each saved pipeline (`model/*.joblib`).
- **Split:** 75% train / 25% test, stratified, `random_state=7` -> 5,282 train / 1,761 test. The test part is `test_data.csv`.
- **Tuning:** `GridSearchCV` (3-fold, scored on AUC) on the training part only. Chosen values are in `model/best_params.json`.
- **Note:** the classes are imbalanced, so Precision, Recall, F1 and MCC are reported for the churn class alongside Accuracy.

## c. GitHub Repository Link
https://github.com/Shyamkumarvn/telco-churn-classification

Live app: https://telco-churn-models.streamlit.app

## d. Models Used

### Comparison table (1,761 test customers)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8069 | 0.8581 | 0.6658 | 0.5460 | 0.6000 | 0.4784 |
| Decision Tree | 0.7939 | 0.8369 | 0.6327 | 0.5310 | 0.5774 | 0.4454 |
| KNN | 0.7956 | 0.8466 | 0.6192 | 0.5953 | 0.6070 | 0.4691 |
| Naive Bayes | 0.7087 | 0.8321 | 0.4733 | 0.8715 | 0.6134 | 0.4605 |
| Support Vector Machine | 0.8098 | 0.8305 | 0.7012 | 0.4925 | 0.5786 | 0.4726 |

Tuned settings: LR `C=5`; Decision Tree `max_depth=5, min_samples_leaf=50`; KNN `k=31`; Naive Bayes `var_smoothing=0.1`; SVM (RBF) `C=0.5`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.8581) and highest MCC (0.4784), with accuracy only 0.003 behind the best. It ranks churners better than any other model, which fits a problem where contract type, tenure and charges act in a mostly additive way. |
| Decision Tree | Weakest overall (MCC 0.4454). Even with depth and leaf-size limits it makes coarse, step-like predictions, so its AUC (0.8369) trails LR and KNN. Its main advantage is that the rules are easy to read. |
| KNN | Balanced: second-best AUC (0.8466) and the best recall among the non-NB models (0.5953). The tuning picked k=31, the top of the grid, meaning heavy smoothing helps in this noisy one-hot feature space. |
| Naive Bayes | Catches 87% of churners (recall 0.8715) but at the cost of precision (0.4733) and accuracy (0.7087); it predicts churn too often. The independence assumption does not hold for correlated one-hot columns. It suits a use case where missing a churner is far worse than a false alarm. |
| Support Vector Machine | Highest accuracy (0.8098) and precision (0.7012) but the lowest recall (0.4925), so it is cautious about flagging churn. Its AUC (0.8305) is lower, partly because probabilities come from Platt scaling. It was also the slowest to train. |

**Overall winner: Logistic Regression.** It has the best AUC and MCC, accuracy close to the top, and is the simplest and fastest model. SVM is best if precision matters most, Naive Bayes if recall matters most. The metrics of the top models are close (MCC 0.47-0.48), so differences of about 0.01 should not be over-interpreted on a test set of 1,761 rows.

## Repository Structure
```
app.py                  Streamlit app (upload, model dropdown, metrics, confusion matrix, ROC, comparison)
build_models.py         tunes and trains the five models, writes test_data.csv
data_prep.py            cleaning shared by training and the app
evaluation.py           the six metrics
telco_churn_full.csv    full dataset
test_data.csv           held-out test split
requirements.txt
model/                  saved pipelines (.joblib), metrics_summary.csv, best_params.json
```

## Run locally
```
pip install -r requirements.txt
python build_models.py   # optional, models are included
streamlit run app.py
```
