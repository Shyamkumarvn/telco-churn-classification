"""Metric helpers (positive class = customer churned)."""
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def score_model(fitted, features, labels):
    hard = fitted.predict(features)
    prob_churn = fitted.predict_proba(features)[:, 1]
    return {
        "Accuracy": accuracy_score(labels, hard),
        "AUC": roc_auc_score(labels, prob_churn),
        "Precision": precision_score(labels, hard, zero_division=0),
        "Recall": recall_score(labels, hard, zero_division=0),
        "F1": f1_score(labels, hard, zero_division=0),
        "MCC": matthews_corrcoef(labels, hard),
    }
