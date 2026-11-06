"""Cleaning helpers shared by the training script and the Streamlit app."""
import pandas as pd

TARGET_COL = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def clean_telco(frame):
    """Return (features, labels) from a raw Telco-style dataframe.

    - drops the customer id
    - TotalCharges is text in the source file; blanks (tenure 0 customers) become 0
    - Churn Yes/No -> 1/0 (already-numeric labels are kept)
    """
    data = frame.copy()
    data = data.drop(columns=["customerID"], errors="ignore")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0.0)
    data["SeniorCitizen"] = data["SeniorCitizen"].astype(int)

    labels = data[TARGET_COL]
    if not pd.api.types.is_numeric_dtype(labels):
        labels = labels.map({"Yes": 1, "No": 0})
    features = data.drop(columns=[TARGET_COL])
    return features, labels.astype(int)
