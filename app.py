import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from data_prep import clean_telco, TARGET_COL
from evaluation import score_model, METRIC_ORDER

st.set_page_config(page_title="Churn Model Lab", page_icon="📉", layout="wide")

SAVED = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
GLOSSARY = {
    "Accuracy": "Share of all customers classified correctly.",
    "AUC": "How well the model ranks churners above non-churners (1.0 = perfect, 0.5 = random).",
    "Precision": "Of customers flagged as churn, how many really churned.",
    "Recall": "Of customers who really churned, how many were caught.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Correlation between prediction and truth; robust on imbalanced data (-1 to 1).",
}


@st.cache_resource
def get_model(path):
    return joblib.load(path)


st.title("📉 Churn Model Lab")
st.markdown("Evaluate five pre-trained customer-churn classifiers on your own **test CSV**. "
            "The models are loaded from disk and are **never retrained** on what you upload.")

with st.sidebar:
    st.header("1. Data")
    upload = st.file_uploader("Test CSV (19 features + Churn)", type="csv")
    st.header("2. Model")
    picked = st.selectbox("Choose a classifier", list(SAVED))
    with st.expander("What do the metrics mean?"):
        for name, text in GLOSSARY.items():
            st.markdown(f"**{name}** - {text}")

if upload is not None:
    frame = pd.read_csv(upload)
elif os.path.exists("test_data.csv"):
    st.info("No upload yet - showing the bundled hold-out file `test_data.csv`.")
    frame = pd.read_csv("test_data.csv")
else:
    st.warning("Upload a CSV to begin.")
    st.stop()

if TARGET_COL not in frame.columns:
    st.error(f"The CSV needs a `{TARGET_COL}` column (Yes/No).")
    st.stop()

clf = get_model(SAVED[picked])
needed = list(clf.named_steps["prep"].feature_names_in_)
absent = [c for c in needed if c not in frame.columns]
if absent:
    st.error("Missing feature columns: " + ", ".join(absent))
    st.stop()

feats, truth = clean_telco(frame)
feats = feats[needed]
guess = clf.predict(feats)
scores = score_model(clf, feats, truth)

c1, c2, c3 = st.columns(3)
c1.metric("Rows evaluated", len(frame))
c2.metric("Actual churn rate", f"{truth.mean():.1%}")
c3.metric("Predicted churn rate", f"{guess.mean():.1%}")

tab_metrics, tab_cm, tab_roc, tab_all = st.tabs(
    ["Metrics", "Confusion matrix & report", "ROC curve", "Compare all models"])

with tab_metrics:
    st.subheader(picked)
    for col, key in zip(st.columns(6), METRIC_ORDER):
        col.metric(key, f"{scores[key]:.4f}")

with tab_cm:
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(confusion_matrix(truth, guess), annot=True, fmt="d", cmap="Purples", ax=ax,
                    xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with b:
        rep = classification_report(truth, guess, target_names=["Stay", "Churn"],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3))

with tab_roc:
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, path in SAVED.items():
        m = get_model(path)
        fpr, tpr, _ = roc_curve(truth, m.predict_proba(feats[needed])[:, 1])
        ax.plot(fpr, tpr, lw=3 if name == picked else 1, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=7)
    st.pyplot(fig)

with tab_all:
    board = pd.DataFrame([{"ML Model Name": n, **score_model(get_model(p), feats, truth)}
                          for n, p in SAVED.items()]).set_index("ML Model Name").round(4)
    st.dataframe(board.style.highlight_max(axis=0, color="#d4edda"))
    st.download_button("Download comparison as CSV", board.to_csv().encode(), "model_comparison.csv")
