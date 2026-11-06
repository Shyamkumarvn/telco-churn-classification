"""Tune and train the five classifiers, then save them with test_data.csv."""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from data_prep import clean_telco, TARGET_COL, NUMERIC_FEATURES
from evaluation import score_model

SEED = 7
TEST_FRACTION = 0.25

raw_df = pd.read_csv("telco_churn_full.csv")
features, labels = clean_telco(raw_df)
category_cols = [c for c in features.columns if c not in NUMERIC_FEATURES]

x_tr, x_te, y_tr, y_te = train_test_split(
    features, labels, test_size=TEST_FRACTION, stratify=labels, random_state=SEED)


def encoder():
    return ColumnTransformer([
        ("scale", StandardScaler(), NUMERIC_FEATURES),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), category_cols),
    ])

# (estimator, hyper-parameter grid) for each model
search_space = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=SEED),
                            {"clf__C": [0.05, 0.1, 0.5, 1, 5]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                      {"clf__max_depth": [3, 4, 5, 7], "clf__min_samples_leaf": [5, 20, 50]}),
    "KNN": (KNeighborsClassifier(),
            {"clf__n_neighbors": [7, 11, 15, 21, 31]}),
    "Naive Bayes": (GaussianNB(),
                    {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
    "Support Vector Machine": (SVC(kernel="rbf", probability=True, random_state=SEED),
                               {"clf__C": [0.5, 1, 2]}),
}

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
table, chosen = [], {}
for label, (est, grid) in search_space.items():
    pipe = Pipeline([("prep", encoder()), ("clf", est)])
    gs = GridSearchCV(pipe, grid, cv=folds, scoring="roc_auc", n_jobs=-1)
    gs.fit(x_tr, y_tr)
    best = gs.best_estimator_
    joblib.dump(best, "model/" + label.lower().replace(" ", "_") + ".joblib")
    chosen[label] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    table.append({"ML Model Name": label, **score_model(best, x_te, y_te)})
    print(label, chosen[label])

summary = pd.DataFrame(table).round(4)
summary.to_csv("model/metrics_summary.csv", index=False)
json.dump(chosen, open("model/best_params.json", "w"), indent=2, default=str)

holdout = x_te.copy()
holdout[TARGET_COL] = y_te.map({1: "Yes", 0: "No"}).values
holdout.to_csv("test_data.csv", index=False)
print(summary.to_markdown(index=False))
print("train", x_tr.shape, "test", x_te.shape, "churn rate", round(labels.mean(), 4))
