"""
Train 5 classifiers on the UCI Dry Bean dataset, evaluate them, and save:
  model/*.joblib            -> trained models (full pipelines incl. scaler)
  model/label_info.json     -> feature names + class names
  test_data.csv             -> 20% hold-out set (features + Class) for the Streamlit app
  dry_bean_classification_results.csv -> comparison table (present in README)
Run:  python train_models.py
"""
import json
import os
import urllib.request
import zipfile
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATASET_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load data ---------------------------------------------------------------
raw = urllib.request.urlopen(DATASET_URL).read()
zf = zipfile.ZipFile(BytesIO(raw))
xlsx = [n for n in zf.namelist() if n.endswith(".xlsx")][0]
df = pd.read_excel(BytesIO(zf.read(xlsx)))
print("Shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()), "| Duplicates:", int(df.duplicated().sum()))
print(df["Class"].value_counts())

# 2. Split -------------------------------------------------------------------
X, y = df.drop(columns=["Class"]), df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Save test data for the Streamlit app (features + target)
pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)
json.dump(
    {"features": list(X.columns), "classes": sorted(y.unique().tolist())},
    open(os.path.join(MODEL_DIR, "label_info.json"), "w"),
)

# 3. Models ------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ]),
}

# 4. Train, evaluate, save ---------------------------------------------------
results, predictions = [], {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    predictions[name] = y_pred
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro",
                             labels=model.classes_),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join(MODEL_DIR, fname))

results_df = (pd.DataFrame(results)
              .sort_values("Accuracy", ascending=False).reset_index(drop=True))
results_df.to_csv("dry_bean_classification_results.csv", index=False)
print("\n", results_df.round(4).to_string(index=False))
print("\nMarkdown table for README:\n")
print(results_df.round(4).to_markdown(index=False) if hasattr(results_df, "to_markdown") else "")

for name, y_pred in predictions.items():
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(classification_report(y_test, y_pred, zero_division=0))

# 5. Confusion matrices (saved as an image for the README) -------------------
labels = sorted(y.unique())
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()
for i, (name, y_pred) in enumerate(predictions.items()):
    sns.heatmap(confusion_matrix(y_test, y_pred, labels=labels), annot=True, fmt="d",
                cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[i])
    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=120)

best = results_df.iloc[0]
print(f"\nBest model by accuracy: {best['ML Model Name']} ({best['Accuracy']:.4f})")
