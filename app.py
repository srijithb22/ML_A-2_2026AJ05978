import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET = "Class"


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_data
def load_info():
    with open(os.path.join(MODEL_DIR, "label_info.json")) as f:
        return json.load(f)


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    try:
        auc = roc_auc_score(y, model.predict_proba(X), multi_class="ovr",
                            average="macro", labels=model.classes_)
    except ValueError:  # e.g. uploaded file is missing some classes
        auc = np.nan
    return y_pred, {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


st.title("🌱 Dry Bean Classification")
st.write("Submission by Sarthak Rout - 2026AK05289")
st.write("Evaluate 5 pre-trained classifiers on **test data** (UCI Dry Bean dataset). "
         "Models are loaded from disk - nothing is retrained.")

info = load_info()
features = info["features"]

# ---- Sidebar: upload + model selection ------------------------------------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV (features + 'Class')", type="csv")
use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=uploaded is None)
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))

if uploaded is not None:
    data = pd.read_csv(uploaded)
elif use_sample and os.path.exists("test_data.csv"):
    data = pd.read_csv("test_data.csv")
else:
    st.info("Upload a test CSV (or tick the bundled sample) to begin.")
    st.stop()

missing = [c for c in features + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}")
    st.stop()

data = data.dropna(subset=features + [TARGET])
X, y = data[features], data[TARGET]
st.subheader("Data preview")
st.write(f"{len(data)} rows | {len(features)} features")
st.dataframe(data.head())

# ---- Selected model --------------------------------------------------------
model = load_model(model_name)
y_pred, metrics = compute_metrics(model, X, y)

st.subheader(f"Evaluation metrics - {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    labels = sorted(info["classes"])
    cm = confusion_matrix(y, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y, y_pred, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3))

# ---- All models comparison -------------------------------------------------
st.subheader("Comparison of all models on this test data")
rows = []
for name in MODEL_FILES:
    _, m = compute_metrics(load_model(name), X, y)
    rows.append({"ML Model Name": name, **m})
comp = pd.DataFrame(rows).set_index("ML Model Name")
st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"))
st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
