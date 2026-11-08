import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix,
    ConfusionMatrixDisplay, classification_report)

st.set_page_config(page_title="Spam Mail Detector", page_icon="📧", layout="wide")

TARGET = "is_spam"
MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}

st.title("📧 Spam Mail Detector")
st.write("Pick a saved model, upload test emails (as 57 numeric features plus the "
         "`is_spam` column), and see how well the model separates spam from normal mail. "
         "The models are only loaded, never retrained.")

st.sidebar.header("Step 1: Test data")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
use_default = st.sidebar.checkbox("Use bundled test_data.csv", value=False)
st.sidebar.header("Step 2: Model")
choice = st.sidebar.selectbox("Select model", list(MODELS))

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_default:
    df = pd.read_csv("test_data.csv")
else:
    st.info("Upload test_data.csv in the sidebar (or tick the bundled file) to begin.")
    st.stop()

if TARGET not in df.columns:
    st.error(f"The file must contain the target column '{TARGET}'.")
    st.stop()

model = joblib.load(f"model/{MODELS[choice]}.joblib")
needed = list(model.feature_names_in_)
missing = [c for c in needed if c not in df.columns]
if missing:
    st.error(f"Missing {len(missing)} feature columns, for example: {missing[:5]}")
    st.stop()

X = df[needed]
y = df[TARGET]
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]

st.subheader(f"Results for {choice} on {len(df)} emails")
c = st.columns(6)
c[0].metric("Accuracy", f"{accuracy_score(y, pred):.4f}")
c[1].metric("AUC", f"{roc_auc_score(y, prob):.4f}")
c[2].metric("Precision", f"{precision_score(y, pred):.4f}")
c[3].metric("Recall", f"{recall_score(y, pred):.4f}")
c[4].metric("F1 Score", f"{f1_score(y, pred):.4f}")
c[5].metric("MCC", f"{matthews_corrcoef(y, pred):.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix** (0 = not spam, 1 = spam)")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay(confusion_matrix(y, pred),
                           display_labels=["Not spam", "Spam"]).plot(ax=ax, cmap="Blues")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    st.code(classification_report(y, pred, target_names=["Not spam", "Spam"]))

with st.expander("Preview of uploaded data"):
    st.dataframe(df.head(20))
