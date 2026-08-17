"""
app.py
------
Interactive dashboard for the early dropout-prediction artefact.

Loads the tuned model saved by run_full_experiment.py and lets a user enter a
student's early indicators to get a dropout risk score, with a SHAP explanation
of WHY the model reached that score (interpretability, RQ2 and ethics).

Run with:
    streamlit run app.py

Important design note
---------------------
The fitted pipeline expects EVERY column it was trained on (7 enrolment +
18 engagement features for the 2-month window). Sending only the handful of
fields shown in the sidebar would raise a column-mismatch error. So we load the
"feature template" saved alongside the model, fill it with sensible defaults
(median for numeric, mode for categorical), and override only the fields the
user edits. Defaults are clearly labelled in the UI so the user knows which
values are their own and which are population averages.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "results", "tuned_model.joblib")

st.set_page_config(page_title="Early Dropout Predictor", layout="wide")

st.title("Early Student Dropout Predictor")
st.caption(
    "Estimates dropout risk from a student's enrolment details and first two "
    "months of online engagement, so support can be offered while it still helps."
)


# ----------------------------------------------------------------------
# Load model bundle
# ----------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


bundle = load_bundle()

if bundle is None:
    st.error("No trained model found at `results/tuned_model.joblib`.")
    st.info("Run `python run_full_experiment.py` first to train and save the model.")
    st.stop()

model = bundle["model"]
threshold = bundle["threshold"]
feature_columns = bundle["feature_columns"]
feature_defaults = bundle["feature_defaults"]
metrics = bundle.get("metrics", {})

# Show the model's honest measured performance up front, so risk scores
# are never read as more certain than the evidence supports.
if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test AUC-ROC", metrics.get("AUC_ROC", "-"))
    c2.metric("Recall (dropouts caught)", metrics.get("recall", "-"))
    c3.metric("Precision", metrics.get("precision", "-"))
    c4.metric("Decision threshold", metrics.get("threshold_used", round(threshold, 2)))

st.divider()


# ----------------------------------------------------------------------
# Sidebar inputs — the strongest predictors found by the SHAP analysis
# ----------------------------------------------------------------------
st.sidebar.header("Student details")
st.sidebar.caption(
    "Fields left untouched use population averages from the training data."
)

editable = {}

if "nota10_hash" in feature_columns:
    editable["nota10_hash"] = st.sidebar.number_input(
        "Admission grade (0-10)", min_value=0.0, max_value=10.0,
        value=float(np.clip(feature_defaults.get("nota10_hash", 6.5), 0, 10)),
        step=0.1,
    )

month1_fields = [
    ("n_wifi_days_2022_9", "Wi-Fi days on campus, month 1", 0, 31),
    ("pft_days_logged_2022_9", "Days logged into platform, month 1", 0, 31),
    ("pft_visits_2022_9", "Platform visits, month 1", 0, 500),
    ("pft_assignment_submissions_2022_9", "Assignment submissions, month 1", 0, 60),
]
month2_fields = [
    ("n_wifi_days_2022_10", "Wi-Fi days on campus, month 2", 0, 31),
    ("pft_days_logged_2022_10", "Days logged into platform, month 2", 0, 31),
    ("resource_events_2022_10", "Resource accesses, month 2", 0, 1000),
]

st.sidebar.subheader("Month 1 engagement")
for col, label, lo, hi in month1_fields:
    if col in feature_columns:
        editable[col] = st.sidebar.number_input(
            label, min_value=float(lo), max_value=float(hi),
            value=float(np.clip(feature_defaults.get(col, 0), lo, hi)), step=1.0,
        )

st.sidebar.subheader("Month 2 engagement")
for col, label, lo, hi in month2_fields:
    if col in feature_columns:
        editable[col] = st.sidebar.number_input(
            label, min_value=float(lo), max_value=float(hi),
            value=float(np.clip(feature_defaults.get(col, 0), lo, hi)), step=1.0,
        )

go = st.sidebar.button("Estimate dropout risk", type="primary")


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
def build_input_row():
    """Fill the full feature template, then override the user's fields."""
    row = dict(feature_defaults)
    row.update(editable)
    return pd.DataFrame([row], columns=feature_columns)


if go:
    input_df = build_input_row()

    try:
        proba = float(model.predict_proba(input_df)[0, 1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    flagged = proba >= threshold

    left, right = st.columns([1, 2])
    with left:
        st.metric("Estimated dropout risk", f"{proba:.1%}")
        if flagged:
            st.warning("Above the alert threshold — worth a supportive check-in.")
        else:
            st.success("Below the alert threshold.")
        st.caption(
            f"Threshold {threshold:.2f} was chosen to balance precision and "
            "recall on validation data, not set arbitrarily at 50%."
        )

    with right:
        st.info(
            "**How to read this.** This is a screening aid, not a verdict on a "
            "student. Roughly 7% of students in the training data dropped out, so "
            "most flagged students will not actually leave. Use a flag as a prompt "
            "for a conversation and an offer of support — never for any decision "
            "that disadvantages a student."
        )

    # ---------------- SHAP explanation ----------------
    st.subheader("Why the model gave this score")
    try:
        import shap

        pre = model.named_steps["pre"]
        clf = model.named_steps["clf"]
        X_enc = pre.transform(input_df)
        if hasattr(X_enc, "toarray"):
            X_enc = X_enc.toarray()
        names = pre.get_feature_names_out()

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_enc)
        if isinstance(sv, list):          # older SHAP returns one array per class
            sv = sv[1]
        sv = np.asarray(sv).reshape(-1)

        contrib = (
            pd.DataFrame({"feature": names, "contribution": sv})
            .assign(magnitude=lambda d: d["contribution"].abs())
            .sort_values("magnitude", ascending=False)
            .head(10)
            .sort_values("contribution")
        )

        fig, ax = plt.subplots(figsize=(7, 4.5))
        colours = ["#c0392b" if v > 0 else "#2874a6" for v in contrib["contribution"]]
        ax.barh(contrib["feature"], contrib["contribution"], color=colours)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("SHAP contribution  (right = pushes risk up)")
        ax.set_title("Top 10 factors behind this prediction")
        fig.tight_layout()
        st.pyplot(fig)

        st.caption(
            "Red bars pushed the predicted risk up; blue bars pushed it down. "
            "Bars for fields you did not edit reflect population defaults."
        )
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
        st.caption("The risk score above is unaffected.")

else:
    st.info("Enter the student's details in the sidebar, then select "
            "**Estimate dropout risk**.")

st.divider()
st.caption(
    "Data: StudentDropoutDataset (Igualde-Sáez et al., 2025), Zenodo, "
    "DOI 10.5281/zenodo.17239943, CC BY 4.0. "
    "Built with scikit-learn, XGBoost, SHAP and Streamlit."
)
