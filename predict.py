"""
predict.py
----------
Turns the trained model into a usable early-warning tool.

Loads the tuned model saved by run_full_experiment.py and scores students,
producing a ranked at-risk list a tutor or support team could actually use.

Usage:
    python predict.py                          # scores dataset_2022_hash.csv
    python predict.py path\\to\\other_year.csv   # scores another year's file

Output:
    results/at_risk_list.csv  - every student with dropout probability,
                                flagged if above the tuned threshold,
                                sorted most-at-risk first.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from data_prep import load_raw, aggregate_to_student, build_feature_set

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "results", "tuned_model.joblib")
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "dataset_2022_hash.csv")
OUT_PATH = os.path.join(HERE, "results", "at_risk_list.csv")


def main():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Run run_full_experiment.py first.")
        return

    bundle = joblib.load(MODEL_PATH)
    model, threshold = bundle["model"], bundle["threshold"]
    print(f"Loaded tuned model (decision threshold = {threshold:.2f})")

    df = load_raw(DATA_PATH)
    student, eng_cols = aggregate_to_student(df)

    # Score with the same early window the model was trained on (2 months)
    X, y, enrol_cols, win_cols = build_feature_set(student, 2, eng_cols)
    proba = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "student_id": student["dni_hash"],
        "dropout_probability": proba.round(3),
        "flagged_at_risk": (proba >= threshold),
    }).sort_values("dropout_probability", ascending=False)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    n_flag = int(out["flagged_at_risk"].sum())
    print(f"Scored {len(out)} students | {n_flag} flagged at risk "
          f"({n_flag/len(out)*100:.1f}%)")
    print(f"Saved ranked list -> {OUT_PATH}")
    print("\nTop 10 highest-risk students:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
