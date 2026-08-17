"""
test_pipeline.py
----------------
Automated checks for the artefact — the "testing" evidence in
"rigorous implementation, testing and evaluation".

Run:
    python test_pipeline.py

Each test prints PASS or FAIL. All should PASS before you trust results.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data_prep import (load_raw, aggregate_to_student, build_feature_set,
                       MONTH_ORDER, ENGAGEMENT_METRICS)
from modelling_v2 import build_preprocessor, make_pipeline

DATA_PATH = "dataset_2022_hash.csv"
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append(status == PASS)
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    # --- Test 1: data loads with expected shape ---
    df = load_raw(DATA_PATH)
    check("Dataset loads", df.shape == (159173, 169), f"shape={df.shape}")

    # --- Test 2: comma decimals were converted to numeric ---
    check("Admission grade is numeric after cleaning",
          pd.api.types.is_numeric_dtype(df["nota10_hash"]))

    # --- Test 3: aggregation gives one row per student ---
    student, eng_cols = aggregate_to_student(df)
    check("One row per student after aggregation",
          student["dni_hash"].is_unique and len(student) == 20427,
          f"students={len(student)}")

    # --- Test 4: dropout label is binary with the known rate ---
    rate = student["dropout"].mean()
    check("Dropout label binary, rate ~7.3%",
          set(student["dropout"].unique()) <= {0, 1} and 0.06 < rate < 0.09,
          f"rate={rate:.3f}")

    # --- Test 5: NO FUTURE LEAKAGE — a 2-month window must contain
    #     only September and October columns, nothing later ---
    X, y, enrol_cols, win_cols = build_feature_set(student, 2, eng_cols)
    allowed = {f"{m}_{yy}_{mm}" for (yy, mm) in MONTH_ORDER[:2]
               for m in ENGAGEMENT_METRICS}
    leaked = [c for c in win_cols if c not in allowed]
    check("No future-month leakage in early window", len(leaked) == 0,
          f"leaked={leaked[:3]}")

    # --- Test 6: pipeline trains and predicts probabilities on a sample ---
    sample = student.sample(3000, random_state=42)
    Xs, ys, ec, wc = build_feature_set(sample, 2, eng_cols)
    pre = build_preprocessor(ec, wc)
    pipe = make_pipeline("Logistic Regression", pre, ys)
    pipe.fit(Xs, ys)
    proba = pipe.predict_proba(Xs)[:, 1]
    check("Pipeline trains & outputs valid probabilities",
          proba.min() >= 0 and proba.max() <= 1 and len(proba) == len(ys))

    # --- Test 7: stratified split preserves class ratio (guard vs accidents) ---
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, ys, test_size=0.2,
                                              stratify=ys, random_state=42)
    check("Stratified split preserves imbalance",
          abs(y_tr.mean() - y_te.mean()) < 0.02,
          f"train={y_tr.mean():.3f} test={y_te.mean():.3f}")

    print("\n" + ("ALL TESTS PASSED" if all(results)
                  else f"{results.count(False)} TEST(S) FAILED"))


if __name__ == "__main__":
    main()
