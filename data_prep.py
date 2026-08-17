"""
data_prep.py
------------
Loads the StudentDropoutDataset (Igualde-Saez et al., 2025, Zenodo),
aggregates course-enrolment rows to one row per student, and builds
feature sets for different EARLY observation windows (e.g. first 1, 2, 3 months
of online engagement) so we can test how early dropout can be predicted.

Dataset columns are Spanish/hashed. Key columns used:
  dni_hash        -> unique student id
  abandono_hash   -> dropout label ('A' = dropped out, 'B' = stayed)
  tipo_ingreso    -> entry route (categorical)
  nota10_hash     -> admission grade (numeric, ~27% missing, comma decimals)
  campus_hash     -> campus (categorical)
  estudios_p_hash -> father's education (categorical)
  estudios_m_hash -> mother's education (categorical)
  dedicacion      -> full/part-time (categorical)
  desplazado_hash -> displaced/relocated student (categorical)
  pft_*_YYYY_M    -> monthly LMS engagement (events, logins, visits,
                     submissions, minutes, wifi days, resource use)
"""

import pandas as pd
import numpy as np

# Academic year runs Sept -> July. Ordered list of (year, month) engagement blocks.
MONTH_ORDER = [
    (2022, 9), (2022, 10), (2022, 11), (2022, 12),
    (2023, 1), (2023, 2), (2023, 3), (2023, 4),
    (2023, 5), (2023, 6), (2023, 7),
]

# The 9 engagement metrics recorded per month
ENGAGEMENT_METRICS = [
    "pft_events", "pft_days_logged", "pft_visits",
    "pft_assignment_submissions", "pft_test_submissions",
    "pft_total_minutes", "n_wifi_days",
    "resource_events", "n_resource_days",
]

# Enrolment / background features known at the START (no leakage from outcomes)
ENROLMENT_FEATURES = [
    "tipo_ingreso", "nota10_hash", "campus_hash",
    "estudios_p_hash", "estudios_m_hash", "dedicacion", "desplazado_hash",
]


def _comma_to_float(series):
    """Convert Spanish-style comma-decimal strings ('8,373') to floats (8.373)."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def load_raw(path):
    """Load the semicolon-separated dataset and fix comma-decimal numeric columns."""
    df = pd.read_csv(path, sep=";", low_memory=False)

    # Admission grades use comma decimals
    for col in ["nota10_hash", "nota14_hash"]:
        if col in df.columns:
            df[col] = _comma_to_float(df[col])

    # All monthly engagement metrics can contain comma decimals (e.g. minutes)
    for (y, m) in MONTH_ORDER:
        for metric in ENGAGEMENT_METRICS:
            col = f"{metric}_{y}_{m}"
            if col in df.columns:
                df[col] = _comma_to_float(df[col])

    return df


def aggregate_to_student(df):
    """
    Collapse course-enrolment rows to one row per student.
    - label + enrolment fields: take the first value per student
    - monthly engagement: sum across a student's enrolments (total activity)
    """
    eng_cols = []
    for (y, m) in MONTH_ORDER:
        for metric in ENGAGEMENT_METRICS:
            col = f"{metric}_{y}_{m}"
            if col in df.columns:
                eng_cols.append(col)

    first_cols = ["abandono_hash"] + [c for c in ENROLMENT_FEATURES if c in df.columns]
    agg_map = {c: "first" for c in first_cols}
    for c in eng_cols:
        agg_map[c] = "sum"

    student = df.groupby("dni_hash").agg(agg_map).reset_index()

    # Binary target: 1 = dropped out ('A'), 0 = stayed ('B')
    student["dropout"] = (student["abandono_hash"] == "A").astype(int)
    student = student.drop(columns=["abandono_hash"])

    return student, eng_cols


def build_feature_set(student, n_months, eng_cols):
    """
    Build a feature set using ONLY the first `n_months` of engagement,
    plus the enrolment/background features. n_months=1 -> September only,
    n_months=2 -> Sept+Oct, etc.
    """
    window = MONTH_ORDER[:n_months]
    window_cols = []
    for (y, m) in window:
        for metric in ENGAGEMENT_METRICS:
            col = f"{metric}_{y}_{m}"
            if col in student.columns:
                window_cols.append(col)

    enrol_cols = [c for c in ENROLMENT_FEATURES if c in student.columns]
    feature_cols = enrol_cols + window_cols

    X = student[feature_cols].copy()
    y = student["dropout"].copy()
    return X, y, enrol_cols, window_cols


if __name__ == "__main__":
    # Smoke test: expects the CSV in the same folder
    df = load_raw("dataset_2022_hash.csv")
    print("Raw shape:", df.shape)
    student, eng_cols = aggregate_to_student(df)
    print("Students:", len(student), "| dropout rate:",
          round(student["dropout"].mean() * 100, 1), "%")
    X, y, enrol, win = build_feature_set(student, n_months=2, eng_cols=eng_cols)
    print("Feature set (2 months): X shape", X.shape)
