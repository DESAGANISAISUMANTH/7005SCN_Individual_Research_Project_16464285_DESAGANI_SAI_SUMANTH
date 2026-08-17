"""
advanced_features.py
--------------------
Two additions to the baseline feature set:

1. COHORT IDENTIFICATION
   Each student's entry year (`anyo_ingreso`) is extracted so the model can be
   restricted to, or validated across, cohorts. This matters: first-year
   students drop out at roughly 13% while students who entered three years
   earlier drop out at 2-3%, so pooling them dilutes the signal for the group
   early intervention actually targets.

2. TRAJECTORY FEATURES
   The baseline features sum activity per month, capturing VOLUME. These
   features capture SHAPE: is the student's engagement falling, erratic, or
   has it stopped altogether? A student with declining activity is a different
   risk from one who was quiet throughout, even if their totals match.
"""

import numpy as np
import pandas as pd

from data_prep import (MONTH_ORDER, ENGAGEMENT_METRICS, ENROLMENT_FEATURES,
                       _comma_to_float)

# The metric used as the primary activity signal for trajectory shape
PRIMARY = "pft_events"


def add_entry_cohort(df, student):
    """Attach each student's entry year (cohort) to the student-level frame."""
    if "anyo_ingreso" not in df.columns:
        student["entry_year"] = np.nan
        return student
    ay = df.groupby("dni_hash")["anyo_ingreso"].first()
    ay = _comma_to_float(ay)
    student = student.merge(
        ay.rename("entry_year"), left_on="dni_hash", right_index=True, how="left")
    return student


def build_trajectory_features(student, n_months):
    """
    Build shape-based features from the first `n_months` of activity.

    Requires n_months >= 2 (a slope needs at least two points).
    Returns a DataFrame of new features aligned to `student`.
    """
    if n_months < 2:
        raise ValueError("Trajectory features need at least 2 months")

    months = MONTH_ORDER[:n_months]
    out = pd.DataFrame(index=student.index)

    # Matrix of the primary activity metric across the early window
    cols = [f"{PRIMARY}_{y}_{m}" for (y, m) in months
            if f"{PRIMARY}_{y}_{m}" in student.columns]
    if len(cols) < 2:
        return out
    A = student[cols].to_numpy(dtype=float)
    A = np.nan_to_num(A, nan=0.0)

    # --- Level and spread ---
    out["traj_mean"] = A.mean(axis=1)
    out["traj_std"] = A.std(axis=1)

    # Volatility relative to level: erratic engagement vs steady engagement
    with np.errstate(divide="ignore", invalid="ignore"):
        out["traj_cv"] = np.where(out["traj_mean"] > 0,
                                  out["traj_std"] / out["traj_mean"], 0.0)

    # --- Direction: least-squares slope across the window ---
    x = np.arange(A.shape[1], dtype=float)
    x_centred = x - x.mean()
    denom = (x_centred ** 2).sum()
    out["traj_slope"] = ((A - A.mean(axis=1, keepdims=True)) * x_centred).sum(axis=1) / denom

    # --- Change from first to last month of the window ---
    out["traj_delta"] = A[:, -1] - A[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["traj_pct_change"] = np.where(A[:, 0] > 0,
                                          (A[:, -1] - A[:, 0]) / A[:, 0], 0.0)
    out["traj_pct_change"] = out["traj_pct_change"].clip(-5, 5)

    # --- Silence: months with no recorded activity at all ---
    silent = (A == 0)
    out["traj_silent_months"] = silent.sum(axis=1)
    # Longest consecutive run of silent months (an interpretable warning sign)
    run, best = np.zeros(A.shape[0]), np.zeros(A.shape[0])
    for j in range(A.shape[1]):
        run = np.where(silent[:, j], run + 1, 0)
        best = np.maximum(best, run)
    out["traj_max_silent_run"] = best

    # --- Did activity drop in the final month of the window? ---
    out["traj_declined_last"] = (A[:, -1] < A[:, -2]).astype(int)

    # --- Breadth: how many different activity types were used at all ---
    breadth_cols = []
    for metric in ENGAGEMENT_METRICS:
        mcols = [f"{metric}_{y}_{m}" for (y, m) in months
                 if f"{metric}_{y}_{m}" in student.columns]
        if mcols:
            breadth_cols.append(student[mcols].fillna(0).sum(axis=1) > 0)
    if breadth_cols:
        out["traj_metric_breadth"] = np.sum(breadth_cols, axis=0)

    return out


def build_feature_set_v2(student, n_months, use_trajectory=True,
                         include_cohort=False):
    """
    Feature set builder that optionally adds trajectory features.

    use_trajectory=False reproduces the baseline set, so the two can be
    compared on identical data and splits.
    """
    window = MONTH_ORDER[:n_months]
    window_cols = []
    for (y, m) in window:
        for metric in ENGAGEMENT_METRICS:
            col = f"{metric}_{y}_{m}"
            if col in student.columns:
                window_cols.append(col)

    enrol_cols = [c for c in ENROLMENT_FEATURES if c in student.columns]
    X = student[enrol_cols + window_cols].copy()
    numeric_extra = []

    if include_cohort and "entry_year" in student.columns:
        X["entry_year"] = student["entry_year"]
        numeric_extra.append("entry_year")

    if use_trajectory and n_months >= 2:
        traj = build_trajectory_features(student, n_months)
        X = pd.concat([X, traj], axis=1)
        numeric_extra += list(traj.columns)

    y = student["dropout"].copy()
    # numeric columns = engagement + admission grade + the new numeric extras
    return X, y, enrol_cols, window_cols + numeric_extra
