"""
operational_eval.py
-------------------
Evaluation that answers the question a university actually asks.

AUC-ROC tells you whether the model ranks students well. It does not tell a
support team how many real dropouts they will catch if they can only contact
200 students. These functions translate model output into operational terms:

  precision_at_k   - of the k highest-risk students, how many really left
  lift_at_k        - how much better than contacting k students at random
  calibration      - does a stated "30% risk" mean 30% actually leave
  simple_rule      - an honest baseline: flag anyone below an activity threshold
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


# ----------------------------------------------------------------------
# Intervention-budget metrics
# ----------------------------------------------------------------------
def precision_at_k(y_true, scores, k):
    """Share of true dropouts among the k highest-risk students."""
    y_true = np.asarray(y_true); scores = np.asarray(scores)
    k = min(k, len(scores))
    top = np.argsort(-scores)[:k]
    return float(y_true[top].mean())


def recall_at_k(y_true, scores, k):
    """Share of ALL dropouts that appear in the top k."""
    y_true = np.asarray(y_true); scores = np.asarray(scores)
    k = min(k, len(scores))
    top = np.argsort(-scores)[:k]
    total = y_true.sum()
    return float(y_true[top].sum() / total) if total else 0.0


def budget_table(y_true, scores, budgets=(50, 100, 200, 500, 1000)):
    """
    Build the table that makes the model's value concrete:
    for each contact budget, how many real dropouts are found, and how
    that compares with contacting the same number of students at random.
    """
    y_true = np.asarray(y_true)
    base_rate = y_true.mean()
    rows = []
    for k in budgets:
        if k > len(y_true):
            continue
        p = precision_at_k(y_true, scores, k)
        r = recall_at_k(y_true, scores, k)
        rows.append({
            "contacts_k": k,
            "pct_of_cohort": round(k / len(y_true) * 100, 1),
            "precision_at_k": round(p, 3),
            "dropouts_found": int(round(p * k)),
            "expected_if_random": int(round(base_rate * k)),
            "lift_vs_random": round(p / base_rate, 2) if base_rate > 0 else np.nan,
            "recall_at_k": round(r, 3),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------
def calibration_table(y_true, proba, n_bins=10):
    """
    Group predictions into probability bands and compare the predicted risk
    with the observed dropout rate in each band. A trustworthy risk score
    should track the diagonal.
    """
    y_true = np.asarray(y_true); proba = np.asarray(proba)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 10:
            continue
        rows.append({
            "band": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
            "n": int(m.sum()),
            "mean_predicted": round(float(proba[m].mean()), 3),
            "observed_rate": round(float(y_true[m].mean()), 3),
            "gap": round(float(proba[m].mean() - y_true[m].mean()), 3),
        })
    return pd.DataFrame(rows)


def calibration_summary(y_true, proba):
    """Brier score (lower is better) plus a no-skill reference."""
    y_true = np.asarray(y_true)
    base = np.full_like(proba, y_true.mean(), dtype=float)
    return {
        "brier_score": round(float(brier_score_loss(y_true, proba)), 4),
        "brier_baseline_always_base_rate": round(float(brier_score_loss(y_true, base)), 4),
    }


# ----------------------------------------------------------------------
# Honest simple-rule baseline
# ----------------------------------------------------------------------
def simple_rule_baseline(student_frame, label_col, activity_col, percentiles=(10, 20, 30)):
    """
    The rule a tutor could apply without any model: flag students whose early
    activity falls in the lowest X% of the cohort. If the trained model cannot
    clearly beat this, that is a finding worth reporting.
    """
    rows = []
    y = student_frame[label_col].to_numpy()
    a = student_frame[activity_col].fillna(0).to_numpy(dtype=float)
    base = y.mean()
    for p in percentiles:
        thr = np.percentile(a, p)
        flagged = a <= thr
        if flagged.sum() == 0:
            continue
        prec = y[flagged].mean()
        rows.append({
            "rule": f"activity in lowest {p}%",
            "threshold": round(float(thr), 1),
            "students_flagged": int(flagged.sum()),
            "precision": round(float(prec), 3),
            "recall": round(float(y[flagged].sum() / y.sum()), 3),
            "lift_vs_random": round(float(prec / base), 2) if base > 0 else np.nan,
        })
    # Ranking quality of the raw signal alone (inverted: less activity = more risk)
    auc = roc_auc_score(y, -a)
    return pd.DataFrame(rows), round(float(auc), 3)
