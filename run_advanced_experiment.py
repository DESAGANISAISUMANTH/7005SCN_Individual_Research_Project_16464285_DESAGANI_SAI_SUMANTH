"""
run_advanced_experiment.py
--------------------------
The four additions that give the project its distinctive contribution.

  EXPERIMENT A - Do trajectory features beat monthly sums?
                 Same data, same split, features are the only difference.

  EXPERIMENT B - Does restricting to first-year students help?
                 First-years drop out far more often than continuing students,
                 and they are the population early intervention targets.

  EXPERIMENT C - Cross-cohort validation.
                 Train on earlier entry cohorts, test on the newest one.
                 Answers "would this model work for next year's intake?"

  EXPERIMENT D - Operational evaluation.
                 precision@k / lift under realistic contact budgets,
                 probability calibration, and an honest simple-rule baseline.

Usage:
    python run_advanced_experiment.py [path_to_csv]

Outputs -> results_advanced/
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, recall_score, precision_score

from data_prep import load_raw, aggregate_to_student
from advanced_features import add_entry_cohort, build_feature_set_v2
from modelling_v2 import build_preprocessor, make_pipeline, cross_validated_scores
from operational_eval import (budget_table, calibration_table,
                             calibration_summary, simple_rule_baseline)

DATA = sys.argv[1] if len(sys.argv) > 1 else "dataset_2022_hash.csv"
OUT = "results_advanced"
os.makedirs(OUT, exist_ok=True)
WINDOW = 2          # months of early engagement
MODEL = "XGBoost"   # falls back inside make_pipeline if unavailable
findings = {}


def fit_and_score(X_tr, y_tr, X_te, y_te, enrol_cols, num_cols, model=MODEL):
    """Fit one pipeline and return (probabilities, headline metrics)."""
    pre = build_preprocessor(enrol_cols, num_cols)
    pipe = make_pipeline(model, pre, y_tr)
    if pipe is None:
        pipe = make_pipeline("Random Forest", pre, y_tr)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return proba, {
        "AUC_ROC": round(float(roc_auc_score(y_te, proba)), 3),
        "PR_AUC": round(float(average_precision_score(y_te, proba)), 3),
        "recall": round(float(recall_score(y_te, pred, zero_division=0)), 3),
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 3),
        "macro_F1": round(float(f1_score(y_te, pred, average="macro")), 3),
    }, pipe


def main():
    print("Loading data...")
    df = load_raw(DATA)
    student, eng_cols = aggregate_to_student(df)
    student = add_entry_cohort(df, student)
    print(f"  {len(student):,} students | overall dropout "
          f"{student['dropout'].mean()*100:.2f}%")

    # Cohort profile - context for everything that follows
    coh = (student.dropna(subset=["entry_year"])
                  .groupby("entry_year")["dropout"].agg(["count", "mean"]))
    coh = coh[coh["count"] >= 100]
    coh["dropout_rate_pct"] = (coh["mean"] * 100).round(2)
    print("\nDropout rate by entry cohort:")
    print(coh[["count", "dropout_rate_pct"]].to_string())
    coh[["count", "dropout_rate_pct"]].to_csv(f"{OUT}/table_cohort_profile.csv")

    # ==================================================================
    # EXPERIMENT A - trajectory features vs monthly sums
    # ==================================================================
    print("\n" + "=" * 66)
    print("EXPERIMENT A: Do trajectory features beat monthly sums?")
    print("=" * 66)
    rows = []
    for use_traj in (False, True):
        X, y, ec, nc = build_feature_set_v2(student, WINDOW, use_trajectory=use_traj)
        pre = build_preprocessor(ec, nc)
        pipe = make_pipeline(MODEL, pre, y) or make_pipeline("Random Forest", pre, y)
        sc = cross_validated_scores(pipe, X, y, n_splits=5)
        label = "Baseline (monthly sums)" if not use_traj else "Baseline + trajectory"
        rows.append({"feature_set": label, "n_features": X.shape[1],
                     "AUC_ROC": sc["AUC_ROC"], "AUC_CI": sc["AUC_ROC_CI"],
                     "PR_AUC": sc["PR_AUC"], "recall": sc["recall"],
                     "macro_F1": sc["macro_F1"]})
        print(f"  {label:24s} feats={X.shape[1]:3d}  "
              f"AUC={sc['AUC_ROC']}±{sc['AUC_ROC_CI']}  PR-AUC={sc['PR_AUC']}")
    expA = pd.DataFrame(rows)
    expA.to_csv(f"{OUT}/expA_trajectory_vs_baseline.csv", index=False)
    gain = expA.loc[1, "AUC_ROC"] - expA.loc[0, "AUC_ROC"]
    ci_sum = expA.loc[0, "AUC_CI"] + expA.loc[1, "AUC_CI"]
    findings["trajectory_auc_gain"] = round(float(gain), 3)
    findings["trajectory_gain_exceeds_ci"] = bool(abs(gain) > ci_sum)
    print(f"  --> AUC change {gain:+.3f}; "
          f"{'larger' if abs(gain) > ci_sum else 'NOT larger'} than combined CIs")

    # ==================================================================
    # EXPERIMENT B - restrict to first-year students
    # ==================================================================
    print("\n" + "=" * 66)
    print("EXPERIMENT B: Does restricting to first-year students help?")
    print("=" * 66)
    # Pick the newest cohort that has a usable sample size. Using max() alone
    # is wrong: a handful of stray records (for example a single 2023 entrant)
    # would be selected and the comparison would be meaningless.
    MIN_COHORT = 500
    counts = student["entry_year"].value_counts()
    eligible = sorted([yr for yr, n in counts.items() if n >= MIN_COHORT])
    if not eligible:
        raise ValueError("No entry cohort has enough students to compare")
    newest = eligible[-1]
    print(f"  Newest cohort with n>={MIN_COHORT}: entry year {int(newest)}")
    first_years = student[student["entry_year"] == newest].copy()
    print(f"  First-year cohort (entry {int(newest)}): {len(first_years):,} students | "
          f"dropout {first_years['dropout'].mean()*100:.2f}%")

    rows = []
    for name, frame in [("All students", student), (f"First-years only", first_years)]:
        X, y, ec, nc = build_feature_set_v2(frame, WINDOW, use_trajectory=True)
        pre = build_preprocessor(ec, nc)
        pipe = make_pipeline(MODEL, pre, y) or make_pipeline("Random Forest", pre, y)
        sc = cross_validated_scores(pipe, X, y, n_splits=5)
        rows.append({"population": name, "n_students": len(frame),
                     "dropout_rate_pct": round(float(frame["dropout"].mean()*100), 2),
                     "AUC_ROC": sc["AUC_ROC"], "AUC_CI": sc["AUC_ROC_CI"],
                     "PR_AUC": sc["PR_AUC"], "recall": sc["recall"]})
        print(f"  {name:20s} n={len(frame):6,}  AUC={sc['AUC_ROC']}±{sc['AUC_ROC_CI']}  "
              f"PR-AUC={sc['PR_AUC']}  recall={sc['recall']}")
    expB = pd.DataFrame(rows)
    expB.to_csv(f"{OUT}/expB_population_comparison.csv", index=False)
    findings["first_year_dropout_rate"] = float(expB.loc[1, "dropout_rate_pct"])
    findings["first_year_pr_auc"] = float(expB.loc[1, "PR_AUC"])

    # ==================================================================
    # EXPERIMENT C - cross-cohort validation
    # ==================================================================
    print("\n" + "=" * 66)
    print("EXPERIMENT C: Cross-cohort validation (train older, test newest)")
    print("=" * 66)
    have_year = student.dropna(subset=["entry_year"])
    train_pool = have_year[have_year["entry_year"] < newest]
    test_pool = have_year[have_year["entry_year"] == newest]
    print(f"  Train: cohorts < {int(newest)}  n={len(train_pool):,} "
          f"(dropout {train_pool['dropout'].mean()*100:.2f}%)")
    print(f"  Test:  cohort  {int(newest)}    n={len(test_pool):,} "
          f"(dropout {test_pool['dropout'].mean()*100:.2f}%)")

    Xtr, ytr, ec, nc = build_feature_set_v2(train_pool, WINDOW, use_trajectory=True)
    Xte, yte, _, _ = build_feature_set_v2(test_pool, WINDOW, use_trajectory=True)
    Xte = Xte[Xtr.columns]                      # identical column order
    proba_cc, m_cc, _ = fit_and_score(Xtr, ytr, Xte, yte, ec, nc)
    print(f"  Cross-cohort performance: {m_cc}")

    # Same-cohort reference: random split inside the newest cohort only
    Xa, ya, ec2, nc2 = build_feature_set_v2(test_pool, WINDOW, use_trajectory=True)
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        Xa, ya, test_size=0.3, stratify=ya, random_state=42)
    proba_same, m_same, _ = fit_and_score(Xa_tr, ya_tr, Xa_te, ya_te, ec2, nc2)
    print(f"  Same-cohort reference:    {m_same}")

    expC = pd.DataFrame([
        {"validation": f"Cross-cohort (train <{int(newest)} -> test {int(newest)})",
         "n_test": len(yte), **m_cc},
        {"validation": f"Same-cohort random split ({int(newest)} only)",
         "n_test": len(ya_te), **m_same},
    ])
    expC.to_csv(f"{OUT}/expC_cross_cohort.csv", index=False)
    findings["cross_cohort_auc"] = m_cc["AUC_ROC"]
    findings["same_cohort_auc"] = m_same["AUC_ROC"]
    findings["generalisation_drop"] = round(m_same["AUC_ROC"] - m_cc["AUC_ROC"], 3)

    # ==================================================================
    # EXPERIMENT D - operational evaluation on the cross-cohort predictions
    # ==================================================================
    print("\n" + "=" * 66)
    print("EXPERIMENT D: Operational evaluation (contact budgets)")
    print("=" * 66)
    budgets = [b for b in (50, 100, 200, 300, 500) if b <= len(yte)]
    bt = budget_table(yte, proba_cc, budgets=budgets)
    print(bt.to_string(index=False))
    bt.to_csv(f"{OUT}/expD_budget_table.csv", index=False)
    if len(bt):
        findings["lift_at_100"] = float(bt.iloc[min(1, len(bt)-1)]["lift_vs_random"])

    print("\n  Calibration:")
    ct = calibration_table(yte, proba_cc)
    print(ct.to_string(index=False))
    ct.to_csv(f"{OUT}/expD_calibration.csv", index=False)
    cs = calibration_summary(yte, proba_cc)
    print(f"  {cs}")
    findings.update(cs)

    print("\n  Honest simple-rule baseline (no model):")
    act_col = "pft_events_2022_9"
    if act_col in test_pool.columns:
        sr, sr_auc = simple_rule_baseline(test_pool, "dropout", act_col)
        print(sr.to_string(index=False))
        print(f"  AUC of raw month-1 activity alone: {sr_auc}")
        sr.to_csv(f"{OUT}/expD_simple_rule.csv", index=False)
        findings["simple_rule_auc"] = sr_auc
        findings["model_beats_simple_rule_by"] = round(m_cc["AUC_ROC"] - sr_auc, 3)

    # ==================================================================
    # FIGURES
    # ==================================================================
    print("\nGenerating figures...")

    # Lift curve
    if len(bt):
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        ax.plot(bt["contacts_k"], bt["lift_vs_random"], marker="o", color="#1f77b4")
        ax.axhline(1.0, ls="--", color="grey", label="random targeting")
        ax.set_xlabel("Students contacted (budget)")
        ax.set_ylabel("Lift over random")
        ax.set_title("Figure A. Value of the model under a fixed contact budget")
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(f"{OUT}/figA_lift_curve.png", dpi=150); plt.close()

    # Calibration curve
    if len(ct):
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        ax.plot([0, ct["mean_predicted"].max()], [0, ct["mean_predicted"].max()],
                ls="--", color="grey", label="perfect calibration")
        ax.plot(ct["mean_predicted"], ct["observed_rate"], marker="o", color="#c0392b")
        ax.set_xlabel("Mean predicted risk"); ax.set_ylabel("Observed dropout rate")
        ax.set_title("Figure B. Calibration")
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(f"{OUT}/figB_calibration.png", dpi=150); plt.close()

    # Cohort dropout rates
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(coh.index.astype(int).astype(str), coh["dropout_rate_pct"], color="#8e44ad")
    ax.set_xlabel("Entry cohort"); ax.set_ylabel("Dropout rate (%)")
    ax.set_title("Figure C. Dropout rate by entry cohort")
    ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(f"{OUT}/figC_cohort_rates.png", dpi=150); plt.close()

    with open(f"{OUT}/advanced_findings.json", "w") as f:
        json.dump(findings, f, indent=2)

    print("\n" + "=" * 66)
    print("KEY FINDINGS")
    print("=" * 66)
    for k, v in findings.items():
        print(f"  {k}: {v}")
    print(f"\nAll outputs saved to ./{OUT}/")


if __name__ == "__main__":
    main()
