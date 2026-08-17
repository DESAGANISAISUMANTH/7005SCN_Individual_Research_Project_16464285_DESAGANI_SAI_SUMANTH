"""
run_full_experiment.py
----------------------
The complete, rigorous experiment for the CW2 artefact (local-machine version).

Usage:
    python run_full_experiment.py [path_to_dataset_csv]

If no path is given, it looks for 'dataset_2022_hash.csv' in this folder.

Produces (in results/):
  cv_results.csv            - CV metrics + 95% CIs, all models x all windows (RQ1, RQ3)
  tuned_model_metrics.json  - held-out test metrics for the tuned XGBoost
  confusion_matrix.png      - tuned model confusion matrix
  roc_curve.png             - ROC curve
  pr_curve.png              - Precision-Recall curve
  auc_vs_window.png         - ablation figure (RQ3)
  feature_importance.csv/png- SHAP top predictors (RQ2)
  fairness_check.csv        - subgroup performance (ethics section)
"""

import os, sys, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score,
)

from data_prep import load_raw, aggregate_to_student, build_feature_set
from modelling_v2 import (
    build_preprocessor, make_pipeline, cross_validated_scores,
    tune_xgboost, HAS_XGB, USE_GPU,
)

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "dataset_2022_hash.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WINDOWS = [1, 2, 3, 6]
MODELS = ["Logistic Regression", "Random Forest"] + (["XGBoost"] if HAS_XGB else [])


def main():
    print(f"GPU acceleration: {'ON (CUDA)' if USE_GPU else 'off (CPU)'}")
    print("Loading and aggregating data...")
    df = load_raw(DATA_PATH)
    student, eng_cols = aggregate_to_student(df)
    print(f"  {len(student)} students | dropout {student['dropout'].mean()*100:.1f}%")

    # ============================================================
    # PART 1: Cross-validated ablation across early windows + models
    # ============================================================
    print("\n[1/5] Cross-validated ablation (5-fold)...")
    rows = []
    for n in WINDOWS:
        X, y, enrol_cols, win_cols = build_feature_set(student, n, eng_cols)
        for model in MODELS:
            pre = build_preprocessor(enrol_cols, win_cols)
            pipe = make_pipeline(model, pre, y)
            scores = cross_validated_scores(pipe, X, y, n_splits=5)
            scores.update({"model": model, "window_months": n})
            rows.append(scores)
            print(f"  {model:20s} {n}mo  AUC={scores['AUC_ROC']}±{scores['AUC_ROC_CI']}  "
                  f"recall={scores['recall']}  F1={scores['macro_F1']}")
    cv_results = pd.DataFrame(rows)
    cv_results.to_csv(os.path.join(OUT, "cv_results.csv"), index=False)

    # ============================================================
    # PART 2: Tune XGBoost on the 2-month window (early-prediction focus)
    # ============================================================
    print("\n[2/5] Hyperparameter tuning (XGBoost, 2-month window, 25 configs x 5-fold)...")
    X2, y2, enrol2, win2 = build_feature_set(student, 2, eng_cols)
    X_tr, X_te, y_tr, y_te = train_test_split(X2, y2, test_size=0.2,
                                              stratify=y2, random_state=42)
    best_params = None
    pre2 = build_preprocessor(enrol2, win2)
    if HAS_XGB:
        best_model, best_params = tune_xgboost(pre2, X_tr, y_tr, n_iter=25)
        print("  Best params:", best_params)
    else:
        print("  NOTE: XGBoost is not installed - falling back to Random Forest.")
        print("        Install it with: pip install xgboost")
        best_model = make_pipeline("Random Forest", pre2, y_tr)

    # --- Threshold tuning (crucial for a 7.3%-imbalanced problem) ---
    # Split training data again to get a clean validation set for choosing
    # the decision threshold; the test set stays untouched until final scoring.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)
    best_model.fit(X_fit, y_fit)
    val_proba = best_model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_val, (val_proba >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    print(f"  Tuned decision threshold (max F1 on validation): {best_t:.2f}")

    # Refit on the full training set, then score the untouched test set
    best_model.fit(X_tr, y_tr)
    proba = best_model.predict_proba(X_te)[:, 1]
    pred_default = (proba >= 0.5).astype(int)
    pred = (proba >= best_t).astype(int)
    final_metrics = {
        "AUC_ROC": round(roc_auc_score(y_te, proba), 3),
        "PR_AUC": round(average_precision_score(y_te, proba), 3),
        "threshold_used": round(best_t, 2),
        "precision": round(precision_score(y_te, pred, zero_division=0), 3),
        "recall": round(recall_score(y_te, pred, zero_division=0), 3),
        "macro_F1": round(f1_score(y_te, pred, average="macro"), 3),
        "precision_at_0.5": round(precision_score(y_te, pred_default, zero_division=0), 3),
        "recall_at_0.5": round(recall_score(y_te, pred_default, zero_division=0), 3),
        "macro_F1_at_0.5": round(f1_score(y_te, pred_default, average="macro"), 3),
        "best_params": best_params,
        "gpu_used": USE_GPU,
    }

    # --- Save the trained model so predict.py and app.py can reuse it ---
    # We also store a "feature template": the exact column order the pipeline
    # was fitted on, plus a sensible default for each column (median for
    # numeric, mode for categorical). The dashboard fills this template and
    # overrides only the fields a user edits - without it, passing a partial
    # DataFrame to the fitted pipeline would raise a column-mismatch error.
    import joblib
    defaults = {}
    for col in X2.columns:
        series = X2[col]
        if pd.api.types.is_numeric_dtype(series):
            defaults[col] = float(series.median())
        else:
            mode = series.mode()
            defaults[col] = (mode.iloc[0] if len(mode) else "")
    bundle = {
        "model": best_model,
        "threshold": best_t,
        "feature_columns": list(X2.columns),
        "feature_defaults": defaults,
        "window_months": 2,
        "metrics": final_metrics,
    }
    joblib.dump(bundle, os.path.join(OUT, "tuned_model.joblib"))
    print("  Saved trained model + feature template -> results/tuned_model.joblib")

    # Save the winning hyperparameters separately so they are never lost
    with open(os.path.join(OUT, "best_params.json"), "w") as f:
        json.dump({"best_params": best_params, "threshold": round(best_t, 3)},
                  f, indent=2, default=str)

    # --- Log environment versions for reproducibility ---
    import sklearn, imblearn
    versions = {"python": sys.version.split()[0], "pandas": pd.__version__,
                "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                "imbalanced-learn": imblearn.__version__}
    try:
        import xgboost; versions["xgboost"] = xgboost.__version__
    except Exception: pass
    with open(os.path.join(OUT, "environment_versions.json"), "w") as f:
        json.dump(versions, f, indent=2)
    with open(os.path.join(OUT, "tuned_model_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    print("  Tuned test metrics:", {k: final_metrics[k] for k in
          ["AUC_ROC", "PR_AUC", "precision", "recall", "macro_F1"]})

    # ============================================================
    # PART 3: Figures — confusion matrix, ROC, PR curves, ablation
    # ============================================================
    print("\n[3/5] Generating evaluation figures...")
    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (tuned model, 2 months)")
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > cm.max()/2 else "black", fontsize=12)
    plt.xticks([0, 1], ["Stayed", "Dropout"]); plt.yticks([0, 1], ["Stayed", "Dropout"])
    plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(y_te, proba)
    plt.figure(figsize=(5, 4.2))
    plt.plot(fpr, tpr, label=f"AUC = {final_metrics['AUC_ROC']}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC Curve (tuned model, 2 months)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "roc_curve.png"), dpi=150); plt.close()

    prec, rec, _ = precision_recall_curve(y_te, proba)
    plt.figure(figsize=(5, 4.2))
    plt.plot(rec, prec, label=f"PR-AUC = {final_metrics['PR_AUC']}")
    plt.axhline(y2.mean(), ls="--", color="grey", label=f"baseline = {y2.mean():.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (tuned model)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "pr_curve.png"), dpi=150); plt.close()

    plt.figure(figsize=(7, 4.5))
    for model in MODELS:
        sub = cv_results[cv_results["model"] == model].sort_values("window_months")
        plt.errorbar(sub["window_months"], sub["AUC_ROC"], yerr=sub["AUC_ROC_CI"],
                     marker="o", capsize=3, label=model)
    plt.xlabel("Months of early engagement used")
    plt.ylabel("AUC-ROC (5-fold CV, 95% CI)")
    plt.title("How prediction improves with more early data (RQ3)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "auc_vs_window.png"), dpi=150); plt.close()

    # ============================================================
    # PART 4: SHAP feature importance (RQ2)
    # ============================================================
    print("\n[4/5] SHAP feature importance...")
    try:
        import shap
        pre_fitted = best_model.named_steps["pre"]
        clf = best_model.named_steps["clf"]
        X_te_enc = pre_fitted.transform(X_te)
        feat_names = pre_fitted.get_feature_names_out()
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_te_enc[:2000])
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        imp = pd.DataFrame({"feature": feat_names, "importance": mean_abs}) \
                .sort_values("importance", ascending=False)
        imp.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
        top = imp.head(12).iloc[::-1]
        plt.figure(figsize=(7, 5))
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Mean |SHAP value|")
        plt.title("Top early predictors of dropout (RQ2)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150); plt.close()
        print("  Top 5:", imp.head(5)["feature"].tolist())
    except Exception as e:
        print("  SHAP skipped:", e)

    # ============================================================
    # PART 5: Fairness check across a subgroup (full/part-time)
    # ============================================================
    print("\n[5/5] Fairness check across student subgroups...")
    try:
        # Guard: only works if X_te kept its DataFrame index (it does here,
        # but this makes the step fail safely rather than crash if that changes)
        if not hasattr(X_te, "index"):
            raise TypeError("X_te is not a DataFrame; cannot map back to subgroups")
        test_idx = X_te.index
        subgroup = student.loc[test_idx, "dedicacion"] if "dedicacion" in student.columns else None
        if subgroup is not None:
            fair_rows = []
            for val in subgroup.dropna().unique():
                mask = (subgroup == val).values
                if mask.sum() < 30:
                    continue
                try:
                    auc = roc_auc_score(y_te.values[mask], proba[mask])
                    rec = recall_score(y_te.values[mask], pred[mask], zero_division=0)
                    fair_rows.append({"subgroup": str(val), "n": int(mask.sum()),
                                      "AUC_ROC": round(auc, 3), "recall": round(rec, 3)})
                except Exception:
                    pass
            fair = pd.DataFrame(fair_rows)
            fair.to_csv(os.path.join(OUT, "fairness_check.csv"), index=False)
            print(fair.to_string(index=False))
    except Exception as e:
        print("  Fairness check skipped:", e)

    print("\nAll done. Everything saved to:", OUT)


if __name__ == "__main__":
    main()
