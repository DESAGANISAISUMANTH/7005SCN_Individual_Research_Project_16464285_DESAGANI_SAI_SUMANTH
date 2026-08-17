"""
modelling_v2.py
---------------
Modelling module for the CW2 artefact — full-power settings for a local
machine (uses your NVIDIA GPU for XGBoost automatically if available).

Models: Logistic Regression (baseline), Random Forest, XGBoost.
Imbalance: SMOTE (train folds only, baseline) + class weights / scale_pos_weight.
Metrics: AUC-ROC, PR-AUC, precision, recall, macro-F1 with
         stratified cross-validation and 95% confidence intervals.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, make_scorer,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def _gpu_available():
    """Detect an NVIDIA GPU for XGBoost acceleration."""
    try:
        import subprocess
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except Exception:
        return False


USE_GPU = _gpu_available()


def build_preprocessor(enrol_cols, engagement_cols):
    numeric_cols = ["nota10_hash"] + engagement_cols
    numeric_cols = [c for c in numeric_cols if c in (enrol_cols + engagement_cols)]
    categorical_cols = [c for c in enrol_cols if c not in numeric_cols]

    numeric_pipe = SkPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = SkPipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ], remainder="drop")


def make_xgb(spw, **overrides):
    """Create an XGBoost classifier, on GPU if available."""
    params = dict(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
        eval_metric="logloss", n_jobs=-1, random_state=42,
        tree_method="hist",
    )
    if USE_GPU:
        params["device"] = "cuda"
    params.update(overrides)
    return XGBClassifier(**params)


def make_pipeline(name, preprocessor, y_train):
    n_pos = int(np.sum(y_train == 1)); n_neg = int(np.sum(y_train == 0))
    spw = round(n_neg / max(n_pos, 1), 2)

    if name == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        steps = [("pre", preprocessor), ("smote", SMOTE(random_state=42)), ("clf", clf)]
    elif name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=42)
        steps = [("pre", preprocessor), ("clf", clf)]
    elif name == "XGBoost" and HAS_XGB:
        clf = make_xgb(spw)
        steps = [("pre", preprocessor), ("clf", clf)]
    else:
        return None
    return ImbPipeline(steps)


SCORERS = {
    "AUC_ROC": "roc_auc",
    "PR_AUC": "average_precision",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "macro_F1": make_scorer(f1_score, average="macro"),
}


def cross_validated_scores(pipe, X, y, n_splits=5, n_jobs=None):
    """
    Return mean and 95% CI for each metric across stratified CV folds.

    n_jobs: how many folds to run in parallel. Default is 2 (a safe balance).
      - Set higher (or -1) on a machine with plenty of RAM to go faster.
      - Keep low if XGBoost is running on the GPU, since parallel folds all
        compete for the same GPU and can exhaust its memory.
    """
    if n_jobs is None:
        n_jobs = 1 if USE_GPU else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cvres = cross_validate(pipe, X, y, cv=cv, scoring=SCORERS, n_jobs=n_jobs)
    out = {}
    for metric in SCORERS:
        vals = cvres[f"test_{metric}"]
        mean = np.mean(vals)
        ci = 1.96 * np.std(vals) / np.sqrt(len(vals))
        out[metric] = round(mean, 3)
        out[f"{metric}_CI"] = round(ci, 3)
    return out


def tune_xgboost(preprocessor, X, y, n_iter=25):
    """Randomised hyperparameter search for XGBoost (GPU-accelerated if available)."""
    if not HAS_XGB:
        return None, None
    n_pos = int(np.sum(y == 1)); n_neg = int(np.sum(y == 0))
    spw = round(n_neg / max(n_pos, 1), 2)
    pipe = ImbPipeline([("pre", preprocessor), ("clf", make_xgb(spw))])
    param_dist = {
        "clf__n_estimators": [200, 400, 600, 800],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.02, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__gamma": [0, 0.1, 0.3],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipe, param_dist, n_iter=n_iter, scoring="roc_auc",
                                cv=cv, n_jobs=1, random_state=42, verbose=1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_
