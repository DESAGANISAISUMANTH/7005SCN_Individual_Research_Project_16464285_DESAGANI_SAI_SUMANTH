# SETUP GUIDE — Run the Dropout Prediction Artefact on Your PC

Follow these steps in order. Total setup time: ~15 minutes.
Full experiment runtime on an RTX machine: roughly 15–40 minutes.

---

## STEP 1 — Install Python (skip if you have Python 3.10+)

1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or 3.12 for Windows.
3. Run the installer. **IMPORTANT: tick "Add python.exe to PATH"** on the first screen.
4. Verify: open Command Prompt (press Win+R, type `cmd`, Enter) and run:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`.

---

## STEP 2 — Create a project folder

1. Make a folder, e.g. `C:\dropout_project`
2. Put these files in it (from the package I gave you):
   - `data_prep.py`
   - `modelling_v2.py`
   - `run_full_experiment.py`
   - `requirements.txt`
3. Put your dataset file `dataset_2022_hash.csv` in the SAME folder.
   (This is the file you downloaded from Zenodo: https://zenodo.org/records/17239943 —
   download the 2022 zip and extract the CSV.)

Your folder should look like:
```
C:\dropout_project\
    data_prep.py
    modelling_v2.py
    run_full_experiment.py
    requirements.txt
    dataset_2022_hash.csv
```

---

## STEP 3 — Create a virtual environment (keeps things clean)

Open Command Prompt and run these one at a time:

```
cd C:\dropout_project
python -m venv venv
venv\Scripts\activate
```

You should now see `(venv)` at the start of the prompt line.
(Every time you come back later, run `venv\Scripts\activate` again first.)

---

## STEP 4 — Install the libraries

With `(venv)` active:

```
pip install -r requirements.txt
```

This installs pandas, scikit-learn, xgboost, imbalanced-learn, shap, matplotlib.
Takes 2–5 minutes.

**GPU note (your RTX card):** XGBoost 2.x supports NVIDIA GPUs out of the box —
no CUDA toolkit install needed. The code auto-detects your GPU (via `nvidia-smi`)
and switches XGBoost to `device="cuda"` automatically. When you run the experiment,
the first line printed tells you: `GPU acceleration: ON (CUDA)` or `off (CPU)`.
Either way the results are the same — GPU is just faster.

---

## STEP 5 — Quick smoke test (30 seconds)

Check the data loads correctly:

```
python data_prep.py
```

Expected output (numbers should match exactly):
```
Raw shape: (159173, 169)
Students: 20427 | dropout rate: 7.3 %
Feature set (2 months): X shape (20427, 25)
```

If you see this, everything is wired up correctly.

---

## STEP 6 — Run the automated tests (1 minute)

```
python test_pipeline.py
```

Seven checks run: data shape, decimal cleaning, one-row-per-student,
label rate, **no future-data leakage in the early windows**, pipeline
training, and stratified splitting. All seven should print `[PASS]`.
Screenshot this output — it is direct evidence of testing for your report.

---

## STEP 7 — Run the full experiment

```
python run_full_experiment.py
```

What it does, in order:
1. **[1/5] Cross-validated ablation** — 3 models × 4 early windows × 5-fold CV
   (answers RQ1 and RQ3). ~5–15 min.
2. **[2/5] Hyperparameter tuning** — 25 XGBoost configurations × 5-fold CV
   on the 2-month window. ~5–20 min (fast on GPU).
3. **[3/5] Figures** — confusion matrix, ROC curve, PR curve, ablation plot.
4. **[4/5] SHAP analysis** — which early features predict dropout (RQ2).
5. **[5/5] Fairness check** — model performance across full-time vs part-time students.

---

## STEP 8 — Generate the at-risk list (the artefact in action)

```
python predict.py
```

This loads the tuned model saved by Step 7 and produces
`results\at_risk_list.csv` — every student ranked by dropout probability,
flagged using the tuned decision threshold. This is your artefact working
as a real early-warning tool, not just an experiment. Include the top of
this table (anonymised IDs are already hashed) as a figure in your report.

---

## STEP 9 — Launch the interactive dashboard (optional but impressive)

```
streamlit run app.py
```

Your browser opens a dashboard where you can enter a student's admission grade
and early engagement, get a risk score, and see a SHAP chart explaining which
factors drove it. Screenshot this for your Artefact chapter — it demonstrates
the model working as a usable tool, not just a script.

Note: your brief accepts "data analysis outputs or models" as a valid artefact,
so the pipeline alone already qualifies. The dashboard is an enhancement that
makes the artefact easier to demonstrate and evaluate.

---

## STEP 10 — Collect your results

Everything is saved into `C:\dropout_project\results\`:

| File | What it is | Report section it feeds |
|------|-----------|------------------------|
| `cv_results.csv` | CV metrics ± 95% CI for every model × window | Evaluation (RQ1, RQ3) |
| `tuned_model_metrics.json` | Final tuned-model test metrics + best hyperparameters | Artefact & Evaluation |
| `confusion_matrix.png` | Confusion matrix figure | Evaluation |
| `roc_curve.png` | ROC curve figure | Evaluation |
| `pr_curve.png` | Precision-Recall curve figure | Evaluation |
| `auc_vs_window.png` | AUC vs months of data (with error bars) | Evaluation (RQ3) |
| `feature_importance.csv/.png` | SHAP top predictors | Evaluation (RQ2) |
| `fairness_check.csv` | Subgroup performance | Ethics/Social section |
| `tuned_model.joblib` | The saved trained model | Artefact (reusability) |
| `at_risk_list.csv` | Ranked at-risk students from predict.py | Artefact in action |
| `environment_versions.json` | Library versions used | Reproducibility/Methodology |
| `best_params.json` | Winning hyperparameters + tuned threshold | Artefact (tuning evidence) |

Note: `tuned_model_metrics.json` now reports metrics at BOTH the default 0.5
threshold and the tuned threshold — comparing them is a genuinely good
paragraph for your evaluation section (why threshold choice matters at 7.3%).

---

## Troubleshooting

- **`python` not recognised** → reinstall Python and tick "Add to PATH", or use `py` instead of `python`.
- **`FileNotFoundError: dataset_2022_hash.csv`** → the CSV isn't in the folder, or is named differently. You can also pass the path directly:
  `python run_full_experiment.py C:\path\to\dataset_2022_hash.csv`
- **Out-of-memory during tuning** → close other apps; or open `run_full_experiment.py` and change `n_iter=25` to `n_iter=10`.
- **GPU shows "off (CPU)"** → your NVIDIA driver may be old. It still runs fine on CPU, just slower. To enable GPU, update your GeForce driver from nvidia.com.
- **SHAP step errors** → run `pip install shap --upgrade` and re-run; the rest of the results are unaffected either way.

---

## What to do AFTER it runs (important for your report)

1. Open `results\cv_results.csv` in Excel — this is your main results table.
2. Look at the four PNG figures — these go directly into your report's evaluation section.
3. Open `tuned_model_metrics.json` in Notepad — note the best hyperparameters; you
   will describe the tuning process in the Artefact section.
4. Compare the tuned test AUC against the untuned CV AUC — the improvement (or lack
   of it) is itself a finding to discuss honestly.
5. Read every script top to bottom and make sure you can explain each step —
   the aggregation, the comma-decimal fix, why SMOTE is only applied to training
   data, why PR-AUC matters at 7.3% dropout, what SHAP values mean. Your report
   (and any viva questions) depend on YOUR understanding, not just the outputs.
