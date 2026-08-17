"""
exploration.py
--------------
Exploratory Data Analysis for Student Dropout Prediction
Runs all cells from the notebook in sequence.
Outputs figures and tables to eda_outputs/
"""

import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_prep import (
    load_raw,
    aggregate_to_student,
    MONTH_ORDER,
    ENGAGEMENT_METRICS,
    ENROLMENT_FEATURES,
)

warnings.filterwarnings("ignore")

# --- Setup ---
OUT = "eda_outputs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
C_STAY, C_DROP = "#2874a6", "#c0392b"
print("Setup complete")

# --- 1. Data integrity check ---
df = load_raw("dataset_2022_hash.csv")

print(f"Raw shape:            {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Duplicated rows:      {df.duplicated().sum()}")
print(f"Unique students:      {df['dni_hash'].nunique():,}")
print(f"Rows per student:     mean {len(df)/df['dni_hash'].nunique():.2f}, max {df['dni_hash'].value_counts().max()}")
print(f"Missing target label: {df['abandono_hash'].isna().sum()}")
print()
print("Column types:")
print(df.dtypes.value_counts().to_string())

# --- 2. Aggregation to student level ---
student, eng_cols = aggregate_to_student(df)
print(f"After aggregation: {len(student):,} students x {student.shape[1]} columns")
print(f"Engagement columns retained: {len(eng_cols)}")
print(f"One row per student: {student['dni_hash'].is_unique}")
print(student.head(3))

# --- 3. Target class balance ---
vc = student["dropout"].value_counts().sort_index()
pct = student["dropout"].value_counts(normalize=True).sort_index() * 100
bal = pd.DataFrame(
    {
        "class": ["Stayed (0)", "Dropped out (1)"],
        "count": vc.values,
        "percent": pct.values.round(2),
    }
)
print(bal.to_string(index=False))
print(f"\nImbalance ratio: 1 dropout to {vc[0]/vc[1]:.1f} continuing students")
bal.to_csv(f"{OUT}/table_class_balance.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4))
ax[0].bar(["Stayed", "Dropped out"], vc.values, color=[C_STAY, C_DROP])
for i, v in enumerate(vc.values):
    ax[0].text(i, v, f"{v:,}", ha="center", va="bottom")
ax[0].set_ylabel("Number of students")
ax[0].set_title("Class counts")
ax[1].pie(
    vc.values,
    labels=[f"Stayed\n{pct[0]:.1f}%", f"Dropped out\n{pct[1]:.1f}%"],
    colors=[C_STAY, C_DROP],
    startangle=90,
    wedgeprops={"edgecolor": "white"},
)
ax[1].set_title("Class proportion")
fig.suptitle("Figure 1. Target class distribution", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig01_class_balance.png", bbox_inches="tight")
plt.show()

# --- 4. Missing data ---
labels, miss = [], []
for (y, m) in MONTH_ORDER:
    col = f"pft_events_{y}_{m}"
    if col in student.columns:
        labels.append(f"{m}/{str(y)[2:]}")
        miss.append(student[col].isna().mean() * 100)
mt = pd.DataFrame({"month": labels, "missing_percent": np.round(miss, 1)})
print("Engagement missingness by month:")
print(mt.to_string(index=False))

enr = [c for c in ENROLMENT_FEATURES if c in student.columns]
enr_miss = pd.DataFrame(
    {
        "feature": enr,
        "missing_percent": [round(student[c].isna().mean() * 100, 1) for c in enr],
    }
).sort_values("missing_percent", ascending=False)
print("\nEnrolment feature missingness:")
print(enr_miss.to_string(index=False))
enr_miss.to_csv(f"{OUT}/table_missing_enrolment.csv", index=False)

# --- 5. Engagement trajectory by outcome ---
traj = []
for (y, m) in MONTH_ORDER:
    col = f"pft_events_{y}_{m}"
    if col in student.columns:
        traj.append(
            {
                "month": f"{m}/{str(y)[2:]}",
                "stayed_mean": student.loc[student.dropout == 0, col].mean(),
                "dropout_mean": student.loc[student.dropout == 1, col].mean(),
            }
        )
tj = pd.DataFrame(traj)
tj["ratio"] = (tj.stayed_mean / tj.dropout_mean).round(2)
print(tj.round(1).to_string(index=False))
tj.to_csv(f"{OUT}/table_engagement_trajectory.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(tj.month, tj.stayed_mean, marker="o", color=C_STAY, label="Stayed")
ax.plot(tj.month, tj.dropout_mean, marker="s", color=C_DROP, label="Dropped out")
ax.set_ylabel("Mean platform events")
ax.set_xlabel("Month of academic year")
ax.set_title("Figure 3. Platform engagement over the year, by final outcome")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig03_engagement_trajectory.png")
plt.show()

# --- 6. Effect size of early features ---
early = [f"{m}_2022_9" for m in ENGAGEMENT_METRICS] + [
    f"{m}_2022_10" for m in ENGAGEMENT_METRICS
]
early = [c for c in early if c in student.columns] + ["nota10_hash"]

rows = []
for col in early:
    a = student.loc[student.dropout == 0, col].dropna()
    b = student.loc[student.dropout == 1, col].dropna()
    if len(a) > 10 and len(b) > 10:
        pooled = np.sqrt((a.var() + b.var()) / 2)
        rows.append(
            {
                "feature": col,
                "stayed_mean": round(a.mean(), 2),
                "dropout_mean": round(b.mean(), 2),
                "cohens_d": round((a.mean() - b.mean()) / pooled if pooled > 0 else 0, 3),
            }
        )
comp = pd.DataFrame(rows).sort_values("cohens_d", key=abs, ascending=False)
print(comp.head(12))
comp.to_csv(f"{OUT}/table_early_feature_comparison.csv", index=False)

top8 = comp.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.barh(
    top8.feature,
    top8.cohens_d,
    color=[C_STAY if v > 0 else C_DROP for v in top8.cohens_d],
)
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("Cohen's d  (positive = higher among students who stayed)")
ax.set_title("Figure 4. Effect size of early features on outcome")
fig.tight_layout()
fig.savefig(f"{OUT}/fig04_effect_sizes.png")
plt.show()

# --- 7. Distributions of key early features ---
keys = [
    c
    for c in [
        "nota10_hash",
        "n_wifi_days_2022_9",
        "pft_days_logged_2022_9",
        "pft_events_2022_10",
    ]
    if c in student.columns
]
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for axx, col in zip(axes.ravel(), keys):
    s0 = student.loc[student.dropout == 0, col].dropna()
    s1 = student.loc[student.dropout == 1, col].dropna()
    hi = np.nanpercentile(pd.concat([s0, s1]), 97)
    bins = np.linspace(0, max(hi, 1), 30)
    axx.hist(s0, bins=bins, alpha=0.6, density=True, color=C_STAY, label="Stayed")
    axx.hist(s1, bins=bins, alpha=0.6, density=True, color=C_DROP, label="Dropped out")
    axx.set_title(col, fontsize=9)
    axx.legend(fontsize=7)
fig.suptitle("Figure 5. Distribution of key early features by outcome", y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/fig05_distributions.png", bbox_inches="tight")
plt.show()

# --- 8. Dropout rate by student characteristic ---
for col in ["dedicacion", "tipo_ingreso", "estudios_m_hash"]:
    if col in student.columns:
        g = student.groupby(col)["dropout"].agg(["count", "mean"])
        g = g[g["count"] >= 100].sort_values("mean", ascending=False)
        g["dropout_rate_pct"] = (g["mean"] * 100).round(1)
        print(f"\n{col} (groups with n>=100):")
        print(g[["count", "dropout_rate_pct"]].head(8).to_string())
        g[["count", "dropout_rate_pct"]].to_csv(f"{OUT}/table_dropout_by_{col}.csv")

# --- 9. Correlation and multicollinearity ---
corr_cols = [c for c in early if c in student.columns][:14]
cm = student[corr_cols].corr()

hi = [
    (cm.index[i], cm.columns[j], round(cm.iloc[i, j], 2))
    for i in range(len(cm))
    for j in range(i + 1, len(cm))
    if abs(cm.iloc[i, j]) > 0.8
]
print(f"Feature pairs with |r| > 0.8: {len(hi)}")
for a, b, r in hi[:8]:
    print(f"  {a}  <->  {b}   r={r}")
pd.DataFrame(hi, columns=["feature_a", "feature_b", "r"]).to_csv(
    f"{OUT}/table_high_correlations.csv", index=False
)

fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=90, fontsize=6)
ax.set_yticks(range(len(corr_cols)))
ax.set_yticklabels(corr_cols, fontsize=6)
fig.colorbar(im, shrink=0.8)
ax.set_title("Figure 7. Correlation matrix of early features")
fig.tight_layout()
fig.savefig(f"{OUT}/fig07_correlation.png")
plt.show()

# --- 10. Summary of key numbers ---
summary = {
    "raw_rows": int(df.shape[0]),
    "raw_cols": int(df.shape[1]),
    "duplicate_rows": int(df.duplicated().sum()),
    "students": int(len(student)),
    "engagement_columns": int(len(eng_cols)),
    "dropout_rate_pct": round(float(student.dropout.mean() * 100), 2),
    "imbalance_ratio": round(float((1 - student.dropout.mean()) / student.dropout.mean()), 1),
    "strongest_early_signal": comp.iloc[0]["feature"],
    "strongest_effect_size": float(comp.iloc[0]["cohens_d"]),
    "high_corr_pairs": len(hi),
}
for k, v in summary.items():
    print(f"{k}: {v}")
with open(f"{OUT}/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll EDA outputs saved to:", OUT)