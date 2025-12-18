import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import BernoulliNB
from ucimlrepo import fetch_ucirepo
import matplotlib.ticker as ticker

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score
)


# =========================
# 1) Veri
# =========================

secondary_mushroom = fetch_ucirepo(id=848)
df_X = secondary_mushroom.data.features
df_y = secondary_mushroom.data.targets

TARGET_COL = "class"      # hedef sütun
POS_LABEL  = "p"          # pozitif sınıf etiketi (poisonous = p)

X = df_X.copy()
y = df_y.copy()

# sütun tipleri
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# Ön işleme: sayısal + kategorik
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"
)

# ------------------------------------------------------------
# 2) MODEL (Naive Bayes - Bernoulli)
# ------------------------------------------------------------
clf = BernoulliNB()

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", clf)
])

# ------------------------------------------------------------
# 3) 5-FOLD CROSS VALIDATION + METRİKLER + ROC
# ------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 200)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train["class"].to_numpy())

    # Sınıf tahmini
    y_pred = model.predict(X_test)

    # Olasılık (ROC için)
    # Pipeline içindeki clf proba veriyorsa çalışır (DecisionTree verir)
    proba = model.predict_proba(X_test)
    # POS_LABEL hangi index'te?
    class_list = list(model.named_steps["clf"].classes_)
    pos_index = class_list.index(POS_LABEL)
    y_score = proba[:, pos_index]

    # Metrikler
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)

    # ROC-AUC (fold bazlı)
    fpr, tpr, _ = roc_curve((y_test == POS_LABEL).astype(int), y_score)
    fold_auc = auc(fpr, tpr)

    # mean ROC için tpr interpolate
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(fold_auc)

    rows.append({
        "Fold": fold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": fold_auc
    })

results_df = pd.DataFrame(rows)

print("=== 5-Fold Performans Tablosu ===")
print(results_df.to_string(index=False))

summary_df = results_df.drop(columns=["Fold"]).agg(["mean", "std"]).T
summary_df.columns = ["Mean", "Std"]

print("\n=== Özet (Mean ± Std) ===")
print(summary_df.to_string())

# ------------------------------------------------------------
# 4) BOXPLOT (Accuracy, Precision, Recall, F1)
# ------------------------------------------------------------
print("\n=== BOXPLOT (Accuracy, Precision, Recall, F1) ===")
metrics = ["Accuracy", "Precision", "Recall", "F1"]


plt.figure(figsize=(8, 4))

plt.boxplot(
    [results_df[m].values for m in ["Accuracy", "Precision", "Recall", "F1"]],
    tick_labels=["Accuracy", "Precision", "Recall", "F1"],
    showmeans=True,vert=False
)

plt.title("Decision Tree – 5-Fold Metric Boxplot")
plt.ylabel("Metrics")
plt.xlabel("Scores")

plt.grid(True, axis="y", linestyle="--", alpha=0.4)

plt.show()


# ------------------------------------------------------------
# 5) ROC-AUC GRAFİĞİ (Fold'lar + Ortalama)
# ------------------------------------------------------------
print("\n=== ROC-AUC GRAFİĞİ ===")
plt.figure(figsize=(6, 6))

# Fold ROC’ları (ortalama ROC + std bandı)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

std_tpr = np.std(tprs, axis=0)
tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

plt.plot(mean_fpr, mean_tpr, linewidth=2,
         label=f"Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})")
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.2, label="± 1 std. dev.")

plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")

plt.title("Naive Bayes - ROC Curve (5-Fold)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="lower right")
plt.show()