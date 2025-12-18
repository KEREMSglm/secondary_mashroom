import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
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
# 2) MODEL (Decision Tree)
# ------------------------------------------------------------
clf = DecisionTreeClassifier(
    criterion="gini",       # "entropy" da seçebilirsin
    random_state=42
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", clf)
])

# y bazen DataFrame olarak gelir; 1D seriye çevir 
if isinstance(y, pd.DataFrame):
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    else:
        raise ValueError("y birden fazla sütun içeriyor; tek hedef sütun olmalı.")

# Modeli eğit
model.fit(X, y)

# Pipeline içinden preprocess ve tree estimator'ı al
pre = model.named_steps["preprocess"]
tree = model.named_steps["clf"]

# OneHot sonrası feature isimlerini al
# (num + onehot(cat) sırasıyla)
num_feature_names = num_cols

ohe = pre.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()

feature_names = list(num_feature_names) + cat_feature_names

# sınıf isimleri
class_names = [str(c) for c in tree.classes_]

# Çizim
plt.figure(figsize=(16, 10))
plot_tree(
    tree,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    max_depth=2,     # <-- ilk 2 seviye
    fontsize=12
)
plt.title("Decision Tree (Entropy) – İlk 2 Seviye")
plt.tight_layout()
plt.show()