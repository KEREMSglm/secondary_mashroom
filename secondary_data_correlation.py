import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

secondary_mushroom = fetch_ucirepo(id=848)
# Hedef sütunu çıkar (sadece özellikler kalsın)
X_df = secondary_mushroom.data.features
# Kategorik verileri sayısallaştır
for col in X_df.columns:
    X_df.loc[:, col] = LabelEncoder().fit_transform(X_df[col].astype(str))

# Korelasyon matrisi oluştur
corr_matrix = X_df.corr()

# Görselleştir (heatmap)
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="RdBu_r",
    center=0,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={'label': 'Correlation Strength'}
)
plt.title("Secondary Data", fontsize=16, weight="bold")
plt.tight_layout()
plt.show()