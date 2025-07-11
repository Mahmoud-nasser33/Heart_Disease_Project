# 01_data_preprocessing.py  (inside notebooks/)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "heart_disease.csv"
print("Loading from:", DATA_PATH)

# 2. Load the dataset
df = pd.read_csv(DATA_PATH)
print("First 5 rows:\n", df.head())

# 3. Check & drop missing values
print("\nMissing values per column:\n", df.isnull().sum())
df.dropna(inplace=True)

# 4. Scale numerical features
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]),
                         columns=numerical_cols)

# 5. Save cleaned, scaled data back to ../data
clean_path = DATA_PATH.parent / "heart_disease_cleaned.csv"
df_scaled.to_csv(clean_path, index=False)
print("\nScaled data saved to:", clean_path)

# 6. Histograms
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# 7. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_scaled.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Scaled Features)")
plt.show()
