# 02_pca_analysis.py     (inside notebooks/)
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# 1. Load the cleaned & scaled data from Step
#    (.. = go up from notebooks/ to project root)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "heart_disease_cleaned.csv"
df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)


# 2. Fit PCA keeping all components to inspect variance

pca_full = PCA()
pca_full.fit(df)

explained = pca_full.explained_variance_ratio_
cum_var   = explained.cumsum()


# 3. Plots

# Scree plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(explained)+1), explained, marker='o')
plt.title("Scree Plot – Variance per Component")
plt.xlabel("Component number")
plt.ylabel("Explained variance ratio")
plt.grid(True)
plt.show()

# Cumulative variance plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
plt.axhline(0.95, linestyle='--', label='95 % threshold')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of components")
plt.ylabel("Cumulative variance ratio")
plt.legend()
plt.grid(True)
plt.show()


# 4. Choose number of components that keeps ≥95
#    (change 0.95 to 0.90 etc. if you prefer)

n_components = (cum_var < 0.95).sum() + 1
print(f"Keeping {n_components} components "
      f"→ {cum_var[n_components-1]:.2%} variance retained")

# Fit PCA with chosen components and transform data
pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(df)

# Build a DataFrame with principal components
pca_cols = [f"PC{i+1}" for i in range(n_components)]
df_pca = pd.DataFrame(pca_features, columns=pca_cols)


# 5. Save the PCA‑reduced dataset for the next step

PCA_PATH = DATA_PATH.parent / "heart_disease_pca.csv"
df_pca.to_csv(PCA_PATH, index=False)
print("PCA‑transformed data saved to:", PCA_PATH)
print("Shape after PCA:", df_pca.shape)
