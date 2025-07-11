import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# Load cleaned & scaled data from Step 2
data_path = Path(__file__).resolve().parent.parent / "data" / "heart_disease_cleaned.csv"
df = pd.read_csv(data_path)
print("Original shape:", df.shape)

# Fit PCA
pca = PCA()
pca.fit(df)

# Scree plot
explained = pca.explained_variance_ratio_
cum_var = explained.cumsum()

plt.plot(range(1, len(explained) + 1), explained, marker='o')
plt.title("Scree Plot")
plt.xlabel("Component number")
plt.ylabel("Explained variance")
plt.grid(True)
plt.show()

plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
plt.axhline(0.95, linestyle='--', color='r', label='95% variance')
plt.title("Cumulative Variance")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.grid(True)
plt.legend()
plt.show()

# Keep components to preserve 95% variance
n_components = (cum_var < 0.95).sum() + 1
print(f"Using {n_components} components to retain {cum_var[n_components-1]:.2%} variance")

pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(df)
pca_df = pd.DataFrame(pca_features, columns=[f"PC{i+1}" for i in range(n_components)])

# Save to data folder
pca_out = data_path.parent / "heart_disease_pca.csv"
pca_df.to_csv(pca_out, index=False)
print("Saved PCA features to:", pca_out)
