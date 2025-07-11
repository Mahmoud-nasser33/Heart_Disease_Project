import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Load PCA-transformed data (299 rows)
X = pd.read_csv("../data/heart_disease_pca.csv")

# Load original labels and drop rows with NaN to match X
original_df = pd.read_csv("../data/heart_disease.csv")
original_df.dropna(inplace=True)

# Convert to binary: 1 = has disease, 0 = no disease
y = (original_df["num"] > 0).astype(int)

# Elbow Method to choose best k
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Apply KMeans with k=2 (because binary classification)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# Compare clusters with true labels
ari_kmeans = adjusted_rand_score(y, kmeans_labels)
print("Adjusted Rand Index (KMeans vs True Labels):", ari_kmeans)

# Visualize KMeans clustering
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans_labels, cmap='viridis')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering (k=2)")
plt.show()

# Hierarchical Clustering
Z = linkage(X, method="ward")

# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Apply flat clustering (e.g., 2 clusters)
hier_labels = fcluster(Z, t=2, criterion='maxclust')

# Compare with real labels
ari_hier = adjusted_rand_score(y, hier_labels)
print("Adjusted Rand Index (Hierarchical vs True Labels):", ari_hier)

# Visualize Hierarchical clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=hier_labels, cmap='plasma')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Hierarchical Clustering (2 clusters)")
plt.show()
