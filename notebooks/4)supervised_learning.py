import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load PCA features (299 rows)
X = pd.read_csv("../data/heart_disease_pca.csv")

# Load original labels and drop rows to match PCA
original_df = pd.read_csv("../data/heart_disease.csv")
original_df.dropna(inplace=True)

# Convert to binary: 1 = has disease, 0 = no disease
y = (original_df["num"] > 0).astype(int)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
