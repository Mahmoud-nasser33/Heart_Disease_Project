import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load PCA features
X = pd.read_csv("../data/heart_disease_pca.csv")

# Load and clean labels
original_df = pd.read_csv("../data/heart_disease.csv")
original_df.dropna(inplace=True)
y = (original_df["num"] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model (you can replace with your best model)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model to a .pkl file
joblib.dump(model, "../models/final_model.pkl")
print("Model saved to: models/final_model.pkl")

# (Optional) To test loading it:
# loaded_model = joblib.load("../models/final_model.pkl")
# print("Loaded model predicts:", loaded_model.predict(X_test[:5]))
