import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load PCA features (299 rows)
X = pd.read_csv("../data/heart_disease_pca.csv")

# Load and clean labels (299 rows)
original_df = pd.read_csv("../data/heart_disease.csv")
original_df.dropna(inplace=True)
y = (original_df["num"] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression + GridSearch
log_params = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}
log_model = LogisticRegression(max_iter=1000)
log_grid = GridSearchCV(log_model, log_params, cv=5)
log_grid.fit(X_train, y_train)
print("Logistic Regression Best Params:", log_grid.best_params_)
log_preds = log_grid.predict(X_test)
print("Logistic Regression Report")
print(classification_report(y_test, log_preds, zero_division=0))

# Random Forest + RandomizedSearch
rf_params = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 4, 6]
}
rf_model = RandomForestClassifier()
rf_search = RandomizedSearchCV(rf_model, rf_params, n_iter=10, cv=5, random_state=42)
rf_search.fit(X_train, y_train)
print("Random Forest Best Params:", rf_search.best_params_)
rf_preds = rf_search.predict(X_test)
print("Random Forest Report")
print(classification_report(y_test, rf_preds, zero_division=0))

# SVM + GridSearch
svm_params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}
svm_model = SVC(probability=True)
svm_grid = GridSearchCV(svm_model, svm_params, cv=5)
svm_grid.fit(X_train, y_train)
print("SVM Best Params:", svm_grid.best_params_)
svm_preds = svm_grid.predict(X_test)
print("SVM Report")
print(classification_report(y_test, svm_preds, zero_division=0))
