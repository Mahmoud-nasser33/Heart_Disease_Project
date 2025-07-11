Heart Disease Prediction Project

A complete machine learning pipeline for predicting heart disease using the UCI dataset. It includes preprocessing, PCA, supervised and unsupervised learning, model tuning, and a Streamlit web interface.

Project Structure:
Heart\_Disease\_Project/
├── data/                  # CSV files (raw, cleaned, PCA)
├── models/                # Saved .pkl model files
├── notebooks/             # Step-by-step ML scripts
├── deployment/            # ngrok setup instructions
├── results/               # Evaluation metrics
├── predictor.py           # Streamlit app
├── requirements.txt       # Required Python packages
├── README.md              # Project description
└── .gitignore             # GitHub ignore list

Features:

* Logistic Regression, Random Forest, SVM classifiers
* Principal Component Analysis (PCA) for dimensionality reduction
* KMeans and Hierarchical Clustering
* Hyperparameter tuning with GridSearchCV & RandomizedSearchCV
* Export trained model to `.pkl`
* Real-time prediction UI with Streamlit
* Optional: Deploy publicly via ngrok

How to Run:

1. Install Required Packages:
   pip install -r requirements.txt

2. Run Streamlit Web App:
   streamlit run predictor.py
   Then open browser at: [http://localhost:8501](http://localhost:8501)


Evaluation Metrics:
See `results/evaluation_metrics.txt` for final performance scores (accuracy, precision, F1-score, etc.).

Requirements:

* Python 3.7+
* Streamlit
* pandas, numpy, matplotlib, seaborn
* scikit-learn, joblib

Author:
Made by Mahmoud

For questions or improvements, feel free to contact or fork this repository.
