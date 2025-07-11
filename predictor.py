import streamlit as st
import pandas as pd
import joblib

# Load the trained model (we trained it before and saved it)
model = joblib.load("models/final_model.pkl")

# Set the title of the web page
st.set_page_config(page_title="Heart Disease Predictor", )
st.title("Heart Disease Prediction")

st.write("Fill in the patient’s information below to see if they may have heart disease.")

# Take inputs from the user
age = st.number_input("Age (years)", min_value=20, max_value=100)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Type of Chest Pain (0: typical angina, 1–3: other types)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
chol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220)
exang = st.selectbox("Exercise-Induced Angina?", ["No", "Yes"])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia Type (0–3)", [0, 1, 2, 3])

# Convert choices into numbers the model understands
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Create a row of input data
input_data = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]])

# Show user what they entered
st.subheader("You Entered:")
st.write(input_data)

# When the user clicks the "Predict" button
if st.button("Predict"):
    st.warning("This is just a preview. The model needs scaled and PCA data.")
    st.info("In the next step, we'll process the input to match the model training.")
else:
    st.info("Fill out the form and click Predict to get a result.")

