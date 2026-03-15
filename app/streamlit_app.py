import streamlit as st
import pickle
import pandas as pd
import sys
import os

# Allow importing from src folder
sys.path.append(os.path.abspath("src"))

from data_preprocessing import preprocess_data

# Load model and training columns
model = pickle.load(open("models/churn_model.pkl", "rb"))
model_columns = pickle.load(open("models/model_columns.pkl", "rb"))

st.title("Customer Churn Prediction")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=0)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict"):

    sample_customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    df = pd.DataFrame([sample_customer])

    # Apply preprocessing
    df = preprocess_data(df)

    # Match training columns
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)

    if prediction[0] == 1:
        st.error("Customer will churn")
    else:
        st.success("Customer will stay")
