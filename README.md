# Customer Churn Prediction

## Project Overview

Customer churn prediction is an important problem for businesses, especially in the telecom industry. This project uses machine learning to predict whether a customer is likely to leave the service based on various features such as tenure, monthly charges, contract type, and services used.

The model analyzes customer data and classifies customers into two categories:

- Customer will stay
- Customer will churn

Predicting churn helps companies take proactive actions to retain customers and improve customer satisfaction.

---

## Objectives

- Analyze customer data to understand churn patterns
- Perform data preprocessing and feature encoding
- Train a machine learning model for churn prediction
- Evaluate model performance using standard metrics
- Deploy the model using a simple web interface

---

## Technologies Used

- Python
- Pandas for data manipulation
- Scikit-learn for building the model
- Matplotlib and Seaborn for data visualization
- Streamlit for deploying the web application

---

## Project Structure
customer_churn_prediction
│
├── data
│ └── churn.csv
│
├── models
│ ├── churn_model.pkl
│ └── model_columns.pkl
│
├── src
│ ├── data_preprocessing.py
│ ├── train_model.py
│ └── predict.py
│
├── app
│ └── streamlit_app.py
│
├── notebooks
│ └── churn_analysis.ipynb
│
├── requirements.txt
│
└── README.md


---

## Installation

### 1 Clone the repository


git clone https://github.com/your-username/customer_churn_prediction.git


### 2 Navigate to the project folder


cd customer_churn_prediction


### 3 Install required libraries


pip install -r requirements.txt


Or install manually:


pip install pandas numpy scikit-learn matplotlib seaborn streamlit


---

## Dataset

The dataset contains telecom customer information including:

- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- PhoneService
- InternetService
- Contract
- MonthlyCharges
- TotalCharges
- Churn

### Target Variable

`Churn`

- **0 → Customer stays**
- **1 → Customer churns**

---

## Data Preprocessing

The following preprocessing steps were applied:

- Removed unnecessary columns such as `customerID`
- Converted `TotalCharges` to numeric format
- Handled missing values
- Encoded categorical variables
- Split dataset into training and testing sets

---

## Model Used

The project uses the **Random Forest Classifier** from scikit-learn for predicting customer churn.

### Advantages

- Handles categorical and numerical features
- Works well with structured datasets
- Reduces overfitting through ensemble learning

---

## Model Evaluation

The model performance is evaluated using:

- Accuracy
- Classification Report
- Confusion Matrix

Example output:


Accuracy: 0.79


Confusion matrix visualization helps analyze prediction performance and understand model errors.

---

## Running the Project

### 1 Train the model


python src/train_model.py


### 2 Run prediction script


python src/predict.py


### 3 Run Streamlit Web App


streamlit run app/streamlit_app.py


After running the command, open the **local URL displayed in the terminal** (usually `http://localhost:8501`) in your browser.

---

## Streamlit Application

The web application built using Streamlit allows users to enter customer details and predict whether the customer will churn.

### Features

- Interactive user interface
- Real-time churn prediction
- Simple and user-friendly design

---

## Future Improvements

- Hyperparameter tuning for improved model accuracy
- Feature importance analysis
- Deploying the model on cloud platforms
- Adding churn probability visualization
- Integrating dashboards for business insights

---

## Conclusion

This project demonstrates how machine learning can be used to analyze customer behavior and pr