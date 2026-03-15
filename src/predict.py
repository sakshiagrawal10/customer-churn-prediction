import pickle
import pandas as pd

from data_preprocessing import preprocess_data

# Load model
model = pickle.load(open("models/churn_model.pkl", "rb"))

# Load training columns
model_columns = pickle.load(open("models/model_columns.pkl", "rb"))


def predict_churn(data):

    df = pd.DataFrame([data])

    # Apply same preprocessing
    df = preprocess_data(df)

    # Align columns with training data
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)

    if prediction[0] == 1:
        return "Customer will churn"
    else:
        return "Customer will stay"


if __name__ == "__main__":

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70,
        "TotalCharges": 700
    }

    print(predict_churn(sample_customer))
