import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop unnecessary column
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Encode target variable only during training
    if "Churn" in df.columns:
        le = LabelEncoder()
        df["Churn"] = le.fit_transform(df["Churn"])
    

    # Convert categorical variables
    df = pd.get_dummies(df, drop_first=True)

    return df
