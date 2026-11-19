"""
Data Loader and Preprocessing Module for Loan Approval Prediction
COMP1818 Artificial Intelligence Applications - University of Greenwich
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def load_raw_dataset(filepath="data/train.csv"):
    """
    Loads raw CSV data and handles target mapping.
    Target: Loan_Status (Y -> 1, N -> 0)
    """
    if not os.path.exists(filepath):
        # Fallback to local data folder or relative path
        alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train.csv")
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath} or {alt_path}")

    df = pd.read_csv(filepath)

    # Drop Loan_ID if present
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # Drop records with missing target labels
    df = df.dropna(subset=["Loan_Status"]).copy()

    # Clean and map target
    df["Loan_Status"] = (
        df["Loan_Status"].astype(str).str.strip().str.upper().map({"Y": 1, "N": 0})
    )
    df = df.dropna(subset=["Loan_Status"]).copy()
    df["Loan_Status"] = df["Loan_Status"].astype(int)

    return df


def engineer_financial_features(df):
    """
    Constructs domain-specific financial ratios:
    - TotalIncome: ApplicantIncome + CoapplicantIncome
    - Loan_to_Income_Ratio: (LoanAmount * 1000) / TotalIncome
    - EMI_Estimate: Monthly installment proxy
    - BalanceIncome: Disposable cashflow buffer
    """
    df = df.copy()
    total_income = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["TotalIncome"] = total_income

    # Safe loan amount in actual currency units (LoanAmount is in thousands)
    loan_val = df["LoanAmount"].fillna(df["LoanAmount"].median()) * 1000
    df["Loan_to_Income_Ratio"] = loan_val / (total_income + 1.0)

    # Estimated Monthly Installment proxy (assuming term in months)
    term_months = df["Loan_Amount_Term"].fillna(360.0).replace(0, 360.0)
    df["EMI_Estimate"] = loan_val / term_months
    df["BalanceIncome"] = total_income - df["EMI_Estimate"]

    return df


def get_data_splits(filepath="data/train.csv", test_size=0.20, random_state=42, engineer_features=False):
    """
    Partitions the dataset into stratified training and testing splits.
    Returns: X_train, X_test, y_train, y_test, numerical_cols, categorical_cols
    """
    df = load_raw_dataset(filepath)

    if engineer_features:
        df = engineer_financial_features(df)

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    """
    Constructs a leak-free ColumnTransformer for numerical and categorical features.
    - Numerical: Median imputation + Z-score standardization
    - Categorical: Most frequent imputation + OneHotEncoder (ignoring unseen classes)
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor
