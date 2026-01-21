import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def check_null_values(df):
    """Check for null values in the dataframe"""
    return df.isnull().sum()

def encode_data(df):
    """Encode categorical variables using one-hot encoding"""
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded

def standard_scale(df):
    """Standardize features using StandardScaler"""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled