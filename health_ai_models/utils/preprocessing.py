# Data preprocessing utilities
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_dataframe(df):
    encoders = {}

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders