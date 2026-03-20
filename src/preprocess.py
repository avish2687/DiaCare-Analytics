import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

TARGET = "readmitted"

def preprocess_diabetes_data(df, training=True, encoder=None, scaler=None):
    df = df.copy()

    # Handle target safely
    if training:
        df[TARGET] = df[TARGET].apply(lambda x: 1 if x == "<30" else 0)
        y = df[TARGET]
        df = df.drop(columns=[TARGET])
    else:
        y = None
        if TARGET in df.columns:
            df = df.drop(columns=[TARGET])

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    if training:
        encoder = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", StandardScaler(), numerical_cols),
            ]
        )
        X = encoder.fit_transform(df)
    else:
        X = encoder.transform(df)

    # 🔑 ALWAYS return 4 values
    return X, y, encoder, scaler
