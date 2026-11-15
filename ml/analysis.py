from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

def analyze_dataframe(df, target_column, problem_type, feature_columns=None):
    """Validate a supervised learning task and split its raw columns."""
    if target_column not in df.columns:
        raise ValueError("Target column not found")
    if problem_type not in ("classification", "regression"):
        raise ValueError("Problem type must be classification or regression")

    df = df.dropna(subset=[target_column]).copy()
    if feature_columns is None:
        feature_columns = [column for column in df.columns if column != target_column]

    if not feature_columns:
        raise ValueError("No feature columns selected")
    if target_column in feature_columns:
        raise ValueError("The target cannot also be an input feature")

    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")

    X = df[feature_columns]
    y = df[target_column]
    if X.empty:
        raise ValueError("No usable rows or features")

    if problem_type == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression requires a numeric target")
        stratify = None
    else:
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two classes")
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, feature_columns


def analyze_csv(filename, target_column, problem_type, feature_columns=None):
    from tools.csv_tools import load_csv

    df = load_csv(filename)
    parts = analyze_dataframe(df, target_column, problem_type, feature_columns)
    return (*parts, Path(filename).name)
