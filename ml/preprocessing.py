import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def prepare_data(df, target_column, problem_type,feature_columns = None):

    if target_column not in df.columns:
        raise ValueError("Target column not found")

    df = df.dropna(subset=[target_column]).copy()

    if feature_columns is not None:

        missing_columns = [
            column for column in feature_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Feature columns not found: {missing_columns}")

        if target_column in feature_columns:
            raise ValueError("Target column cannot be a feature")

        X = df[feature_columns]

    else:
        X = df.drop(columns=[target_column])

    y = df[target_column]

    if X.empty:
        raise ValueError("No feature columns available")

    if problem_type == "classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes")

    if problem_type not in ["classification", "regression"]:
        raise ValueError("Invalid problem type")

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor