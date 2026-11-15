"""Data preparation, model training, evaluation, and persistence."""

from math import sqrt
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_DIR = Path("workspace/output/models")



def create_preprocessor(X):

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

    return preprocessor


def prepare_raw_data(df, target_column, problem_type, feature_columns=None):

    if target_column not in df.columns:
        raise ValueError("Target column not found")

    if problem_type not in ["classification", "regression"]:
        raise ValueError("Invalid problem type")

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

    preprocessor = create_preprocessor(X)

    if problem_type == "classification":
        stratify_data = y
    else:
        stratify_data = None

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=stratify_data)

    return X_train, X_test, y_train, y_test, preprocessor


def prepare_data(df, target_column, problem_type, feature_columns=None):

    X_train, X_test, y_train, y_test, preprocessor = prepare_raw_data(
        df,
        target_column,
        problem_type,
        feature_columns
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor


def get_models(problem_type):

    if problem_type == "classification":

        models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

    elif problem_type == "regression":

        models = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }

    else:
        raise ValueError("Invalid problem type")

    return models


def train_models(problem_type, X_train, y_train):

    models = get_models(problem_type)

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models


def evaluate_models(problem_type, trained_models, X_test, y_test):

    results = {}
    predictions = {}

    for name, model in trained_models.items():

        y_pred = model.predict(X_test)

        predictions[name] = y_pred

        if problem_type == "classification":

            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }

        elif problem_type == "regression":

            mse = mean_squared_error(y_test, y_pred)

            results[name] = {
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mse,
                "rmse": sqrt(mse),
                "r2_score": r2_score(y_test, y_pred)
            }

        else:
            raise ValueError("Invalid problem type")

    return results, predictions


def cross_validate_models(problem_type, models, X_train, y_train, preprocessor, n_splits=5):

    if problem_type == "classification":

        smallest_class = y_train.value_counts().min()

        if smallest_class < 2:
            raise ValueError("Each class needs at least 2 samples for cross-validation")

        folds = min(n_splits, smallest_class)

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

        scoring = {
            "accuracy": "accuracy",
            "f1_score": "f1_weighted"
        }

    elif problem_type == "regression":

        folds = min(n_splits, len(X_train))

        if folds < 2:
            raise ValueError("At least 2 samples are required for cross-validation")

        cv = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "r2_score": "r2"
        }

    else:
        raise ValueError("Invalid problem type")

    results = {}

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", clone(model))
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring
        )

        if problem_type == "classification":

            results[name] = {
                "folds": folds,
                "accuracy_mean": float(scores["test_accuracy"].mean()),
                "accuracy_std": float(scores["test_accuracy"].std()),
                "f1_score_mean": scores["test_f1_score"].mean(),
                "f1_score_std": scores["test_f1_score"].std()
            }

        else:

            results[name] = {
                "folds": folds,
                "rmse_mean": -scores["test_rmse"].mean(),
                "rmse_std": scores["test_rmse"].std(),
                "r2_score_mean": scores["test_r2_score"].mean(),
                "r2_score_std": scores["test_r2_score"].std()
            }

    return results


def save_model(model, preprocessor, model_name, problem_type,metadata):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": model,
        "preprocessor": preprocessor,
        "model_name": model_name,
        "problem_type": problem_type,
        "metadata" : metadata
    }

    file_path = MODEL_DIR / f"{model_name}.joblib"

    joblib.dump(model_data, file_path)

    return file_path


def load_model(file_path):

    model_data = joblib.load(file_path)

    return model_data


def predict_with_saved_model(file_path, input_data):

    model_data = load_model(file_path)

    model = model_data["model"]
    preprocessor = model_data["preprocessor"]

    processed_data = preprocessor.transform(input_data)

    predictions = model.predict(processed_data)

    return predictions
