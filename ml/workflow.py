"""Preprocess, compare, evaluate, save, and use ML models."""

from pathlib import Path

import joblib
from ml.analysis import analyze_dataframe
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_DIR = Path("workspace/output/models")


def create_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def get_models(problem_type):
    if problem_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        }
    if problem_type == "regression":
        return {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        }
    raise ValueError("Invalid problem type")


def train_and_evaluate(X_train, X_test, y_train, y_test, problem_type):
    if problem_type == "classification":
        folds = min(5, y_train.value_counts().min())
        if folds < 2:
            raise ValueError("Each class needs at least two training rows")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scoring = {"accuracy": "accuracy", "f1": "f1_weighted"}
    elif problem_type == "regression":
        folds = min(5, len(X_train))
        if folds < 2:
            raise ValueError("Not enough training rows")
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        scoring = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}
    else:
        raise ValueError("Invalid problem type")

    test_metrics = {}
    cv_metrics = {}
    trained_models = {}

    for name, estimator in get_models(problem_type).items():
        pipeline = Pipeline([
            ("preprocessor", create_preprocessor(X_train)),
            ("model", estimator),
        ])
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring
        )
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        trained_models[name] = pipeline

        if problem_type == "classification":
            cv_metrics[name] = {
                "folds": int(folds),
                "accuracy_mean": float(scores["test_accuracy"].mean()),
                "accuracy_std": float(scores["test_accuracy"].std()),
                "f1_score_mean": float(scores["test_f1"].mean()),
                "f1_score_std": float(scores["test_f1"].std()),
            }
            test_metrics[name] = {
                "accuracy": accuracy_score(y_test, predicted),
                "precision": precision_score(
                    y_test, predicted, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test, predicted, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, predicted, average="weighted", zero_division=0
                ),
                "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
            }
        else:
            mse = mean_squared_error(y_test, predicted)
            cv_metrics[name] = {
                "folds": int(folds),
                "rmse_mean": float(-scores["test_rmse"].mean()),
                "rmse_std": float(scores["test_rmse"].std()),
                "r2_score_mean": float(scores["test_r2"].mean()),
                "r2_score_std": float(scores["test_r2"].std()),
            }
            test_metrics[name] = {
                "mae": mean_absolute_error(y_test, predicted),
                "mse": mse,
                "rmse": mse ** 0.5,
                "r2_score": r2_score(y_test, predicted),
            }

    return test_metrics, cv_metrics, trained_models


def save_model(pipeline, model_name, problem_type, metadata):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump({
        "model": pipeline.named_steps["model"],
        "preprocessor": pipeline.named_steps["preprocessor"],
        "model_name": model_name,
        "problem_type": problem_type,
        "metadata": metadata,
    }, path)
    return path


def load_model(path):
    return joblib.load(path)


def predict_with_saved_model(path, input_data):
    saved = load_model(path)
    prepared = saved["preprocessor"].transform(input_data)
    return saved["model"].predict(prepared)


def prepare_data(df, target_column, problem_type, feature_columns=None):
    """Keep the existing preprocessing example working."""
    X_train, X_test, y_train, y_test, _ = analyze_dataframe(
        df, target_column, problem_type, feature_columns
    )
    preprocessor = create_preprocessor(X_train)
    return (
        preprocessor.fit_transform(X_train),
        preprocessor.transform(X_test),
        y_train,
        y_test,
        preprocessor,
    )
