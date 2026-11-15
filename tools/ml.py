"""Agent-facing tools for training, selecting, and using models."""

from pathlib import Path

import pandas as pd

from ml.analysis import analyze_csv
from ml.workflow import (
    MODEL_DIR, load_model, predict_with_saved_model,
    save_model, train_and_evaluate,
)


def train_model(csv_path, target_column, problem_type, feature_columns=None):
    (
        X_train, X_test, y_train, y_test, features_used, dataset_name
    ) = analyze_csv(csv_path, target_column, problem_type, feature_columns)

    metrics, cv_results, trained_models = train_and_evaluate(
        X_train, X_test, y_train, y_test, problem_type
    )

    if problem_type == "classification":
        best_name = max(
            cv_results, key=lambda name: cv_results[name]["f1_score_mean"]
        )
    else:
        best_name = min(
            cv_results, key=lambda name: cv_results[name]["rmse_mean"]
        )

    dataset_stem = Path(dataset_name).stem.lower().replace(" ", "_")
    saved_name = f"{dataset_stem}_{target_column}_{problem_type}_{best_name}"

    metadata = {
        "dataset_name": dataset_name,
        "algorithm": best_name,
        "target_column": target_column,
        "feature_columns": features_used,
        "problem_type": problem_type,
        "metrics": metrics,
        "cross_validation": cv_results,
    }
    path = save_model(
        trained_models[best_name], saved_name, problem_type, metadata
    )

    return {
        "status": "success",
        "best_model": saved_name,
        "algorithm": best_name,
        "metrics": metrics,
        "cross_validation": cv_results,
        "model_path": str(path),
    }


def predict(input_data, model_name):
    if Path(model_name).name != model_name:
        raise ValueError("Invalid model name")
    path = MODEL_DIR / f"{model_name}.joblib"
    if not path.is_file():
        raise FileNotFoundError("Saved model not found")
    row = pd.DataFrame([input_data])
    return predict_with_saved_model(path, row).tolist()


def list_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    for path in MODEL_DIR.glob("*.joblib"):
        saved = load_model(path)
        info = saved.get("metadata", {})
        models.append({
            "model_name": saved["model_name"],
            "algorithm": info.get("algorithm", saved["model_name"]),
            "problem_type": saved["problem_type"],
            "dataset_name": info.get("dataset_name", "Unknown"),
            "target_column": info.get("target_column", "Unknown"),
            "feature_columns": info.get("feature_columns", []),
            "metrics": info.get("metrics", {}),
            "cross_validation": info.get("cross_validation", {}),
        })
    return models


def select_model(dataset_name, problem_type, target_column=None):
    matches = [
        model for model in list_models()
        if model["dataset_name"].lower() == dataset_name.lower()
        and model["problem_type"] == problem_type
        and (target_column is None or model["target_column"] == target_column)
    ]
    if not matches:
        return {"status": "error", "message": "No matching model found."}

    if problem_type == "classification":
        metric, fallback = "f1_score_mean", "f1_score"
        default = -1
    elif problem_type == "regression":
        metric, fallback = "rmse_mean", "rmse"
        default = float("inf")
    else:
        return {"status": "error", "message": "Invalid problem type."}

    def score(model):
        algorithm = model["algorithm"]
        cv_score = model["cross_validation"].get(algorithm, {}).get(metric)
        if cv_score is not None:
            return cv_score
        return model["metrics"].get(algorithm, {}).get(fallback, default)

    best = (
        max(matches, key=score) if problem_type == "classification"
        else min(matches, key=score)
    )
    score_name = (
        metric if metric in best["cross_validation"].get(best["algorithm"], {})
        else fallback
    )
    return {
        "status": "success",
        "model_name": best["model_name"],
        "algorithm": best["algorithm"],
        "dataset_name": best["dataset_name"],
        "target_column": best["target_column"],
        "features": best["feature_columns"],
        "selection_metric": score_name,
        "selection_score": score(best),
    }
