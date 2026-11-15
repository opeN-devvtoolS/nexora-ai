
import pandas as pd
from pathlib import Path
from tools.csv_tools import load_csv

from ml.workflow import (
    prepare_raw_data,
    train_models,
    get_models,
    evaluate_models,
    cross_validate_models,
    save_model,
    load_model,
    predict_with_saved_model,
)


MODEL_DIR = Path("workspace/output/models")



def train_model(csv_path, target_column, problem_type, feature_columns=None):

    df = load_csv(csv_path)

    features_used = (
        feature_columns
        if feature_columns is not None
        else [column for column in df.columns if column != target_column]
    )
    X_train_raw, X_test_raw, y_train, y_test, preprocessor = prepare_raw_data(
        df,
        target_column,
        problem_type,
        feature_columns
    )

    cv_results = cross_validate_models(
        problem_type,
        get_models(problem_type),
        X_train_raw,
        y_train,
        preprocessor
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    trained_models = train_models(
        problem_type,
        X_train,
        y_train
    )

    results, predictions = evaluate_models(
        problem_type,
        trained_models,
        X_test,
        y_test
    )

    if problem_type == "classification":
        best_model_name = max(
            cv_results,
            key=lambda name: cv_results[name]["f1_score_mean"]
        )
    else:
        best_model_name = min(
            cv_results,
            key=lambda name: cv_results[name]["rmse_mean"]
        )

    best_model = trained_models[best_model_name]

    dataset_name = Path(csv_path).name
    dataset_stem = Path(csv_path).stem.lower().replace(" ", "_")

    saved_model_name = f"{dataset_stem}_{target_column}_{problem_type}_{best_model_name}"

    metadata = {
        "dataset_name": dataset_name,
        "algorithm": best_model_name,
        "target_column": target_column,
        "feature_columns": features_used,
        "problem_type": problem_type,
        "metrics": results,
        "cross_validation": cv_results
    }

    model_path = save_model(
        best_model,
        preprocessor,
        saved_model_name,
        problem_type,
        metadata
    )

    return {
        "status": "success",
        "best_model": saved_model_name,
        "algorithm": best_model_name,
        "metrics": results,
        "cross_validation": cv_results,
        "model_path": str(model_path)
    }


def predict(input_data, model_name):

    if Path(model_name).name != model_name:
        raise ValueError("Invalid model name")

    model_path = MODEL_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Saved model not found")

    input_df = pd.DataFrame([input_data])

    predictions = predict_with_saved_model(
        model_path,
        input_df
    )

    return predictions.tolist()


def list_models():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    models = []

    for file in MODEL_DIR.glob("*.joblib"):

        model_data = load_model(file)

        metadata = model_data.get("metadata", {})

        models.append({
            "model_name": model_data["model_name"],
            "algorithm": metadata.get("algorithm", model_data["model_name"]),
            "problem_type": model_data["problem_type"],
            "dataset_name": metadata.get("dataset_name", "Unknown"),
            "target_column": metadata.get("target_column", "Unknown"),
            "feature_columns": metadata.get("feature_columns", []),
            "metrics": metadata.get("metrics", {}),
            "cross_validation": metadata.get("cross_validation", {})
        })

    return models





def select_model(dataset_name, problem_type, target_column=None):

    available_models = list_models()

    matching_models = []

    for model in available_models:

        if model["dataset_name"].lower() != dataset_name.lower():
            continue

        if model["problem_type"] != problem_type:
            continue

        if target_column and model["target_column"] != target_column:
            continue

        matching_models.append(model)

    if not matching_models:
        return {
            "status": "error",
            "message": "No matching model found."
        }

    if problem_type == "classification":

        best_model = max(
            matching_models,
            key=lambda model: model["cross_validation"].get(
                model["algorithm"], {}
            ).get(
                "f1_score_mean",
                model["metrics"].get(model["algorithm"], {}).get("f1_score", -1)
            )
        )

        cv_metrics = best_model["cross_validation"].get(
            best_model["algorithm"], {}
        )

        if "f1_score_mean" in cv_metrics:
            score_name = "f1_score_mean"
            score = cv_metrics["f1_score_mean"]
        else:
            score_name = "f1_score"
            score = best_model["metrics"].get(
                best_model["algorithm"], {}
            ).get("f1_score")

    elif problem_type == "regression":

        best_model = min(
            matching_models,
            key=lambda model: model["cross_validation"].get(
                model["algorithm"], {}
            ).get(
                "rmse_mean",
                model["metrics"].get(model["algorithm"], {}).get("rmse", float("inf"))
            )
        )

        cv_metrics = best_model["cross_validation"].get(
            best_model["algorithm"], {}
        )

        if "rmse_mean" in cv_metrics:
            score_name = "rmse_mean"
            score = cv_metrics["rmse_mean"]
        else:
            score_name = "rmse"
            score = best_model["metrics"].get(
                best_model["algorithm"], {}
            ).get("rmse")

    else:
        return {
            "status": "error",
            "message": "Unsupported problem type."
        }

    return {
        "status": "success",
        "model_name": best_model["model_name"],
        "algorithm": best_model["algorithm"],
        "dataset_name": best_model["dataset_name"],
        "target_column": best_model["target_column"],
        "features": best_model["feature_columns"],
        "selection_metric": score_name,
        "selection_score": score
    }
