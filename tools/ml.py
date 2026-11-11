
import pandas as pd
from pathlib import Path

from ml.preprocessing import prepare_data
from ml.training import train_models
from ml.evaluation import evaluate_models
from ml.model_persistence import save_model, load_model
from ml.prediction import predict_with_saved_model


MODEL_DIR = Path("workspace/output/models")


def train_model(csv_path, target_column, problem_type, feature_columns=None):

    df = pd.read_csv(csv_path)

    features_used = (
        feature_columns
        if feature_columns is not None
        else [column for column in df.columns if column != target_column]
    )

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        df,
        target_column,
        problem_type,
        feature_columns
    )

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
            results,
            key=lambda name: results[name]["f1_score"]
        )
    else:
        best_model_name = min(
            results,
            key=lambda name: results[name]["rmse"]
        )

    best_model = trained_models[best_model_name]

    dataset_name = Path(csv_path).name
    dataset_stem = Path(csv_path).stem.lower().replace(" ", "_")

    saved_model_name = f"{dataset_stem}_{best_model_name}"

    metadata = {
        "dataset_name": dataset_name,
        "algorithm": best_model_name,
        "target_column": target_column,
        "feature_columns": features_used,
        "problem_type": problem_type,
        "metrics": results
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
            "metrics": metadata.get("metrics", {})
        })

    return models
