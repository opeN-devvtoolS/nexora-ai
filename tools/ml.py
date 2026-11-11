
import pandas as pd
from pathlib import Path

from ml.preprocessing import prepare_data
from ml.training import train_models
from ml.evaluation import evaluate_models
from ml.model_persistence import save_model
from ml.prediction import predict_with_saved_model


MODEL_DIR = Path("workspace/output/models")


def train_model(csv_path, target_column, problem_type,feature_columns=None):

    df = pd.read_csv(csv_path)

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

    model_path = save_model(
        best_model,
        preprocessor,
        best_model_name,
        problem_type
    )

    return {
        "status": "success",
        "best_model": best_model_name,
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

    models = [
        file.stem
        for file in MODEL_DIR.glob("*.joblib")
    ]

    return models
