
import joblib
from pathlib import Path


MODEL_DIR = Path("workspace/output/models")


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
