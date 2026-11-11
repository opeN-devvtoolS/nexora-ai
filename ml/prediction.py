
from ml.model_persistence import load_model


def predict_with_saved_model(file_path, input_data):

    model_data = load_model(file_path)

    model = model_data["model"]
    preprocessor = model_data["preprocessor"]

    processed_data = preprocessor.transform(input_data)

    predictions = model.predict(processed_data)

    return predictions
