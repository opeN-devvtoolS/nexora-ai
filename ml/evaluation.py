from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from math import sqrt


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
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()}

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
