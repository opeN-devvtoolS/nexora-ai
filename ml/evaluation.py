
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

from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    KFold
)

from sklearn.pipeline import Pipeline
from sklearn.base import clone
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
