from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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