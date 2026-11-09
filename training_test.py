
import pandas as pd
from ml.model_persistence import save_model, load_model
from ml.preprocessing import prepare_data
from ml.training import train_models
from ml.evaluation import evaluate_models


df = pd.read_csv("workspace/input/students.csv")

df = df.drop(columns=["student_id", "name"])

target_column = "passed"
problem_type = "classification"

X_train, X_test, y_train, y_test, preprocessor = prepare_data(
    df,
    target_column,
    problem_type
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

print("\nModel Evaluation Results")

for name, metrics in results.items():

    print(f"\nModel: {name}")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

print("\nActual values:")
print(y_test.to_list())

for name, prediction in predictions.items():
    print(f"\n{name} predictions:")
    print(prediction)


print("\nSaving trained models...")

for name, model in trained_models.items():

    file_path = save_model(model,preprocessor,name,problem_type)

    print(f"Saved: {file_path}")


print("\nLoading Random Forest model...")

loaded_data = load_model(
    "workspace/output/models/random_forest.joblib"
)

loaded_model = loaded_data["model"]

print("Loaded model:", loaded_data["model_name"])
print("Problem type:", loaded_data["problem_type"])