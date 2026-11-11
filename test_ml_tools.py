
from tools.ml import train_model, predict, list_models

result = train_model(
    "workspace/input/students.csv",
    "passed",
    "classification",
    ["class", "score", "attendance", "study_hours"]
)

print("Training result:")
print(result)

print("\nSaved models:")
print(list_models())

prediction = predict(
    {
        "class": "A",
        "score": 75,
        "attendance": 90,
        "study_hours": 5.0
    },
    result["best_model"]
)

print("\nPrediction:")
print(prediction)
