
from tools.ml import train_model, list_models, predict


# STEP 1: Train a model
result = train_model(
    "workspace/input/students.csv",
    "passed",
    "classification",
    ["class", "score", "attendance", "study_hours"]
)

print("\nTRAINING RESULT:")
print(result)


# STEP 2: Retrieve saved model information
models = list_models()

print("\nSAVED MODEL REGISTRY:")

for model in models:
    print(model)


# STEP 3: Predict using the newly saved model
prediction = predict(
    {
        "class": "A",
        "score": 75,
        "attendance": 90,
        "study_hours": 5.0
    },
    result["best_model"]
)

print("\nPREDICTION:")
print(prediction)
