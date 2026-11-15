
from tools.ml import train_model, list_models, select_model

result = train_model(
    "students.csv",
    "score",
    "regression",
    ["class", "attendance", "study_hours"]
)

print("\nREGRESSION TRAINING RESULT:")
print(result)

print("\nCROSS-VALIDATION RESULTS:")
print(result["cross_validation"])

print("\nMODEL SELECTION:")
print(
    select_model(
        dataset_name="students.csv",
        problem_type="regression",
        target_column="score"
    )
)
