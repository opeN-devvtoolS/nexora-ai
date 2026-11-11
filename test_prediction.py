
import pandas as pd

from ml.prediction import predict_with_saved_model


new_student = pd.DataFrame([
    {
        "class": "A",
        "score": 75,
        "attendance": 90,
        "study_hours": 5.0
    }
])


predictions = predict_with_saved_model(
    "workspace/output/models/random_forest.joblib",
    new_student
)


print("New student data:")
print(new_student)

print("\nPredicted result:")
print(predictions[0])   
