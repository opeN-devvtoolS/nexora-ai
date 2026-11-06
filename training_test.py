import pandas as pd

from ml.preprocessing import prepare_data
from ml.training import train_models


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

print("Training completed!")

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTrained models:")

for name in trained_models:
    print(name)