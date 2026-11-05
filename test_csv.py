import pandas as pd

from ml.preprocessing import prepare_data

data = {
    "Age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "City": [
        "Karachi", "Lahore", "Karachi", "Islamabad",
        "Lahore", "Karachi", "Islamabad", "Lahore",
        "Karachi", "Islamabad"
    ],
    "Marks": [65, 75, 80, 45, 90, 55, 88, 70, 95, 60],
    "Result": [
        "Pass", "Pass", "Pass", "Fail", "Pass",
        "Fail", "Pass", "Pass", "Pass", "Fail"
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test, preprocessor = prepare_data(
    df,
    "Result",
    "classification"
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("Training targets:")
print(y_train)

print("Testing targets:")
print(y_test)