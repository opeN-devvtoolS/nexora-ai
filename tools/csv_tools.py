
import pandas as pd
from tools.paths import get_path


def load_csv(filename):

    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported")

    path = get_path("input", filename)

    if not path.is_file():
        raise FileNotFoundError("CSV file not found")

    df = pd.read_csv(path, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("CSV file is empty")

    return df


def preview_csv(filename):

    df = load_csv(filename)

    return df.head().to_string(index=False)


def describe_csv(filename):

    df = load_csv(filename)

    result = "DATASET INFORMATION\n"

    result += "Rows: " + str(df.shape[0]) + "\n"
    result += "Columns: " + str(df.shape[1]) + "\n\n"

    result += "COLUMN NAMES:\n"
    result += str(list(df.columns)) + "\n\n"

    result += "DATA TYPES:\n"
    result += df.dtypes.to_string() + "\n\n"

    result += "MISSING VALUES:\n"
    result += df.isnull().sum().to_string() + "\n\n"

    result += "NUMERICAL STATISTICS:\n"

    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        result += numeric_df.describe().round(2).to_string()
    else:
        result += "No numerical columns found."

    result += "\n\n"

    result += "UNIQUE VALUES:\n"

    for column in df.columns:

        result += column + ": "
        result += str(df[column].nunique()) + "\n"

    return result
