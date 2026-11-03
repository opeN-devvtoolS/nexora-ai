
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



def filter_csv(filename, conditions, logic="and"):

    df = load_csv(filename)

    if logic.lower() not in ["and", "or"]:
        raise ValueError("Logic must be and or or")

    if not conditions:
        raise ValueError("At least one condition is required")

    masks = []

    for column, operator, value in conditions:

        if column not in df.columns:
            raise ValueError("Column not found: " + column)

        if operator == "contains":

            mask = df[column].astype("string").str.contains(
                str(value),
                case=False,
                na=False
            )

        elif operator in ["==", "!=", ">", ">=", "<", "<="]:

            series = df[column]

            if operator in [">", ">=", "<", "<="]:

                if not pd.api.types.is_numeric_dtype(series):
                    raise ValueError("Comparison requires a numerical column")

                value = float(value)

            elif pd.api.types.is_numeric_dtype(series):
                value = float(value)

            if operator == "==":
                mask = series == value
            elif operator == "!=":
                mask = series != value
            elif operator == ">":
                mask = series > value
            elif operator == ">=":
                mask = series >= value
            elif operator == "<":
                mask = series < value
            else:
                mask = series <= value

        else:
            raise ValueError("Unsupported operator")

        masks.append(mask)

    result = masks[0]

    for mask in masks[1:]:

        if logic.lower() == "and":
            result = result & mask
        else:
            result = result | mask

    filtered_df = df[result]

    return filtered_df.to_string(index=False)


def sort_csv(filename, column, ascending=True):

    df = load_csv(filename)

    if column not in df.columns:
        raise ValueError("Column not found")

    result = df.sort_values(
        by=column,
        ascending=ascending
    )

    return result.to_string(index=False)


def aggregate_csv(filename, column, operation):

    df = load_csv(filename)

    if column not in df.columns:
        raise ValueError("Column not found")

    if operation not in ["sum", "mean", "median", "min", "max", "count"]:
        raise ValueError("Unsupported operation")

    if operation != "count":

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError("Operation requires a numerical column")

    if operation == "sum":
        result = df[column].sum()

    elif operation == "mean":
        result = df[column].mean()

    elif operation == "median":
        result = df[column].median()

    elif operation == "min":
        result = df[column].min()

    elif operation == "max":
        result = df[column].max()

    else:
        result = df[column].count()

    return str(result)



def group_csv(filename, group_column, value_column, operation):

    df = load_csv(filename)

    if group_column not in df.columns:
        raise ValueError("Group column not found")

    if value_column not in df.columns:
        raise ValueError("Value column not found")

    if operation not in ["sum", "mean", "median", "min", "max", "count"]:
        raise ValueError("Unsupported operation")

    if operation != "count":

        if not pd.api.types.is_numeric_dtype(df[value_column]):
            raise ValueError("Operation requires a numerical column")

    grouped = df.groupby(group_column, dropna=False)[value_column]

    result = grouped.agg(operation)

    return result.to_string()


def correlation_csv(filename):

    df = load_csv(filename)

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        raise ValueError("At least two numerical columns are required")

    result = numeric_df.corr()

    return result.round(3).to_string()


