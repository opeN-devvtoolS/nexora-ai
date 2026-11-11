
from .files import list_files, read_file
from .csv_tools import (
    preview_csv,
    describe_csv,
    filter_csv,
    sort_csv,
    aggregate_csv,
    group_csv,
    correlation_csv
)
from .ml import train_model, predict, list_models


TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "preview_csv": preview_csv,
    "describe_csv": describe_csv,
    "filter_csv": filter_csv,
    "sort_csv": sort_csv,
    "aggregate_csv": aggregate_csv,
    "group_csv": group_csv,
    "correlation_csv": correlation_csv,
    "train_model": train_model,
    "predict": predict,
    "list_models": list_models
}


def tool(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or []
            }
        }
    }


TOOL_LIST = [
    tool(
        "list_files",
        "List files in the input, knowledge, and output folders."
    ),

    tool(
        "read_file",
        "Read a text, JSON, or Markdown file from the knowledge or input folder.",
        {
            "filename": {"type": "string"}
        },
        ["filename"]
    ),

    tool(
        "preview_csv",
        "Show the first five rows of a CSV dataset.",
        {
            "filename": {"type": "string"}
        },
        ["filename"]
    ),

    tool(
        "describe_csv",
        "Describe a CSV dataset, including columns, data types, missing values, and numerical statistics.",
        {
            "filename": {"type": "string"}
        },
        ["filename"]
    ),

    tool(
        "filter_csv",
        "Filter CSV rows using one or more conditions.",
        {
            "filename": {"type": "string"},
            "conditions": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 3
                }
            },
            "logic": {
                "type": "string",
                "enum": ["and", "or"]
            }
        },
        ["filename", "conditions"]
    ),

    tool(
        "sort_csv",
        "Sort a CSV dataset by a selected column.",
        {
            "filename": {"type": "string"},
            "column": {"type": "string"},
            "ascending": {"type": "boolean"}
        },
        ["filename", "column"]
    ),

    tool(
        "aggregate_csv",
        "Calculate sum, mean, median, minimum, maximum, or count for a CSV column.",
        {
            "filename": {"type": "string"},
            "column": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": ["sum", "mean", "median", "min", "max", "count"]
            }
        },
        ["filename", "column", "operation"]
    ),

    tool(
        "group_csv",
        "Group CSV data by one column and aggregate another column.",
        {
            "filename": {"type": "string"},
            "group_column": {"type": "string"},
            "value_column": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": ["sum", "mean", "median", "min", "max", "count"]
            }
        },
        ["filename", "group_column", "value_column", "operation"]
    ),

    tool(
        "correlation_csv",
        "Calculate correlations between numerical columns in a CSV dataset.",
        {
            "filename": {"type": "string"}
        },
        ["filename"]
    ),

    tool(
        "train_model",
        "Train and evaluate machine learning models using a CSV dataset, then save the best model.",
        {
            "csv_path": {"type": "string"},
            "target_column": {"type": "string"},
            "problem_type": {
                "type": "string",
                "enum": ["classification", "regression"]
            },
            "feature_columns": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        ["csv_path", "target_column", "problem_type"]
    ),

    tool(
        "predict",
        "Make predictions using a previously saved machine learning model.",
        {
            "input_data": {"type": "object"},
            "model_name": {"type": "string"}
        },
        ["input_data", "model_name"]
    ),

    tool(
        "list_models",
        "List all saved machine learning models."
    )
]
