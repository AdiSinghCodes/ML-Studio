from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from app.core.config import DATASET_DIR


def get_dataset_path(stored_filename: str) -> Path:
    """
    Safely locate a dataset stored in the datasets directory.
    """

    filename = Path(stored_filename).name
    file_path = DATASET_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    return file_path


def load_dataset_by_name(stored_filename: str) -> pd.DataFrame:
    """
    Load a previously uploaded CSV or XLSX dataset.
    """

    file_path = get_dataset_path(stored_filename)
    extension = file_path.suffix.lower()

    try:
        if extension == ".csv":
            return pd.read_csv(file_path)

        if extension == ".xlsx":
            return pd.read_excel(file_path)

        raise HTTPException(
            status_code=400,
            detail="Unsupported dataset format."
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read dataset: {str(error)}"
        )


def get_dataset_columns(stored_filename: str) -> dict:
    """
    Return all available columns in the uploaded dataset.
    """

    df = load_dataset_by_name(stored_filename)

    return {
        "stored_filename": stored_filename,
        "columns": [str(column) for column in df.columns],
        "total_columns": int(len(df.columns))
    }


def detect_problem_type(
    stored_filename: str,
    target_column: str
) -> dict:
    """
    Detect whether the selected target column is more suitable for
    classification or regression.

    Heuristic:
    - Non-numeric targets -> classification
    - Numeric targets with a small number of unique values -> classification
    - Numeric targets with many unique values -> regression
    """

    df = load_dataset_by_name(stored_filename)

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' does not exist in the dataset."
        )

    target = df[target_column].dropna()

    if target.empty:
        raise HTTPException(
            status_code=400,
            detail="The selected target column contains only missing values."
        )

    unique_values = int(target.nunique())
    total_samples = int(len(target))
    data_type = str(target.dtype)

    # Categorical or text targets are classification problems.
    if not pd.api.types.is_numeric_dtype(target):
        problem_type = "classification"
        reason = "The target column is categorical or text-based."

    else:
        # Numeric columns with relatively few unique values are usually
        # class labels (for example 0/1 or ratings 1-5).
        classification_threshold = max(20, int(total_samples * 0.05))

        if unique_values <= classification_threshold:
            problem_type = "classification"
            reason = (
                "The numeric target has a limited number of unique values, "
                "so it is treated as a class label."
            )
        else:
            problem_type = "regression"
            reason = (
                "The numeric target has many unique values, "
                "so it is treated as a continuous value."
            )

    response = {
        "stored_filename": stored_filename,
        "target_column": target_column,
        "problem_type": problem_type,
        "target_data_type": data_type,
        "samples": total_samples,
        "unique_target_values": unique_values,
        "reason": reason
    }

    if problem_type == "classification":
        response["classes"] = [
            str(value) for value in sorted(target.unique(), key=lambda x: str(x))
        ]

    return response
