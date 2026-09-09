import uuid
from pathlib import Path

import pandas as pd
from fastapi import UploadFile, HTTPException

from app.core.config import DATASET_DIR


def save_uploaded_dataset(file: UploadFile) -> Path:
    """Save an uploaded dataset with a unique filename."""

    allowed_extensions = [".csv", ".xlsx"]
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only CSV and XLSX files are supported."
        )

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = DATASET_DIR / unique_filename

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return file_path


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load CSV or XLSX into a Pandas DataFrame."""

    extension = file_path.suffix.lower()

    try:
        if extension == ".csv":
            df = pd.read_csv(file_path)
        elif extension == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format.")

        return df

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read dataset: {str(error)}"
        )


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Generate basic dataset statistics."""

    total_rows = int(df.shape[0])
    total_columns = int(df.shape[1])

    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    total_cells = total_rows * total_columns
    missing_percentage = 0

    if total_cells > 0:
        missing_percentage = round(
            (missing_values / total_cells) * 100,
            2
        )

    duplicate_percentage = 0

    if total_rows > 0:
        duplicate_percentage = round(
            (duplicate_rows / total_rows) * 100,
            2
        )

    column_information = []

    for column in df.columns:
        column_information.append({
            "name": str(column),
            "data_type": str(df[column].dtype),
            "missing_values": int(df[column].isnull().sum()),
            "unique_values": int(df[column].nunique())
        })

    health_score = calculate_health_score(
        missing_percentage,
        duplicate_percentage
    )

    return {
        "rows": total_rows,
        "columns": total_columns,
        "missing_values": missing_values,
        "missing_percentage": missing_percentage,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": duplicate_percentage,
        "health_score": health_score,
        "column_information": column_information
    }


def calculate_health_score(
    missing_percentage: float,
    duplicate_percentage: float
) -> int:
    """Calculate a simple dataset quality score from 0 to 100."""

    score = 100

    score -= min(missing_percentage * 2, 60)
    score -= min(duplicate_percentage * 2, 30)

    return max(0, round(score))
