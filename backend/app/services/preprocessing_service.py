import re
import uuid
from datetime import datetime

import joblib
import pandas as pd
from fastapi import HTTPException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import PREPROCESSOR_DIR
from app.services.target_service import load_dataset_by_name


HIGH_CARDINALITY_RATIO = 0.50
HIGH_CARDINALITY_MIN_UNIQUE = 30


def _build_one_hot_encoder():
    """Build a version-compatible OneHotEncoder."""

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def _detect_id_columns(df: pd.DataFrame) -> list:
    """
    Detect identifier-like columns.

    Explicit ID columns and near-unique columns with ID-like names
    are removed because they can encourage overfitting.
    """

    id_columns = []

    for column in df.columns:
        column_name = str(column).lower()
        unique_ratio = (
            df[column].nunique(dropna=True) / max(len(df), 1)
        )

        if (
            column_name == "id"
            or column_name.endswith("id")
            or "identifier" in column_name
        ):
            id_columns.append(column)
            continue

        if (
            unique_ratio >= 0.98
            and (
                "number" in column_name
                or "code" in column_name
                or "index" in column_name
            )
        ):
            id_columns.append(column)

    return id_columns


def _is_high_cardinality(series: pd.Series) -> bool:
    """Detect categorical columns with many distinct categories."""

    unique_values = series.nunique(dropna=True)
    unique_ratio = unique_values / max(len(series), 1)

    return (
        unique_values >= HIGH_CARDINALITY_MIN_UNIQUE
        and unique_ratio >= HIGH_CARDINALITY_RATIO
    )


def _extract_name_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Extract title and name length from a person-name column.
    """

    extracted = df.copy()

    titles = (
        extracted[column]
        .astype("string")
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
    )

    common_titles = {
        "Mr", "Mrs", "Miss", "Master",
        "Dr", "Rev", "Major", "Col",
        "Mlle", "Mme", "Ms"
    }

    extracted[f"{column}_Title"] = titles.where(
        titles.isin(common_titles),
        "Rare"
    ).fillna("Unknown")

    extracted[f"{column}_Length"] = (
        extracted[column]
        .astype("string")
        .str.len()
        .fillna(0)
    )

    extracted = extracted.drop(columns=[column])

    return extracted


def _extract_cabin_features(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Extract the deck letter and number of cabins from a cabin column.
    """

    extracted = df.copy()

    cabin_text = (
        extracted[column]
        .astype("string")
        .fillna("Unknown")
    )

    extracted[f"{column}_Deck"] = (
        cabin_text
        .str.extract(r"([A-Za-z])", expand=False)
        .str.upper()
        .fillna("Unknown")
    )

    extracted[f"{column}_Count"] = (
        cabin_text
        .str.split()
        .str.len()
        .fillna(0)
    )

    extracted = extracted.drop(columns=[column])

    return extracted


def _extract_ticket_features(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Extract ticket prefix and ticket length.

    The raw ticket value is removed because it is high-cardinality.
    """

    extracted = df.copy()

    ticket_text = (
        extracted[column]
        .astype("string")
        .fillna("Unknown")
        .str.strip()
    )

    prefix = (
        ticket_text
        .str.replace(r"\d", "", regex=True)
        .str.replace(r"[^A-Za-z]", "", regex=True)
        .str.upper()
        .str.strip()
    )

    extracted[f"{column}_Prefix"] = prefix.replace(
        "",
        "NUMERIC"
    )

    extracted[f"{column}_Length"] = ticket_text.str.len()

    extracted = extracted.drop(columns=[column])

    return extracted


def _smart_feature_engineering(
    X: pd.DataFrame
) -> tuple[pd.DataFrame, list, list]:
    """
    Apply dataset-aware feature engineering.

    Special handling:
    - Name -> title and name length
    - Cabin -> deck and cabin count
    - Ticket -> prefix and ticket length

    Remaining high-cardinality categorical columns are dropped.
    """

    engineered = X.copy()

    engineered_columns = []
    dropped_high_cardinality = []

    for column in list(engineered.columns):
        column_name = str(column).lower()

        if column not in engineered.columns:
            continue

        if column_name == "name":
            engineered = _extract_name_features(
                engineered,
                column
            )
            engineered_columns.extend([
                f"{column}_Title",
                f"{column}_Length"
            ])

        elif column_name == "cabin":
            engineered = _extract_cabin_features(
                engineered,
                column
            )
            engineered_columns.extend([
                f"{column}_Deck",
                f"{column}_Count"
            ])

        elif column_name == "ticket":
            engineered = _extract_ticket_features(
                engineered,
                column
            )
            engineered_columns.extend([
                f"{column}_Prefix",
                f"{column}_Length"
            ])

    # Drop any remaining high-cardinality categorical columns.
    for column in list(engineered.columns):
        if pd.api.types.is_numeric_dtype(
            engineered[column]
        ):
            continue

        if _is_high_cardinality(
            engineered[column]
        ):
            dropped_high_cardinality.append(
                str(column)
            )
            engineered = engineered.drop(
                columns=[column]
            )

    return (
        engineered,
        engineered_columns,
        dropped_high_cardinality
    )


def preprocess_dataset(
    stored_filename: str,
    target_column: str,
    scale_numeric: bool = True
) -> dict:
    """
    Intelligently preprocess a tabular dataset for ML.
    """

    df = load_dataset_by_name(stored_filename)

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' "
                "does not exist in the dataset."
            )
        )

    original_rows = int(df.shape[0])
    original_columns = int(df.shape[1])

    empty_rows_before = int(
        df.isna().all(axis=1).sum()
    )
    df = df.dropna(axis=0, how="all").copy()

    fully_empty_columns = [
        column
        for column in df.columns
        if (
            column != target_column
            and df[column].isna().all()
        )
    ]

    if fully_empty_columns:
        df = df.drop(
            columns=fully_empty_columns
        )

    missing_target_rows = int(
        df[target_column].isna().sum()
    )
    df = df.dropna(
        subset=[target_column]
    ).copy()

    y = df[target_column].copy()
    X = df.drop(
        columns=[target_column]
    ).copy()

    if X.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                "No feature columns remain after "
                "removing the target column."
            )
        )

    id_columns = _detect_id_columns(X)

    if id_columns:
        X = X.drop(columns=id_columns)

    (
        X,
        engineered_columns,
        dropped_high_cardinality
    ) = _smart_feature_engineering(X)

    if X.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                "No usable features remain after "
                "feature engineering."
            )
        )

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_steps = [
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]

        if scale_numeric:
            numeric_steps.append(
                ("scaler", StandardScaler())
            )

        numeric_pipeline = Pipeline(
            steps=numeric_steps
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    _build_one_hot_encoder()
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    X_processed = preprocessor.fit_transform(X)

    try:
        transformed_feature_names = [
            str(name)
            for name in (
                preprocessor.get_feature_names_out()
            )
        ]
    except Exception:
        transformed_feature_names = [
            f"feature_{index}"
            for index in range(
                X_processed.shape[1]
            )
        ]

    artifact_id = str(uuid.uuid4())

    artifact = {
        "preprocessor": preprocessor,
        "feature_columns": X.columns.tolist(),
        "target_column": target_column,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "scale_numeric": scale_numeric,
        "engineered_columns": engineered_columns,
        "dropped_high_cardinality": (
            dropped_high_cardinality
        ),
        "created_at": (
            datetime.utcnow().isoformat()
        )
    }

    artifact_filename = (
        f"{artifact_id}_preprocessor.joblib"
    )

    artifact_path = (
        PREPROCESSOR_DIR / artifact_filename
    )

    joblib.dump(artifact, artifact_path)

    return {
        "message": (
            "Intelligent dataset preprocessing "
            "completed successfully."
        ),
        "stored_filename": stored_filename,
        "target_column": target_column,
        "original_shape": {
            "rows": original_rows,
            "columns": original_columns
        },
        "training_shape": {
            "rows": int(X_processed.shape[0]),
            "features_after_preprocessing": (
                int(X_processed.shape[1])
            )
        },
        "target_samples": int(len(y)),
        "removed_empty_rows": empty_rows_before,
        "removed_rows_with_missing_target": (
            missing_target_rows
        ),
        "removed_fully_empty_columns": [
            str(column)
            for column in fully_empty_columns
        ],
        "removed_identifier_columns": [
            str(column)
            for column in id_columns
        ],
        "engineered_features": (
            engineered_columns
        ),
        "dropped_high_cardinality_columns": (
            dropped_high_cardinality
        ),
        "numeric_columns": [
            str(column)
            for column in numeric_columns
        ],
        "categorical_columns": [
            str(column)
            for column in categorical_columns
        ],
        "scaling_applied": scale_numeric,
        "transformed_feature_count": int(
            X_processed.shape[1]
        ),
        "preprocessor_artifact": (
            artifact_filename
        ),
        "sample_transformed_features": (
            transformed_feature_names[:30]
        )
    }
