import re

import joblib
import pandas as pd

from fastapi import HTTPException

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, r2_score

from app.core.config import MODEL_DIR
from app.services.target_service import load_dataset_by_name


def _resolve_model_path(model_artifact: str):
    safe_name = str(model_artifact).replace("\\", "").replace("/", "")
    model_path = MODEL_DIR / safe_name

    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model artifact '{model_artifact}' was not found."
        )

    return model_path


def _load_model_artifact(model_artifact: str):
    try:
        artifact = joblib.load(_resolve_model_path(model_artifact))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load model artifact: {str(error)}"
        )

    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise HTTPException(
            status_code=500,
            detail="Invalid model artifact. A trained pipeline was not found."
        )

    return artifact


def _extract_name_title(value):
    if pd.isna(value):
        return "Rare"

    match = re.search(r",\s*([^.]*)\.", str(value))

    if not match:
        return "Rare"

    title = match.group(1).strip()

    # Keep the same broad grouping used by intelligent preprocessing.
    if title in ["Mlle"]:
        return "Mlle"
    if title in ["Mme"]:
        return "Mme"
    if title in [
        "Mr", "Mrs", "Miss", "Master",
        "Dr", "Rev", "Col", "Major", "Ms"
    ]:
        return title

    return "Rare"


def _extract_ticket_prefix(value):
    if pd.isna(value):
        return "UNKNOWN"

    ticket = str(value).strip()

    # Remove digits and normalize remaining prefix.
    prefix = re.sub(r"\d", "", ticket)
    prefix = re.sub(r"[^A-Za-z]", "", prefix).upper()

    if not prefix:
        return "NUMERIC"

    return prefix


def _engineer_features(df: pd.DataFrame):
    """
    Reproduce the intelligent feature-engineering stage used
    before model training.
    """

    data = df.copy()

    # Remove the Titanic identifier column.
    if "PassengerId" in data.columns:
        data = data.drop(columns=["PassengerId"])

    # Name features.
    if "Name" in data.columns:
        data["Name_Title"] = data["Name"].apply(
            _extract_name_title
        )
        data["Name_Length"] = data["Name"].astype(str).str.len()

    # Ticket features.
    if "Ticket" in data.columns:
        data["Ticket_Prefix"] = data["Ticket"].apply(
            _extract_ticket_prefix
        )
        data["Ticket_Length"] = data["Ticket"].astype(str).str.len()

    # Cabin features.
    if "Cabin" in data.columns:
        cabin_text = data["Cabin"].fillna("").astype(str)

        data["Cabin_Deck"] = cabin_text.apply(
            lambda value: value[0].upper()
            if value and value[0].isalpha()
            else "UNKNOWN"
        )

        data["Cabin_Count"] = cabin_text.apply(
            lambda value: len(value.split())
            if value.strip()
            else 0
        )

    # The training stage used engineered versions of these
    # high-cardinality source columns rather than their raw forms.
    data = data.drop(
        columns=["Name", "Ticket", "Cabin"],
        errors="ignore"
    )

    return data


def _align_features_with_pipeline(X, pipeline):
    """
    Align input columns with the columns expected by the
    ColumnTransformer inside the saved sklearn pipeline.
    """

    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
    except Exception:
        preprocessor = None

    if preprocessor is None:
        return X

    expected_columns = getattr(
        preprocessor,
        "feature_names_in_",
        None
    )

    if expected_columns is None:
        return X

    expected_columns = list(expected_columns)

    missing = [
        column
        for column in expected_columns
        if column not in X.columns
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "The dataset could not be aligned with the "
                "saved model. Missing engineered columns: "
                f"{missing}"
            )
        )

    # Remove unexpected columns and preserve exact training order.
    return X[expected_columns].copy()


def _prepare_data(
    stored_filename,
    target_column,
    model_artifact
):
    df = load_dataset_by_name(stored_filename)

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' does not exist."
        )

    df = df.dropna(axis=0, how="all").copy()
    df = df.dropna(subset=[target_column]).copy()

    artifact = _load_model_artifact(model_artifact)

    y = df[target_column].copy()

    X_original = df.drop(columns=[target_column]).copy()

    # Recreate the same intelligent feature engineering.
    X = _engineer_features(X_original)

    pipeline = artifact["pipeline"]

    # Match the exact schema expected by the saved model.
    X = _align_features_with_pipeline(X, pipeline)

    target_encoder = artifact.get("target_encoder")

    if target_encoder is not None:
        try:
            y = target_encoder.transform(y.astype(str))
        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The dataset target values are incompatible "
                    f"with the saved model: {str(error)}"
                )
            )

    return X, y, artifact


def _baseline_score(pipeline, X, y, problem_type):
    predictions = pipeline.predict(X)

    if problem_type == "classification":
        return float(
            f1_score(
                y,
                predictions,
                average="weighted",
                zero_division=0
            )
        )

    return float(r2_score(y, predictions))


def explain_feature_importance(
    stored_filename: str,
    target_column: str,
    model_artifact: str,
    top_n: int = 10,
    n_repeats: int = 10
) -> dict:

    top_n = max(1, min(top_n, 50))
    n_repeats = max(3, min(n_repeats, 50))

    X, y, artifact = _prepare_data(
        stored_filename,
        target_column,
        model_artifact
    )

    pipeline = artifact["pipeline"]

    problem_type = artifact.get(
        "problem_type",
        "classification"
    )

    scoring = (
        "f1_weighted"
        if problem_type == "classification"
        else "r2"
    )

    try:
        baseline = _baseline_score(
            pipeline,
            X,
            y,
            problem_type
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to calculate baseline model performance: "
                f"{str(error)}"
            )
        )

    try:
        result = permutation_importance(
            estimator=pipeline,
            X=X,
            y=y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Permutation importance calculation failed: "
                f"{str(error)}"
            )
        )

    features = []

    for index, feature in enumerate(X.columns):
        features.append(
            {
                "feature": str(feature),
                "importance": round(
                    float(result.importances_mean[index]),
                    6
                ),
                "importance_std": round(
                    float(result.importances_std[index]),
                    6
                )
            }
        )

    features.sort(
        key=lambda item: item["importance"],
        reverse=True
    )

    top_features = features[:top_n]

    positive_count = sum(
        1
        for item in features
        if item["importance"] > 0
    )

    model_name = artifact.get(
        "best_model_name",
        artifact.get("model_name", "Unknown Model")
    )

    top_names = ", ".join(
        item["feature"]
        for item in top_features[:5]
    )

    return {
        "message": (
            "Model explainability analysis completed successfully."
        ),
        "stored_filename": stored_filename,
        "target_column": target_column,
        "model_artifact": model_artifact,
        "model_name": model_name,
        "problem_type": problem_type,
        "explainability_method": (
            "Permutation Feature Importance"
        ),
        "configuration": {
            "scoring": scoring,
            "permutation_repeats": n_repeats,
            "features_requested": top_n,
            "total_features_analyzed": len(X.columns)
        },
        "baseline_score": round(baseline, 4),
        "top_features": top_features,
        "positive_importance_feature_count": positive_count,
        "feature_engineering_applied": [
            "Name_Title",
            "Name_Length",
            "Ticket_Prefix",
            "Ticket_Length",
            "Cabin_Deck",
            "Cabin_Count"
        ],
        "explanation_summary": (
            f"For the {problem_type} model '{model_name}', "
            f"the features with the largest measured impact were: "
            f"{top_names}. Importance was calculated by repeatedly "
            f"shuffling each feature and measuring the decrease in "
            f"{scoring} performance."
        )
    }
