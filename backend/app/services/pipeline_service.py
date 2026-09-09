from fastapi import HTTPException

from app.services.target_service import (
    load_dataset_by_name,
    detect_problem_type
)

from app.services.preprocessing_service import (
    preprocess_dataset
)

from app.services.training_service import (
    train_and_compare_models,
    tune_models
)

from app.services.explain_service import (
    explain_feature_importance
)


def _detect_target_column(
    stored_filename: str
) -> dict:
    """
    Automatically select a likely target column.

    Priority:
    1. Common ML target names.
    2. Last suitable column as fallback.
    """

    df = load_dataset_by_name(
        stored_filename
    )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The dataset is empty."
        )

    columns = list(df.columns)

    if len(columns) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "The dataset must contain at least "
                "one feature column and one target column."
            )
        )

    # Common target column names.
    target_keywords = [
        "target",
        "label",
        "class",
        "outcome",
        "output",
        "result",
        "response",
        "y",
        "survived",
        "diagnosis",
        "price",
        "saleprice",
        "salary",
        "income",
        "score"
    ]

    for column in columns:
        column_name = str(
            column
        ).lower().replace(
            "_",
            ""
        ).replace(
            " ",
            ""
        )

        for keyword in target_keywords:
            if (
                keyword in column_name
                or column_name == keyword
            ):
                return {
                    "stored_filename": (
                        stored_filename
                    ),
                    "target_column": (
                        str(column)
                    ),
                    "detection_method": (
                        "common_target_name"
                    ),
                    "confidence": "high"
                }

    # Fallback:
    # In many tabular datasets the target is the last column.
    target_column = columns[-1]

    return {
        "stored_filename": (
            stored_filename
        ),
        "target_column": (
            str(target_column)
        ),
        "detection_method": (
            "last_column_fallback"
        ),
        "confidence": "medium"
    }


def run_complete_pipeline(
    stored_filename: str
) -> dict:

    pipeline_results = {}

    # -----------------------------------------
    # STEP 1 - TARGET DETECTION
    # -----------------------------------------

    try:

        target_result = (
            _detect_target_column(
                stored_filename
            )
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Automatic target detection failed: "
                f"{str(error)}"
            )
        )

    target_column = (
        target_result["target_column"]
    )

    pipeline_results[
        "target_detection"
    ] = target_result

    # -----------------------------------------
    # STEP 2 - PROBLEM TYPE DETECTION
    # -----------------------------------------

    try:

        problem_result = (
            detect_problem_type(
                stored_filename,
                target_column
            )
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Problem type detection failed: "
                f"{str(error)}"
            )
        )

    pipeline_results[
        "problem_detection"
    ] = problem_result

    # -----------------------------------------
    # STEP 3 - PREPROCESSING
    # -----------------------------------------

    try:

        preprocessing_result = (
            preprocess_dataset(
                stored_filename=(
                    stored_filename
                ),
                target_column=(
                    target_column
                )
            )
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Dataset preprocessing failed: "
                f"{str(error)}"
            )
        )

    pipeline_results[
        "preprocessing"
    ] = preprocessing_result

    # -----------------------------------------
    # STEP 4 - MODEL TRAINING
    # -----------------------------------------

    try:

        training_result = (
            train_and_compare_models(
                stored_filename=(
                    stored_filename
                ),
                target_column=(
                    target_column
                )
            )
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model training failed: "
                f"{str(error)}"
            )
        )

    pipeline_results[
        "model_training"
    ] = training_result

    # -----------------------------------------
    # STEP 5 - HYPERPARAMETER TUNING
    # -----------------------------------------

    try:

        tuning_result = (
            tune_models(
                stored_filename=(
                    stored_filename
                ),
                target_column=(
                    target_column
                ),
                n_iter=15,
                max_models_to_tune=3
            )
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Hyperparameter tuning failed: "
                f"{str(error)}"
            )
        )

    pipeline_results[
        "hyperparameter_tuning"
    ] = tuning_result

    # -----------------------------------------
    # STEP 6 - GET BEST MODEL
    # -----------------------------------------

    best_tuned_model = (
        tuning_result.get(
            "best_tuned_model"
        )
    )

    if not best_tuned_model:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to identify the "
                "best tuned model."
            )
        )

    model_artifact = (
        best_tuned_model.get(
            "model_artifact"
        )
    )

    model_name = (
        best_tuned_model.get(
            "name"
        )
    )

    if not model_artifact:

        raise HTTPException(
            status_code=500,
            detail=(
                "Final model artifact "
                "was not created."
            )
        )

    # -----------------------------------------
    # STEP 7 - EXPLAINABLE AI
    # -----------------------------------------

    try:

        explainability_result = (
            explain_feature_importance(
                stored_filename=(
                    stored_filename
                ),
                target_column=(
                    target_column
                ),
                model_artifact=(
                    model_artifact
                ),
                top_n=10,
                n_repeats=10
            )
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Explainable AI analysis failed: "
                f"{str(error)}"
            )
        )

    pipeline_results[
        "explainability"
    ] = explainability_result

    # -----------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------

    final_result = {

        "status": "completed",

        "target_column": (
            target_column
        ),

        "problem_type": (
            problem_result.get(
                "problem_type"
            )
        ),

        "best_model": (
            model_name
        ),

        "model_artifact": (
            model_artifact
        ),

        "model_score": (
            best_tuned_model.get(
                "primary_test_score"
            )
        ),

        "top_features": (
            explainability_result.get(
                "top_features",
                []
            )
        )
    }

    pipeline_results[
        "final_result"
    ] = final_result

    return {

        "message": (
            "Automated machine learning "
            "pipeline completed successfully."
        ),

        "stored_filename": (
            stored_filename
        ),

        "pipeline": (
            pipeline_results
        )
    }
