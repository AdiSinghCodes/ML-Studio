import uuid
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from app.core.config import MODEL_DIR, TRAINING_DATA_DIR
from app.services.target_service import detect_problem_type, load_dataset_by_name


def _build_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _detect_id_columns(df: pd.DataFrame) -> list:
    id_columns = []

    for column in df.columns:
        name = str(column).lower()
        unique_ratio = df[column].nunique(dropna=True) / max(len(df), 1)

        if name == "id" or name.endswith("id") or "identifier" in name:
            id_columns.append(column)
        elif unique_ratio >= 0.98 and (
            "number" in name or "code" in name or "index" in name
        ):
            id_columns.append(column)

    return id_columns


def _extract_name_features(df, column):
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
        extracted[column].astype("string").str.len().fillna(0)
    )

    return extracted.drop(columns=[column])


def _extract_cabin_features(df, column):
    extracted = df.copy()

    cabin_text = extracted[column].astype("string").fillna("Unknown")

    extracted[f"{column}_Deck"] = (
        cabin_text
        .str.extract(r"([A-Za-z])", expand=False)
        .str.upper()
        .fillna("Unknown")
    )

    extracted[f"{column}_Count"] = (
        cabin_text.str.split().str.len().fillna(0)
    )

    return extracted.drop(columns=[column])


def _extract_ticket_features(df, column):
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

    extracted[f"{column}_Prefix"] = prefix.replace("", "NUMERIC")
    extracted[f"{column}_Length"] = ticket_text.str.len()

    return extracted.drop(columns=[column])


def _is_high_cardinality(series):
    unique_values = series.nunique(dropna=True)
    unique_ratio = unique_values / max(len(series), 1)

    return unique_values >= 30 and unique_ratio >= 0.50


def _prepare_features(df, target_column):
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    id_columns = _detect_id_columns(X)

    if id_columns:
        X = X.drop(columns=id_columns)

    for column in list(X.columns):
        name = str(column).lower()

        if name == "name":
            X = _extract_name_features(X, column)
        elif name == "cabin":
            X = _extract_cabin_features(X, column)
        elif name == "ticket":
            X = _extract_ticket_features(X, column)

    high_cardinality_columns = []

    for column in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[column]):
            if _is_high_cardinality(X[column]):
                high_cardinality_columns.append(str(column))
                X = X.drop(columns=[column])

    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                ("encoder", _build_one_hot_encoder())
            ]
        )
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    if not transformers:
        raise HTTPException(
            status_code=400,
            detail="No usable features found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return (
        X,
        y,
        preprocessor,
        id_columns,
        high_cardinality_columns
    )


def _classification_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
        "Support Vector Machine": SVC(
            probability=True,
            random_state=42
        )
    }


def _regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=42
        ),
        "Support Vector Regressor": SVR()
    }


def _safe_cv_folds(y, problem_type, requested_folds):
    requested_folds = max(2, min(requested_folds, 10))

    if problem_type == "classification":
        smallest_class = int(
            pd.Series(y).value_counts().min()
        )
        return max(2, min(requested_folds, smallest_class))

    return max(2, min(requested_folds, len(y)))


def _evaluate_classification(model, X_test, y_test):
    predictions = model.predict(X_test)

    average = (
        "binary"
        if len(np.unique(y_test)) == 2
        else "weighted"
    )

    result = {
        "accuracy": round(
            float(accuracy_score(y_test, predictions)),
            4
        ),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    predictions,
                    average=average,
                    zero_division=0
                )
            ),
            4
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    predictions,
                    average=average,
                    zero_division=0
                )
            ),
            4
        ),
        "f1_score": round(
            float(
                f1_score(
                    y_test,
                    predictions,
                    average=average,
                    zero_division=0
                )
            ),
            4
        )
    }

    if (
        len(np.unique(y_test)) == 2
        and hasattr(model, "predict_proba")
    ):
        probabilities = model.predict_proba(X_test)[:, 1]

        result["roc_auc"] = round(
            float(
                roc_auc_score(
                    y_test,
                    probabilities
                )
            ),
            4
        )

    return result


def _evaluate_regression(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return {
        "mae": round(
            float(mean_absolute_error(y_test, predictions)),
            4
        ),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2_score": round(
            float(r2_score(y_test, predictions)),
            4
        )
    }


def _prepare_training_context(
    stored_filename,
    target_column,
    test_size,
    cv_folds
):
    if not 0.05 <= test_size <= 0.50:
        raise HTTPException(
            status_code=400,
            detail="test_size must be between 0.05 and 0.50."
        )

    df = load_dataset_by_name(stored_filename)

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' does not exist."
            )
        )

    df = df.dropna(axis=0, how="all").copy()
    df = df.dropna(subset=[target_column]).copy()

    (
        X,
        y,
        preprocessor,
        removed_ids,
        dropped_high_cardinality
    ) = _prepare_features(df, target_column)

    detection = detect_problem_type(
        stored_filename,
        target_column
    )
    problem_type = detection["problem_type"]

    target_encoder = None

    if problem_type == "classification":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

        if len(np.unique(y)) < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Classification requires at least two classes."
                )
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        models = _classification_models()
        scoring = "f1_weighted"

    else:
        y = pd.to_numeric(y, errors="coerce")
        valid_mask = ~pd.isna(y)

        X = X.loc[valid_mask].copy()
        y = y.loc[valid_mask].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

        models = _regression_models()
        scoring = "r2"

    safe_folds = _safe_cv_folds(
        y_train,
        problem_type,
        cv_folds
    )

    if problem_type == "classification":
        cv_strategy = StratifiedKFold(
            n_splits=safe_folds,
            shuffle=True,
            random_state=42
        )
    else:
        cv_strategy = KFold(
            n_splits=safe_folds,
            shuffle=True,
            random_state=42
        )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "removed_ids": removed_ids,
        "dropped_high_cardinality": (
            dropped_high_cardinality
        ),
        "problem_type": problem_type,
        "target_encoder": target_encoder,
        "models": models,
        "scoring": scoring,
        "safe_folds": safe_folds,
        "cv_strategy": cv_strategy
    }


def train_and_compare_models(
    stored_filename: str,
    target_column: str,
    test_size: float = 0.20,
    cv_folds: int = 5
) -> dict:
    context = _prepare_training_context(
        stored_filename,
        target_column,
        test_size,
        cv_folds
    )

    X = context["X"]
    X_train = context["X_train"]
    X_test = context["X_test"]
    y_train = context["y_train"]
    y_test = context["y_test"]
    preprocessor = context["preprocessor"]
    problem_type = context["problem_type"]
    models = context["models"]
    scoring = context["scoring"]
    cv_strategy = context["cv_strategy"]

    model_results = {}
    fitted_models = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator)
            ]
        )

        try:
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=-1
            )

            pipeline.fit(X_train, y_train)

            if problem_type == "classification":
                metrics = _evaluate_classification(
                    pipeline,
                    X_test,
                    y_test
                )
                primary_score = metrics["f1_score"]
            else:
                metrics = _evaluate_regression(
                    pipeline,
                    X_test,
                    y_test
                )
                primary_score = metrics["r2_score"]

            model_results[model_name] = {
                "cross_validation_mean": round(
                    float(np.mean(cv_scores)),
                    4
                ),
                "cross_validation_std": round(
                    float(np.std(cv_scores)),
                    4
                ),
                "test_metrics": metrics,
                "primary_test_score": round(
                    float(primary_score),
                    4
                )
            }

            fitted_models[model_name] = pipeline

        except Exception as error:
            model_results[model_name] = {
                "error": str(error)
            }

    successful_models = {
        name: result
        for name, result in model_results.items()
        if "primary_test_score" in result
    }

    if not successful_models:
        raise HTTPException(
            status_code=500,
            detail="All models failed during training."
        )

    best_model_name = max(
        successful_models,
        key=lambda name: successful_models[name][
            "primary_test_score"
        ]
    )

    best_model = fitted_models[best_model_name]

    training_id = str(uuid.uuid4())
    model_filename = f"{training_id}_best_model.joblib"

    model_artifact = {
        "pipeline": best_model,
        "problem_type": problem_type,
        "target_column": target_column,
        "target_encoder": context["target_encoder"],
        "best_model_name": best_model_name,
        "training_timestamp": datetime.utcnow().isoformat()
    }

    joblib.dump(
        model_artifact,
        MODEL_DIR / model_filename
    )

    summary = {
        "message": (
            "Model training and comparison completed successfully."
        ),
        "stored_filename": stored_filename,
        "target_column": target_column,
        "problem_type": problem_type,
        "dataset_split": {
            "total_samples": int(len(X)),
            "training_samples": int(len(X_train)),
            "testing_samples": int(len(X_test)),
            "test_size": test_size
        },
        "cross_validation": {
            "folds": context["safe_folds"],
            "scoring": scoring
        },
        "feature_preparation": {
            "removed_identifier_columns": [
                str(column)
                for column in context["removed_ids"]
            ],
            "dropped_high_cardinality_columns": (
                context["dropped_high_cardinality"]
            )
        },
        "models": model_results,
        "best_model": {
            "name": best_model_name,
            "primary_test_score": (
                successful_models[best_model_name][
                    "primary_test_score"
                ]
            ),
            "model_artifact": model_filename
        }
    }

    joblib.dump(
        summary,
        TRAINING_DATA_DIR /
        f"{training_id}_training_summary.joblib"
    )

    return summary


def _classification_search_spaces():
    return {
        "Logistic Regression": {
            "model__C": [
                0.01, 0.1, 0.5, 1.0, 2.0,
                5.0, 10.0, 20.0
            ],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
            "model__class_weight": [
                None, "balanced"
            ]
        },
        "Random Forest": {
            "model__n_estimators": [
                150, 250, 400, 600
            ],
            "model__max_depth": [
                None, 5, 10, 15, 25
            ],
            "model__min_samples_split": [
                2, 5, 10, 20
            ],
            "model__min_samples_leaf": [
                1, 2, 4, 8
            ],
            "model__max_features": [
                "sqrt", "log2", None
            ]
        },
        "Gradient Boosting": {
            "model__n_estimators": [
                50, 100, 150, 250
            ],
            "model__learning_rate": [
                0.01, 0.03, 0.05,
                0.1, 0.2
            ],
            "model__max_depth": [
                2, 3, 4, 5
            ],
            "model__subsample": [
                0.6, 0.8, 1.0
            ],
            "model__min_samples_split": [
                2, 5, 10
            ]
        },
        "Support Vector Machine": {
            "model__C": [
                0.1, 0.5, 1.0, 2.0,
                5.0, 10.0
            ],
            "model__kernel": [
                "linear", "rbf", "poly"
            ],
            "model__gamma": [
                "scale", "auto",
                0.001, 0.01, 0.1
            ]
        }
    }


def _regression_search_spaces():
    return {
        "Ridge Regression": {
            "model__alpha": [
                0.01, 0.1, 0.5, 1.0,
                2.0, 5.0, 10.0,
                50.0, 100.0
            ]
        },
        "Random Forest Regressor": {
            "model__n_estimators": [
                150, 250, 400, 600
            ],
            "model__max_depth": [
                None, 5, 10, 15, 25
            ],
            "model__min_samples_split": [
                2, 5, 10, 20
            ],
            "model__min_samples_leaf": [
                1, 2, 4, 8
            ],
            "model__max_features": [
                "sqrt", "log2", 1.0
            ]
        },
        "Gradient Boosting Regressor": {
            "model__n_estimators": [
                50, 100, 150, 250
            ],
            "model__learning_rate": [
                0.01, 0.03, 0.05,
                0.1, 0.2
            ],
            "model__max_depth": [
                2, 3, 4, 5
            ],
            "model__subsample": [
                0.6, 0.8, 1.0
            ],
            "model__min_samples_split": [
                2, 5, 10
            ]
        },
        "Support Vector Regressor": {
            "model__C": [
                0.1, 0.5, 1.0, 2.0,
                5.0, 10.0, 50.0
            ],
            "model__epsilon": [
                0.01, 0.05, 0.1,
                0.2, 0.5
            ],
            "model__kernel": [
                "linear", "rbf", "poly"
            ],
            "model__gamma": [
                "scale", "auto",
                0.001, 0.01, 0.1
            ]
        }
    }


def tune_models(
    stored_filename: str,
    target_column: str,
    test_size: float = 0.20,
    cv_folds: int = 5,
    n_iter: int = 15,
    max_models_to_tune: int = 3
) -> dict:
    """
    Automatically tune the strongest baseline models.

    Phase 1:
    Train baseline models and rank them by CV score.

    Phase 2:
    Tune the top N supported models with RandomizedSearchCV.

    Phase 3:
    Evaluate tuned models on the held-out test set and save
    the final best model.
    """

    n_iter = max(3, min(n_iter, 100))
    max_models_to_tune = max(
        1,
        min(max_models_to_tune, 5)
    )

    context = _prepare_training_context(
        stored_filename,
        target_column,
        test_size,
        cv_folds
    )

    X_train = context["X_train"]
    X_test = context["X_test"]
    y_train = context["y_train"]
    y_test = context["y_test"]
    preprocessor = context["preprocessor"]
    problem_type = context["problem_type"]
    models = context["models"]
    scoring = context["scoring"]
    cv_strategy = context["cv_strategy"]

    baseline_results = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator)
            ]
        )

        try:
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=-1
            )

            baseline_results[model_name] = {
                "cross_validation_mean": round(
                    float(np.mean(cv_scores)),
                    4
                ),
                "cross_validation_std": round(
                    float(np.std(cv_scores)),
                    4
                )
            }

        except Exception as error:
            baseline_results[model_name] = {
                "error": str(error)
            }

    successful_baselines = {
        name: result
        for name, result in baseline_results.items()
        if "cross_validation_mean" in result
    }

    if not successful_baselines:
        raise HTTPException(
            status_code=500,
            detail="Baseline evaluation failed for all models."
        )

    ranked_models = sorted(
        successful_baselines.keys(),
        key=lambda name: successful_baselines[name][
            "cross_validation_mean"
        ],
        reverse=True
    )

    if problem_type == "classification":
        search_spaces = _classification_search_spaces()
    else:
        search_spaces = _regression_search_spaces()

    selected_models = [
        name
        for name in ranked_models
        if name in search_spaces
    ][:max_models_to_tune]

    if not selected_models:
        raise HTTPException(
            status_code=500,
            detail=(
                "No supported models were available "
                "for hyperparameter tuning."
            )
        )

    tuning_results = {}
    tuned_models = {}

    for model_name in selected_models:
        estimator = models[model_name]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator)
            ]
        )

        try:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=(
                    search_spaces[model_name]
                ),
                n_iter=n_iter,
                scoring=scoring,
                cv=cv_strategy,
                random_state=42,
                n_jobs=-1,
                refit=True,
                return_train_score=False
            )

            search.fit(
                X_train,
                y_train
            )

            tuned_pipeline = search.best_estimator_

            if problem_type == "classification":
                test_metrics = (
                    _evaluate_classification(
                        tuned_pipeline,
                        X_test,
                        y_test
                    )
                )
                primary_score = test_metrics[
                    "f1_score"
                ]
            else:
                test_metrics = (
                    _evaluate_regression(
                        tuned_pipeline,
                        X_test,
                        y_test
                    )
                )
                primary_score = test_metrics[
                    "r2_score"
                ]

            baseline_cv = successful_baselines[
                model_name
            ]["cross_validation_mean"]

            tuned_cv = round(
                float(search.best_score_),
                4
            )

            tuning_results[model_name] = {
                "baseline_cross_validation_score": (
                    baseline_cv
                ),
                "tuned_cross_validation_score": (
                    tuned_cv
                ),
                "cross_validation_improvement": round(
                    tuned_cv - baseline_cv,
                    4
                ),
                "best_parameters": {
                    key.replace("model__", ""): value
                    for key, value in (
                        search.best_params_.items()
                    )
                },
                "test_metrics": test_metrics,
                "primary_test_score": round(
                    float(primary_score),
                    4
                )
            }

            tuned_models[model_name] = (
                tuned_pipeline
            )

        except Exception as error:
            tuning_results[model_name] = {
                "error": str(error)
            }

    successful_tuned_models = {
        name: result
        for name, result in tuning_results.items()
        if "primary_test_score" in result
    }

    if not successful_tuned_models:
        raise HTTPException(
            status_code=500,
            detail="All hyperparameter tuning runs failed."
        )

    best_model_name = max(
        successful_tuned_models,
        key=lambda name: successful_tuned_models[name][
            "primary_test_score"
        ]
    )

    best_pipeline = tuned_models[
        best_model_name
    ]

    tuning_id = str(uuid.uuid4())

    model_filename = (
        f"{tuning_id}_tuned_best_model.joblib"
    )

    model_artifact = {
        "pipeline": best_pipeline,
        "problem_type": problem_type,
        "target_column": target_column,
        "target_encoder": context["target_encoder"],
        "best_model_name": best_model_name,
        "best_parameters": (
            successful_tuned_models[
                best_model_name
            ]["best_parameters"]
        ),
        "tuning_timestamp": (
            datetime.utcnow().isoformat()
        )
    }

    joblib.dump(
        model_artifact,
        MODEL_DIR / model_filename
    )

    summary = {
        "message": (
            "Automated hyperparameter tuning completed successfully."
        ),
        "stored_filename": stored_filename,
        "target_column": target_column,
        "problem_type": problem_type,
        "dataset_split": {
            "training_samples": int(
                len(X_train)
            ),
            "testing_samples": int(
                len(X_test)
            ),
            "test_size": test_size
        },
        "tuning_configuration": {
            "cross_validation_folds": (
                context["safe_folds"]
            ),
            "scoring": scoring,
            "random_search_iterations_per_model": (
                n_iter
            ),
            "models_tuned": selected_models
        },
        "baseline_ranking": ranked_models,
        "baseline_models": baseline_results,
        "tuned_models": tuning_results,
        "best_tuned_model": {
            "name": best_model_name,
            "primary_test_score": (
                successful_tuned_models[
                    best_model_name
                ]["primary_test_score"]
            ),
            "best_parameters": (
                successful_tuned_models[
                    best_model_name
                ]["best_parameters"]
            ),
            "model_artifact": model_filename
        }
    }

    joblib.dump(
        summary,
        TRAINING_DATA_DIR /
        f"{tuning_id}_tuning_summary.joblib"
    )

    return summary
