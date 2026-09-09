import math
import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.services.target_service import (
    load_dataset_by_name,
    detect_problem_type
)


def perform_eda(
    stored_filename: str,
    target_column: str = None
) -> dict:
    """
    Perform comprehensive Exploratory Data Analysis (EDA) on tabular & NLP datasets.
    Supports both Supervised (with target_column) and Unsupervised (target_column=None) workflows.
    """
    df = load_dataset_by_name(stored_filename)

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The dataset is empty."
        )

    total_rows = int(df.shape[0])
    total_columns = int(df.shape[1])
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)

    # Validate target column if provided
    is_supervised = False
    problem_type_info = None

    if target_column and str(target_column).strip() != "":
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Selected target column '{target_column}' was not found in the dataset."
            )
        is_supervised = True
        try:
            problem_type_info = detect_problem_type(stored_filename, target_column)
        except Exception:
            problem_type_info = {"problem_type": "unknown"}

    # Calculate duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    duplicate_percentage = round((duplicate_rows / max(total_rows, 1)) * 100, 2)

    # Separate column types
    numeric_cols = []
    categorical_cols = []
    text_cols = []
    datetime_cols = []

    column_profiles = {}

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        missing_cnt = int(series.isnull().sum())
        missing_pct = round((missing_cnt / max(total_rows, 1)) * 100, 2)
        unique_cnt = int(series.nunique(dropna=True))

        profile = {
            "name": col_str,
            "data_type": str(series.dtype),
            "missing_values": missing_cnt,
            "missing_percentage": missing_pct,
            "unique_values": unique_cnt,
            "is_target": (col_str == target_column) if is_supervised else False
        }

        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col_str)
            profile["type_category"] = "datetime"

        # Check if numeric
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col_str)
            profile["type_category"] = "numeric"

            clean_series = series.dropna()
            if not clean_series.empty:
                q1 = float(clean_series.quantile(0.25))
                q3 = float(clean_series.quantile(0.75))
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = clean_series[
                    (clean_series < lower_bound) | (clean_series > upper_bound)
                ]
                outlier_cnt = int(len(outliers))
                outlier_pct = round((outlier_cnt / len(clean_series)) * 100, 2)

                std_val = float(clean_series.std())
                skew_val = float(clean_series.skew()) if len(clean_series) > 2 else 0.0
                kurt_val = float(clean_series.kurtosis()) if len(clean_series) > 3 else 0.0

                profile["numeric_stats"] = {
                    "min": float(clean_series.min()),
                    "max": float(clean_series.max()),
                    "mean": round(float(clean_series.mean()), 4),
                    "std": round(std_val, 4),
                    "median": round(float(clean_series.median()), 4),
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr": round(iqr, 4),
                    "skewness": round(skew_val, 4) if not math.isnan(skew_val) else 0.0,
                    "kurtosis": round(kurt_val, 4) if not math.isnan(kurt_val) else 0.0,
                    "outliers_count": outlier_cnt,
                    "outliers_percentage": outlier_pct
                }
            else:
                profile["numeric_stats"] = None

        # Categorical or Text
        else:
            # Check if it's text/NLP (long strings, high average word count)
            sample_str = series.dropna().astype(str)
            avg_word_count = (
                sample_str.str.split().str.len().mean() if not sample_str.empty else 0
            )

            if avg_word_count > 2.5 and unique_cnt > 3:
                text_cols.append(col_str)
                profile["type_category"] = "text_nlp"
                profile["nlp_stats"] = {
                    "avg_word_count": round(float(avg_word_count), 2),
                    "max_word_count": int(sample_str.str.split().str.len().max()) if not sample_str.empty else 0,
                    "avg_char_length": round(float(sample_str.str.len().mean()), 2) if not sample_str.empty else 0,
                    "vocabulary_size": int(len(set(" ".join(sample_str.tolist()).split()))) if not sample_str.empty else 0
                }
            else:
                categorical_cols.append(col_str)
                profile["type_category"] = "categorical"
                top_values = (
                    series.value_counts(dropna=True).head(5).to_dict()
                )
                profile["categorical_stats"] = {
                    "top_frequencies": {str(k): int(v) for k, v in top_values.items()},
                    "is_high_cardinality": unique_cnt > 50
                }

        column_profiles[col_str] = profile

    # Compute correlation matrix for numeric columns
    correlation_matrix = {}
    high_correlations = []

    if len(numeric_cols) >= 2:
        try:
            corr_df = df[numeric_cols].corr(method="pearson").round(3)
            corr_df = corr_df.fillna(0)

            for c1 in numeric_cols:
                correlation_matrix[c1] = {}
                for c2 in numeric_cols:
                    val = float(corr_df.loc[c1, c2])
                    correlation_matrix[c1][c2] = val
                    if c1 < c2 and abs(val) >= 0.85:
                        high_correlations.append({
                            "feature_1": c1,
                            "feature_2": c2,
                            "correlation": val
                        })
        except Exception:
            correlation_matrix = {}

    # Target & Imbalance Analysis
    target_analysis = None
    if is_supervised and target_column in df.columns:
        target_series = df[target_column].dropna()
        prob_type = problem_type_info.get("problem_type", "classification") if problem_type_info else "classification"

        if prob_type == "classification":
            val_counts = target_series.value_counts()
            total_target_cnt = len(target_series)
            class_dist = {}
            for k, v in val_counts.items():
                class_dist[str(k)] = {
                    "count": int(v),
                    "percentage": round((v / max(total_target_cnt, 1)) * 100, 2)
                }

            if len(val_counts) > 1:
                maj_cnt = val_counts.iloc[0]
                min_cnt = val_counts.iloc[-1]
                imbalance_ratio = round(maj_cnt / max(min_cnt, 1), 2)

                if imbalance_ratio > 5.0:
                    imbalance_status = "severe_imbalance"
                elif imbalance_ratio > 2.5:
                    imbalance_status = "moderate_imbalance"
                else:
                    imbalance_status = "balanced"
            else:
                imbalance_ratio = 1.0
                imbalance_status = "single_class"

            target_analysis = {
                "target_column": target_column,
                "mode": "supervised",
                "problem_type": "classification",
                "num_classes": int(len(val_counts)),
                "class_distribution": class_dist,
                "imbalance_ratio": imbalance_ratio,
                "imbalance_status": imbalance_status
            }
        else:
            # Regression target
            target_analysis = {
                "target_column": target_column,
                "mode": "supervised",
                "problem_type": "regression",
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "mean": round(float(target_series.mean()), 4),
                "median": round(float(target_series.median()), 4),
                "std": round(float(target_series.std()), 4),
                "skewness": round(float(target_series.skew()), 4) if len(target_series) > 2 else 0.0
            }
    else:
        # Unsupervised Mode
        zero_var_cols = []
        for nc in numeric_cols:
            if df[nc].nunique(dropna=True) <= 1:
                zero_var_cols.append(nc)

        target_analysis = {
            "mode": "unsupervised",
            "message": "No target column specified. Dataset is evaluated for unsupervised tasks (Clustering / PCA).",
            "zero_variance_columns": zero_var_cols,
            "scaling_recommended": len(numeric_cols) > 0
        }

    # Data Health Score Calculation
    total_missing_cnt = int(df.isnull().sum().sum())
    total_cells = total_rows * total_columns
    missing_pct_overall = round((total_missing_cnt / max(total_cells, 1)) * 100, 2)

    health_score = 100
    health_score -= min(missing_pct_overall * 2, 40)
    health_score -= min(duplicate_percentage * 2, 20)
    if high_correlations:
        health_score -= min(len(high_correlations) * 3, 15)
    health_score = max(0, round(health_score))

    # Structured AI Agent Payload
    agent_summary = {
        "dataset_name": stored_filename,
        "mode": "supervised" if is_supervised else "unsupervised",
        "target_column": target_column if is_supervised else None,
        "rows": total_rows,
        "columns": total_columns,
        "health_score": health_score,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "text_nlp_features": text_cols,
        "high_correlation_pairs": high_correlations,
        "target_analysis": target_analysis
    }

    return {
        "status": "success",
        "stored_filename": stored_filename,
        "mode": "supervised" if is_supervised else "unsupervised",
        "target_column": target_column if is_supervised else None,
        "summary": {
            "rows": total_rows,
            "columns": total_columns,
            "memory_mb": memory_usage_mb,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_percentage,
            "health_score": health_score,
            "numeric_column_count": len(numeric_cols),
            "categorical_column_count": len(categorical_cols),
            "text_column_count": len(text_cols),
            "datetime_column_count": len(datetime_cols)
        },
        "column_profiles": column_profiles,
        "correlation_matrix": correlation_matrix,
        "high_correlations": high_correlations,
        "target_analysis": target_analysis,
        "eda_agent_summary": agent_summary
    }
