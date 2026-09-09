from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.training_service import (
    train_and_compare_models,
    tune_models
)


router = APIRouter(
    prefix="/training",
    tags=["Model Training"]
)


class TrainingRequest(BaseModel):
    stored_filename: str
    target_column: str
    test_size: float = Field(
        default=0.20,
        ge=0.05,
        le=0.50
    )
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=10
    )


class TuningRequest(BaseModel):
    stored_filename: str
    target_column: str
    test_size: float = Field(
        default=0.20,
        ge=0.05,
        le=0.50
    )
    cv_folds: int = Field(
        default=5,
        ge=2,
        le=10
    )
    n_iter: int = Field(
        default=15,
        ge=3,
        le=100
    )
    max_models_to_tune: int = Field(
        default=3,
        ge=1,
        le=5
    )


@router.post("/train")
def train_models(
    request: TrainingRequest
):
    return train_and_compare_models(
        stored_filename=request.stored_filename,
        target_column=request.target_column,
        test_size=request.test_size,
        cv_folds=request.cv_folds
    )


@router.post("/tune")
def tune_hyperparameters(
    request: TuningRequest
):
    return tune_models(
        stored_filename=request.stored_filename,
        target_column=request.target_column,
        test_size=request.test_size,
        cv_folds=request.cv_folds,
        n_iter=request.n_iter,
        max_models_to_tune=(
            request.max_models_to_tune
        )
    )
