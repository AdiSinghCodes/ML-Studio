from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.explain_service import (
    explain_feature_importance
)


router = APIRouter(
    prefix="/explain",
    tags=["Explainable AI"]
)


class ExplainRequest(BaseModel):
    stored_filename: str
    target_column: str
    model_artifact: str
    top_n: int = Field(
        default=10,
        ge=1,
        le=50
    )
    n_repeats: int = Field(
        default=10,
        ge=3,
        le=50
    )


@router.post("/feature-importance")
def get_feature_importance(
    request: ExplainRequest
):
    """
    Generate model-agnostic permutation feature
    importance for a saved trained model.
    """

    return explain_feature_importance(
        stored_filename=request.stored_filename,
        target_column=request.target_column,
        model_artifact=request.model_artifact,
        top_n=request.top_n,
        n_repeats=request.n_repeats
    )
