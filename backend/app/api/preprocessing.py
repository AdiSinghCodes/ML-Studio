from fastapi import APIRouter
from pydantic import BaseModel

from app.services.preprocessing_service import (
    preprocess_dataset
)


router = APIRouter(
    prefix="/preprocessing",
    tags=["Preprocessing"]
)


class PreprocessingRequest(BaseModel):
    stored_filename: str
    target_column: str
    scale_numeric: bool = True


@router.post("/prepare")
def prepare_dataset(
    request: PreprocessingRequest
):
    """
    Preprocess an uploaded dataset for machine learning.
    """

    return preprocess_dataset(
        stored_filename=request.stored_filename,
        target_column=request.target_column,
        scale_numeric=request.scale_numeric
    )
