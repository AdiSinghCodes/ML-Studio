from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.eda_service import perform_eda

router = APIRouter(
    prefix="/eda",
    tags=["EDA"]
)


class EDARequest(BaseModel):
    stored_filename: str
    target_column: Optional[str] = None


@router.post("/analyze")
def analyze_eda(request: EDARequest):
    """
    Perform comprehensive EDA on a dataset.
    If target_column is provided, performs Supervised EDA.
    If target_column is null/empty, performs Unsupervised EDA.
    """
    return perform_eda(
        stored_filename=request.stored_filename,
        target_column=request.target_column
    )


@router.get("/{stored_filename}")
def get_eda_report(
    stored_filename: str,
    target_column: Optional[str] = None
):
    """
    GET endpoint to retrieve EDA statistics for a dataset.
    """
    return perform_eda(
        stored_filename=stored_filename,
        target_column=target_column
    )
