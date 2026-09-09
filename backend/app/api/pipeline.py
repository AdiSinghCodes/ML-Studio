from fastapi import APIRouter
from pydantic import BaseModel

from app.services.pipeline_service import (
    run_complete_pipeline
)


router = APIRouter(
    prefix="/pipeline",
    tags=["Automated Pipeline"]
)


class PipelineRequest(BaseModel):
    stored_filename: str


@router.post("/run")
def run_pipeline(
    request: PipelineRequest
):
    """
    Run the complete automated AML pipeline.
    """

    return run_complete_pipeline(
        stored_filename=request.stored_filename
    )