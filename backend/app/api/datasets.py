from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from app.services.dataset_service import (
    save_uploaded_dataset,
    load_dataset,
    analyze_dataset
)
from app.services.target_service import (
    get_dataset_columns,
    detect_problem_type
)


router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"]
)


class TargetSelectionRequest(BaseModel):
    stored_filename: str
    target_column: str


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...)
):
    """
    Upload and analyze a CSV or XLSX dataset.
    """

    file_path = save_uploaded_dataset(file)
    dataframe = load_dataset(file_path)
    analysis = analyze_dataset(dataframe)

    return {
        "message": "Dataset uploaded successfully.",
        "filename": file.filename,
        "stored_filename": file_path.name,
        "analysis": analysis
    }


@router.get("/{stored_filename}/columns")
def list_dataset_columns(stored_filename: str):
    """
    Return the columns available in an uploaded dataset.
    """

    return get_dataset_columns(stored_filename)


@router.post("/select-target")
def select_target_column(request: TargetSelectionRequest):
    """
    Select a target column and automatically detect whether the problem
    is classification or regression.
    """

    return detect_problem_type(
        stored_filename=request.stored_filename,
        target_column=request.target_column
    )
