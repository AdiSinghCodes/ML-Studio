from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.ai_agent_service import run_ai_agent_recommendation

router = APIRouter(
    prefix="/agent",
    tags=["AI Agent"]
)


class AgentRecommendationRequest(BaseModel):
    stored_filename: str
    target_column: Optional[str] = None


@router.post("/recommend")
def get_recommendation(request: AgentRecommendationRequest):
    """
    Connects EDA JSON with AI Agent reasoning, evaluates 4 candidate ML models,
    and returns leaderboard scores and AI justification.
    """
    return run_ai_agent_recommendation(
        stored_filename=request.stored_filename,
        target_column=request.target_column
    )
