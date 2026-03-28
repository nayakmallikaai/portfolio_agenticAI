from typing import Literal, Optional
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    user_id: str
    session_id: str
    mode: Literal["goal", "feedback"] = "goal"
    goal: Optional[str] = None   # required when mode="goal", ignored when mode="feedback"


class ExecuteRequest(BaseModel):
    user_id: str
    session_id: str
    approved: bool
