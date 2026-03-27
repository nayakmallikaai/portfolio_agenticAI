from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    user_id: str
    session_id: str
    goal: str


class ExecuteRequest(BaseModel):
    user_id: str
    session_id: str
    approved: bool
