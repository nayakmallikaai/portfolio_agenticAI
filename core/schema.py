# Pydantic models for Type Safety
from pydantic import BaseModel, Field
from typing import List, Optional

class Trade(BaseModel):
    ticker: str
    action: str = Field(description="BUY or SELL")
    quantity: int
    reasoning: str

class PortfolioReview(BaseModel):
    approved: bool
    trades: List[Trade]
    risk_score: float
    rejection_reason: Optional[str] = None

class AnalystOutput(BaseModel):
    thought_process: str
    proposed_trades: List[Trade]

class RiskReview(BaseModel):
    approved: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    comments: str