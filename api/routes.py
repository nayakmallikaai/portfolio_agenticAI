"""
API route handlers. Pure HTTP concerns only — no LLM or graph logic here.
"""
import asyncio
import json
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from api.schemas import AnalyzeRequest, ExecuteRequest
from agent.graph import run_analysis
from db.engine import SessionLocal
import db.crud as crud

router = APIRouter(prefix="/api")

_mcp_holder: Dict[str, Any] = {}

MIN_GOAL_LENGTH = 10
MAX_GOAL_LENGTH = 500

PORTFOLIO_KEYWORDS = [
    "portfolio", "stock", "buy", "sell", "invest", 
    "trade", "rebalance", "risk", "holdings", "price",
    "shares", "equity", "cash", "return", "diversif",
    "trim", "reduce", "increase", "allocation"
]

BLOCKED_PATTERNS = [
    "how are you", "what are you", "who are you",
    "hello", "hi ", "hey ", "test", "ignore previous",
    "forget your instructions", "pretend you are"
]

# Predefined goal injected when mode="feedback"
_FEEDBACK_GOAL = (
    "Do a comprehensive health check of my portfolio. "
    "Review concentration risk, sector diversification, cash-to-equity ratio, and how each holding is performing. "
    "If the portfolio looks healthy and balanced, say so clearly and propose no trades. "
    "Only suggest trades if there is a clear, specific problem — such as dangerous concentration, "
    "excessive idle cash, or a significantly underperforming holding. "
    "Quality over quantity: one well-reasoned trade suggestion is better than several forced ones."
)


def set_mcp_holder(holder: Dict[str, Any]) -> None:
    global _mcp_holder
    _mcp_holder = holder


# ── GET /api/portfolio/{user_id} ──────────────────────────────────────────────
@router.get("/portfolio/{user_id}")
async def get_portfolio_view(user_id: str):
    """
    Returns the user's current portfolio with live prices for each holding.
    Auto-seeds the user with default holdings on first call.
    """
    db = SessionLocal()
    try:
        portfolio = crud.get_portfolio(db, user_id)
    finally:
        db.close()

    mcp_session = _mcp_holder["session"]
    tickers = list(portfolio["holdings"].keys())

    # Fetch all live prices concurrently
    price_results = await asyncio.gather(*[
        mcp_session.call_tool("get_live_price", {"ticker": t, "user_id": user_id})
        for t in tickers
    ])

    holdings = []
    equity_total = 0.0
    total_gain_loss = 0.0
    for ticker, result in zip(tickers, price_results):
        info      = portfolio["holdings"][ticker]
        qty       = info["qty"]
        buy_price = info.get("buy_price")
        buy_date  = info.get("buy_date")
        price     = float(result.content[0].text)
        value     = qty * price
        equity_total += value

        gain_loss     = round((price - buy_price) * qty, 2) if buy_price else None
        gain_loss_pct = round((price - buy_price) / buy_price * 100, 2) if buy_price else None
        if gain_loss is not None:
            total_gain_loss += gain_loss

        holdings.append({
            "ticker":        ticker,
            "qty":           qty,
            "buy_price":     round(buy_price, 2) if buy_price else None,
            "buy_date":      buy_date,
            "live_price":    round(price, 2),
            "value":         round(value, 2),
            "gain_loss":     gain_loss,
            "gain_loss_pct": gain_loss_pct,
        })

    holdings.sort(key=lambda x: x["value"], reverse=True)
    total_value = round(equity_total + portfolio["cash"], 2)

    return {
        "user_id":         user_id,
        "cash":            portfolio["cash"],
        "holdings":        holdings,
        "equity_value":    round(equity_total, 2),
        "total_value":     total_value,
        "total_gain_loss": round(total_gain_loss, 2),
    }


# ── POST /api/analyze ─────────────────────────────────────────────────────────
@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Run multi-agent analysis.
    mode="goal"     → uses the user-supplied goal string
    mode="feedback" → uses a predefined portfolio health review prompt
    """
    if req.mode == "goal" and not req.goal:
        raise HTTPException(status_code=400, detail="goal is required when mode is 'goal'.")

    effective_goal = _FEEDBACK_GOAL if req.mode == "feedback" else req.goal
    # Guardrail 1 : Evaluate if the goal is relevant and adheres to portfolio queries 
    is_valid, error_msg = validate_goal(effective_goal)
    
    if not is_valid:
        return {"decision_summary": error_msg, "risk_approved": False, "proposed_trades": [], "tool_calls_log": []}

    db = SessionLocal()
    try:
        crud.get_or_create_user(db, req.user_id)

        result = await run_analysis(effective_goal, req.user_id, _mcp_holder, req.mode)

        crud.save_analysis_session(
            db,
            session_id=req.session_id,
            user_id=req.user_id,
            mode=req.mode,
            goal=effective_goal,
            decision_summary=result["decision_summary"],
            risk_approved=result["risk_approved"],
            proposed_trades=result["proposed_trades"],
            retry_count=result["retry_count"],
        )

        proposed_rows = crud.save_proposed_trades(
            db, req.session_id, req.user_id, result["proposed_trades"]
        )

        risk_status = "APPROVED" if result["risk_approved"] else "REJECTED"
        risk_note   = result.get("risk_note", "")

        tool_calls_log = result.get("tool_calls_log", [])

        if not result["risk_approved"]:
            return {
                "session_id": req.session_id,
                "mode": req.mode,
                "status": risk_status,
                "decision_summary": result["decision_summary"],
                "risk_note": risk_note,
                "risk_approved": False,
                "retry_count": result["retry_count"],
                "proposed_trades": [],
                "tool_calls_log": tool_calls_log,
            }

        # Approved, no trades — portfolio healthy or goal already met
        if result.get("no_trade_reason") == "analyst_no_trade":
            return {
                "session_id": req.session_id,
                "mode": req.mode,
                "status": risk_status,
                "decision_summary": result["decision_summary"],
                "risk_note": risk_note,
                "risk_approved": True,
                "retry_count": result["retry_count"],
                "proposed_trades": [],
                "tool_calls_log": tool_calls_log,
            }

        # Approved with trades
        return {
            "session_id": req.session_id,
            "mode": req.mode,
            "status": risk_status,
            "decision_summary": result["decision_summary"],
            "risk_note": risk_note,
            "risk_approved": True,
            "retry_count": result["retry_count"],
            "proposed_trades": proposed_rows,
            "tool_calls_log": tool_calls_log,
        }
    finally:
        db.close()


# ── POST /api/execute ─────────────────────────────────────────────────────────
@router.post("/execute")
async def execute(req: ExecuteRequest):
    """
    Manual approval gate. Fetches live price per trade before recording.
    """
    db = SessionLocal()
    try:
        session_data = crud.get_analysis_session(db, req.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found.")
        if session_data.executed:
            raise HTTPException(status_code=400, detail="Trades already executed for this session.")

        if not req.approved:
            crud.mark_session_executed(db, req.session_id)
            crud.reject_proposed_trades(db, req.session_id)
            return {
                "status": "rejected",
                "message": "Trade plan rejected. All proposed trades marked as rejected.",
            }

        trades = session_data.proposed_trades or []
        if not trades:
            raise HTTPException(status_code=400, detail="No structured trades found in plan.")

        mcp_session = _mcp_holder["session"]
        trade_results = []

        for trade in trades:
            price_result = await mcp_session.call_tool(
                "get_live_price",
                {"ticker": trade["ticker"], "user_id": req.user_id},
            )
            raw_price = price_result.content[0].text
            if raw_price.startswith("ERROR"):
                raise HTTPException(status_code=502, detail=f"Price unavailable for {trade['ticker']}: {raw_price}")
            live_price = float(raw_price)
            print(f"[EXECUTE] {trade['ticker']} live price: ${live_price}")

            result = await mcp_session.call_tool(
                "record_trade",
                {
                    "user_id": req.user_id,
                    "session_id": req.session_id,
                    "ticker": trade["ticker"],
                    "side": trade["side"],
                    "qty": int(trade["qty"]),
                    "price": live_price,
                },
            )
            result_data = json.loads(result.content[0].text)
            if result_data.get("status") == "ERROR":
                raise HTTPException(
                    status_code=400,
                    detail=f"Trade failed for {trade['ticker']}: {result_data.get('reason', 'Unknown error')}",
                )
            trade_results.append({
                "trade_id": result_data.get("trade_id"),
                "ticker": trade["ticker"],
                "side": trade["side"],
                "qty": int(trade["qty"]),
                "proposed_price": trade.get("price"),
                "executed_price": live_price,
                "total_value": result_data.get("total_value"),
                "status": result_data.get("status"),
            })

        # Mark session executed only after ALL trades complete successfully
        crud.mark_session_executed(db, req.session_id)

        # Open a fresh session so we read the portfolio AFTER MCP commits are visible
        fresh_db = SessionLocal()
        try:
            updated_portfolio = crud.get_portfolio(fresh_db, req.user_id)
        finally:
            fresh_db.close()

        return {
            "status": "executed",
            "trade_results": trade_results,
            "updated_portfolio": updated_portfolio,
        }
    finally:
        db.close()

def validate_goal(goal: str) -> tuple[bool, str]:
    # Too short
    if len(goal.strip()) < MIN_GOAL_LENGTH:
        return False, "Please describe your portfolio goal in more detail"
    
    # Too long
    if len(goal) > MAX_GOAL_LENGTH:
        return False, "Goal is too long. Please keep it under 500 characters"
    
    # Blocked patterns - prompt injection / irrelevant
    goal_lower = goal.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in goal_lower:
            return False, "Please provide a valid portfolio management goal"
    
    # No financial keywords
    if not any(kw in goal_lower for kw in PORTFOLIO_KEYWORDS):
        return False, "I can only help with portfolio management goals"
    
    return True, ""