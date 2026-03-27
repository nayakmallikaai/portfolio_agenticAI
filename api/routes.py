"""
API route handlers. Pure HTTP concerns only — no LLM or graph logic here.
"""
import json
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from api.schemas import AnalyzeRequest, ExecuteRequest
from agent.graph import run_analysis
from db.engine import SessionLocal
import db.crud as crud

router = APIRouter(prefix="/api")

_mcp_holder: Dict[str, Any] = {}


def set_mcp_holder(holder: Dict[str, Any]) -> None:
    global _mcp_holder
    _mcp_holder = holder


# ── POST /api/analyze ─────────────────────────────────────────────────────────
@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Run multi-agent analysis. Saves proposed trades to trade_history immediately
    (proposed=True, accepted=None). Returns trade IDs so the UI can track them.
    """
    db = SessionLocal()
    try:
        crud.get_or_create_user(db, req.user_id)

        result = await run_analysis(req.goal, req.user_id, _mcp_holder)

        crud.save_analysis_session(
            db,
            session_id=req.session_id,
            user_id=req.user_id,
            goal=req.goal,
            decision_summary=result["decision_summary"],
            risk_approved=result["risk_approved"],
            proposed_trades=result["proposed_trades"],
            retry_count=result["retry_count"],
        )

        # Save proposed trades to trade_history now, before user approves
        proposed_rows = crud.save_proposed_trades(
            db, req.session_id, req.user_id, result["proposed_trades"]
        )

        return {
            "session_id": req.session_id,
            "decision_summary": result["decision_summary"],
            "risk_approved": result["risk_approved"],
            "retry_count": result["retry_count"],
            "proposed_trades": proposed_rows,   # includes trade_id per row
        }
    finally:
        db.close()


# ── POST /api/execute ─────────────────────────────────────────────────────────
@router.post("/execute")
async def execute(req: ExecuteRequest):
    """
    Manual approval gate.
    - Reject: marks all proposed trades as accepted=False.
    - Approve: fetches live price per trade, updates trade row in-place (accepted=True).
    Returns trade_id for every trade.
    """
    db = SessionLocal()
    try:
        session_data = crud.get_analysis_session(db, req.session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found.")
        if session_data.executed:
            raise HTTPException(status_code=400, detail="Trades already executed for this session.")

        # Lock session immediately to prevent double execution
        crud.mark_session_executed(db, req.session_id)

        if not req.approved:
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
            # ── Fetch real-time price right before recording ──────────────────
            price_result = await mcp_session.call_tool(
                "get_live_price",
                {"ticker": trade["ticker"], "user_id": req.user_id},
            )
            live_price = float(price_result.content[0].text)
            print(f"[EXECUTE] {trade['ticker']} live price: ₹{live_price}")

            # ── Update the existing proposed row in trade_history ─────────────
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

        updated_portfolio = crud.get_portfolio(db, req.user_id)
        return {
            "status": "executed",
            "trade_results": trade_results,
            "updated_portfolio": updated_portfolio,
        }
    finally:
        db.close()
