"""
All database operations. Each function takes a Session and returns plain Python objects.
"""
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy.orm import Session

from db.models import User, Portfolio, AnalysisSession, TradeHistory

_SEED_HOLDINGS = {"HDFC": 100, "RELIANCE": 50, "TCS": 20}
_SEED_CASH = 500_000.0


# ── User ──────────────────────────────────────────────────────────────────────

def get_or_create_user(db: Session, user_id: str) -> User:
    user = db.query(User).filter(User.user_id == user_id).first()
    if user:
        return user
    user = User(user_id=user_id, cash=_SEED_CASH)
    db.add(user)
    db.flush()
    for ticker, qty in _SEED_HOLDINGS.items():
        db.add(Portfolio(user_id=user_id, ticker=ticker, quantity=qty))
    db.commit()
    db.refresh(user)
    return user


# ── Portfolio ─────────────────────────────────────────────────────────────────

def get_portfolio(db: Session, user_id: str) -> dict:
    get_or_create_user(db, user_id)
    user = db.query(User).filter(User.user_id == user_id).first()
    rows = db.query(Portfolio).filter(Portfolio.user_id == user_id, Portfolio.quantity > 0).all()
    return {"holdings": {r.ticker: r.quantity for r in rows}, "cash": user.cash}


def _upsert_holding(db: Session, user_id: str, ticker: str, qty_delta: int) -> None:
    holding = (
        db.query(Portfolio)
        .filter(Portfolio.user_id == user_id, Portfolio.ticker == ticker)
        .first()
    )
    if holding:
        holding.quantity += qty_delta
    else:
        db.add(Portfolio(user_id=user_id, ticker=ticker, quantity=qty_delta))


# ── Analysis session ──────────────────────────────────────────────────────────

def save_analysis_session(
    db: Session, session_id: str, user_id: str, mode: str, goal: str,
    decision_summary: str, risk_approved: bool, proposed_trades: list, retry_count: int,
) -> AnalysisSession:
    row = AnalysisSession(
        session_id=session_id, user_id=user_id, mode=mode, goal=goal,
        decision_summary=decision_summary, risk_approved=risk_approved,
        proposed_trades=proposed_trades, retry_count=retry_count, executed=False,
    )
    db.add(row)
    db.commit()
    return row


def get_analysis_session(db: Session, session_id: str) -> Optional[AnalysisSession]:
    return db.query(AnalysisSession).filter(AnalysisSession.session_id == session_id).first()


def mark_session_executed(db: Session, session_id: str) -> None:
    row = db.query(AnalysisSession).filter(AnalysisSession.session_id == session_id).first()
    if row:
        row.executed = True
        db.commit()


# ── Trade history ─────────────────────────────────────────────────────────────

def save_proposed_trades(
    db: Session, session_id: str, user_id: str, trades: List[dict]
) -> List[dict]:
    """
    Called at analysis time. Inserts one TradeHistory row per trade with:
      proposed=True, accepted=None (pending), proposed_price from LLM output.
    Idempotent: skips rows that already exist for (session_id, ticker, side).
    Returns list of {trade_id, ticker, side, qty, proposed_price}.
    """
    results = []
    for trade in trades:
        side = trade["side"].upper()
        existing = (
            db.query(TradeHistory)
            .filter(
                TradeHistory.session_id == session_id,
                TradeHistory.ticker == trade["ticker"],
                TradeHistory.side == side,
            )
            .first()
        )
        if existing:
            results.append({
                "trade_id": existing.id,
                "ticker": existing.ticker,
                "side": existing.side,
                "qty": existing.qty,
                "proposed_price": existing.proposed_price,
                "accepted": existing.accepted,
            })
            continue

        row = TradeHistory(
            session_id=session_id,
            user_id=user_id,
            ticker=trade["ticker"],
            side=side,
            qty=int(trade["qty"]),
            proposed_price=float(trade.get("price", 0)) or None,
            proposed=True,
            accepted=None,   # pending user decision
        )
        db.add(row)
        db.flush()   # populate row.id before commit
        results.append({
            "trade_id": row.id,
            "ticker": row.ticker,
            "side": row.side,
            "qty": row.qty,
            "proposed_price": row.proposed_price,
            "accepted": row.accepted,
        })

    db.commit()
    return results


def record_trade(
    db: Session, session_id: str, user_id: str,
    ticker: str, side: str, qty: int, price: float,
) -> dict:
    """
    Called at execution time. Finds the existing proposed row and updates it in-place:
      accepted=True, executed_price=price, total_value, executed_at.
    Idempotency: if already accepted=True, returns ALREADY_EXECUTED with the id.
    """
    side = side.upper()

    row = (
        db.query(TradeHistory)
        .filter(
            TradeHistory.session_id == session_id,
            TradeHistory.ticker == ticker,
            TradeHistory.side == side,
        )
        .first()
    )

    # ── Already executed — return early (idempotency) ─────────────────────────
    if row and row.accepted is True:
        return {
            "status": "ALREADY_EXECUTED",
            "trade_id": row.id,
            "ticker": ticker,
            "side": side,
        }

    total = qty * price

    # ── Portfolio update ──────────────────────────────────────────────────────
    user = db.query(User).filter(User.user_id == user_id).first()
    if side == "BUY":
        if user.cash < total:
            return {"status": "ERROR", "trade_id": row.id if row else None, "reason": "Insufficient funds."}
        user.cash -= total
        _upsert_holding(db, user_id, ticker, qty)
    else:
        user.cash += total
        _upsert_holding(db, user_id, ticker, -qty)

    # ── Update existing proposed row OR insert if somehow missing ─────────────
    if row:
        row.executed_price = price
        row.total_value = total
        row.accepted = True
        row.executed_at = datetime.now(timezone.utc)
        db.commit()
        trade_id = row.id
    else:
        new_row = TradeHistory(
            session_id=session_id, user_id=user_id, ticker=ticker,
            side=side, qty=qty, executed_price=price,
            total_value=total, proposed=True, accepted=True,
            executed_at=datetime.now(timezone.utc),
        )
        db.add(new_row)
        db.commit()
        trade_id = new_row.id

    return {
        "status": "SUCCESS",
        "trade_id": trade_id,
        "ticker": ticker,
        "side": side,
        "qty": qty,
        "executed_price": price,
        "total_value": total,
    }


def reject_proposed_trades(db: Session, session_id: str) -> None:
    """Mark all pending proposed trades for a session as rejected."""
    rows = (
        db.query(TradeHistory)
        .filter(TradeHistory.session_id == session_id, TradeHistory.accepted.is_(None))
        .all()
    )
    for row in rows:
        row.accepted = False
    db.commit()
