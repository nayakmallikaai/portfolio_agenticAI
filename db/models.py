from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON
from db.engine import Base


class User(Base):
    __tablename__ = "users"

    user_id:    Mapped[str]   = mapped_column(String, primary_key=True)
    cash:       Mapped[float] = mapped_column(Float, default=500000.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Portfolio(Base):
    __tablename__ = "portfolios"
    __table_args__ = (UniqueConstraint("user_id", "ticker", name="uq_user_ticker"),)

    id:         Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:    Mapped[str]   = mapped_column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    ticker:     Mapped[str]   = mapped_column(String(20), nullable=False)
    quantity:   Mapped[int]   = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    session_id:       Mapped[str]  = mapped_column(String, primary_key=True)
    user_id:          Mapped[str]  = mapped_column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    goal:             Mapped[str]  = mapped_column(Text, nullable=False)
    decision_summary: Mapped[str]  = mapped_column(Text, nullable=True)
    risk_approved:    Mapped[bool] = mapped_column(Boolean, default=False)
    proposed_trades:  Mapped[list] = mapped_column(JSON, default=list)
    retry_count:      Mapped[int]  = mapped_column(Integer, default=0)
    executed:         Mapped[bool] = mapped_column(Boolean, default=False)
    created_at:       Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class TradeHistory(Base):
    __tablename__ = "trade_history"
    # (session_id, ticker, side) is the idempotency key — one row per trade per session
    __table_args__ = (UniqueConstraint("session_id", "ticker", "side", name="uq_session_trade"),)

    id:             Mapped[int]            = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id:     Mapped[str]            = mapped_column(String, ForeignKey("analysis_sessions.session_id"), nullable=False, index=True)
    user_id:        Mapped[str]            = mapped_column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    ticker:         Mapped[str]            = mapped_column(String(20), nullable=False)
    side:           Mapped[str]            = mapped_column(String(4), nullable=False)       # BUY | SELL
    qty:            Mapped[int]            = mapped_column(Integer, nullable=False)

    # Two-phase pricing: proposed_price set at analysis time, executed_price set at execution
    proposed_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    executed_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_value:    Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # calculated at execution

    # Lifecycle flags
    proposed:       Mapped[bool]           = mapped_column(Boolean, default=True)          # always True
    accepted:       Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)         # None=pending, True=accepted, False=rejected

    created_at:     Mapped[datetime]       = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    executed_at:    Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
