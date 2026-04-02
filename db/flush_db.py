"""
Flush all user data from the database.
Deletes in FK-safe order: trade_history → analysis_sessions → portfolios → users.

Usage:
    python -m db.flush_db
"""
from db.engine import SessionLocal
from db.models import TradeHistory, AnalysisSession, Portfolio, User


def flush_all() -> None:
    db = SessionLocal()
    try:
        trades    = db.query(TradeHistory).delete()
        sessions  = db.query(AnalysisSession).delete()
        portfolios = db.query(Portfolio).delete()
        users     = db.query(User).delete()
        db.commit()
        print(
            f"Flushed — trades: {trades}, sessions: {sessions}, "
            f"portfolios: {portfolios}, users: {users}"
        )
        print("Next new user will be seeded with AAPL×10, MSFT×5, JPM×15, $5,000 cash.")
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    flush_all()
