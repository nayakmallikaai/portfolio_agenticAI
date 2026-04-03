"""
Reset eval_user to the seed portfolio state without touching any other user.

Use this before running the eval suite against a live or staging environment
where eval_user's portfolio may have drifted from prior eval runs.

In CI the Postgres service container is always fresh, so this is not needed there.

Usage:
    python -m db.reset_eval_user                 # resets eval_user
    python -m db.reset_eval_user --user my_user  # resets a different user
"""
import argparse
from datetime import datetime, timezone, timedelta
from db.engine import SessionLocal
from db.models import TradeHistory, AnalysisSession, Portfolio, User

SEED_BUY_DATE = datetime.now(timezone.utc) - timedelta(days=180)

EVAL_USER_ID  = "eval_user"
SEED_CASH     = 5_000.0
SEED_HOLDINGS = {
    "AAPL": {"qty": 10, "buy_price": 226.00},
    "MSFT": {"qty":  5, "buy_price": 435.00},
    "JPM":  {"qty": 15, "buy_price": 228.00},
}


def reset_user(user_id: str) -> None:
    db = SessionLocal()
    try:
        # 1. Wipe trade history for this user
        trades = (
            db.query(TradeHistory)
            .filter(TradeHistory.user_id == user_id)
            .delete()
        )

        # 2. Wipe analysis sessions for this user
        sessions = (
            db.query(AnalysisSession)
            .filter(AnalysisSession.user_id == user_id)
            .delete()
        )

        # 3. Wipe portfolio rows for this user
        holdings = (
            db.query(Portfolio)
            .filter(Portfolio.user_id == user_id)
            .delete()
        )

        # 4. Reset or create the user row with seed cash
        user = db.query(User).filter(User.user_id == user_id).first()
        if user:
            user.cash = SEED_CASH
        else:
            user = User(user_id=user_id, cash=SEED_CASH)
            db.add(user)

        db.flush()

        # 5. Re-insert seed holdings with buy prices
        for ticker, info in SEED_HOLDINGS.items():
            db.add(Portfolio(
                user_id=user_id,
                ticker=ticker,
                quantity=info["qty"],
                buy_price=info["buy_price"],
                buy_date=SEED_BUY_DATE,
            ))

        db.commit()
        print(
            f"Reset '{user_id}' — "
            f"cleared: trades={trades}, sessions={sessions}, holdings={holdings}\n"
            f"Reseeded: {SEED_HOLDINGS}, cash=${SEED_CASH:,.0f}"
        )

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=EVAL_USER_ID)
    args = parser.parse_args()
    reset_user(args.user)
