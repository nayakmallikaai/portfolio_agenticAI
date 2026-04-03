import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

# Railway (and some other providers) emit "postgresql://" but SQLAlchemy 2.x
# requires the explicit driver scheme "postgresql+psycopg2://"
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def migrate_db() -> None:
    """Safe incremental migrations — adds columns that don't exist yet."""
    from sqlalchemy import text
    migrations = [
        "ALTER TABLE analysis_sessions ADD COLUMN IF NOT EXISTS mode VARCHAR(10) NOT NULL DEFAULT 'goal'",
        "ALTER TABLE portfolios ADD COLUMN IF NOT EXISTS buy_price FLOAT",
        "ALTER TABLE portfolios ADD COLUMN IF NOT EXISTS buy_date TIMESTAMP WITH TIME ZONE",
    ]
    with engine.connect() as conn:
        for stmt in migrations:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception as e:
                print(f"[MIGRATE] skipped: {e}")
