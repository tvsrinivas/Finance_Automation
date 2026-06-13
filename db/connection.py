"""
Database connection — Neon PostgreSQL via SQLAlchemy.
Uses connection pooling suitable for serverless Neon.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    url = os.getenv("NEON_DATABASE_URL")
    if not url:
        raise ValueError("NEON_DATABASE_URL not set in environment")
    # SQLAlchemy needs postgresql:// not postgres://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


# Engine — Neon works best with small pool + pre-ping
engine = create_engine(
    get_database_url(),
    pool_size=3,
    max_overflow=2,
    pool_pre_ping=True,       # test connection before use
    pool_recycle=300,         # recycle connections every 5 min
    connect_args={"sslmode": "require"},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection() -> bool:
    """Test that the database is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Neon database connection OK")
        return True
    except Exception as e:
        logger.error(f"Neon database connection failed: {e}")
        return False
