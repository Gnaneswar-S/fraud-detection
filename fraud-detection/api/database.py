"""
SQLAlchemy ORM models and session factory.

Tables:
  - transactions  : raw request payload
  - predictions   : model output per transaction
  - audit_log     : every API call with user + timestamp
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.config import settings

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,          # reconnect on stale connections
    pool_size=5,
    max_overflow=10,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------
class TransactionRecord(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(64), unique=True, index=True, nullable=False)
    transaction_amt = Column(Float, nullable=False)
    product_cd = Column(String(8))
    card_type = Column(String(16))
    p_emaildomain = Column(String(64))
    transaction_dt = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(64), index=True, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    fraud_label = Column(Boolean, nullable=False)
    risk_tier = Column(String(16), nullable=False)
    top_features_json = Column(Text)          # JSON string of top 3 SHAP features
    model_version = Column(String(32), nullable=False)
    prediction_timestamp = Column(DateTime(timezone=True), nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String(128))
    username = Column(String(64))
    transaction_id = Column(String(64))
    http_status = Column(Integer)
    latency_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_tables() -> None:
    """Create all tables (idempotent — called at startup)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
