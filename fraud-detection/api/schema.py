"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TransactionRequest(BaseModel):
    """Incoming transaction for fraud scoring."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    transaction_amt: float = Field(..., gt=0, description="Transaction amount in USD")
    product_cd: str = Field(..., description="Product code (W, H, C, S, R)")
    card_type: Optional[str] = Field(None, description="Card type: credit / debit")
    addr1: Optional[float] = Field(None, description="Billing address postal code")
    p_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    transaction_dt: float = Field(..., description="Seconds elapsed from reference date")
    card1: Optional[float] = Field(None, description="Card issuer numeric code")
    card2: Optional[float] = Field(None, description="Card sub-type numeric code")
    dist1: Optional[float] = Field(None, description="Distance between billing/shipping address")
    c1: Optional[float] = Field(None, description="Counting feature C1 (obfuscated)")
    c2: Optional[float] = Field(None, description="Counting feature C2 (obfuscated)")
    d1: Optional[float] = Field(None, description="Timedelta feature D1 (days)")
    d15: Optional[float] = Field(None, description="Timedelta feature D15 (days)")
    v258: Optional[float] = Field(None, description="Vesta engineered feature V258")
    v308: Optional[float] = Field(None, description="Vesta engineered feature V308")

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "txn_001",
                "transaction_amt": 149.99,
                "product_cd": "W",
                "card_type": "credit",
                "addr1": 299.0,
                "p_emaildomain": "gmail.com",
                "transaction_dt": 86400.0,
                "card1": 9500.0,
                "card2": 111.0,
                "dist1": 0.0,
                "c1": 1.0,
                "c2": 1.0,
                "d1": 14.0,
                "d15": 120.0,
                "v258": 1.0,
                "v308": 0.0,
            }
        }
    }


class FeatureImportance(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_fraud_risk" | "decreases_fraud_risk"


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}   # silence model_version warning

    transaction_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    fraud_label: bool
    risk_tier: str  # LOW | MEDIUM | HIGH | CRITICAL
    top_3_features: list[FeatureImportance]
    model_version: str
    prediction_timestamp: datetime


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}   # silence model_loaded warning

    status: str
    model_loaded: bool
    version: str