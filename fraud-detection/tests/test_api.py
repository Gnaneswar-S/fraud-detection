"""
pytest test suite — Fraud Detection API
=========================================
The 4 JPMC HireVue core tests:
  (a) POST /predict returns 200 with valid token
  (b) fraud_probability is between 0 and 1
  (c) missing required field returns 422
  (d) no token returns 401

Plus 7 additional coverage tests.

Fixtures come from conftest.py (session-scoped client, valid_token, etc.)
No PostgreSQL required — DB dependency is overridden with a MagicMock.

Run:
    pytest tests/test_api.py -v
"""
from __future__ import annotations

import pytest

# All fixtures (client, valid_token, auth_headers, valid_txn) are
# injected automatically from conftest.py.

VALID_TXN = {
    "transaction_id": "test_txn_001",
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


# ─────────────────────────────────────────────────────────────────────────────
# (a) POST /predict returns 200 with valid token
# ─────────────────────────────────────────────────────────────────────────────
def test_predict_returns_200_with_valid_token(client, valid_token):
    headers = {"Authorization": f"Bearer {valid_token}"}
    resp = client.post("/predict", json=VALID_TXN, headers=headers)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


# ─────────────────────────────────────────────────────────────────────────────
# (b) fraud_probability is between 0 and 1
# ─────────────────────────────────────────────────────────────────────────────
def test_fraud_probability_in_valid_range(client, valid_token):
    headers = {"Authorization": f"Bearer {valid_token}"}
    resp = client.post("/predict", json=VALID_TXN, headers=headers)
    assert resp.status_code == 200
    prob = resp.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0, f"fraud_probability out of [0,1]: {prob}"


# ─────────────────────────────────────────────────────────────────────────────
# (c) missing required field returns 422
# ─────────────────────────────────────────────────────────────────────────────
def test_missing_required_field_returns_422(client, valid_token):
    headers = {"Authorization": f"Bearer {valid_token}"}
    bad_txn = {k: v for k, v in VALID_TXN.items() if k != "transaction_amt"}
    resp = client.post("/predict", json=bad_txn, headers=headers)
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"


# ─────────────────────────────────────────────────────────────────────────────
# (d) no token returns 401
# ─────────────────────────────────────────────────────────────────────────────
def test_no_token_returns_401(client):
    resp = client.post("/predict", json=VALID_TXN)   # no Authorization header
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text}"


# ─────────────────────────────────────────────────────────────────────────────
# Additional tests
# ─────────────────────────────────────────────────────────────────────────────

def test_health_endpoint_no_auth_required(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "version" in body


def test_token_endpoint_bad_credentials_returns_401(client):
    resp = client.post("/token", data={"username": "nobody", "password": "wrong"})
    assert resp.status_code == 401


def test_response_contains_top_3_shap_features(client, valid_token):
    headers = {"Authorization": f"Bearer {valid_token}"}
    resp = client.post("/predict", json=VALID_TXN, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "top_3_features" in body
    assert len(body["top_3_features"]) == 3
    for feat in body["top_3_features"]:
        assert "feature" in feat
        assert "shap_value" in feat
        assert feat["direction"] in ("increases_fraud_risk", "decreases_fraud_risk")


def test_risk_tier_is_valid_category(client, valid_token):
    headers = {"Authorization": f"Bearer {valid_token}"}
    resp = client.post("/predict", json=VALID_TXN, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["risk_tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def test_transaction_id_echoed_in_response(client, valid_token):
    headers = {"Authorization": f"Bearer {valid_token}"}
    txn = {**VALID_TXN, "transaction_id": "echo_me_123"}
    resp = client.post("/predict", json=txn, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["transaction_id"] == "echo_me_123"


def test_invalid_amount_zero_returns_422(client, valid_token):
    """transaction_amt has gt=0 constraint — zero must fail validation."""
    headers = {"Authorization": f"Bearer {valid_token}"}
    bad_txn = {**VALID_TXN, "transaction_id": "zero_amt", "transaction_amt": 0}
    resp = client.post("/predict", json=bad_txn, headers=headers)
    assert resp.status_code == 422


def test_high_amount_scores_at_least_as_high_as_low_amount(client, valid_token):
    """
    Demonstrates model monotonicity in demo mode:
    large transaction amount should not score lower than a $1 transaction.
    """
    headers = {"Authorization": f"Bearer {valid_token}"}
    small = client.post("/predict", json={**VALID_TXN, "transaction_id": "small", "transaction_amt": 1.0},    headers=headers)
    large = client.post("/predict", json={**VALID_TXN, "transaction_id": "large", "transaction_amt": 9999.0}, headers=headers)
    assert small.status_code == 200
    assert large.status_code == 200
    assert large.json()["fraud_probability"] >= small.json()["fraud_probability"]


def test_high_risk_email_domain_in_payload(client, valid_token):
    """anonymous.com is flagged as high-risk — should score >= clean email."""
    headers = {"Authorization": f"Bearer {valid_token}"}
    clean  = client.post("/predict", json={**VALID_TXN, "transaction_id": "clean",  "p_emaildomain": "gmail.com"},     headers=headers)
    risky  = client.post("/predict", json={**VALID_TXN, "transaction_id": "risky",  "p_emaildomain": "anonymous.com"}, headers=headers)
    assert clean.status_code == 200
    assert risky.status_code == 200
    assert risky.json()["fraud_probability"] >= clean.json()["fraud_probability"]
