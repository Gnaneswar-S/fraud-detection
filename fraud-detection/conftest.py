"""
pytest configuration — shared fixtures for the entire test suite.

DB is fully mocked — no PostgreSQL or SQLite file needed during tests.
TestClient is session-scoped (created once, reused across all tests).
"""
from __future__ import annotations

import unittest.mock as mock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def _patch_db():
    """Replace SQLAlchemy engine + session with MagicMocks for the whole run."""
    mock_db = mock.MagicMock()
    mock_db.add      = mock.MagicMock(return_value=None)
    mock_db.commit   = mock.MagicMock(return_value=None)
    mock_db.rollback = mock.MagicMock(return_value=None)
    mock_db.close    = mock.MagicMock(return_value=None)
    mock_db.query.return_value \
        .order_by.return_value \
        .limit.return_value \
        .all.return_value = []

    with mock.patch("api.database.create_engine"), \
         mock.patch("api.database.create_tables"), \
         mock.patch("api.database.SessionLocal", return_value=mock_db), \
         mock.patch("api.database.get_db", return_value=iter([mock_db])):
        yield mock_db


@pytest.fixture(scope="session")
def client(_patch_db):
    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="session")
def valid_token(client) -> str:
    resp = client.post("/token", data={"username": "analyst", "password": "changeme"})
    assert resp.status_code == 200, f"Token request failed: {resp.text}"
    return resp.json()["access_token"]


@pytest.fixture
def auth_headers(valid_token) -> dict:
    return {"Authorization": f"Bearer {valid_token}"}


@pytest.fixture
def valid_txn() -> dict:
    return {
        "transaction_id":  "fixture_txn_001",
        "transaction_amt": 149.99,
        "product_cd":      "W",
        "card_type":       "credit",
        "addr1":           299.0,
        "p_emaildomain":   "gmail.com",
        "transaction_dt":  86400.0,
        "card1":           9500.0,
        "card2":           111.0,
        "dist1":           0.0,
        "c1":              1.0,
        "c2":              1.0,
        "d1":              14.0,
        "d15":             120.0,
        "v258":            1.0,
        "v308":            0.0,
    }