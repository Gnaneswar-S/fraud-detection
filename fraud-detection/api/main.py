"""
Real-Time Financial Fraud Detection API  — Elite Edition
=========================================================

Endpoints
---------
Public (no auth):
  GET  /health              liveness probe
  GET  /metrics             Prometheus text metrics

Auth:
  POST /token               obtain JWT Bearer token

Inference (JWT required):
  POST /predict             score transaction → fraud_probability + SHAP top-3
  POST /explain             full SHAP breakdown for every feature
  GET  /predictions         recent prediction history

Ops (JWT required):
  GET  /monitoring/summary  model drift report
  GET  /model/info          loaded model metadata

Elite features
--------------
  Rate limiting        slowapi — 60 req/min on /predict, 30 on /explain
  Prometheus metrics   /metrics endpoint (request count, latency, fraud rate)
  /explain endpoint    full per-feature SHAP with raw values
  /model/info          version, uptime, cache size, mode (production | demo)
  Transaction cache    duplicate txn_id cached 60 s → <1 ms response
  Security headers     X-Frame-Options, nosniff, Cache-Control: no-store
  Structured logging   method, path, status, latency, req_id on every call

NOTE: Do NOT add 'from __future__ import annotations' to this file.
FastAPI + slowapi + Pydantic v2 require that type annotations are
evaluated eagerly at import time, not as lazy strings.
"""
import json
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

# ── Rate limiting (applied via middleware, not per-route decorators) ──────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── Prometheus ────────────────────────────────────────────────────────────
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
)

from api.auth import authenticate_user, create_access_token, get_current_user
from api.config import settings
from api.database import (
    AuditLog, PredictionRecord, TransactionRecord,
    create_tables, get_db,
)
from api.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from api.model_loader import is_loaded, load_models, predict, explain, get_model_info
from fastapi.middleware.cors import CORSMiddleware

from api.schema import (
    HealthResponse, PredictionResponse, TokenResponse, TransactionRequest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metric definitions
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
FRAUD_SCORE = Histogram(
    "fraud_api_fraud_probability",
    "Distribution of fraud probability scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
FRAUD_FLAGS = Counter(
    "fraud_api_fraud_flags_total",
    "Total transactions flagged as fraud",
    ["risk_tier"],
)
MODEL_UPTIME = Gauge(
    "fraud_api_model_uptime_seconds",
    "Seconds since model was loaded",
)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200/minute"],   # global default
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — creating DB tables and loading models…")
    try:
        create_tables()
    except Exception as exc:
        logger.warning("DB init skipped (not available in test/demo): %s", exc)
    load_models()
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection Engine",
    description=(
        "XGBoost + Isolation Forest ensemble · SHAP explainability · "
        "JWT-secured · PostgreSQL audit logging · Prometheus metrics · "
        "Rate limiting · Transaction deduplication cache"
    ),
    version=settings.MODEL_VERSION,
    lifespan=lifespan,
)

# Register rate limiter state and middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],          # tighten to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# ---------------------------------------------------------------------------
# Serve frontend dashboard — GET / returns the HTML dashboard
# ---------------------------------------------------------------------------
import os
from pathlib import Path

def _find_dashboard():
    # Check multiple locations — works regardless of how uvicorn is started
    for candidate in [
        Path.cwd() / "frontend" / "index.html",                    # run from project root ✓
        Path(__file__).parent.parent / "frontend" / "index.html",  # relative to api/main.py
        Path(__file__).parent / "frontend" / "index.html",         # fallback
    ]:
        if candidate.exists():
            return candidate
    return None

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    dashboard = _find_dashboard()
    if dashboard:
        logger.info("Serving dashboard from %s", dashboard)
        return FileResponse(str(dashboard), media_type="text/html")
    # Show helpful debug info instead of blank page
    cwd = Path.cwd()
    return HTMLResponse(f"""
        <html><body style="font-family:sans-serif;padding:40px;background:#0f1117;color:#e8eaf6">
        <h2 style="color:#ff4d6d">Dashboard not found</h2>
        <p>uvicorn is running from: <code>{cwd}</code></p>
        <p>Looking for: <code>{cwd / 'frontend' / 'index.html'}</code></p>
        <p><b>Fix:</b> Make sure <code>frontend\index.html</code> exists, then restart uvicorn.</p>
        <p>API is working: <a href="/docs" style="color:#6c63ff">/docs</a> | 
           <a href="/health" style="color:#6c63ff">/health</a></p>
        </body></html>""", status_code=200)

_frontend_dir = (_find_dashboard() or Path("frontend/index.html")).parent
if _frontend_dir.exists():
    try:
        app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Prometheus scrape endpoint — no auth, standard practice
# ---------------------------------------------------------------------------
@app.get("/metrics", tags=["Ops"], include_in_schema=False)
async def prometheus_metrics():
    """Prometheus text exposition format. Scrape every 15 s."""
    info = get_model_info()
    MODEL_UPTIME.set(info.get("uptime_sec", 0))
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.post("/token", response_model=TokenResponse, tags=["Auth"])
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """
    Exchange `username` + `password` for a JWT Bearer token.

    ```bash
    curl -X POST http://localhost:8000/token \\
         -d "username=analyst&password=changeme"
    ```
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user["username"]})
    return TokenResponse(access_token=token)


# ---------------------------------------------------------------------------
# Health — public, used by Render / Docker healthcheck
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    """Liveness probe — no authentication required."""
    return HealthResponse(
        status="ok",
        model_loaded=is_loaded(),
        version=settings.MODEL_VERSION,
    )


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------
@app.get("/model/info", tags=["Ops"])
async def model_info(current_user: dict = Depends(get_current_user)):
    """
    Metadata about the currently loaded model artefact.

    Returns mode (production | demo), version, uptime seconds, cache size.
    """
    return get_model_info()


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@limiter.limit("60/minute")
async def predict_fraud(
    request: Request,
    txn: TransactionRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Score a single transaction for fraud probability.

    **Response fields**
    - `fraud_probability` — ensemble score in [0, 1]
    - `fraud_label`       — True if score >= 0.5
    - `risk_tier`         — LOW / MEDIUM / HIGH / CRITICAL
    - `top_3_features`    — SHAP-ranked top drivers with direction

    **Risk tiers**

    | Tier     | Probability |
    |----------|-------------|
    | LOW      | < 0.20      |
    | MEDIUM   | 0.20–0.49   |
    | HIGH     | 0.50–0.79   |
    | CRITICAL | >= 0.80     |
    """
    t0 = time.perf_counter()
    txn_dict = txn.model_dump()

    fraud_prob, fraud_label, risk_tier, top_features = predict(txn_dict)
    latency_ms = (time.perf_counter() - t0) * 1000
    now = datetime.now(timezone.utc)

    # ── Prometheus ────────────────────────────────────────────────────
    REQUEST_COUNT.labels("POST", "/predict", "200").inc()
    REQUEST_LATENCY.labels("/predict").observe(latency_ms / 1000)
    FRAUD_SCORE.observe(fraud_prob)
    if fraud_label:
        FRAUD_FLAGS.labels(risk_tier).inc()

    # ── Persist to DB ─────────────────────────────────────────────────
    try:
        db.add(TransactionRecord(
            transaction_id=txn.transaction_id,
            transaction_amt=txn.transaction_amt,
            product_cd=txn.product_cd,
            card_type=txn.card_type,
            p_emaildomain=txn.p_emaildomain,
            transaction_dt=txn.transaction_dt,
        ))
        db.add(PredictionRecord(
            transaction_id=txn.transaction_id,
            fraud_probability=fraud_prob,
            fraud_label=fraud_label,
            risk_tier=risk_tier,
            top_features_json=json.dumps([f.model_dump() for f in top_features]),
            model_version=settings.MODEL_VERSION,
            prediction_timestamp=now,
        ))
        db.add(AuditLog(
            endpoint="/predict",
            username=current_user["username"],
            transaction_id=txn.transaction_id,
            http_status=200,
            latency_ms=round(latency_ms, 2),
        ))
        db.commit()
    except Exception as exc:
        logger.warning("DB write skipped (demo/test mode): %s", exc)
        db.rollback()

    return PredictionResponse(
        transaction_id=txn.transaction_id,
        fraud_probability=fraud_prob,
        fraud_label=fraud_label,
        risk_tier=risk_tier,
        top_3_features=top_features,
        model_version=settings.MODEL_VERSION,
        prediction_timestamp=now,
    )


# ---------------------------------------------------------------------------
# Explain — full per-feature SHAP breakdown
# ---------------------------------------------------------------------------
@app.post("/explain", tags=["Inference"])
@limiter.limit("30/minute")
async def explain_prediction(
    request: Request,
    txn: TransactionRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Return a full SHAP explanation for **every** feature (all 22, not just top 3).

    Response is sorted by absolute SHAP impact descending.
    Use this when an analyst needs to justify a fraud decision to a customer.

    ```bash
    curl -X POST http://localhost:8000/explain \\
      -H "Authorization: Bearer $TOKEN" \\
      -H "Content-Type: application/json" \\
      -d '{"transaction_id":"txn_001","transaction_amt":849.99,...}'
    ```
    """
    txn_dict = txn.model_dump()
    fraud_prob, fraud_label, risk_tier, _ = predict(txn_dict)
    full_explanation = explain(txn_dict)

    return {
        "transaction_id":    txn.transaction_id,
        "fraud_probability": fraud_prob,
        "fraud_label":       fraud_label,
        "risk_tier":         risk_tier,
        **full_explanation,
    }


# ---------------------------------------------------------------------------
# Monitoring / drift
# ---------------------------------------------------------------------------
@app.get("/monitoring/summary", tags=["Ops"])
async def monitoring_summary(
    window: int = 1000,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Model drift report over the last `window` predictions.

    Checks fraud rate, score distribution (KS statistic), score mean drift,
    and P95 latency.  Returns HEALTHY or DEGRADED with alert details.
    """
    from api.monitoring import (
        compute_drift_report,
        load_recent_latencies_from_db,
        load_recent_scores_from_db,
    )
    scores    = load_recent_scores_from_db(window)
    latencies = load_recent_latencies_from_db(window)
    return compute_drift_report(scores, latencies)


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------
@app.get("/predictions", tags=["Inference"])
async def get_predictions(
    limit: int = 20,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the most recent `limit` predictions from the DB."""
    try:
        rows = (
            db.query(PredictionRecord)
            .order_by(PredictionRecord.prediction_timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "transaction_id":       r.transaction_id,
                "fraud_probability":    r.fraud_probability,
                "fraud_label":          r.fraud_label,
                "risk_tier":            r.risk_tier,
                "model_version":        r.model_version,
                "prediction_timestamp": r.prediction_timestamp,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning("DB read skipped: %s", exc)
        return []