# 🔍 Real-Time Financial Fraud Detection Engine

> XGBoost + Isolation Forest ensemble | SHAP explainability | JWT-secured FastAPI | PostgreSQL audit logging | Docker + CI/CD

[![CI](https://github.com/YOUR_USERNAME/fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/fraud-detection/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT REQUEST                           │
└────────────────────────────┬────────────────────────────────────┘
                             │  POST /predict  (Bearer JWT)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Microservice                        │
│                                                                 │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────────┐  │
│  │  JWT Auth    │──▶│  Pydantic Schema │──▶│ Preprocessing  │  │
│  │  (HS256)     │   │  Validation      │   │ Pipeline       │  │
│  └──────────────┘   └──────────────────┘   └───────┬────────┘  │
│                                                     │           │
│  ┌──────────────────────────────────────────────────▼────────┐  │
│  │                   Ensemble Inference                      │  │
│  │   XGBoost (75%) ──┐                                       │  │
│  │                   ├──▶  fraud_probability [0,1]           │  │
│  │   IsoForest (25%) ┘                                       │  │
│  │                                                           │  │
│  │   SHAP TreeExplainer ──▶  top_3_features                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │              PostgreSQL Audit Logging                     │  │
│  │   transactions │ predictions (model_version) │ audit_log  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

Batch Scorer (offline)
  CSV ──▶ batch_scoring.py ──▶ results CSV + DB
```

---

## Model Metrics

> Run `notebooks/model_training.ipynb` on the IEEE-CIS dataset and fill in your actual numbers below.

| Metric         | Value           |
|----------------|-----------------|
| **AUC-ROC**    | `UPDATE AFTER BUILD` |
| Precision      | `UPDATE AFTER BUILD` |
| Recall         | `UPDATE AFTER BUILD` |
| F1 Score       | `UPDATE AFTER BUILD` |
| Training rows  | 590,540          |
| Fraud rate     | ~3.5%            |
| SMOTE balanced | Yes              |
| Features       | 22               |

---

## Quickstart

### Option A — Docker Compose (recommended)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection

# 2. Set secrets
cp .env.example .env
# Edit .env — set SECRET_KEY, API_PASSWORD, etc.

# 3. Spin up Postgres + API
docker compose up --build

# 4. Verify
curl http://localhost:8000/health
```

### Option B — Local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Postgres must be running (or set DATABASE_URL=sqlite:///./dev.db for quick test)
cp .env.example .env

uvicorn api.main:app --reload
```

---

## API Reference

### `POST /token` — Get JWT

```bash
curl -X POST http://localhost:8000/token \
  -d "username=analyst&password=changeme"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

---

### `GET /health` — Liveness probe

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### `POST /predict` — Score a transaction _(protected)_

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -d "username=analyst&password=changeme" | jq -r .access_token)

curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_demo_001",
    "transaction_amt": 849.99,
    "product_cd": "W",
    "card_type": "credit",
    "addr1": 299.0,
    "p_emaildomain": "anonymous.com",
    "transaction_dt": 86400.0,
    "card1": 9500.0,
    "card2": 111.0,
    "dist1": 320.0,
    "c1": 1.0,
    "c2": 1.0,
    "d1": 0.0,
    "d15": 0.0,
    "v258": 0.0,
    "v308": 1.0
  }'
```

**Response:**
```json
{
  "transaction_id": "txn_demo_001",
  "fraud_probability": 0.823,
  "fraud_label": true,
  "risk_tier": "CRITICAL",
  "top_3_features": [
    {
      "feature": "high_risk_email_domain",
      "shap_value": 0.412,
      "direction": "increases_fraud_risk"
    },
    {
      "feature": "dist1_log",
      "shap_value": 0.298,
      "direction": "increases_fraud_risk"
    },
    {
      "feature": "log_transaction_amt",
      "shap_value": 0.187,
      "direction": "increases_fraud_risk"
    }
  ],
  "model_version": "1.0.0",
  "prediction_timestamp": "2024-11-12T14:32:01.124Z"
}
```

**Risk tiers:**

| Tier       | fraud_probability |
|------------|-------------------|
| LOW        | < 0.20            |
| MEDIUM     | 0.20 – 0.49       |
| HIGH       | 0.50 – 0.79       |
| CRITICAL   | ≥ 0.80            |

---

### `GET /predictions` — Recent history _(protected)_

```bash
curl http://localhost:8000/predictions?limit=5 \
  -H "Authorization: Bearer $TOKEN"
```

---

## Testing

```bash
# Run all tests
pytest tests/test_api.py -v

# The 4 JPMC HireVue tests:
pytest tests/test_api.py::test_predict_returns_200_with_valid_token   -v
pytest tests/test_api.py::test_fraud_probability_in_valid_range       -v
pytest tests/test_api.py::test_missing_required_field_returns_422     -v
pytest tests/test_api.py::test_no_token_returns_401                   -v
```

Expected output:
```
tests/test_api.py::test_predict_returns_200_with_valid_token    PASSED
tests/test_api.py::test_fraud_probability_in_valid_range        PASSED
tests/test_api.py::test_missing_required_field_returns_422      PASSED
tests/test_api.py::test_no_token_returns_401                    PASSED
tests/test_api.py::test_health_endpoint_no_auth                 PASSED
tests/test_api.py::test_token_with_bad_credentials_returns_401  PASSED
tests/test_api.py::test_response_schema_has_top_3_features      PASSED
tests/test_api.py::test_risk_tier_is_valid_category             PASSED
tests/test_api.py::test_high_amount_flagged_higher_risk         PASSED
tests/test_api.py::test_invalid_amount_zero_returns_422         PASSED
tests/test_api.py::test_transaction_id_in_response_matches_request PASSED
```

---

## Batch Scorer

```bash
# Score a full CSV of historical transactions
python batch/batch_scoring.py --input data/raw/train_transaction.csv

# Also write to PostgreSQL
python batch/batch_scoring.py --input data/raw/train_transaction.csv --db
```

Output: `results/batch_results_<timestamp>.csv`

```
transaction_id, fraud_probability, fraud_label, risk_tier,
top_feature_1, top_feature_2, top_feature_3, model_version, prediction_timestamp
```

---

## Training the Model

1. Download from Kaggle:
   ```
   https://www.kaggle.com/competitions/ieee-fraud-detection/data
   ```
   Save `train_transaction.csv` → `data/raw/`

2. Run EDA:
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

3. Run training (produces your real AUC-ROC):
   ```bash
   jupyter notebook notebooks/model_training.ipynb
   ```

4. Update `api/preprocessing.py` with the printed `TRAIN_AMT_MEAN` / `TRAIN_AMT_STD`.

5. Update this README's metrics table with your real numbers.

---

## CI/CD Pipeline

```
Push to any branch
  │
  ├── lint     (ruff + mypy)
  │
  ├── test     (pytest — no Postgres needed, uses demo model)
  │
  └── docker build
        │
        └── (main branch only)
              │
              └── deploy → Render
```

GitHub Actions workflow: `.github/workflows/ci.yml`

---

## Deployment (Render)

1. Push to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com) → connect repo.
3. Set **environment variables** in Render dashboard (**never in code**):
   - `SECRET_KEY` — `openssl rand -hex 32`
   - `API_PASSWORD`
   - `DATABASE_URL` — from Render Postgres service
4. Set `RENDER_DEPLOY_HOOK_URL` as a GitHub Actions secret.
5. Every merge to `main` auto-deploys.

---

## Project Structure

```
fraud-detection/
├── api/
│   ├── __init__.py
│   ├── auth.py           JWT token creation + get_current_user dependency
│   ├── config.py         Pydantic settings (env-driven)
│   ├── database.py       SQLAlchemy ORM: transactions, predictions, audit_log
│   ├── main.py           FastAPI app: /token, /health, /predict, /predictions
│   ├── model_loader.py   Singleton model load + ensemble predict + SHAP
│   ├── preprocessing.py  22-feature engineering pipeline
│   └── schema.py         Pydantic request/response models
├── batch/
│   └── batch_scoring.py  Offline CSV scorer → results CSV + DB
├── notebooks/
│   ├── eda.ipynb          Class imbalance, missing values, distributions
│   └── model_training.ipynb  SMOTE, XGBoost, IsoForest, SHAP, artefact export
├── tests/
│   └── test_api.py       11 pytest tests (4 JPMC HireVue + extras)
├── model/                fraud_model.pkl + scaler.pkl (gitignored)
├── data/raw/             train_transaction.csv (gitignored)
├── .github/workflows/
│   └── ci.yml            lint → test → docker build → deploy
├── conftest.py           Shared pytest fixtures
├── Dockerfile            python:3.11-slim, non-root user, healthcheck
├── docker-compose.yml    API + Postgres for local dev
├── render.yaml           Render deploy manifest
├── requirements.txt
└── .env.example
```

---

## Engineered Features (22)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `log_transaction_amt` | log1p of amount — reduces right skew |
| 2 | `amt_zscore` | z-score vs training population |
| 3 | `time_of_day_bucket` | 4-hour bucket (0–5) |
| 4 | `is_night_txn` | 1 if 22:00–05:59 |
| 5 | `is_weekend` | 1 if Saturday/Sunday |
| 6 | `days_since_ref` | days elapsed from reference date |
| 7 | `velocity_1h_flag` | same-card transaction count in 1 hr |
| 8 | `high_risk_email_domain` | anonymous.com, mail.com, etc. |
| 9 | `is_credit` | card type binary |
| 10 | `product_cd_encoded` | label-encoded product code |
| 11 | `addr1_missing` | missingness indicator |
| 12 | `dist1_log` | log1p of billing–shipping distance |
| 13 | `dist1_missing` | missingness indicator |
| 14 | `c_ratio` | C1 / (C2 + 1) |
| 15 | `d1_missing` | missingness indicator |
| 16 | `d15_missing` | missingness indicator |
| 17 | `card1_norm` | card1 / max(card1_train) |
| 18 | `card2_norm` | card2 / max(card2_train) |
| 19 | `v258_flag` | V258 > 0 |
| 20 | `v308_flag` | V308 > 0 |
| 21 | `amt_x_c1` | interaction: amount × C1 |
| 22 | `amt_bin` | amount quantile bucket (0–4) |

---

## Security Notes

- JWT tokens expire after 60 minutes (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)
- All secrets loaded from environment variables — never hard-coded
- Docker image runs as non-root `appuser`
- All endpoints except `/health` and `/token` require authentication
- Pydantic validates all inputs; invalid payloads return `422 Unprocessable Entity`
