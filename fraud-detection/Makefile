# ============================================================
# Fraud Detection Engine — Makefile
# ============================================================
# Usage:  make <target>
# Requires: python 3.11+, docker, docker compose

.PHONY: help install dev test lint fmt build up down train batch \
        migrate shell clean

PYTHON     = python3
PIP        = $(PYTHON) -m pip
PYTEST     = $(PYTHON) -m pytest
UVICORN    = $(PYTHON) -m uvicorn
DATA_CSV   = data/raw/train_transaction.csv

# ─── Help ────────────────────────────────────────────────────
help:        ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ─── Setup ───────────────────────────────────────────────────
install:     ## Install Python dependencies
	$(PIP) install -r requirements.txt

dev:         ## Run API locally with hot-reload (no Docker)
	@cp -n .env.example .env 2>/dev/null || true
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

# ─── Testing ─────────────────────────────────────────────────
test:        ## Run full pytest suite
	$(PYTEST) tests/test_api.py -v

test-4:      ## Run only the 4 JPMC HireVue tests
	$(PYTEST) tests/test_api.py \
	  -k "test_predict_returns_200 or test_fraud_probability or \
	      test_missing_required_field or test_no_token" \
	  -v

# ─── Code quality ────────────────────────────────────────────
lint:        ## Run ruff linter
	$(PYTHON) -m ruff check api/ batch/ tests/ model/

fmt:         ## Auto-format with ruff
	$(PYTHON) -m ruff format api/ batch/ tests/ model/

# ─── Docker ──────────────────────────────────────────────────
build:       ## Build Docker image
	docker build -t fraud-detection-api:latest .

up:          ## Start API + Postgres with docker compose
	docker compose up --build

down:        ## Stop all containers
	docker compose down -v

# ─── Model training ──────────────────────────────────────────
train:       ## Train model on full dataset (needs data/raw/train_transaction.csv)
	@test -f $(DATA_CSV) || (echo "Download dataset first: see README.md" && exit 1)
	$(PYTHON) model/train.py --data $(DATA_CSV)

train-quick: ## Quick training on 50K rows (for dev)
	@test -f $(DATA_CSV) || (echo "Download dataset first: see README.md" && exit 1)
	$(PYTHON) model/train.py --data $(DATA_CSV) --sample 50000

train-tune:  ## Full training with hyperparameter search
	@test -f $(DATA_CSV) || (echo "Download dataset first: see README.md" && exit 1)
	$(PYTHON) model/train.py --data $(DATA_CSV) --tune --n-iter 30

# ─── Batch scoring ───────────────────────────────────────────
batch:       ## Run batch scorer on full training CSV
	@test -f $(DATA_CSV) || (echo "Download dataset first: see README.md" && exit 1)
	$(PYTHON) batch/batch_scoring.py --input $(DATA_CSV)

# ─── Database ────────────────────────────────────────────────
migrate:     ## Run Alembic migrations (requires running Postgres)
	alembic upgrade head

migrate-gen: ## Auto-generate new migration (usage: make migrate-gen MSG="add column x")
	alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Roll back last migration
	alembic downgrade -1

# ─── Utilities ───────────────────────────────────────────────
token:       ## Get a JWT token from the local API
	@curl -s -X POST http://localhost:8000/token \
	  -d "username=analyst&password=changeme" | python3 -m json.tool

health:      ## Check API health
	@curl -s http://localhost:8000/health | python3 -m json.tool

shell:       ## Open a Python shell with project on path
	$(PYTHON) -c "import sys; sys.path.insert(0, '.'); import IPython; IPython.start_ipython()"

clean:       ## Remove generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage test.db results/
