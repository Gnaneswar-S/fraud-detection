#!/usr/bin/env bash
# ============================================================
# Fraud Detection Engine — Local Setup Script
# ============================================================
# Usage: bash setup.sh
# Tested on: macOS 13+, Ubuntu 22.04
# Requires: python 3.11+, git

set -euo pipefail

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn] ${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ─── Python version check ──────────────────────────────────
PYTHON=$(command -v python3.11 || command -v python3 || error "python3 not found")
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python version: $PY_VER"
[[ "$PY_VER" < "3.10" ]] && error "Python 3.10+ required (found $PY_VER)"

# ─── Virtual environment ───────────────────────────────────
if [ ! -d ".venv" ]; then
  info "Creating virtual environment …"
  $PYTHON -m venv .venv
fi
source .venv/bin/activate
info "Virtual environment activated."

# ─── Dependencies ──────────────────────────────────────────
info "Installing dependencies …"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
info "Dependencies installed."

# ─── .env file ─────────────────────────────────────────────
if [ ! -f ".env" ]; then
  cp .env.example .env
  # Generate a real SECRET_KEY
  SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  sed -i.bak "s/your_openssl_rand_hex_32_here/$SECRET/" .env && rm .env.bak
  warn ".env created. Set API_PASSWORD before deploying."
else
  info ".env already exists — skipping."
fi

# ─── Model directories ─────────────────────────────────────
mkdir -p model data/raw results
info "Directory structure ready."

# ─── Docker check ──────────────────────────────────────────
if command -v docker &> /dev/null; then
  info "Docker found: $(docker --version | cut -d' ' -f3)"
else
  warn "Docker not found — install from https://docker.com to use docker compose."
fi

# ─── Dataset reminder ──────────────────────────────────────
if [ ! -f "data/raw/train_transaction.csv" ]; then
  warn "Dataset not found. Download it from Kaggle:"
  warn "  https://www.kaggle.com/competitions/ieee-fraud-detection/data"
  warn "  → Save train_transaction.csv to data/raw/"
fi

# ─── Quick smoke test ──────────────────────────────────────
info "Running quick smoke test (preprocessing pipeline) …"
python3 - <<'PYEOF'
import sys
sys.path.insert(0, '.')
from api.preprocessing import build_features, FEATURE_NAMES
txn = dict(transaction_amt=100.0, transaction_dt=86400.0, product_cd='W',
           p_emaildomain='gmail.com', card_type='credit',
           addr1=None, dist1=None, c1=None, c2=None, d1=None, d15=None,
           v258=None, v308=None, card1=None, card2=None)
f = build_features(txn)
assert len(f) == 22, f"Expected 22 features, got {len(f)}"
print(f"  Preprocessing: OK ({len(f)} features)")
PYEOF

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1.  Train model:     make train-quick      # 50K rows, quick dev"
echo "  2.  Start API:       make dev               # or: make up (Docker)"
echo "  3.  Run tests:       make test"
echo "  4.  Get token:       make token"
echo ""
echo "Full dataset training (after downloading CSV):"
echo "  make train"
echo ""
echo "AUC-ROC will be printed after training — update your resume with that number."
