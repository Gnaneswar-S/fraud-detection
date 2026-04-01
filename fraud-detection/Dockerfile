# ============================================================
# Fraud Detection API — Docker image
# Base: python:3.11-slim (minimal attack surface)
# ============================================================
FROM python:3.11-slim

# System dependencies for psycopg2 + XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source
# (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY api/ ./api/
COPY batch/ ./batch/

# Copy model artefacts if present (optional — falls back to demo mode)
COPY model/ ./model/

# Non-root user for security
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
