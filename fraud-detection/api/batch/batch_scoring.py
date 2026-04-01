"""
Batch Scorer
============
Reads a CSV of historical transactions, runs the fraud model on each row,
and writes results to:
  - results CSV: results/batch_results_<timestamp>.csv
  - PostgreSQL predictions table (optional, skipped if DB unavailable)

Usage:
  python batch/batch_scoring.py --input data/raw/transactions.csv [--db]

Simulates the offline bank retraining pipeline pattern:
  1. Ingest raw CSV
  2. Feature-engineer each row
  3. Score with current model artefact
  4. Persist scores + timestamp for audit
  5. Log aggregate fraud rate for monitoring drift
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from api.model_loader import load_models, predict
from api.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

EXPECTED_COLUMNS = [
    "TransactionID",
    "TransactionAmt",
    "ProductCD",
    "card_type",
    "addr1",
    "P_emaildomain",
    "TransactionDT",
    "card1",
    "card2",
    "dist1",
    "C1",
    "C2",
    "D1",
    "D15",
    "V258",
    "V308",
]


def _row_to_dict(row: pd.Series) -> dict:
    """Map a DataFrame row to the TransactionRequest dict shape."""
    return {
        "transaction_id": str(row.get("TransactionID", "unknown")),
        "transaction_amt": float(row.get("TransactionAmt", 1.0)),
        "product_cd": str(row.get("ProductCD", "W")),
        "card_type": str(row.get("card_type", "")),
        "addr1": row.get("addr1") if pd.notna(row.get("addr1")) else None,
        "p_emaildomain": str(row.get("P_emaildomain", "")) if pd.notna(row.get("P_emaildomain")) else None,
        "transaction_dt": float(row.get("TransactionDT", 0)),
        "card1": float(row.get("card1", 0)) if pd.notna(row.get("card1")) else None,
        "card2": float(row.get("card2", 0)) if pd.notna(row.get("card2")) else None,
        "dist1": float(row.get("dist1", 0)) if pd.notna(row.get("dist1")) else None,
        "c1": float(row.get("C1", 1)) if pd.notna(row.get("C1")) else None,
        "c2": float(row.get("C2", 1)) if pd.notna(row.get("C2")) else None,
        "d1": float(row.get("D1", 0)) if pd.notna(row.get("D1")) else None,
        "d15": float(row.get("D15", 0)) if pd.notna(row.get("D15")) else None,
        "v258": float(row.get("V258", 0)) if pd.notna(row.get("V258")) else None,
        "v308": float(row.get("V308", 0)) if pd.notna(row.get("V308")) else None,
    }


def run_batch(input_path, write_db=False, limit=None)-> Path:
    """
    Score all transactions in *input_path* CSV.

    Parameters
    ----------
    input_path  Path to raw transactions CSV
    write_db    If True, persist results to PostgreSQL

    Returns
    -------
    Path to the results CSV written
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading model artefacts…")
    load_models()

    logger.info("Reading %s …", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    if limit:
        df = df.head(limit)
        logger.info("Limiting to %d rows for batch scoring.", limit)
    total = len(df)
    logger.info("Loaded %d rows.", total)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"batch_results_{ts}.csv"

    results = []
    fraud_count = 0
    t_start = time.perf_counter()

    with open(out_path, "w", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "transaction_id",
                "fraud_probability",
                "fraud_label",
                "risk_tier",
                "top_feature_1",
                "top_feature_2",
                "top_feature_3",
                "model_version",
                "prediction_timestamp",
            ],
        )
        writer.writeheader()

        for idx, row in df.iterrows():
            try:
                txn_dict = _row_to_dict(row)
                fraud_prob, fraud_label, risk_tier, top_features = predict(txn_dict)
                now = datetime.now(timezone.utc).isoformat()

                record = {
                    "transaction_id": txn_dict["transaction_id"],
                    "fraud_probability": fraud_prob,
                    "fraud_label": fraud_label,
                    "risk_tier": risk_tier,
                    "top_feature_1": top_features[0].feature if len(top_features) > 0 else "",
                    "top_feature_2": top_features[1].feature if len(top_features) > 1 else "",
                    "top_feature_3": top_features[2].feature if len(top_features) > 2 else "",
                    "model_version": settings.MODEL_VERSION,
                    "prediction_timestamp": now,
                }
                writer.writerow(record)
                results.append(record)

                if fraud_label:
                    fraud_count += 1

                if (idx + 1) % 1000 == 0:
                    logger.info(
                        "  Scored %d / %d rows  (%.1f%% fraud so far)",
                        idx + 1, total, 100.0 * fraud_count / (idx + 1),
                    )
            except Exception as exc:
                logger.warning("Row %d failed: %s", idx, exc)
                continue

    elapsed = time.perf_counter() - t_start
    fraud_rate = 100.0 * fraud_count / max(len(results), 1)

    logger.info("=" * 55)
    logger.info("Batch scoring complete in %.1f s", elapsed)
    logger.info("  Rows scored     : %d", len(results))
    logger.info("  Fraud flagged   : %d  (%.2f%%)", fraud_count, fraud_rate)
    logger.info("  Output          : %s", out_path)
    logger.info("=" * 55)

    # --- Optional DB write ---
    if write_db:
        _write_to_db(results)

    return out_path


def _write_to_db(results: list[dict]) -> None:
    """Write batch results to PostgreSQL predictions table."""
    try:
        from api.database import SessionLocal, PredictionRecord
        import json

        db = SessionLocal()
        try:
            for r in results:
                db.add(PredictionRecord(
                    transaction_id=r["transaction_id"],
                    fraud_probability=r["fraud_probability"],
                    fraud_label=bool(r["fraud_label"]),
                    risk_tier=r["risk_tier"],
                    top_features_json=json.dumps([r["top_feature_1"], r["top_feature_2"], r["top_feature_3"]]),
                    model_version=r["model_version"],
                    prediction_timestamp=datetime.fromisoformat(r["prediction_timestamp"]),
                ))
            db.commit()
            logger.info("DB write: %d records committed.", len(results))
        finally:
            db.close()
    except Exception as exc:
        logger.warning("DB write failed (skipping): %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection Batch Scorer")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to transactions CSV (e.g. data/raw/train_transaction.csv)",
        
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Write results to PostgreSQL in addition to CSV",
    )
    parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Limit number of rows for testing (e.g. 1000)"
    )
    args = parser.parse_args()
    run_batch(args.input, write_db=args.db, limit=args.limit)
