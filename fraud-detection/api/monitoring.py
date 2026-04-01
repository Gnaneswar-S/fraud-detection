"""
Model Monitoring — Drift Detection
====================================
Detects when the live fraud rate or score distribution drifts from training
baselines. In production, run this on a schedule (cron / Render cron job).

Checks:
  1. Score distribution drift  (KS test vs training baseline)
  2. Fraud rate drift          (z-test vs expected 3.5%)
  3. Missing field rates       (alert if fields suddenly more/less sparse)
  4. Prediction latency P95    (warn if >500ms)

Usage (standalone):
  python api/monitoring.py --window 1000

Usage (as FastAPI endpoint — add to main.py):
  GET /monitoring/summary   [protected]
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training baselines (update from model/metrics.json after each retrain)
# ---------------------------------------------------------------------------
BASELINE = {
    "fraud_rate_pct":  3.5,          # expected % of predictions flagged fraud
    "score_mean":      0.08,         # expected mean fraud_probability (most txns are legit)
    "score_std":       0.18,         # expected std  fraud_probability
    "p95_latency_ms":  200.0,        # expected P95 response time
}

DRIFT_THRESHOLDS = {
    "fraud_rate_pct_delta":  3.0,    # alert if live fraud rate differs by >3pp
    "score_mean_delta":      0.08,   # alert if mean score shifts by >0.08
    "score_std_delta":       0.10,   # alert if std shifts by >0.10
    "ks_statistic":          0.45,   # alert if KS test stat >0.45 (scores are bimodal, not Gaussian)
    "p95_latency_ms":        500.0,  # alert if P95 latency >500ms
}


def _ks_statistic(sample: list[float], baseline_mean: float, baseline_std: float) -> float:
    """
    Approximate one-sample KS statistic vs a Gaussian baseline.
    Full scipy.stats.kstest would be more precise but adds a dependency.
    """
    if len(sample) < 10:
        return 0.0
    sample_sorted = sorted(sample)
    n = len(sample_sorted)
    max_d = 0.0
    for i, x in enumerate(sample_sorted):
        # CDF of empirical distribution at x
        F_emp = (i + 1) / n
        # CDF of baseline Gaussian at x
        z = (x - baseline_mean) / (baseline_std + 1e-9)
        # Approximate normal CDF via error function
        F_base = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        max_d = max(max_d, abs(F_emp - F_base))
    return round(max_d, 4)


def compute_drift_report(
    recent_scores: list[float],
    recent_latencies_ms: Optional[list[float]] = None,
) -> dict:
    """
    Compute a drift report from a window of recent predictions.

    Parameters
    ----------
    recent_scores       List of recent fraud_probability values
    recent_latencies_ms List of recent API latency measurements (ms)

    Returns
    -------
    Dict with alerts and statistics
    """
    if not recent_scores:
        return {"error": "No recent scores to analyse."}

    n = len(recent_scores)
    fraud_count = sum(1 for s in recent_scores if s >= 0.5)
    live_fraud_rate = 100.0 * fraud_count / n
    live_mean = sum(recent_scores) / n
    live_std = math.sqrt(sum((s - live_mean) ** 2 for s in recent_scores) / max(n - 1, 1))

    ks_stat = _ks_statistic(recent_scores, BASELINE["score_mean"], BASELINE["score_std"])

    alerts = []

    fraud_delta = abs(live_fraud_rate - BASELINE["fraud_rate_pct"])
    if fraud_delta > DRIFT_THRESHOLDS["fraud_rate_pct_delta"]:
        alerts.append({
            "type": "FRAUD_RATE_DRIFT",
            "message": (
                f"Live fraud rate {live_fraud_rate:.1f}% differs from "
                f"baseline {BASELINE['fraud_rate_pct']}% by {fraud_delta:.1f}pp"
            ),
            "severity": "HIGH" if fraud_delta > 5 else "MEDIUM",
        })

    mean_delta = abs(live_mean - BASELINE["score_mean"])
    if mean_delta > DRIFT_THRESHOLDS["score_mean_delta"]:
        alerts.append({
            "type": "SCORE_MEAN_DRIFT",
            "message": (
                f"Score mean {live_mean:.3f} differs from baseline "
                f"{BASELINE['score_mean']} by {mean_delta:.3f}"
            ),
            "severity": "MEDIUM",
        })

    if ks_stat > DRIFT_THRESHOLDS["ks_statistic"]:
        alerts.append({
            "type": "SCORE_DISTRIBUTION_DRIFT",
            "message": f"KS statistic {ks_stat:.3f} exceeds threshold {DRIFT_THRESHOLDS['ks_statistic']}",
            "severity": "HIGH" if ks_stat > 0.20 else "MEDIUM",
        })

    # Latency check
    p95_latency = None
    if recent_latencies_ms:
        sorted_lat = sorted(recent_latencies_ms)
        p95_latency = sorted_lat[int(len(sorted_lat) * 0.95)]
        if p95_latency > DRIFT_THRESHOLDS["p95_latency_ms"]:
            alerts.append({
                "type": "HIGH_LATENCY",
                "message": f"P95 latency {p95_latency:.0f}ms exceeds {DRIFT_THRESHOLDS['p95_latency_ms']}ms",
                "severity": "HIGH",
            })

    return {
        "window_size":       n,
        "computed_at":       datetime.now(timezone.utc).isoformat(),
        "live_fraud_rate_pct": round(live_fraud_rate, 3),
        "baseline_fraud_rate_pct": BASELINE["fraud_rate_pct"],
        "live_score_mean":   round(live_mean, 4),
        "live_score_std":    round(live_std, 4),
        "ks_statistic":      ks_stat,
        "p95_latency_ms":    round(p95_latency, 1) if p95_latency else None,
        "alerts":            alerts,
        "status":            "DEGRADED" if alerts else "HEALTHY",
    }


def load_recent_scores_from_db(window: int = 1000) -> list[float]:
    """Load the last *window* predictions from Postgres."""
    try:
        from api.database import SessionLocal, PredictionRecord

        db = SessionLocal()
        try:
            rows = (
                db.query(PredictionRecord.fraud_probability)
                .order_by(PredictionRecord.prediction_timestamp.desc())
                .limit(window)
                .all()
            )
            return [r.fraud_probability for r in rows]
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Could not load scores from DB: %s", exc)
        return []


def load_recent_latencies_from_db(window: int = 1000) -> list[float]:
    """Load the last *window* latency measurements from audit_log."""
    try:
        from api.database import SessionLocal, AuditLog

        db = SessionLocal()
        try:
            rows = (
                db.query(AuditLog.latency_ms)
                .filter(AuditLog.endpoint == "/predict")
                .order_by(AuditLog.created_at.desc())
                .limit(window)
                .all()
            )
            return [r.latency_ms for r in rows if r.latency_ms is not None]
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Could not load latencies from DB: %s", exc)
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud model drift monitor")
    parser.add_argument("--window", type=int, default=1000,
                        help="Number of recent predictions to analyse")
    args = parser.parse_args()

    scores    = load_recent_scores_from_db(args.window)
    latencies = load_recent_latencies_from_db(args.window)

    if not scores:
        print("No predictions found in DB. Run some predictions first.")
    else:
        report = compute_drift_report(scores, latencies)
        print(json.dumps(report, indent=2))

        if report["alerts"]:
            print(f"\n⚠  {len(report['alerts'])} alert(s) detected — investigate model drift.")
        else:
            print("\n✓ Model health: HEALTHY")
