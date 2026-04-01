"""
Tests for the model drift monitoring module (api/monitoring.py).

Run: pytest tests/test_monitoring.py -v
"""
from __future__ import annotations

import random

import pytest

from api.monitoring import (
    BASELINE,
    DRIFT_THRESHOLDS,
    _ks_statistic,
    compute_drift_report,
)

random.seed(42)

# ---------------------------------------------------------------------------
# Helpers — generate realistic score distributions
# ---------------------------------------------------------------------------

def _healthy_scores(n: int = 1000, seed: int = 42) -> list[float]:
    """Bimodal: ~96.5% legit (low score), ~3.5% fraud (high score)."""
    r = random.Random(seed)
    legit = int(n * 0.965)
    fraud = n - legit
    scores = (
        [r.uniform(0.01, 0.12) for _ in range(legit)] +
        [r.uniform(0.50, 0.95) for _ in range(fraud)]
    )
    r.shuffle(scores)
    return scores


def _high_fraud_scores(n: int = 1000, fraud_pct: float = 0.25, seed: int = 42) -> list[float]:
    r = random.Random(seed)
    fraud = int(n * fraud_pct)
    legit = n - fraud
    return (
        [r.uniform(0.01, 0.10) for _ in range(legit)] +
        [r.uniform(0.55, 0.99) for _ in range(fraud)]
    )


# ---------------------------------------------------------------------------
# Healthy baseline
# ---------------------------------------------------------------------------
class TestHealthy:
    def test_healthy_status(self):
        r = compute_drift_report(_healthy_scores())
        assert r["status"] == "HEALTHY", (
            f"Expected HEALTHY, got {r['status']}. Alerts: {r['alerts']}"
        )

    def test_healthy_no_alerts(self):
        r = compute_drift_report(_healthy_scores())
        assert r["alerts"] == []

    def test_fraud_rate_close_to_baseline(self):
        r = compute_drift_report(_healthy_scores())
        assert abs(r["live_fraud_rate_pct"] - BASELINE["fraud_rate_pct"]) < 1.0

    def test_window_size_matches_input(self):
        scores = _healthy_scores(500)
        r = compute_drift_report(scores)
        assert r["window_size"] == 500


# ---------------------------------------------------------------------------
# Fraud rate drift
# ---------------------------------------------------------------------------
class TestFraudRateDrift:
    def test_high_fraud_rate_triggers_alert(self):
        r = compute_drift_report(_high_fraud_scores(fraud_pct=0.25))
        assert r["status"] == "DEGRADED"
        types = [a["type"] for a in r["alerts"]]
        assert "FRAUD_RATE_DRIFT" in types

    def test_zero_fraud_rate_triggers_alert(self):
        all_legit = [0.02] * 500
        r = compute_drift_report(all_legit)
        assert r["status"] == "DEGRADED"

    def test_alert_contains_severity(self):
        r = compute_drift_report(_high_fraud_scores(fraud_pct=0.30))
        fraud_alerts = [a for a in r["alerts"] if a["type"] == "FRAUD_RATE_DRIFT"]
        assert len(fraud_alerts) == 1
        assert fraud_alerts[0]["severity"] in ("MEDIUM", "HIGH")

    def test_large_spike_is_high_severity(self):
        r = compute_drift_report(_high_fraud_scores(fraud_pct=0.50))
        fraud_alerts = [a for a in r["alerts"] if a["type"] == "FRAUD_RATE_DRIFT"]
        assert any(a["severity"] == "HIGH" for a in fraud_alerts)


# ---------------------------------------------------------------------------
# Score distribution drift
# ---------------------------------------------------------------------------
class TestScoreDistributionDrift:
    def test_stuck_high_scores_detected(self):
        stuck = [0.92] * 600
        r = compute_drift_report(stuck)
        assert r["status"] == "DEGRADED"
        types = [a["type"] for a in r["alerts"]]
        assert "SCORE_MEAN_DRIFT" in types or "SCORE_DISTRIBUTION_DRIFT" in types

    def test_stuck_low_scores_detected(self):
        stuck = [0.001] * 600
        r = compute_drift_report(stuck)
        assert r["status"] == "DEGRADED"

    def test_mean_reported(self):
        r = compute_drift_report(_healthy_scores())
        assert "live_score_mean" in r
        assert 0.0 <= r["live_score_mean"] <= 1.0

    def test_std_reported(self):
        r = compute_drift_report(_healthy_scores())
        assert "live_score_std" in r
        assert r["live_score_std"] >= 0.0


# ---------------------------------------------------------------------------
# KS statistic
# ---------------------------------------------------------------------------
class TestKSStatistic:
    def test_shifted_distribution_has_high_ks(self):
        shifted = [random.uniform(0.7, 1.0) for _ in range(300)]
        ks = _ks_statistic(shifted, BASELINE["score_mean"], BASELINE["score_std"])
        assert ks > DRIFT_THRESHOLDS["ks_statistic"], (
            f"Expected KS > {DRIFT_THRESHOLDS['ks_statistic']}, got {ks}"
        )

    def test_matching_distribution_has_low_ks(self):
        import math
        matching = [
            max(0.0, min(1.0, random.gauss(BASELINE["score_mean"], BASELINE["score_std"])))
            for _ in range(300)
        ]
        ks = _ks_statistic(matching, BASELINE["score_mean"], BASELINE["score_std"])
        assert ks < DRIFT_THRESHOLDS["ks_statistic"], (
            f"Expected KS < {DRIFT_THRESHOLDS['ks_statistic']}, got {ks}"
        )

    def test_ks_small_sample_returns_zero(self):
        ks = _ks_statistic([0.1, 0.2], 0.08, 0.18)
        assert ks == 0.0

    def test_ks_is_between_zero_and_one(self):
        sample = [random.random() for _ in range(100)]
        ks = _ks_statistic(sample, 0.08, 0.18)
        assert 0.0 <= ks <= 1.0

    def test_ks_reported_in_output(self):
        r = compute_drift_report(_healthy_scores())
        assert "ks_statistic" in r
        assert 0.0 <= r["ks_statistic"] <= 1.0


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
class TestLatency:
    def test_high_latency_triggers_alert(self):
        scores = _healthy_scores()
        latencies = [random.uniform(550, 900) for _ in range(300)]
        r = compute_drift_report(scores, latencies)
        types = [a["type"] for a in r["alerts"]]
        assert "HIGH_LATENCY" in types

    def test_normal_latency_no_alert(self):
        scores = _healthy_scores()
        latencies = [random.uniform(20, 150) for _ in range(300)]
        r = compute_drift_report(scores, latencies)
        types = [a["type"] for a in r["alerts"]]
        assert "HIGH_LATENCY" not in types

    def test_p95_reported_when_latency_provided(self):
        scores = _healthy_scores()
        latencies = [random.uniform(50, 200) for _ in range(100)]
        r = compute_drift_report(scores, latencies)
        assert r["p95_latency_ms"] is not None
        assert r["p95_latency_ms"] > 0

    def test_p95_none_when_no_latency(self):
        r = compute_drift_report(_healthy_scores())
        assert r["p95_latency_ms"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_scores_returns_error(self):
        r = compute_drift_report([])
        assert "error" in r

    def test_single_score_does_not_crash(self):
        r = compute_drift_report([0.05])
        assert "status" in r

    def test_all_fraud_scores(self):
        r = compute_drift_report([0.99] * 200)
        assert r["status"] == "DEGRADED"

    def test_report_has_required_keys(self):
        r = compute_drift_report(_healthy_scores())
        required = [
            "window_size", "computed_at", "live_fraud_rate_pct",
            "baseline_fraud_rate_pct", "live_score_mean", "live_score_std",
            "ks_statistic", "p95_latency_ms", "alerts", "status",
        ]
        for key in required:
            assert key in r, f"Missing key: {key}"

    def test_status_is_valid_value(self):
        r = compute_drift_report(_healthy_scores())
        assert r["status"] in ("HEALTHY", "DEGRADED")

    def test_baseline_fraud_rate_matches_constant(self):
        r = compute_drift_report(_healthy_scores())
        assert r["baseline_fraud_rate_pct"] == BASELINE["fraud_rate_pct"]
