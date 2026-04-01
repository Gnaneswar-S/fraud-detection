"""
Automated Retraining Trigger
==============================
Monitors model drift and triggers retraining when the KS statistic
exceeds the configured threshold (default: 0.45).

What it does:
  Every N minutes (configurable), it:
  1. Loads recent prediction scores from the database
  2. Computes the KS statistic vs training distribution
  3. If KS > threshold → runs model/train.py on fresh data
  4. Reloads the new model artefacts without restarting the server
  5. Logs all retraining events to retraining_log.json

Why automatic retraining matters:
  Fraud patterns change monthly. A model trained in Dec 2017 will
  degrade as fraudsters adapt their techniques. The KS statistic is
  a distribution-free test that detects when live scores diverge
  from what the model saw during training.

How to run:
  python api/retraining.py --data data/raw/train_transaction.csv --watch

  Or as a background thread started from main.py:
  from api.retraining import RetrainingWatcher
  watcher = RetrainingWatcher(data_path="data/raw/train_transaction.csv")
  watcher.start()

Configuration:
  KS_THRESHOLD        float   0.45   Trigger retraining above this KS
  CHECK_INTERVAL_MIN  int     60     How often to check (minutes)
  MIN_SAMPLES         int     100    Minimum predictions needed to check
  DATA_PATH           str           Path to training CSV for retraining
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

KS_THRESHOLD       = 0.45    # alert + retrain threshold
CHECK_INTERVAL_MIN = 60      # check every 60 minutes
MIN_SAMPLES        = 100     # need at least 100 recent predictions
RETRAIN_LOG        = Path("model/retraining_log.json")


def _load_recent_scores(window: int = 500) -> list[float]:
    """Pull recent fraud probabilities from the database."""
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
            return [float(r.fraud_probability) for r in rows]
        finally:
            db.close()
    except Exception as e:
        logger.warning("Could not load scores from DB: %s", e)
        return []


def _compute_ks(scores: list[float]) -> float:
    """KS statistic vs Gaussian baseline (from monitoring.py)."""
    from api.monitoring import _ks_statistic, BASELINE
    return _ks_statistic(scores, BASELINE["score_mean"], BASELINE["score_std"])


def _log_event(event: dict) -> None:
    """Append a retraining event to the JSON log."""
    RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    events = []
    if RETRAIN_LOG.exists():
        try:
            events = json.loads(RETRAIN_LOG.read_text())
        except Exception:
            events = []
    events.append(event)
    RETRAIN_LOG.write_text(json.dumps(events, indent=2))


def _reload_models() -> bool:
    """Hot-reload model artefacts without restarting the server."""
    try:
        # Ensure project root is on sys.path (needed when called from subprocess context)
        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import api.model_loader as ml
        ml._bundle = None          # clear singleton
        ml._CACHE.clear()          # clear prediction cache
        from api.model_loader import load_models
        load_models()              # reload from disk
        logger.info("Models hot-reloaded successfully.")
        return True
    except Exception as e:
        logger.warning(
            "Hot-reload skipped (expected when running as standalone script, "
            "not as part of the running API server): %s", e
        )
        return False


def run_retraining(data_path: str, sample: int | None = None) -> dict:
    """
    Execute model/train.py as a subprocess and return result summary.

    Using subprocess (not importing directly) so retraining runs in a
    separate process — it won't block the API server and won't share
    memory with the running model.
    """
    logger.info("Starting retraining on %s …", data_path)
    cmd = [sys.executable, "model/train.py", "--data", data_path]
    if sample:
        cmd += ["--sample", str(sample)]

    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),   # run from project root
    )
    elapsed = round(time.perf_counter() - start, 1)

    success = result.returncode == 0
    auc_roc = None

    if success:
        # Parse AUC-ROC from stdout
        for line in result.stdout.splitlines():
            if "AUC-ROC" in line and ":" in line:
                try:
                    auc_roc = float(line.split(":")[-1].strip().split()[0])
                except Exception:
                    pass
        # Try reading from metrics.json
        metrics_path = Path("model/metrics.json")
        if metrics_path.exists() and auc_roc is None:
            try:
                auc_roc = json.loads(metrics_path.read_text()).get("auc_roc")
            except Exception:
                pass

    event = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "trigger":          "ks_threshold_exceeded",
        "success":          success,
        "training_time_s":  elapsed,
        "new_auc_roc":      auc_roc,
        "return_code":      result.returncode,
        "stdout_tail":      result.stdout[-500:] if result.stdout else "",
        "stderr_tail":      result.stderr[-300:] if result.stderr else "",
    }
    _log_event(event)

    if success:
        logger.info("Retraining complete in %.0fs. AUC-ROC: %s", elapsed, auc_roc)
        _reload_models()
    else:
        logger.error("Retraining failed (exit %d):\n%s", result.returncode, result.stderr[-500:])

    return event


class RetrainingWatcher:
    """
    Background thread that monitors drift and triggers retraining.

    Usage:
        watcher = RetrainingWatcher(data_path="data/raw/train_transaction.csv")
        watcher.start()    # non-blocking — runs in daemon thread
        watcher.stop()     # graceful shutdown
    """

    def __init__(
        self,
        data_path:          str,
        ks_threshold:       float = KS_THRESHOLD,
        check_interval_min: int   = CHECK_INTERVAL_MIN,
        min_samples:        int   = MIN_SAMPLES,
        sample:             int | None = None,
    ):
        self.data_path          = data_path
        self.ks_threshold       = ks_threshold
        self.check_interval_sec = check_interval_min * 60
        self.min_samples        = min_samples
        self.sample             = sample
        self._stop_event        = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_check_time: str | None = None
        self.last_ks: float | None = None
        self.retrain_count      = 0

    def start(self) -> None:
        """Start the background watcher thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Retraining watcher started. Checking every %d min, KS threshold %.2f",
            self.check_interval_sec // 60, self.ks_threshold,
        )

    def stop(self) -> None:
        """Signal the watcher to stop."""
        self._stop_event.set()

    def status(self) -> dict:
        """Return current watcher status for the /model/info endpoint."""
        return {
            "active":           self._thread is not None and self._thread.is_alive(),
            "last_check":       self.last_check_time,
            "last_ks":          self.last_ks,
            "retrain_count":    self.retrain_count,
            "ks_threshold":     self.ks_threshold,
            "check_interval_min": self.check_interval_sec // 60,
        }

    def _watch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._check_and_maybe_retrain()
            except Exception as e:
                logger.error("Watcher error: %s", e)
            self._stop_event.wait(timeout=self.check_interval_sec)

    def _check_and_maybe_retrain(self) -> None:
        scores = _load_recent_scores(window=500)
        self.last_check_time = datetime.now(timezone.utc).isoformat()

        if len(scores) < self.min_samples:
            logger.info(
                "Drift check: only %d scores (need %d). Skipping.",
                len(scores), self.min_samples,
            )
            return

        ks = _compute_ks(scores)
        self.last_ks = ks
        logger.info("Drift check: KS statistic = %.4f (threshold %.2f)", ks, self.ks_threshold)

        if ks > self.ks_threshold:
            logger.warning(
                "KS %.4f > threshold %.2f — triggering retraining!",
                ks, self.ks_threshold,
            )
            result = run_retraining(self.data_path, self.sample)
            if result["success"]:
                self.retrain_count += 1
        else:
            logger.info("Model healthy — no retraining needed.")


# Singleton watcher (not started by default — call watcher.start() explicitly)
_watcher: RetrainingWatcher | None = None


def get_watcher() -> RetrainingWatcher | None:
    return _watcher


def init_watcher(data_path: str, **kwargs) -> RetrainingWatcher:
    """Initialise and start the global watcher singleton."""
    global _watcher
    _watcher = RetrainingWatcher(data_path=data_path, **kwargs)
    _watcher.start()
    return _watcher


# ---------------------------------------------------------------------------
# CLI — run as standalone watcher process
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Fraud model drift watcher + auto-retraining")
    parser.add_argument("--data",     required=True,   help="Path to training CSV")
    parser.add_argument("--watch",    action="store_true", help="Run continuous watcher")
    parser.add_argument("--force",    action="store_true", help="Force retraining immediately")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (minutes)")
    parser.add_argument("--threshold",type=float, default=KS_THRESHOLD, help="KS threshold")
    parser.add_argument("--sample",   type=int, default=None, help="Row sample for quick retrain")
    args = parser.parse_args()

    if args.force:
        print("Force-retraining...")
        result = run_retraining(args.data, args.sample)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)

    if args.watch:
        watcher = RetrainingWatcher(
            data_path=args.data,
            ks_threshold=args.threshold,
            check_interval_min=args.interval,
            sample=args.sample,
        )
        watcher.start()
        print(f"Watching for drift... (KS threshold={args.threshold}, interval={args.interval}min)")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
                s = watcher.status()
                print(f"  Last check: {s['last_check']} | KS: {s['last_ks']} | Retrains: {s['retrain_count']}")
        except KeyboardInterrupt:
            watcher.stop()
            print("Watcher stopped.")