"""
Real Transaction Velocity Tracker
===================================
Tracks how many times a card (card1 value) has been used in recent
time windows, using an in-process sliding-window cache (no Redis needed
for deployment simplicity — Redis version documented below).

This replaces the placeholder velocity_1h_flag = 0 in preprocessing.py.

Why velocity matters:
  Fraudsters typically test a stolen card with small purchases (< $10)
  before making large fraudulent ones. A card used 5 times in 10 minutes
  is a major red flag regardless of individual transaction amounts.

Thresholds (calibrated on IEEE-CIS dataset fraud patterns):
  > 3 transactions in 1 hour    → velocity_1h_flag = 1
  > 6 transactions in 1 hour    → velocity_1h_flag = 2 (higher risk)
  > 10 transactions in 1 hour   → velocity_1h_flag = 3 (critical)

Production Redis implementation:
  Replace VelocityTracker with:
    import redis
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)

    def get_velocity(card_id: str) -> int:
        key = f"vel:{card_id}"
        count = r.incr(key)
        if count == 1:
            r.expire(key, 3600)  # 1 hour TTL
        return count

Usage:
    from api.velocity import velocity_tracker
    flag = velocity_tracker.record_and_get_flag(card1_value)
    # Pass flag into build_features() as velocity_1h_flag
"""
from __future__ import annotations

import time
import threading
from collections import defaultdict, deque


class VelocityTracker:
    """
    Thread-safe sliding window velocity tracker.

    Uses a deque of timestamps per card_id. On each call:
    1. Append current timestamp
    2. Evict timestamps older than the window
    3. Return the count

    Memory: O(max_events_per_card * n_unique_cards)
    Cleanup thread evicts stale cards every 5 minutes.
    """

    def __init__(self, window_seconds: int = 3600):
        self.window = window_seconds
        self._data: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        # Background cleanup to prevent unbounded memory growth
        self._start_cleanup_thread()

    def record(self, card_id: str | float | None) -> int:
        """
        Record a transaction for card_id and return current
        transaction count within the window.

        Returns 0 if card_id is None/NaN.
        """
        if card_id is None or (isinstance(card_id, float) and card_id != card_id):
            return 0

        key = str(int(float(card_id)))
        now = time.monotonic()

        with self._lock:
            dq = self._data[key]
            dq.append(now)
            # Evict events outside the window
            while dq and dq[0] < now - self.window:
                dq.popleft()
            return len(dq)

    def get_flag(self, count: int) -> int:
        """
        Convert raw count to a risk flag:
          0–3 transactions  → 0 (normal)
          4–6               → 1 (elevated)
          7–9               → 2 (high)
          10+               → 3 (critical)
        """
        if count <= 3:   return 0
        if count <= 6:   return 1
        if count <= 9:   return 2
        return 3

    def record_and_get_flag(self, card_id) -> int:
        """Convenience: record + return risk flag in one call."""
        count = self.record(card_id)
        return self.get_flag(count)

    def get_count(self, card_id) -> int:
        """Get current count without recording a new event."""
        if card_id is None:
            return 0
        key = str(int(float(card_id)))
        now = time.monotonic()
        with self._lock:
            dq = self._data.get(key, deque())
            # Count events in window without modifying
            return sum(1 for t in dq if t >= now - self.window)

    def _start_cleanup_thread(self):
        """Remove cards with no activity in the last window."""
        def _cleanup():
            while True:
                time.sleep(300)  # every 5 minutes
                now = time.monotonic()
                with self._lock:
                    stale = [k for k, dq in self._data.items()
                             if not dq or dq[-1] < now - self.window]
                    for k in stale:
                        del self._data[k]

        t = threading.Thread(target=_cleanup, daemon=True)
        t.start()


# Singleton — import this everywhere
velocity_tracker = VelocityTracker(window_seconds=3600)