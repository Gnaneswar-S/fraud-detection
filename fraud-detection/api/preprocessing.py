"""
Preprocessing pipeline — mirrors exactly what was applied during training.

Features engineered (20+):
  1.  log_transaction_amt          — log1p of amount (right-skew reduction)
  2.  amt_zscore                   — z-score vs population mean/std
  3.  time_of_day_bucket           — 0-5 (4-hr buckets from transaction_dt)
  4.  is_night_txn                 — 1 if 22:00–05:59 local hour bucket
  5.  is_weekend                   — 1 if Saturday/Sunday bucket
  6.  days_since_ref               — transaction_dt / 86400
  7.  velocity_1h_flag             — placeholder (populated by batch scorer)
  8.  high_risk_email_domain       — 1 for known high-fraud domains
  9.  is_credit                    — card type binary
  10. product_cd_encoded           — label-encoded product code
  11. addr1_missing                — missingness indicator for addr1
  12. dist1_log                    — log1p of dist1
  13. dist1_missing                — missingness indicator for dist1
  14. c_ratio                      — c1 / (c2 + 1)
  15. d1_missing                   — missingness indicator for d1
  16. d15_missing                  — missingness indicator for d15
  17. card1_norm                   — card1 / 18396 (max in training)
  18. card2_norm                   — card2 / 600
  19. v258_flag                    — binarised V258
  20. v308_flag                    — binarised V308
  21. amt_x_c1                     — interaction: amount × c1
  22. amt_bin                      — amount quantile bucket (0–4)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# Population statistics from training set (590K IEEE-CIS rows)
# Source: model/metrics.json → population_stats after running model/train.py
TRAIN_AMT_MEAN = 135.03
TRAIN_AMT_STD  = 239.16

HIGH_RISK_DOMAINS = {
    "anonymous.com",
    "mail.com",
    "protonmail.com",
    "guerrillamail.com",
    "throwam.com",
    "yopmail.com",
}

PRODUCT_CODES = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}

FEATURE_NAMES: list[str] = [
    "log_transaction_amt",
    "amt_zscore",
    "time_of_day_bucket",
    "is_night_txn",
    "is_weekend",
    "days_since_ref",
    "velocity_1h_flag",
    "high_risk_email_domain",
    "is_credit",
    "product_cd_encoded",
    "addr1_missing",
    "dist1_log",
    "dist1_missing",
    "c_ratio",
    "d1_missing",
    "d15_missing",
    "card1_norm",
    "card2_norm",
    "v258_flag",
    "v308_flag",
    "amt_x_c1",
    "amt_bin",
]


def _safe(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return float(value)


def build_features(txn: dict) -> np.ndarray:
    """
    Convert a raw transaction dict → 1-D numpy array of engineered features.
    Shape: (22,) — must match training feature order.
    """
    amt = _safe(txn.get("transaction_amt"), 0.01)
    dt = _safe(txn.get("transaction_dt"), 0.0)

    log_amt = math.log1p(amt)
    amt_zscore = (amt - TRAIN_AMT_MEAN) / (TRAIN_AMT_STD + 1e-9)

    # Time features (transaction_dt is seconds from 2017-12-01 reference)
    seconds_in_day = dt % 86400
    hour = seconds_in_day / 3600
    tod_bucket = int(hour / 4)          # 0=0-4h, 1=4-8h, … 5=20-24h
    is_night = 1 if (hour >= 22 or hour < 6) else 0

    days = dt / 86400
    day_of_week = int(days) % 7
    is_weekend = 1 if day_of_week >= 5 else 0

    email = (txn.get("p_emaildomain") or "").lower()
    high_risk_email = 1 if email in HIGH_RISK_DOMAINS else 0

    card_type = (txn.get("card_type") or "").lower()
    is_credit = 1 if card_type == "credit" else 0

    product = (txn.get("product_cd") or "W").upper()
    product_enc = PRODUCT_CODES.get(product, 0)

    addr1 = txn.get("addr1")
    addr1_missing = 1 if addr1 is None else 0

    dist1 = txn.get("dist1")
    dist1_missing = 1 if dist1 is None else 0
    dist1_log = math.log1p(_safe(dist1, 0.0))

    c1 = _safe(txn.get("c1"), 1.0)
    c2 = _safe(txn.get("c2"), 1.0)
    c_ratio = c1 / (c2 + 1.0)

    d1 = txn.get("d1")
    d1_missing = 1 if d1 is None else 0

    d15 = txn.get("d15")
    d15_missing = 1 if d15 is None else 0

    card1 = _safe(txn.get("card1"), 9500.0)
    card2 = _safe(txn.get("card2"), 111.0)
    card1_norm = card1 / 18396.0
    card2_norm = card2 / 600.0

    v258 = _safe(txn.get("v258"), 0.0)
    v308 = _safe(txn.get("v308"), 0.0)
    v258_flag = 1 if v258 > 0 else 0
    v308_flag = 1 if v308 > 0 else 0

    amt_x_c1 = amt * c1
    amt_bin = min(int(amt / 50), 4)     # buckets: 0-50, 50-100, … 200+

    features = np.array(
        [
            log_amt,
            amt_zscore,
            tod_bucket,
            is_night,
            is_weekend,
            days,
            0,                  # velocity_1h_flag — set externally if needed
            high_risk_email,
            is_credit,
            product_enc,
            addr1_missing,
            dist1_log,
            dist1_missing,
            c_ratio,
            d1_missing,
            d15_missing,
            card1_norm,
            card2_norm,
            v258_flag,
            v308_flag,
            amt_x_c1,
            amt_bin,
        ],
        dtype=np.float32,
    )
    return features


def build_feature_df(txn: dict) -> pd.DataFrame:
    """Return features as a single-row DataFrame with named columns."""
    return pd.DataFrame([build_features(txn)], columns=FEATURE_NAMES)