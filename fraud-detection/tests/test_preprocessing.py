"""
Unit tests for the feature engineering pipeline.

These test the core ML logic independently of the API layer
and run with zero external dependencies.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from api.preprocessing import (
    FEATURE_NAMES,
    build_feature_df,
    build_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
BASE_TXN = {
    "transaction_id": "test_001",
    "transaction_amt": 100.0,
    "product_cd": "W",
    "card_type": "credit",
    "addr1": 299.0,
    "p_emaildomain": "gmail.com",
    "transaction_dt": 86400.0,
    "card1": 9500.0,
    "card2": 111.0,
    "dist1": 10.0,
    "c1": 2.0,
    "c2": 1.0,
    "d1": 14.0,
    "d15": 120.0,
    "v258": 1.0,
    "v308": 0.0,
}


# ---------------------------------------------------------------------------
# Shape / schema tests
# ---------------------------------------------------------------------------
class TestFeatureShape:
    def test_feature_vector_shape(self):
        f = build_features(BASE_TXN)
        assert f.shape == (22,), f"Expected (22,), got {f.shape}"

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 22

    def test_feature_names_unique(self):
        assert len(set(FEATURE_NAMES)) == 22, "Duplicate feature name detected"

    def test_dataframe_shape(self):
        df = build_feature_df(BASE_TXN)
        assert df.shape == (1, 22)
        assert list(df.columns) == FEATURE_NAMES

    def test_no_nan_in_output(self):
        f = build_features(BASE_TXN)
        assert not any(math.isnan(v) for v in f), "NaN leaked into features"

    def test_dtype_is_float32(self):
        f = build_features(BASE_TXN)
        assert f.dtype == np.float32


# ---------------------------------------------------------------------------
# Amount features
# ---------------------------------------------------------------------------
class TestAmountFeatures:
    def test_log_amt_positive(self):
        f = build_features(BASE_TXN)
        assert f[0] > 0, "log_transaction_amt should be positive for amt>0"

    def test_log_amt_monotone(self):
        """Higher amount → higher log_amt."""
        f_low  = build_features({**BASE_TXN, "transaction_amt": 10.0})
        f_high = build_features({**BASE_TXN, "transaction_amt": 1000.0})
        assert f_high[0] > f_low[0]

    def test_amt_bin_caps_at_4(self):
        f = build_features({**BASE_TXN, "transaction_amt": 9999.0})
        assert f[21] == 4.0, "amt_bin should cap at 4"

    def test_amt_bin_small_amount(self):
        f = build_features({**BASE_TXN, "transaction_amt": 25.0})
        assert f[21] == 0.0, "25 USD → bin 0"

    def test_amt_x_c1_interaction(self):
        f = build_features({**BASE_TXN, "transaction_amt": 100.0, "c1": 3.0})
        assert f[20] == pytest.approx(300.0, abs=0.01)


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------
class TestTimeFeatures:
    def test_night_flag_at_midnight(self):
        # transaction_dt = 0 → hour 0 → night
        f = build_features({**BASE_TXN, "transaction_dt": 0.0})
        assert f[3] == 1.0, "Hour 0 should be night"

    def test_night_flag_at_23h(self):
        f = build_features({**BASE_TXN, "transaction_dt": 3600 * 23})
        assert f[3] == 1.0, "Hour 23 should be night"

    def test_not_night_at_noon(self):
        f = build_features({**BASE_TXN, "transaction_dt": 3600 * 12})
        assert f[3] == 0.0, "Hour 12 should not be night"

    def test_tod_bucket_range(self):
        for hour in range(24):
            f = build_features({**BASE_TXN, "transaction_dt": float(3600 * hour)})
            assert 0 <= f[2] <= 5, f"tod_bucket out of range for hour {hour}"

    def test_days_since_ref(self):
        f = build_features({**BASE_TXN, "transaction_dt": 86400.0 * 10})
        assert f[5] == pytest.approx(10.0, abs=0.01), "days_since_ref should be 10"


# ---------------------------------------------------------------------------
# Email domain features
# ---------------------------------------------------------------------------
class TestEmailFeatures:
    HIGH_RISK = ["anonymous.com", "mail.com", "protonmail.com",
                 "guerrillamail.com", "throwam.com", "yopmail.com"]
    LOW_RISK  = ["gmail.com", "yahoo.com", "outlook.com", "company.com", ""]

    def test_high_risk_domains_flagged(self):
        for domain in self.HIGH_RISK:
            f = build_features({**BASE_TXN, "p_emaildomain": domain})
            assert f[7] == 1.0, f"{domain} should be flagged"

    def test_low_risk_domains_not_flagged(self):
        for domain in self.LOW_RISK:
            f = build_features({**BASE_TXN, "p_emaildomain": domain})
            assert f[7] == 0.0, f"{domain} should NOT be flagged"

    def test_none_email_not_flagged(self):
        f = build_features({**BASE_TXN, "p_emaildomain": None})
        assert f[7] == 0.0

    def test_case_insensitive(self):
        f = build_features({**BASE_TXN, "p_emaildomain": "Anonymous.COM"})
        assert f[7] == 1.0, "Email domain check should be case-insensitive"


# ---------------------------------------------------------------------------
# Card features
# ---------------------------------------------------------------------------
class TestCardFeatures:
    def test_credit_card_flagged(self):
        f = build_features({**BASE_TXN, "card_type": "credit"})
        assert f[8] == 1.0

    def test_debit_not_flagged(self):
        f = build_features({**BASE_TXN, "card_type": "debit"})
        assert f[8] == 0.0

    def test_none_card_type_not_flagged(self):
        f = build_features({**BASE_TXN, "card_type": None})
        assert f[8] == 0.0

    def test_credit_case_insensitive(self):
        f = build_features({**BASE_TXN, "card_type": "CREDIT"})
        assert f[8] == 1.0

    def test_card1_normalised(self):
        f = build_features({**BASE_TXN, "card1": 18396.0})
        assert f[16] == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# Product code
# ---------------------------------------------------------------------------
class TestProductCode:
    def test_known_codes(self):
        expected = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
        for code, enc in expected.items():
            f = build_features({**BASE_TXN, "product_cd": code})
            assert f[9] == float(enc), f"ProductCD {code} → expected {enc}"

    def test_unknown_code_defaults_to_zero(self):
        f = build_features({**BASE_TXN, "product_cd": "Z"})
        assert f[9] == 0.0


# ---------------------------------------------------------------------------
# Missingness indicators
# ---------------------------------------------------------------------------
class TestMissingnessIndicators:
    @pytest.mark.parametrize("field, feat_idx", [
        ("addr1", 10),
        ("dist1", 12),
        ("d1",    14),
        ("d15",   15),
    ])
    def test_missing_flag_set_when_none(self, field: str, feat_idx: int):
        txn = {**BASE_TXN, field: None}
        f = build_features(txn)
        assert f[feat_idx] == 1.0, f"{field}=None → feature[{feat_idx}] should be 1"

    @pytest.mark.parametrize("field, feat_idx", [
        ("addr1", 10),
        ("dist1", 12),
        ("d1",    14),
        ("d15",   15),
    ])
    def test_missing_flag_clear_when_present(self, field: str, feat_idx: int):
        txn = {**BASE_TXN, field: 5.0}
        f = build_features(txn)
        assert f[feat_idx] == 0.0, f"{field}=5.0 → feature[{feat_idx}] should be 0"


# ---------------------------------------------------------------------------
# NaN safety
# ---------------------------------------------------------------------------
class TestNaNSafety:
    def test_nan_c1_handled(self):
        import math as _math
        f = build_features({**BASE_TXN, "c1": float("nan")})
        assert not any(_math.isnan(v) for v in f)

    def test_nan_c2_handled(self):
        import math as _math
        f = build_features({**BASE_TXN, "c2": float("nan")})
        assert not any(_math.isnan(v) for v in f)

    def test_all_optional_none(self):
        import math as _math
        txn = {
            "transaction_amt": 50.0,
            "transaction_dt":  0.0,
            "product_cd":      "W",
            "card_type":       None,
            "addr1":           None,
            "p_emaildomain":   None,
            "card1":           None,
            "card2":           None,
            "dist1":           None,
            "c1":              None,
            "c2":              None,
            "d1":              None,
            "d15":             None,
            "v258":            None,
            "v308":            None,
        }
        f = build_features(txn)
        assert f.shape == (22,)
        assert not any(_math.isnan(v) for v in f)
