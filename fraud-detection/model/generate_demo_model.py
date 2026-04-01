"""
generate_demo_model.py
======================
Generates lightweight placeholder model artefacts for:
  - Local development without the full IEEE-CIS dataset
  - CI/CD pipeline smoke tests
  - Docker image validation

The demo model uses synthetic data shaped like the real feature matrix.
AUC-ROC on synthetic data is ~0.93 (not meaningful — replace with real
model by running notebooks/model_training.ipynb on the actual dataset).

Usage:
    python model/generate_demo_model.py
    # → writes model/fraud_model.pkl and model/scaler.pkl
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Using sklearn GradientBoosting fallback.")
    from sklearn.ensemble import GradientBoostingClassifier

from api.preprocessing import FEATURE_NAMES

N_LEGIT = 5000
N_FRAUD = 200
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

print("Generating synthetic training data…")

# Legit transactions: centred features
X_legit = rng.normal(loc=0.0, scale=1.0, size=(N_LEGIT, len(FEATURE_NAMES)))

# Fraud transactions: shifted means to create a learnable signal
shift = np.zeros(len(FEATURE_NAMES))
shift[0] = 1.5   # log_transaction_amt higher
shift[1] = 2.0   # amt_zscore higher
shift[7] = 3.0   # high_risk_email_domain
shift[11] = 1.0  # dist1_log
X_fraud = rng.normal(loc=shift, scale=0.8, size=(N_FRAUD, len(FEATURE_NAMES)))
X_fraud[:, 7] = rng.choice([0, 1], size=N_FRAUD, p=[0.3, 0.7])  # binary flag

X = np.vstack([X_legit, X_fraud])
y = np.array([0] * N_LEGIT + [1] * N_FRAUD)

# Shuffle
idx = rng.permutation(len(y))
X, y = X[idx], y[idx]

# Scale
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

print(f"Training on {len(y):,} rows ({y.sum()} fraud)…")

# Train XGBoost (or fallback)
if HAS_XGB:
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
    )
else:
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, random_state=RANDOM_STATE
    )
clf.fit(X_sc, y)

# Train Isolation Forest on legit only
legit_mask = y == 0
iso = IsolationForest(
    n_estimators=100,
    contamination=N_FRAUD / (N_LEGIT + N_FRAUD),
    random_state=RANDOM_STATE,
)
iso.fit(X_sc[legit_mask])

# Quick eval
from sklearn.metrics import roc_auc_score
proba = clf.predict_proba(X_sc)[:, 1]
auc = roc_auc_score(y, proba)
print(f"Demo XGBoost AUC-ROC (synthetic data): {auc:.4f}")
print("NOTE: This is a DEMO model. Run notebooks/model_training.ipynb for real metrics.")

# Save
MODEL_DIR = Path(__file__).parent
artefacts = {"xgb": clf, "iso": iso}
joblib.dump(artefacts, MODEL_DIR / "fraud_model.pkl", compress=3)
joblib.dump(scaler,    MODEL_DIR / "scaler.pkl",      compress=3)
print(f"\nSaved artefacts to {MODEL_DIR}/")
print("  fraud_model.pkl")
print("  scaler.pkl")
