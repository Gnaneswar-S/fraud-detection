"""
Standalone Model Training Script — auto-detects column names
Works with: train_transaction.csv  OR  fraud_train_preprocessed.csv
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE; SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: pip install xgboost"); sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
MODEL_DIR = Path("model")

COLUMN_ALIASES = {
    "transactionamt":"TransactionAmt","transaction_amt":"TransactionAmt","amount":"TransactionAmt","amt":"TransactionAmt",
    "transactiondt":"TransactionDT","transaction_dt":"TransactionDT",
    "productcd":"ProductCD","product_cd":"ProductCD","product":"ProductCD",
    "p_emaildomain":"P_emaildomain","p_email_domain":"P_emaildomain","emaildomain":"P_emaildomain",
    "c1":"C1","c2":"C2","d1":"D1","d15":"D15",
    "v258":"V258","v308":"V308",
    "isfraud":"isFraud","is_fraud":"isFraud","fraud":"isFraud","label":"isFraud","target":"isFraud",
}

HIGH_RISK_DOMAINS = {"anonymous.com","mail.com","protonmail.com","guerrillamail.com","throwam.com","yopmail.com"}
PRODUCT_CODES = {"W":0,"H":1,"C":2,"S":3,"R":4}

def normalise_columns(df):
    rename = {c: COLUMN_ALIASES[c.lower().strip()] for c in df.columns if c.lower().strip() in COLUMN_ALIASES and COLUMN_ALIASES[c.lower().strip()] != c}
    if rename:
        logger.info("Renaming columns: %s", rename)
        df = df.rename(columns=rename)
    return df

def check_cols(df):
    missing = [c for c in ["TransactionAmt","TransactionDT","ProductCD","isFraud"] if c not in df.columns]
    if missing:
        logger.error("MISSING COLUMNS: %s", missing)
        logger.error("Your CSV has: %s", list(df.columns))
        sys.exit(1)

def col(df, name, default):
    return df[name] if name in df.columns else pd.Series(default, index=df.index)

def engineer_features(df):
    d = pd.DataFrame(index=df.index)
    amt_mean, amt_std = float(df["TransactionAmt"].mean()), float(df["TransactionAmt"].std())
    d["log_transaction_amt"]    = np.log1p(df["TransactionAmt"])
    d["amt_zscore"]             = (df["TransactionAmt"] - amt_mean) / (amt_std + 1e-9)
    hour                        = (df["TransactionDT"] % 86400) / 3600
    d["time_of_day_bucket"]     = (hour / 4).astype(int)
    d["is_night_txn"]           = ((hour >= 22) | (hour < 6)).astype(int)
    days                        = df["TransactionDT"] / 86400
    d["is_weekend"]             = (days.astype(int) % 7 >= 5).astype(int)
    d["days_since_ref"]         = days
    d["velocity_1h_flag"]       = 0
    d["high_risk_email_domain"] = col(df,"P_emaildomain","").fillna("").str.lower().isin(HIGH_RISK_DOMAINS).astype(int)
    d["is_credit"]              = (col(df,"card_type","").fillna("").str.lower() == "credit").astype(int)
    d["product_cd_encoded"]     = df["ProductCD"].map(PRODUCT_CODES).fillna(0)
    d["addr1_missing"]          = col(df,"addr1", np.nan).isnull().astype(int)
    dist1                       = col(df,"dist1", np.nan)
    d["dist1_log"]              = np.log1p(dist1.fillna(0))
    d["dist1_missing"]          = dist1.isnull().astype(int)
    c1                          = col(df,"C1",1.0).fillna(1)
    c2                          = col(df,"C2",1.0).fillna(1)
    d["c_ratio"]                = c1 / (c2 + 1)
    d["d1_missing"]             = col(df,"D1",np.nan).isnull().astype(int)
    d["d15_missing"]            = col(df,"D15",np.nan).isnull().astype(int)
    d["card1_norm"]             = col(df,"card1",9500.0).fillna(9500) / 18396.0
    d["card2_norm"]             = col(df,"card2",111.0).fillna(111) / 600.0
    d["v258_flag"]              = (col(df,"V258",0.0).fillna(0) > 0).astype(int)
    d["v308_flag"]              = (col(df,"V308",0.0).fillna(0) > 0).astype(int)
    d["amt_x_c1"]               = df["TransactionAmt"] * c1
    d["amt_bin"]                = (df["TransactionAmt"] / 50).clip(upper=4).astype(int)
    return d, {"TRAIN_AMT_MEAN": round(amt_mean,4), "TRAIN_AMT_STD": round(amt_std,4)}

def train(data_path, sample=None, tune=False, n_iter=20, output_dir=MODEL_DIR, identity_path=None):
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s …", data_path)
    df = pd.read_csv(data_path, low_memory=False)
    # 🔥 Merge identity data if provided
    if identity_path:
        try:
            logger.info("Loading identity data from %s …", identity_path)
            df_id = pd.read_csv(identity_path, low_memory=False)
            df = df.merge(df_id, on="TransactionID", how="left")
            logger.info("After merging identity: %s", df.shape)
        except Exception as e:
            logger.warning("Failed to merge identity data: %s", e)
    logger.info("Raw shape: %s   columns: %s", df.shape, list(df.columns[:10]))

    df = normalise_columns(df)
    check_cols(df)

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
        logger.info("Sampled %d rows.", len(df))

    fraud_rate = df["isFraud"].mean() * 100
    logger.info("Fraud rate: %.2f%%  (%d / %d)", fraud_rate, int(df["isFraud"].sum()), len(df))

    logger.info("Engineering features …")
    X, pop_stats = engineer_features(df)
    y = df["isFraud"].values
    logger.info("Feature matrix: %s", X.shape)
    logger.info("TRAIN_AMT_MEAN=%.2f  TRAIN_AMT_STD=%.2f  <- update api/preprocessing.py",
                pop_stats["TRAIN_AMT_MEAN"], pop_stats["TRAIN_AMT_STD"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if SMOTE_AVAILABLE:
        logger.info("Applying SMOTE …")
        X_res, y_res = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train_sc, y_train)
        logger.info("After SMOTE: %s", np.bincount(y_res))
    else:
        logger.warning("SMOTE not available — run: pip install imbalanced-learn")
        X_res, y_res = X_train_sc, y_train

    logger.info("Training XGBoost …")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="auc", early_stopping_rounds=30,
        random_state=42, tree_method="hist", device="cpu",
    )
    xgb_model.fit(X_res, y_res, eval_set=[(X_test_sc, y_test)], verbose=100)

    logger.info("Training Isolation Forest …")
    iso = IsolationForest(n_estimators=200, contamination=fraud_rate/100, random_state=42, n_jobs=-1)
    iso.fit(X_train_sc[y_train == 0])

    logger.info("Evaluating ensemble …")
    xgb_p  = xgb_model.predict_proba(X_test_sc)[:, 1]
    iso_p  = 1 / (1 + np.exp(iso.decision_function(X_test_sc)))
    ens_p  = 0.75 * xgb_p + 0.25 * iso_p
    ens_pred = (ens_p >= 0.5).astype(int)

    metrics = {
        "auc_roc":   round(float(roc_auc_score(y_test, ens_p)), 4),
        "precision": round(float(precision_score(y_test, ens_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, ens_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, ens_pred, zero_division=0)), 4),
        "fraud_rate_pct":    round(fraud_rate, 4),
        "train_rows":        int(len(y_train)),
        "test_rows":         int(len(y_test)),
        "n_features":        int(X.shape[1]),
        "model_version":     "1.0.0",
        "population_stats":  pop_stats,
        "training_time_sec": round(time.perf_counter() - t0, 1),
    }

    print("\n" + "="*52)
    print("  RESULTS — COPY THESE TO YOUR RESUME")
    print("="*52)
    print(f"  AUC-ROC   : {metrics['auc_roc']}   <-- PUT THIS ON RESUME")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1        : {metrics['f1']}")
    print(f"  Train rows: {metrics['train_rows']:,}")
    print("="*52)
    print(classification_report(y_test, ens_pred, target_names=["Legit","Fraud"]))

    fi = pd.DataFrame({"feature": X.columns, "importance": xgb_model.feature_importances_}).sort_values("importance", ascending=False)
    fi.to_csv(output_dir / "feature_importance.csv", index=False)

    joblib.dump({"xgb": xgb_model, "iso": iso}, output_dir / "fraud_model.pkl", compress=3)
    joblib.dump(scaler, output_dir / "scaler.pkl", compress=3)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved fraud_model.pkl, scaler.pkl, metrics.json, feature_importance.csv → %s/", output_dir)
    logger.info("Done in %.0f seconds.", time.perf_counter() - t0)
    return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data",       required=True)
    p.add_argument("--sample",     type=int, default=None)
    p.add_argument("--identity", default=None)
    p.add_argument("--tune",       action="store_true")
    p.add_argument("--n-iter",     type=int, default=20)
    p.add_argument("--output-dir", default="model")
    args = p.parse_args()
    r = train(args.data, args.sample, args.tune, args.n_iter, Path(args.output_dir), args.identity)
    print(f"\nFinal AUC-ROC = {r['auc_roc']}  <-- update your resume")
    print(f"Full metrics  -> {args.output_dir}\\metrics.json")