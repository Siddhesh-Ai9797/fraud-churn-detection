# src/train.py
# Job: Train XGBoost with class weighting, log everything to MLflow, save model to disk.
# Airflow calls run_training() as Step 2 in the retraining pipeline.

import os
import pickle
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support
)
from preprocess import run_preprocessing


# ── 1. Compute Class Weight ───────────────────────────────────────────────────

def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    XGBoost needs to know how imbalanced the data is.
    scale_pos_weight = normal_count / fraud_count
    This tells XGBoost: missing a fraud = scale_pos_weight times more costly
    than missing a normal transaction.
    """
    fraud_count  = y_train.sum()
    normal_count = (y_train == 0).sum()
    ratio        = normal_count / fraud_count
    print(f"Class ratio (scale_pos_weight): {ratio:.2f}")
    return ratio


# ── 2. Train ──────────────────────────────────────────────────────────────────

def train_model(X_train: np.ndarray, y_train: np.ndarray, scale_pos_weight: float) -> XGBClassifier:
    """
    Train XGBoost with class weighting.
    We use class weighting over SMOTE here because:
    - Precision 0.88 vs SMOTE's 0.73 (fewer false alarms)
    - Recall 0.85 — still catches most fraud
    - No synthetic data generation needed, simpler and faster
    """
    clf = XGBClassifier(
        scale_pos_weight=scale_pos_weight,  # fraud penalty multiplier
        n_estimators=100,                   # number of trees to build
        max_depth=6,                        # how deep each tree can go
        learning_rate=0.1,                  # how fast each tree corrects mistakes
        subsample=0.8,                      # use 80% of rows per tree (prevents overfitting)
        colsample_bytree=0.8,               # use 80% of features per tree (prevents overfitting)
        random_state=42,                    # reproducibility
        eval_metric="aucpr",                # optimize for precision-recall AUC (better than accuracy for imbalance)
        verbosity=0                         # stay quiet during training
    )

    clf.fit(X_train, y_train)
    print("Model training complete.")
    return clf


# ── 3. Evaluate ───────────────────────────────────────────────────────────────

def evaluate_model(clf: XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model at default threshold (0.5) first.
    We do business cost threshold optimization in evaluate.py.
    Returns a dict of metrics for MLflow logging.
    """
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # fraud probability for each transaction

    auc = roc_auc_score(y_test, y_proba)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average="binary")

    print("\nEvaluation at default threshold (0.5):")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"AUC-ROC: {auc:.4f}")

    return {
        "auc_roc"         : auc,
        "fraud_precision" : p,
        "fraud_recall"    : r,
        "fraud_f1"        : f
    }


# ── 4. Save Model ─────────────────────────────────────────────────────────────

def save_model(clf: XGBClassifier, path: str = "outputs/model.pkl"):
    """
    Save trained model to disk using pickle.
    FastAPI loads this at startup to serve predictions.
    MLflow also logs its own copy — this local copy is for FastAPI.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved → {path}")


def load_model(path: str = "outputs/model.pkl") -> XGBClassifier:
    """Load model from disk. Called by FastAPI at startup."""
    with open(path, "rb") as f:
        clf = pickle.load(f)
    return clf


# ── 5. Master Function ────────────────────────────────────────────────────────

def run_training(data_path: str = "data/creditcard.csv"):
    """
    Master function. Airflow calls this single function as Step 2.
    Runs preprocessing → trains model → evaluates → logs to MLflow → saves model.
    Everything is logged to MLflow so you can compare runs in the dashboard.
    """
    # Step 1 — Preprocess
    X_train, X_test, y_train, y_test = run_preprocessing(data_path)

    # Step 2 — Class weight
    scale_pos_weight = compute_scale_pos_weight(y_train)

    # Step 3 — Train and log to MLflow
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="xgb_class_weight"):

        # Log hyperparameters
        params = {
            "model"             : "XGBClassifier",
            "scale_pos_weight"  : scale_pos_weight,
            "n_estimators"      : 100,
            "max_depth"         : 6,
            "learning_rate"     : 0.1,
            "subsample"         : 0.8,
            "colsample_bytree"  : 0.8,
            "imbalance_strategy": "class_weighting"
        }
        mlflow.log_params(params)

        # Train
        clf = train_model(X_train, y_train, scale_pos_weight)

        # Evaluate
        metrics = evaluate_model(clf, X_test, y_test)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model to MLflow (separate from local pickle)
        mlflow.xgboost.log_model(clf, "fraud_model")

        print(f"\nMLflow run logged.")
        print(f"AUC-ROC        : {metrics['auc_roc']:.4f}")
        print(f"Fraud Precision: {metrics['fraud_precision']:.4f}")
        print(f"Fraud Recall   : {metrics['fraud_recall']:.4f}")
        print(f"Fraud F1       : {metrics['fraud_f1']:.4f}")

    # Step 4 — Save locally for FastAPI
    save_model(clf)

    return clf, X_test, y_test


# ── Run directly from terminal ────────────────────────────────────────────────
# python src/train.py

if __name__ == "__main__":
    clf, X_test, y_test = run_training()
    print("\nTraining pipeline complete.")