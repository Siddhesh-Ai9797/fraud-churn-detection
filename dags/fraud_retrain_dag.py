# dags/fraud_retrain_dag.py
# Job: Automatically retrain the fraud detection model every week.
# Airflow reads this file and schedules it to run every Monday at 2am.
# Each step runs in order. If one step fails, the rest don't run.
# You can also trigger it manually from the Airflow UI anytime.

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import sys
import os

# Add src to path so Airflow can find our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


# ── 1. Default Arguments ──────────────────────────────────────────────────────
# These settings apply to every task in the DAG unless overridden.

default_args = {
    "owner"             : "sidpa",              # who owns this pipeline
    "depends_on_past"   : False,                # don't wait for last week's run to succeed
    "start_date"        : datetime(2026, 1, 1), # when to start scheduling
    "email_on_failure"  : False,                # set to True and add email to get alerts
    "email_on_retry"    : False,
    "retries"           : 1,                    # if a task fails, retry once
    "retry_delay"       : timedelta(minutes=5)  # wait 5 minutes before retrying
}


# ── 2. DAG Definition ─────────────────────────────────────────────────────────
# DAG = Directed Acyclic Graph.
# Directed   = tasks flow in one direction (no going back)
# Acyclic    = no loops (task A can't depend on task B if B depends on A)
# Graph      = a network of connected tasks
#
# schedule_interval="0 2 * * 1" is a cron expression meaning:
# minute=0, hour=2, any day of month, any month, day of week=1 (Monday)
# → runs every Monday at 2:00 AM

with DAG(
    dag_id="fraud_detection_retraining",
    default_args=default_args,
    description="Weekly retraining pipeline for fraud detection model",
    schedule_interval="0 2 * * 1",  # every Monday at 2am
    catchup=False,                  # don't backfill missed runs
    tags=["fraud", "ml", "retraining"]
) as dag:


    # ── Task 1: Preprocess ────────────────────────────────────────────────────
    # Loads raw data, scales features, splits into train/test, saves scaler.
    # If this fails → training never starts → safe, no bad model gets deployed.

    def task_preprocess():
        from preprocess import run_preprocessing
        print("Starting preprocessing...")
        X_train, X_test, y_train, y_test = run_preprocessing(
            data_path="/opt/airflow/data/creditcard.csv"
        )
        print(f"Preprocessing complete. Train: {len(X_train):,} | Test: {len(X_test):,}")
        # Note: we don't pass data between tasks via return values.
        # Each task saves its outputs to disk and the next task reads from disk.
        # This is the standard Airflow pattern.

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess
    )


    # ── Task 2: Train ─────────────────────────────────────────────────────────
    # Trains XGBoost with class weighting.
    # Logs all parameters and metrics to MLflow automatically.
    # Saves model to outputs/model.pkl for FastAPI to use.

    def task_train():
        from train import run_training
        print("Starting training...")
        clf, X_test, y_test = run_training(
            data_path="/opt/airflow/data/creditcard.csv"
        )
        print("Training complete. Model saved to outputs/model.pkl")

    train_task = PythonOperator(
        task_id="train",
        python_callable=task_train
    )


    # ── Task 3: Evaluate ──────────────────────────────────────────────────────
    # Finds the business cost threshold that minimizes total Rs. loss.
    # Saves best threshold to outputs/threshold.txt
    # FastAPI reads this file — so threshold updates automatically after retraining.

    def task_evaluate():
        from evaluate import run_evaluation
        print("Starting evaluation...")
        clf, best_thresh = run_evaluation(
            data_path="/opt/airflow/data/creditcard.csv"
        )
        print(f"Evaluation complete. Best threshold: {best_thresh:.2f}")

    evaluate_task = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate
    )


    # ── Task 4: Validate ──────────────────────────────────────────────────────
    # Sanity check before deploying.
    # Reads the saved threshold and checks model meets minimum performance bar.
    # If fraud recall drops below 0.75 → raise an error → pipeline stops.
    # This prevents a degraded model from going to production automatically.
    # Cricket analogy: you don't field a bowler with 10+ economy. Minimum bar required.

    def task_validate():
        import pickle
        import numpy as np
        from sklearn.metrics import recall_score
        from preprocess import run_preprocessing

        print("Starting validation...")

        # Load saved model
        with open("outputs/model.pkl", "rb") as f:
            clf = pickle.load(f)

        # Load saved threshold
        with open("outputs/threshold.txt", "r") as f:
            threshold = float(f.read().strip())

        # Quick evaluation on test set
        X_train, X_test, y_train, y_test = run_preprocessing(
            data_path="/opt/airflow/data/creditcard.csv"
        )

        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= threshold).astype(int)
        recall  = recall_score(y_test, y_pred)

        print(f"Fraud recall at threshold {threshold:.2f}: {recall:.4f}")

        # Minimum performance bar
        MIN_RECALL = 0.75
        if recall < MIN_RECALL:
            raise ValueError(
                f"Model failed validation. Recall {recall:.4f} is below minimum {MIN_RECALL}. "
                f"Not deploying. Check MLflow for details."
            )

        print(f"Validation passed. Recall {recall:.4f} >= {MIN_RECALL}")

    validate_task = PythonOperator(
        task_id="validate",
        python_callable=task_validate
    )


    # ── Task 5: Deploy ────────────────────────────────────────────────────────
    # In a real production system this would push the new model to S3,
    # update the model registry in MLflow, and trigger a Kubernetes rolling update.
    # For local development, we just confirm the files are in place
    # and print deployment instructions.

    def task_deploy():
        import os
        from datetime import datetime

        print("Starting deployment...")

        # Check all required files exist
        required_files = [
            "outputs/model.pkl",
            "outputs/scaler.pkl",
            "outputs/threshold.txt"
        ]

        for path in required_files:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file missing: {path}")
            size = os.path.getsize(path)
            print(f"✅ {path} ({size:,} bytes)")

        # Read and log current threshold
        with open("outputs/threshold.txt", "r") as f:
            threshold = float(f.read().strip())

        # Log deployment timestamp
        deploy_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\nDeployment summary:")
        print(f"Timestamp  : {deploy_time}")
        print(f"Threshold  : {threshold:.2f}")
        print(f"Status     : SUCCESS")
        print(f"\nTo apply new model: docker build -t fraud-detection . && docker run -p 8000:8000 fraud-detection")

        # In production you would do:
        # import boto3
        # s3.upload_file("outputs/model.pkl", "your-bucket", "models/fraud/model.pkl")
        # Then trigger Kubernetes rolling update via kubectl or Kubernetes API

    deploy_task = PythonOperator(
        task_id="deploy",
        python_callable=task_deploy
    )


    # ── Task Order ────────────────────────────────────────────────────────────
    # >> means "then run".
    # preprocess → train → evaluate → validate → deploy
    # If any task fails, the pipeline stops there.
    # The failed task shows in red in the Airflow UI.

    preprocess_task >> train_task >> evaluate_task >> validate_task >> deploy_task