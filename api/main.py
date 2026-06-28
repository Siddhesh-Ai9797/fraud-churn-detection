# api/main.py
# Job: Serve fraud predictions via a REST API.
# Run with: uvicorn api.main:app --reload

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ── 1. App Setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring for credit card transactions.",
    version="1.0.0"
)


# ── 2. Load Model, Scaler, Threshold at Startup ───────────────────────────────
# Loaded once when server starts — not on every request.
# Loading on every request would be extremely slow.

MODEL_PATH     = "outputs/model.pkl"
SCALER_PATH    = "outputs/scaler.pkl"
THRESHOLD_PATH = "outputs/threshold.txt"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run src/evaluate.py first.")

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {SCALER_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"Scaler not found at {SCALER_PATH}. Run src/evaluate.py first.")

try:
    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())
    print(f"Threshold loaded: {threshold}")
except FileNotFoundError:
    threshold = 0.5
    print(f"Threshold file not found. Using default: {threshold}")


# ── 3. Request Schema ─────────────────────────────────────────────────────────
# Pydantic BaseModel defines exactly what JSON the API expects.
# If any field is missing or wrong type, FastAPI returns a clear 422 error.

class Transaction(BaseModel):
    V1: float;  V2: float;  V3: float;  V4: float
    V5: float;  V6: float;  V7: float;  V8: float
    V9: float;  V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    Amount: float
    Time: float


# ── 4. Health Check ───────────────────────────────────────────────────────────
# Kubernetes uses this to know if the container is alive.
# If this returns 200 OK, Kubernetes keeps it running.
# If it fails, Kubernetes restarts the container automatically.

@app.get("/health")
def health():
    return {
        "status"    : "healthy",
        "model"     : MODEL_PATH,
        "threshold" : threshold
    }


# ── 5. Predict Endpoint ───────────────────────────────────────────────────────

@app.post("/predict")
def predict(transaction: Transaction):
    """
    Takes a raw transaction and returns a fraud decision.

    Flow:
    1. Scale Amount and Time using saved scaler
    2. Build feature array in correct order (V1-V28 + Amount_scaled + Time_scaled)
    3. Get fraud probability from model
    4. Apply business threshold to get BLOCK or APPROVE
    5. Return decision + probability + threshold used
    """
    try:
        # Step 1 — Scale Amount and Time separately
        # scaler was fitted on [Amount, Time] together during training
        amount_scaled = scaler.transform([[transaction.Amount]])[0][0]
        time_scaled   = scaler.transform([[transaction.Time]])[0][0]

        # Step 2 — Build feature array in exact training order
        features = np.array([[
            transaction.V1,  transaction.V2,  transaction.V3,  transaction.V4,
            transaction.V5,  transaction.V6,  transaction.V7,  transaction.V8,
            transaction.V9,  transaction.V10, transaction.V11, transaction.V12,
            transaction.V13, transaction.V14, transaction.V15, transaction.V16,
            transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24,
            transaction.V25, transaction.V26, transaction.V27, transaction.V28,
            amount_scaled, time_scaled
        ]])  # shape: (1, 30)

        # Step 3 — Get fraud probability
        fraud_probability = model.predict_proba(features)[0][1]

        # Step 4 — Apply business threshold
        decision = "BLOCK" if fraud_probability >= threshold else "APPROVE"

        return {
            "decision"          : decision,
            "fraud_probability" : round(float(fraud_probability), 4),
            "threshold_used"    : threshold,
            "message"           : (
                f"Transaction BLOCKED. Fraud probability {fraud_probability:.2%} exceeds threshold {threshold}."
                if decision == "BLOCK"
                else
                f"Transaction APPROVED. Fraud probability {fraud_probability:.2%} is below threshold {threshold}."
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── 6. Batch Predict Endpoint ─────────────────────────────────────────────────
# Transactions often arrive in batches in production.
# Much faster than calling /predict 100 times in a loop.

class BatchRequest(BaseModel):
    transactions: list[Transaction]

@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):
    results = []
    for txn in batch.transactions:
        result = predict(txn)
        results.append(result)
    return {"predictions": results, "count": len(results)}


# ── 7. Model Info Endpoint ────────────────────────────────────────────────────

@app.get("/model/info")
def model_info():
    return {
        "model_path"        : MODEL_PATH,
        "threshold"         : threshold,
        "cost_missed_fraud" : "Rs.10,000",
        "cost_false_alarm"  : "Rs.500",
        "features"          : "V1-V28 + Amount + Time (30 total)",
        "scaler"            : "StandardScaler fitted on training data"
    }