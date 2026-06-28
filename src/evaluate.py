# src/evaluate.py
# Job: Find the classification threshold that minimizes real business cost.
# Default threshold of 0.5 treats all mistakes equally.
# We know missing fraud costs Rs.10,000 and false alarms cost Rs.500.
# So we sweep all thresholds and find the cheapest one.
# Airflow calls run_evaluation() as Step 3 in the retraining pipeline.

import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import classification_report, precision_recall_fscore_support
from train import run_training


# ── 1. Business Cost Constants ────────────────────────────────────────────────

COST_MISSED_FRAUD = 10_000   # Rs. cost when model misses a real fraud (false negative)
COST_FALSE_ALARM  = 500      # Rs. cost when model flags innocent transaction (false positive)


# ── 2. Threshold Sweep ────────────────────────────────────────────────────────

def find_best_threshold(clf, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Try every threshold from 0.01 to 0.99.
    At each threshold:
        - Count missed frauds (false negatives) → multiply by Rs.10,000
        - Count false alarms (false positives)  → multiply by Rs.500
        - Total cost = sum of both
    Pick the threshold with the lowest total cost.
    Returns best_threshold, best_cost, all thresholds, all costs.
    """
    y_proba     = clf.predict_proba(X_test)[:, 1]  # fraud probability for each transaction
    thresholds  = np.arange(0.01, 1.0, 0.01)
    costs       = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        missed_frauds = ((y_pred == 0) & (y_test == 1)).sum()  # model said normal, actually fraud
        false_alarms  = ((y_pred == 1) & (y_test == 0)).sum()  # model said fraud, actually normal

        total_cost = (missed_frauds * COST_MISSED_FRAUD) + (false_alarms * COST_FALSE_ALARM)
        costs.append(total_cost)

    costs       = np.array(costs)
    best_idx    = np.argmin(costs)
    best_thresh = thresholds[best_idx]
    best_cost   = costs[best_idx]

    return best_thresh, best_cost, thresholds, costs


# ── 3. Evaluate at Best Threshold ─────────────────────────────────────────────

def evaluate_at_threshold(clf, X_test: np.ndarray, y_test: np.ndarray, threshold: float) -> dict:
    """
    Run classification report at a specific threshold.
    Called twice — once at default 0.5, once at best business threshold.
    So you can see exactly how much the threshold shift helped.
    """
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average="binary")

    missed_frauds = ((y_pred == 0) & (y_test == 1)).sum()
    false_alarms  = ((y_pred == 1) & (y_test == 0)).sum()
    total_cost    = (missed_frauds * COST_MISSED_FRAUD) + (false_alarms * COST_FALSE_ALARM)

    print(f"\nResults at threshold {threshold:.2f}:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"Missed frauds : {missed_frauds} × Rs.{COST_MISSED_FRAUD:,} = Rs.{missed_frauds * COST_MISSED_FRAUD:,}")
    print(f"False alarms  : {false_alarms} × Rs.{COST_FALSE_ALARM:,} = Rs.{false_alarms * COST_FALSE_ALARM:,}")
    print(f"Total cost    : Rs.{total_cost:,}")

    return {
        "threshold"     : threshold,
        "fraud_precision": p,
        "fraud_recall"  : r,
        "fraud_f1"      : f,
        "missed_frauds" : int(missed_frauds),
        "false_alarms"  : int(false_alarms),
        "total_cost"    : int(total_cost)
    }


# ── 4. Plot ───────────────────────────────────────────────────────────────────

def plot_cost_curve(thresholds, costs, best_thresh, best_cost, save_path: str = "outputs/cost_vs_threshold.png"):
    """
    Plot total business cost vs classification threshold.
    Red dashed line marks the best threshold.
    Red dot marks the minimum cost point.
    Saved to outputs folder and logged to MLflow.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, costs, color="steelblue", linewidth=2, label="Total Business Cost")
    plt.axvline(best_thresh, color="red", linestyle="--", label=f"Best threshold = {best_thresh:.2f}")
    plt.scatter([best_thresh], [best_cost], color="red", zorder=5, s=80)
    plt.xlabel("Threshold")
    plt.ylabel("Total Business Cost (Rs.)")
    plt.title("Business Cost vs Classification Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved → {save_path}")


# ── 5. Save Best Threshold ────────────────────────────────────────────────────

def save_threshold(threshold: float, path: str = "outputs/threshold.txt"):
    """
    Save best threshold to a text file.
    FastAPI reads this at startup so it uses the correct threshold for every prediction.
    If you retrain and the best threshold changes, FastAPI picks it up automatically.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(threshold))
    print(f"Threshold saved → {path}")


def load_threshold(path: str = "outputs/threshold.txt") -> float:
    """Load threshold from disk. Called by FastAPI at startup."""
    with open(path, "r") as f:
        return float(f.read().strip())


# ── 6. Master Function ────────────────────────────────────────────────────────

def run_evaluation(data_path: str = "data/creditcard.csv"):
    """
    Master function. Airflow calls this single function as Step 3.
    Trains model → finds best threshold → logs everything to MLflow → saves threshold.
    """
    # Step 1 — Train (which internally runs preprocessing)
    clf, X_test, y_test = run_training(data_path)

    # Step 2 — Results at default threshold 0.5 (baseline comparison)
    print("\n── Baseline: Default Threshold 0.5 ──")
    baseline = evaluate_at_threshold(clf, X_test, y_test, threshold=0.5)

    # Step 3 — Find best business cost threshold
    print("\n── Sweeping all thresholds ──")
    best_thresh, best_cost, thresholds, costs = find_best_threshold(clf, X_test, y_test)
    print(f"Best threshold : {best_thresh:.2f}")
    print(f"Minimum cost   : Rs.{best_cost:,}")

    # Step 4 — Results at best threshold
    print("\n── Optimized: Best Business Threshold ──")
    optimized = evaluate_at_threshold(clf, X_test, y_test, threshold=best_thresh)

    # Step 5 — How much did threshold optimization save?
    cost_saved = baseline["total_cost"] - optimized["total_cost"]
    print(f"\nCost saved by threshold optimization: Rs.{cost_saved:,}")

    # Step 6 — Plot
    plot_cost_curve(thresholds, costs, best_thresh, best_cost)

    # Step 7 — Save threshold for FastAPI
    save_threshold(best_thresh)

    # Step 8 — Log everything to MLflow
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run(run_name="threshold_optimization"):
        mlflow.log_params({
            "cost_missed_fraud" : COST_MISSED_FRAUD,
            "cost_false_alarm"  : COST_FALSE_ALARM,
            "best_threshold"    : best_thresh
        })
        mlflow.log_metrics({
            "baseline_cost"         : baseline["total_cost"],
            "optimized_cost"        : optimized["total_cost"],
            "cost_saved"            : cost_saved,
            "optimized_recall"      : optimized["fraud_recall"],
            "optimized_precision"   : optimized["fraud_precision"],
            "optimized_f1"          : optimized["fraud_f1"],
            "missed_frauds"         : optimized["missed_frauds"],
            "false_alarms"          : optimized["false_alarms"]
        })
        mlflow.log_artifact("outputs/cost_vs_threshold.png")

    return clf, best_thresh


# ── Run directly from terminal ────────────────────────────────────────────────
# python src/evaluate.py

if __name__ == "__main__":
    clf, best_thresh = run_evaluation()
    print(f"\nEvaluation complete. Best threshold: {best_thresh:.2f}")