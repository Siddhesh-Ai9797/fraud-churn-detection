# src/preprocess.py
# Job: Load raw data, scale features, split into train/test, save scaler for prediction time.
# Airflow calls run_preprocessing() as the first step in the retraining pipeline.

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and return dataframe."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows | Fraud rate: {df['Class'].mean():.4%}")
    return df


# ── 2. Scale ──────────────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame):
    """
    Scale Amount and Time using StandardScaler.
    V1-V28 are already PCA-scaled, so we leave them alone.
    Returns scaled dataframe and the fitted scaler object.
    We return the scaler because we must save it and reuse it at prediction time.
    Using a different scaler at prediction time = wrong mean/std = broken model.
    """
    scaler = StandardScaler()

    df = df.copy()  # never modify the original dataframe
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])

    df = df.drop(columns=["Amount", "Time"])  # drop raw columns, keep scaled ones

    return df, scaler


# ── 3. Features and Target ────────────────────────────────────────────────────

def get_features_and_target(df: pd.DataFrame):
    """
    Split dataframe into X (inputs) and y (what we predict).
    X = V1 to V28 + Amount_scaled + Time_scaled  → 30 features total
    y = Class column (0 = normal, 1 = fraud)
    """
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]

    X = df[feature_cols].values  # numpy array, models need numpy not dataframes
    y = df["Class"].values

    print(f"X shape: {X.shape} | Frauds: {y.sum()} | Normal: {(y == 0).sum()}")
    return X, y


# ── 4. Split ──────────────────────────────────────────────────────────────────

def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split.
    stratify=y ensures fraud ratio is maintained in both train and test.
    Without this, random splitting might put all frauds in train and zero in test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Train frauds: {y_train.sum()} | Test frauds: {y_test.sum()}")

    return X_train, X_test, y_train, y_test


# ── 5. Save / Load Scaler ─────────────────────────────────────────────────────

def save_scaler(scaler, path: str = "outputs/scaler.pkl"):
    """
    Freeze the fitted scaler to disk using pickle.
    FastAPI loads this at prediction time to scale incoming transactions
    using the exact same mean and std that was used during training.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:  # wb = write binary
        pickle.dump(scaler, f)
    print(f"Scaler saved → {path}")


def load_scaler(path: str = "outputs/scaler.pkl"):
    """Load scaler from disk. Called by FastAPI at startup."""
    with open(path, "rb") as f:  # rb = read binary
        scaler = pickle.load(f)
    return scaler


# ── 6. Master Function ────────────────────────────────────────────────────────

def run_preprocessing(data_path: str = "data/creditcard.csv"):
    """
    Master function. Airflow calls this single function as Step 1.
    Runs all steps in order and returns everything train.py needs.
    """
    df                            = load_data(data_path)
    df_scaled, scaler             = scale_features(df)
    save_scaler(scaler)
    X, y                          = get_features_and_target(df_scaled)
    X_train, X_test, y_train, y_test = split_data(X, y)

    return X_train, X_test, y_train, y_test


# ── Run directly from terminal ────────────────────────────────────────────────
# python src/preprocess.py
# If another file imports this, this block is skipped automatically.

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_preprocessing()
    print("Preprocessing complete.")