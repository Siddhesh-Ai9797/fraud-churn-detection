import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

import api.main as api
from src.preprocess import scale_features

client = TestClient(api.app)


def transaction(**overrides):
    body = {f"V{i}": 0.0 for i in range(1, 29)}
    body.update(Amount=50.0, Time=40000.0)
    body.update(overrides)
    return body


def test_health_reports_threshold():
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert 0 < body["threshold"] < 1


def test_predict_returns_a_decision():
    body = client.post("/predict", json=transaction()).json()
    assert body["decision"] in {"BLOCK", "APPROVE"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["threshold_used"] == api.threshold


def test_decision_follows_threshold(monkeypatch):
    class FixedModel:
        def __init__(self, p):
            self.p = p

        def predict_proba(self, features):
            return np.array([[1 - self.p, self.p]])

    monkeypatch.setattr(api, "model", FixedModel(api.threshold + 0.01))
    assert client.post("/predict", json=transaction()).json()["decision"] == "BLOCK"
    monkeypatch.setattr(api, "model", FixedModel(api.threshold - 0.01))
    assert client.post("/predict", json=transaction()).json()["decision"] == "APPROVE"


def test_missing_field_is_422():
    body = transaction()
    del body["V14"]
    assert client.post("/predict", json=body).status_code == 422


def test_batch_returns_one_result_per_transaction():
    body = {"transactions": [transaction(), transaction(Amount=900.0)]}
    response = client.post("/predict/batch", json=body).json()
    assert response["count"] == 2
    assert len(response["predictions"]) == 2


def test_per_column_scalers_are_used(monkeypatch):
    amount = StandardScaler().fit([[0.0], [200.0]])      # mean 100, std 100
    time = StandardScaler().fit([[0.0], [100000.0]])     # mean 50000, std 50000
    monkeypatch.setattr(api, "scaler", {"Amount": amount, "Time": time})
    a, t = api.scale_amount_time(300.0, 100000.0)
    assert a == pytest.approx(2.0)
    assert t == pytest.approx(1.0)


def test_preprocess_fits_each_column_separately():
    df = pd.DataFrame({"Amount": [10.0, 30.0], "Time": [1000.0, 3000.0], "Class": [0, 1]})
    scaled, scalers = scale_features(df)
    assert scalers["Amount"].mean_[0] == pytest.approx(20.0)
    assert scalers["Time"].mean_[0] == pytest.approx(2000.0)
    assert scaled["Amount_scaled"].tolist() == pytest.approx([-1.0, 1.0])
