"""
Automated Pytest Suite for Boston Housing Price Prediction Application.
Validates model loading, prediction accuracy, schema validation, batch scoring,
and error handling.
"""

import io
import json
import pytest
from app import create_app

@pytest.fixture
def client():
    """Configures a Flask test client."""
    app = create_app("development")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    """Verifies that the main web UI renders properly."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"NeuroEstate" in response.data
    assert b"Boston Housing Price AI" in response.data

def test_health_check(client):
    """Verifies that the health endpoint reports healthy model state."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model"]["loaded"] is True
    assert data["model"]["input_features"] == 13

def test_metadata_endpoint(client):
    """Verifies the feature metadata and preset schema."""
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.get_json()
    assert "features" in data
    assert len(data["features"]) == 13
    assert "presets" in data
    assert len(data["presets"]) >= 3
    assert data["architecture"]["total_params"] == 3521

def test_single_prediction_benchmark(client):
    """
    Verifies that the exact test vector from Cell 17 in the notebook
    evaluates to ~28.396 ($28,396).
    """
    benchmark_payload = {
        "CRIM": 0.02, "ZN": 18.0, "INDUS": 2.5, "CHAS": 0.0, "NOX": 0.45,
        "RM": 6.5, "AGE": 65.0, "DIS": 4.5, "RAD": 1.0, "TAX": 300.0,
        "PTRATIO": 15.0, "B": 390.0, "LSTAT": 5.0
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(benchmark_payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    pred = data["prediction"]
    # Check that price is around 28.396
    assert abs(pred["price_k"] - 28.396) < 0.05
    assert pred["price_formatted"] == "$28,396"
    assert "adjusted_price_formatted" in pred
    assert "confidence_interval" in pred
    assert "feature_impacts" in data
    assert len(data["feature_impacts"]) == 13

def test_single_prediction_invalid_payload(client):
    """Verifies validation rejection for non-numeric fields."""
    bad_payload = {
        "CRIM": "invalid_string", "RM": 6.0
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(bad_payload),
        content_type="application/json"
    )
    assert response.status_code == 422
    data = response.get_json()
    assert data["success"] is False
    assert "must be a valid number" in data["error"]

def test_single_prediction_missing_features_imputation(client):
    """Verifies that missing features are imputed with dataset averages and return warnings."""
    partial_payload = {
        "RM": 7.5,
        "CRIM": 0.1
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(partial_payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert len(data["warnings"]) > 0

def test_batch_prediction_csv_file(client):
    """Verifies batch scoring via multipart CSV upload."""
    csv_content = """CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,description
0.02,18,2.5,0,0.45,6.5,65,4.5,1,300,15,390,5,Benchmark Home
0.05,40,3.2,0,0.42,7.2,32,5.8,3,250,14.2,395,4.2,Suburban Executive
"""
    data = {
        "file": (io.BytesIO(csv_content.encode("utf-8")), "test_properties.csv")
    }
    response = client.post(
        "/api/predict-batch",
        data=data,
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    result = response.get_json()
    assert result["success"] is True
    assert result["summary"]["total_rows"] == 2
    assert result["summary"]["successful_predictions"] == 2
    assert len(result["results"]) == 2

def test_batch_prediction_json_records(client):
    """Verifies batch scoring via JSON payload."""
    payload = {
        "records": [
            {"CRIM": 0.02, "ZN": 18, "INDUS": 2.5, "CHAS": 0, "NOX": 0.45, "RM": 6.5, "AGE": 65, "DIS": 4.5, "RAD": 1, "TAX": 300, "PTRATIO": 15, "B": 390, "LSTAT": 5},
            {"CRIM": 0.08, "ZN": 25, "INDUS": 4.0, "CHAS": 1, "NOX": 0.48, "RM": 8.1, "AGE": 45, "DIS": 3.6, "RAD": 4, "TAX": 310, "PTRATIO": 13.5, "B": 392, "LSTAT": 3.8}
        ]
    }
    response = client.post(
        "/api/predict-batch",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["summary"]["successful_predictions"] == 2

def test_export_results_endpoint(client):
    """Verifies CSV generation from prediction results."""
    payload = {
        "results": [
            {
                "row_index": 1,
                "status": "success",
                "price_k": 28.396,
                "price_formatted": "$28,396",
                "adjusted_price_formatted": "$215,807",
                "category": "Upper Mid-Market",
                "percentile": 80.2,
                "lower_bound": "$25,473",
                "upper_bound": "$31,319",
                "features": {"RM": 6.5, "CRIM": 0.02}
            }
        ]
    }
    response = client.post(
        "/api/export-results",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/csv")
    assert b"Predicted_Price_1000s" in response.data
    assert b"28.396" in response.data
