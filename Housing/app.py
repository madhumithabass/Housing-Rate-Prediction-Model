"""
Boston Housing Price Prediction Web Application
Simple, clean, and intuitive Flask application powered by the trained ANN model.
"""

import os
import sys
import socket
import numpy as np
import joblib

# Suppress TensorFlow logging messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ANN_regression.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

# Load model and scaler once on startup
print("Loading trained neural network model...")
model = keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
print("Model and scaler loaded successfully!")

# Feature list in exact order expected by the model
FEATURES = [
    {"id": "CRIM", "name": "Neighborhood Crime Rate", "desc": "Per capita crime rate by town", "default": 0.02, "min": 0.0, "max": 90.0, "step": 0.01, "unit": "crimes/person"},
    {"id": "ZN", "name": "Large Residential Land", "desc": "Zoned for lots over 25,000 sq.ft", "default": 18.0, "min": 0.0, "max": 100.0, "step": 1.0, "unit": "%"},
    {"id": "INDUS", "name": "Commercial / Business Area", "desc": "Non-retail business acres per town", "default": 2.5, "min": 0.0, "max": 30.0, "step": 0.1, "unit": "%"},
    {"id": "CHAS", "name": "Charles River Waterfront", "desc": "Property bounds the Charles River", "default": 0, "type": "select", "options": [{"val": 0, "label": "No (Inland)"}, {"val": 1, "label": "Yes (Waterfront)"}]},
    {"id": "NOX", "name": "Air Pollution Level (NOx)", "desc": "Nitric oxides concentration (lower is cleaner)", "default": 0.45, "min": 0.35, "max": 0.90, "step": 0.01, "unit": "parts/10M"},
    {"id": "RM", "name": "Average Number of Rooms", "desc": "Average rooms per dwelling", "default": 6.5, "min": 3.0, "max": 9.0, "step": 0.1, "unit": "rooms"},
    {"id": "AGE", "name": "Proportion of Older Homes", "desc": "Homes built prior to 1940", "default": 65.0, "min": 0.0, "max": 100.0, "step": 1.0, "unit": "%"},
    {"id": "DIS", "name": "Distance to Employment Centers", "desc": "Distance to Boston job hubs", "default": 4.5, "min": 1.0, "max": 12.5, "step": 0.1, "unit": "miles"},
    {"id": "RAD", "name": "Highway Accessibility", "desc": "Index of access to radial freeways", "default": 1.0, "min": 1.0, "max": 24.0, "step": 1.0, "unit": "index (1-24)"},
    {"id": "TAX", "name": "Property Tax Rate", "desc": "Full-value property tax rate", "default": 300.0, "min": 180.0, "max": 720.0, "step": 10.0, "unit": "$ per $10k"},
    {"id": "PTRATIO", "name": "Pupil-to-Teacher Ratio", "desc": "Students per teacher in local schools", "default": 15.0, "min": 12.0, "max": 23.0, "step": 0.1, "unit": "students/teacher"},
    {"id": "B", "name": "Demographic Index (B)", "desc": "Town demographic index", "default": 390.0, "min": 0.0, "max": 400.0, "step": 5.0, "unit": "index"},
    {"id": "LSTAT", "name": "Socioeconomic Status Index", "desc": "% lower socioeconomic status (lower is wealthier)", "default": 5.0, "min": 1.0, "max": 40.0, "step": 0.1, "unit": "%"}
]

# Quick presets for demo
PRESETS = {
    "default": {
        "name": "Standard Suburban Home",
        "values": {"CRIM": 0.02, "ZN": 18.0, "INDUS": 2.5, "CHAS": 0, "NOX": 0.45, "RM": 6.5, "AGE": 65.0, "DIS": 4.5, "RAD": 1.0, "TAX": 300.0, "PTRATIO": 15.0, "B": 390.0, "LSTAT": 5.0}
    },
    "luxury": {
        "name": "Luxury Waterfront Estate",
        "values": {"CRIM": 0.05, "ZN": 35.0, "INDUS": 3.0, "CHAS": 1, "NOX": 0.42, "RM": 8.0, "AGE": 30.0, "DIS": 4.0, "RAD": 3.0, "TAX": 280.0, "PTRATIO": 13.5, "B": 395.0, "LSTAT": 3.5}
    },
    "budget": {
        "name": "Affordable Starter Home",
        "values": {"CRIM": 2.50, "ZN": 0.0, "INDUS": 18.0, "CHAS": 0, "NOX": 0.65, "RM": 5.2, "AGE": 90.0, "DIS": 2.0, "RAD": 24.0, "TAX": 666.0, "PTRATIO": 20.2, "B": 350.0, "LSTAT": 21.0}
    },
    "city": {
        "name": "Urban City Apartment",
        "values": {"CRIM": 0.30, "ZN": 0.0, "INDUS": 9.0, "CHAS": 0, "NOX": 0.52, "RM": 5.9, "AGE": 75.0, "DIS": 2.5, "RAD": 5.0, "TAX": 380.0, "PTRATIO": 17.0, "B": 385.0, "LSTAT": 10.5}
    }
}

def predict_price_from_inputs(input_dict):
    """Processes 13 feature inputs and returns predicted price."""
    vec = []
    for f in FEATURES:
        val = float(input_dict.get(f["id"], f["default"]))
        vec.append(val)
    
    arr = np.array(vec, dtype=np.float32).reshape(1, -1)
    scaled_arr = scaler.transform(arr)
    raw_pred = model(scaled_arr, training=False).numpy()[0][0]
    price_k_raw = max(1.0, float(raw_pred))
    price_usd = price_k_raw * 1000.0
    adjusted_usd = price_usd * 7.6  # Approximate contemporary inflation value
    
    # Simple market tag
    if price_k_raw < 17:
        market_tag = "Budget Friendly"
    elif price_k_raw < 26:
        market_tag = "Mid-Market Value"
    elif price_k_raw < 35:
        market_tag = "Upper Tier"
    else:
        market_tag = "Luxury Premium"

    return {
        "price_k": round(price_k_raw, 2),
        "price_formatted": f"${price_usd:,.0f}",
        "adjusted_formatted": f"${adjusted_usd:,.0f}",
        "market_tag": market_tag
    }

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    selected_values = {f["id"]: f["default"] for f in FEATURES}
    
    if request.method == "POST":
        for f in FEATURES:
            if f["id"] in request.form:
                try:
                    selected_values[f["id"]] = float(request.form[f["id"]])
                except (ValueError, TypeError):
                    pass
        prediction_result = predict_price_from_inputs(selected_values)
    else:
        # Initial default prediction so the user immediately sees a live result
        prediction_result = predict_price_from_inputs(selected_values)

    return render_template(
        "index.html",
        features=FEATURES,
        presets=PRESETS,
        selected=selected_values,
        result=prediction_result
    )

@app.route("/predict-ajax", methods=["POST"])
def predict_ajax():
    """Simple AJAX endpoint so the page updates seamlessly without reload."""
    data = request.get_json() or {}
    result = predict_price_from_inputs(data)
    return jsonify({"success": True, "result": result})

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

if __name__ == "__main__":
    # Choose port: 5050 by default, or 5000 if free
    target_port = 5050
    if not is_port_in_use(5000):
        target_port = 5000

    print("=" * 60)
    print("  BOSTON HOUSING PRICE PREDICTOR")
    print(f"  Open in your browser: http://127.0.0.1:{target_port}")
    print("=" * 60)
    app.run(host="127.0.0.1", port=target_port, debug=False)
