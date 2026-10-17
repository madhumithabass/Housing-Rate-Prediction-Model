# 🏡 NeuroEstate: Boston Housing Price AI Web Application
### Deep Learning Artificial Neural Network (Keras 3 / TensorFlow) & Flask

A production-grade, full-stack housing price valuation web platform powered by a trained Sequential Deep Artificial Neural Network (`ANN_regression.h5`). Features interactive real-time sensitivity analysis, market scenario presets, explainable AI (XAI) feature attribution, batch CSV scoring, responsive visualizations, automated test coverage, and Docker containerization.

---

## 🌟 Key Highlights & Capabilities

- **Deep Neural Network Inference**: Directly integrates your trained `ANN_regression.h5` model (3 hidden Dense layers with 64, 32, and 16 ReLU neurons, and 1 linear output neuron; 3,521 trainable parameters).
- **Exact Benchmark Replication**: Verified against Cell 17 in the training notebook (`[0.02, 18, 2.5, 0, 0.45, 6.5, 65, 4.5, 1, 300, 15, 390, 5]`), reproducing the exact predicted median home value of **$28,396** (`28.395668`).
- **Explainable AI (XAI) Drivers**: Evaluates local marginal feature impacts against regional Boston baseline averages to present an interactive horizontal waterfall/bar chart showing which property traits increased or decreased value.
- **13-Dimensional Radar Footprint**: Real-time multi-axial radar comparing the current home's profile across 13 dimensions against the metropolitan norm.
- **Dual-Synchronized Controls**: Interactive sliders coupled with numeric inputs and a "Live AI" toggle for real-time reactive valuations as you adjust parameters.
- **Scenario Presets**: 1-click loading of realistic property archetypes (*Suburban Executive*, *Charles River Waterfront Estate*, *Urban Academic Flat*, *Affordable Starter Home*, and *Notebook Baseline*).
- **Batch CSV Scoring & Export**: Drag-and-drop CSV dataset upload, processing hundreds of homes concurrently with pagination, filtering, summary statistics, and 1-click CSV result export.
- **Developer REST API**: Fully documented `/api/predict`, `/api/predict-batch`, `/api/metadata`, and `/health` endpoints with copyable cURL, Python, and JavaScript snippets.
- **Production-Ready & Tested**: 100% test pass rate with `pytest`, health checks, security headers, Docker, and `docker-compose`.

---

## 🧠 Neural Network Model Architecture

| Layer | Type | Neurons / Units | Activation | Parameter Count |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Feature Vector | 13 Inputs | None | 0 |
| **Dense 1** | Hidden Layer | 64 | ReLU | 896 |
| **Dense 2** | Hidden Layer | 32 | ReLU | 2,080 |
| **Dense 3** | Hidden Layer | 16 | ReLU | 528 |
| **Output** | Linear Regressor | 1 | Linear | 17 |
| **Total** | **Sequential ANN** | **113 Neurons** | — | **3,521 Parameters** |

### Verified Holdout Evaluation Metrics
- **Mean Absolute Error (MAE)**: $\pm \$2.923\text{k}$ ($\pm \$2,923$)
- **Mean Squared Error (MSE)**: $19.067$
- **Root Mean Squared Error (RMSE)**: $\$4.367\text{k}$
- **Coefficient of Determination ($R^2$)**: $0.771$ ($77.1\%$ explained variance)
- **Inference Latency**: $\approx 9\text{ ms}$ on CPU per sample

---

## 📊 Dataset Feature Dictionary (13 Features)

| Feature | Feature Name | Description | Range | Dataset Mean |
| :--- | :--- | :--- | :--- | :--- |
| `CRIM` | Per Capita Crime Rate | Per capita crime rate by town | 0.0 - 90.0 | 3.09 |
| `ZN` | Large Residential Land % | Proportion of residential land zoned for lots > 25k sq.ft | 0.0 - 100.0% | 10.90% |
| `INDUS` | Non-Retail Business Acres | Proportion of non-retail business acres per town | 0.0 - 30.0% | 11.27% |
| `CHAS` | Charles River Tract | Bounds Charles River (= 1) or inland (= 0) | 0 or 1 | 0.10 |
| `NOX` | Nitric Oxides Concentration | Air pollution concentration (parts per 10 million) | 0.35 - 0.90 | 0.54 |
| `RM` | Average Rooms per Dwelling | Average number of rooms per home | 3.0 - 9.0 | 6.35 |
| `AGE` | Pre-1940 Built Homes % | Proportion of owner-occupied units built before 1940 | 1.0 - 100.0% | 66.85% |
| `DIS` | Distance to Employment Hubs | Weighted distances to 5 Boston employment centres | 1.0 - 12.5 mi | 4.01 mi |
| `RAD` | Radial Highway Accessibility | Accessibility index to radial freeways (1-24) | 1 - 24 | 9.98 |
| `TAX` | Property Tax Rate | Full-value property-tax rate per $10,000 | $180 - $720 | $417.50 |
| `PTRATIO`| Pupil-Teacher Ratio | Pupil-teacher ratio by town school district | 12.0 - 23.0 | 18.38 |
| `B` | Demographic Index B | $1000(B_k - 0.63)^2$ where $B_k$ is town proportion | 0.0 - 400.0 | 364.16 |
| `LSTAT` | Lower Socioeconomic Status % | Percentage of lower socioeconomic status population | 1.0 - 40.0% | 12.31% |

**Target (`MEDV`)**: Median value of owner-occupied homes in $1,000s (e.g. 28.396 = $28,396). Also translated to contemporary estimated purchasing power (~$215,800).

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ installed
- Git

### 1. Clone & Setup Workspace
```bash
git clone <repo-url>
cd Housing
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests/test_app.py -v
```

### 4. Start the Application
```bash
python run.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🐳 Docker Deployment

### Run with Docker Compose (Recommended)
```bash
docker-compose up --build
```
The application will be accessible at `http://localhost:5000` with automated health checks enabled.

### Run with Docker CLI
```bash
docker build -t boston-housing-ai .
docker run -p 5000:5000 boston-housing-ai
```

---

## 📡 REST API Documentation

### 1. Single Property Prediction
- **URL**: `/api/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Sample Request**:
```json
{
  "CRIM": 0.02,
  "ZN": 18.0,
  "INDUS": 2.5,
  "CHAS": 0.0,
  "NOX": 0.45,
  "RM": 6.5,
  "AGE": 65.0,
  "DIS": 4.5,
  "RAD": 1.0,
  "TAX": 300.0,
  "PTRATIO": 15.0,
  "B": 390.0,
  "LSTAT": 5.0
}
```

**Sample Response**:
```json
{
  "success": true,
  "prediction": {
    "price_k": 28.396,
    "price_formatted": "$28,396",
    "adjusted_price_formatted": "$215,807",
    "confidence_interval": {
      "mae_k": 2.923,
      "lower_formatted": "$25,473",
      "upper_formatted": "$31,319",
      "confidence_level": "68% (±1 MAE)"
    },
    "category": {
      "tier": "Upper Mid-Market",
      "badge_class": "badge-upper",
      "color": "#6366f1"
    },
    "percentile": 80.2
  },
  "feature_impacts": [ ... ],
  "normalized_profile": { ... }
}
```

### 2. Batch Prediction
- **URL**: `/api/predict-batch`
- **Method**: `POST`
- **Form Data**: `file=@sample_boston_properties.csv`

Returns summary statistics and an array of individual valuations.

### 3. Health Check
- **URL**: `/health`
- **Method**: `GET`
```json
{
  "status": "healthy",
  "service": "Boston Housing ANN Prediction API",
  "uptime_seconds": 128.4,
  "model": {
    "name": "Boston Housing ANN Regression",
    "loaded": true,
    "input_features": 13
  }
}
```

---

## 📁 Repository Structure

```
Housing/
├── ANN_regression.h5           # Trained Artificial Neural Network model
├── ANN for Regression.ipynb    # Original training Jupyter Notebook
├── housing.md                  # Project specification prompt
├── scaler.joblib               # Fitted StandardScaler artifact
├── model_metadata.json         # Architecture specs, feature bounds, & presets
├── predictor.py                # Core ML inference engine & XAI drivers
├── app.py                      # Production Flask application & API routes
├── config.py                   # Environment & runtime configurations
├── run.py                      # Server bootstrap entrypoint
├── sample_batch.csv            # Sample CSV file for batch scoring
├── requirements.txt            # Locked dependencies
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Container orchestration configuration
├── templates/
│   └── index.html              # Responsive glassmorphic single-page app
├── static/
│   ├── css/
│   │   └── style.css           # Modern design system & themes
│   └── js/
│       └── app.js              # Client controller, Chart.js & reactive UI
└── tests/
    └── test_app.py             # Pytest test suite (9 test cases)
```

---

## 🏆 Portfolio Presentation Notes

This project demonstrates end-to-end Machine Learning Engineering best practices:
1. **Model Archaeology & Verification**: Reverse-engineered training artifacts from Jupyter notebooks, resolved Keras 3 legacy deserialization differences with `compile=False`, and verified exact mathematical reproduction ($28,396) against holdout benchmarks.
2. **Explainable AI (XAI)**: Provides stakeholders with transparent insight into *why* a property was valued at a particular figure using marginal sensitivity attribution.
3. **Robust Backend Engineering**: Built with modular application factories, singleton model caching, schema validation with fallback imputation, and error handling.
4. **Intuitive Frontend UX**: Two-way synchronized sliders and inputs, live reactive AI updates, dark/light themes, and CSV drag-and-drop batch valuation.
