"""
Housing Price Predictor Module
Encapsulates Keras ANN regression model inference, feature preprocessing,
explainability/sensitivity calculation, and input validation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

# Suppress overly verbose TensorFlow / Keras logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", 
    "RM", "AGE", "DIS", "RAD", "TAX", 
    "PTRATIO", "B", "LSTAT"
]

FEATURE_BOUNDS = {
    "CRIM": (0.0, 100.0),
    "ZN": (0.0, 100.0),
    "INDUS": (0.0, 35.0),
    "CHAS": (0.0, 1.0),
    "NOX": (0.3, 1.0),
    "RM": (2.0, 10.0),
    "AGE": (0.0, 100.0),
    "DIS": (0.5, 15.0),
    "RAD": (1.0, 24.0),
    "TAX": (150.0, 800.0),
    "PTRATIO": (10.0, 25.0),
    "B": (0.0, 400.0),
    "LSTAT": (0.5, 50.0)
}

DEFAULT_INFLATION_MULTIPLIER = 7.6  # 1970s nominal index to contemporary value

class HousingPredictor:
    """Singleton-style or reusable predictor for the Boston Housing ANN model."""

    _instance: Optional["HousingPredictor"] = None

    def __init__(self, 
                 model_path: str = "ANN_regression.h5", 
                 scaler_path: str = "scaler.joblib",
                 metadata_path: str = "model_metadata.json"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, model_path)
        self.scaler_path = os.path.join(self.base_dir, scaler_path)
        self.metadata_path = os.path.join(self.base_dir, metadata_path)
        
        self.model = None
        self.scaler = None
        self.metadata = {}
        self.feature_means = np.zeros(13)
        self.baseline_price = 22.5
        
        self._load_artifacts()

    @classmethod
    def get_instance(cls) -> "HousingPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_artifacts(self) -> None:
        """Loads model weights, scaler, and metadata JSON."""
        # 1. Load Metadata
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info("Loaded metadata from %s", self.metadata_path)
        else:
            self.metadata = {}
            logger.warning("Metadata file not found at %s", self.metadata_path)

        # 2. Load Scaler
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"StandardScaler artifact not found at {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        logger.info("Loaded StandardScaler from %s", self.scaler_path)

        # Extract feature means for baseline
        if hasattr(self.scaler, "mean_"):
            self.feature_means = self.scaler.mean_.copy()
        else:
            self.feature_means = np.array([
                3.745, 11.480, 11.104, 0.062, 0.557, 
                6.267, 69.011, 3.740, 9.441, 405.899, 
                18.476, 354.783, 12.741
            ])

        # 3. Load Keras Model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ANN Model file not found at {self.model_path}")
        
        # In Keras 3.x, legacy H5 models are best loaded with compile=False for pure inference
        self.model = keras.models.load_model(self.model_path, compile=False)
        logger.info("Loaded ANN Regression model from %s", self.model_path)

        # Warmup and compute baseline price (at mean values)
        scaled_mean = self.scaler.transform(self.feature_means.reshape(1, -1))
        pred_tensor = self.model(scaled_mean, training=False)
        self.baseline_price = float(pred_tensor.numpy()[0][0])
        logger.info("Predictor warmed up. Dataset baseline price: $%.2fk", self.baseline_price)

    def validate_features(self, input_data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """
        Validates and converts input dictionary to ordered feature vector.
        Returns (feature_vector, warnings_list).
        """
        warnings = []
        vec = []

        for i, fname in enumerate(FEATURE_NAMES):
            if fname not in input_data or input_data[fname] is None or input_data[fname] == "":
                # Fallback to mean/default
                val = float(self.feature_means[i])
                warnings.append(f"Missing feature '{fname}'; imputed dataset mean {val:.2f}.")
            else:
                try:
                    val = float(input_data[fname])
                except (ValueError, TypeError):
                    raise ValueError(f"Feature '{fname}' must be a valid number, got: {input_data[fname]}")

            # Bounds verification
            low, high = FEATURE_BOUNDS.get(fname, (-np.inf, np.inf))
            if val < low:
                warnings.append(f"Feature '{fname}' ({val}) is below expected minimum ({low}). Clamped to {low}.")
                val = low
            elif val > high:
                warnings.append(f"Feature '{fname}' ({val}) is above expected maximum ({high}). Clamped to {high}.")
                val = high

            vec.append(val)

        return np.array(vec, dtype=np.float32), warnings

    def calculate_percentile(self, price_k: float) -> float:
        """Estimates market percentile (0-100) using empirical Boston target distribution."""
        # Empirical percentiles from Boston dataset:
        # p10: 12.75, p25: 16.7, p50: 21.2, p75: 25.0, p90: 34.8, max: 50.0
        points = [
            (5.0, 0.0),
            (12.75, 10.0),
            (16.7, 25.0),
            (21.2, 50.0),
            (25.0, 75.0),
            (34.8, 90.0),
            (50.0, 100.0)
        ]
        if price_k <= points[0][0]:
            return 1.0
        if price_k >= points[-1][0]:
            return 99.0

        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i+1]
            if x0 <= price_k <= x1:
                pct = y0 + (price_k - x0) / (x1 - x0) * (y1 - y0)
                return round(float(pct), 1)
        return 50.0

    def categorize_price(self, price_k: float) -> Dict[str, str]:
        """Categorizes predicted price into intuitive market tier and color token."""
        if price_k < 15.0:
            return {"tier": "Affordable / Starter", "badge_class": "badge-affordable", "color": "#10b981"}
        elif price_k < 23.0:
            return {"tier": "Mid-Market Value", "badge_class": "badge-moderate", "color": "#06b6d4"}
        elif price_k < 32.0:
            return {"tier": "Upper Mid-Market", "badge_class": "badge-upper", "color": "#6366f1"}
        elif price_k < 42.0:
            return {"tier": "Premium Executive", "badge_class": "badge-premium", "color": "#8b5cf6"}
        else:
            return {"tier": "Ultra Luxury Estate", "badge_class": "badge-luxury", "color": "#ec4899"}

    def compute_feature_impacts(self, raw_vector: np.ndarray, current_price: float) -> List[Dict[str, Any]]:
        """
        Calculates local feature contribution vs baseline average.
        Evaluates marginal impact of substituting each individual feature with its mean.
        """
        impacts = []
        scaled_current = self.scaler.transform(raw_vector.reshape(1, -1))
        
        # Create a batch where for each feature i, feature i is replaced by the mean
        batch_perturbed = np.tile(raw_vector, (13, 1))
        for i in range(13):
            batch_perturbed[i, i] = self.feature_means[i]

        scaled_batch = self.scaler.transform(batch_perturbed)
        preds_perturbed = self.model(scaled_batch, training=False).numpy().flatten()

        for i, fname in enumerate(FEATURE_NAMES):
            # If replacing feature i with mean caused prediction to drop from current_price to preds_perturbed[i],
            # then feature i contributed (current_price - preds_perturbed[i])
            impact = float(current_price - preds_perturbed[i])
            
            val = float(raw_vector[i])
            mean_val = float(self.feature_means[i])
            diff = val - mean_val

            # Get human friendly name from metadata if available
            meta_feat = next((f for f in self.metadata.get("features", []) if f["id"] == fname), None)
            display_name = meta_feat["name"] if meta_feat else fname
            unit = meta_feat.get("unit", "") if meta_feat else ""

            impacts.append({
                "feature": fname,
                "name": display_name,
                "unit": unit,
                "value": round(val, 2),
                "mean": round(mean_val, 2),
                "delta_from_mean": round(diff, 2),
                "impact_k": round(impact, 3),
                "impact_usd": round(impact * 1000, 0),
                "direction": "positive" if impact >= 0 else "negative"
            })

        # Sort by absolute impact magnitude descending
        impacts.sort(key=lambda x: abs(x["impact_k"]), reverse=True)
        return impacts

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes single home prediction pipeline.
        Returns complete enriched result with valuation, confidence bounds,
        percentiles, category, and explainability factors.
        """
        raw_vec, warnings = self.validate_features(input_data)
        
        # Scale
        scaled_vec = self.scaler.transform(raw_vec.reshape(1, -1))
        
        # Inference
        pred_tensor = self.model(scaled_vec, training=False)
        price_k = float(pred_tensor.numpy()[0][0])
        
        # Real-estate values cannot be strictly negative
        price_k = max(1.0, price_k)
        
        mae = float(self.metadata.get("metrics", {}).get("test_mae", 2.923))
        inflation_mult = float(self.metadata.get("dataset", {}).get("inflation_multiplier", DEFAULT_INFLATION_MULTIPLIER))
        
        price_usd = price_k * 1000.0
        adjusted_usd = price_usd * inflation_mult
        
        lower_bound_k = max(1.0, price_k - mae)
        upper_bound_k = price_k + mae
        
        percentile = self.calculate_percentile(price_k)
        category = self.categorize_price(price_k)
        impacts = self.compute_feature_impacts(raw_vec, price_k)

        # Prepare normalized comparison for Radar chart (0 to 100 relative to min/max)
        normalized_profile = {}
        for i, fname in enumerate(FEATURE_NAMES):
            low, high = FEATURE_BOUNDS.get(fname, (0.0, 100.0))
            val = raw_vec[i]
            norm_val = ((val - low) / (high - low)) * 100.0 if high > low else 50.0
            normalized_profile[fname] = round(float(np.clip(norm_val, 0, 100)), 1)

        return {
            "success": True,
            "prediction": {
                "price_k": round(price_k, 3),
                "price_formatted": f"${price_usd:,.0f}",
                "adjusted_price_usd": round(adjusted_usd, 0),
                "adjusted_price_formatted": f"${adjusted_usd:,.0f}",
                "confidence_interval": {
                    "mae_k": round(mae, 3),
                    "lower_k": round(lower_bound_k, 3),
                    "upper_k": round(upper_bound_k, 3),
                    "lower_formatted": f"${lower_bound_k * 1000:,.0f}",
                    "upper_formatted": f"${upper_bound_k * 1000:,.0f}",
                    "confidence_level": "68% (±1 MAE)"
                },
                "category": category,
                "percentile": percentile,
                "baseline_comparison": {
                    "baseline_k": round(self.baseline_price, 3),
                    "delta_k": round(price_k - self.baseline_price, 3),
                    "percentage_difference": round(((price_k - self.baseline_price) / self.baseline_price) * 100, 1)
                }
            },
            "features_used": {FEATURE_NAMES[i]: float(raw_vec[i]) for i in range(13)},
            "normalized_profile": normalized_profile,
            "feature_impacts": impacts,
            "warnings": warnings
        }

    def predict_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes vectorized batch prediction on a pandas DataFrame.
        """
        if df.empty:
            raise ValueError("Provided dataset is empty.")

        records = []
        valid_rows = []
        row_indices = []

        # Check required columns
        missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {', '.join(missing_cols)}")

        for idx, row in df.iterrows():
            try:
                row_dict = {col: row[col] for col in FEATURE_NAMES}
                vec, _ = self.validate_features(row_dict)
                valid_rows.append(vec)
                row_indices.append(idx)
            except Exception as e:
                records.append({
                    "row_index": idx,
                    "status": "error",
                    "error": str(e)
                })

        if not valid_rows:
            return {
                "success": False,
                "error": "No valid rows found in dataset for prediction.",
                "results": records
            }

        batch_arr = np.array(valid_rows, dtype=np.float32)
        scaled_batch = self.scaler.transform(batch_arr)
        preds_tensor = self.model(scaled_batch, training=False)
        preds_k = preds_tensor.numpy().flatten()
        preds_k = np.clip(preds_k, 1.0, None)

        inflation_mult = float(self.metadata.get("dataset", {}).get("inflation_multiplier", DEFAULT_INFLATION_MULTIPLIER))
        mae = float(self.metadata.get("metrics", {}).get("test_mae", 2.923))

        for orig_idx, vec, p_k in zip(row_indices, valid_rows, preds_k):
            p_usd = float(p_k * 1000.0)
            adj_usd = float(p_usd * inflation_mult)
            cat = self.categorize_price(float(p_k))
            pct = self.calculate_percentile(float(p_k))

            orig_data = {FEATURE_NAMES[i]: float(vec[i]) for i in range(13)}
            # Keep optional original description if present
            if "description" in df.columns:
                orig_data["description"] = str(df.at[orig_idx, "description"])

            records.append({
                "row_index": int(orig_idx) + 1,
                "status": "success",
                "price_k": round(float(p_k), 3),
                "price_formatted": f"${p_usd:,.0f}",
                "adjusted_price_formatted": f"${adj_usd:,.0f}",
                "category": cat["tier"],
                "percentile": pct,
                "lower_bound": f"${max(1.0, float(p_k - mae)) * 1000:,.0f}",
                "upper_bound": f"${float(p_k + mae) * 1000:,.0f}",
                "features": orig_data
            })

        # Calculate summary stats for the batch
        all_preds = [r["price_k"] for r in records if r["status"] == "success"]
        stats = {
            "total_rows": len(df),
            "successful_predictions": len(all_preds),
            "failed_rows": len(df) - len(all_preds),
            "mean_price_k": round(float(np.mean(all_preds)), 2) if all_preds else 0.0,
            "min_price_k": round(float(np.min(all_preds)), 2) if all_preds else 0.0,
            "max_price_k": round(float(np.max(all_preds)), 2) if all_preds else 0.0,
            "mean_formatted": f"${float(np.mean(all_preds)) * 1000:,.0f}" if all_preds else "$0"
        }

        return {
            "success": True,
            "summary": stats,
            "results": records
        }
