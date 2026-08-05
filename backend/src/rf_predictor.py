"""
rf_predictor.py
Inference wrapper for the trained Random Forest flood model.

Feature order (must match training):
    ['distance_water', 'elevation', 'landcover', 'latitude',
     'longitude', 'rainfall', 'slope']
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import List

logger = logging.getLogger(__name__)


def load_rf_model(model_path: str):
    """Load and return the saved RandomForest model."""
    return joblib.load(model_path)


def predict_rf(model, points_df: pd.DataFrame, today_rainfall: List[float]) -> dict:
    """
    Run RF prediction on nearby points.

    Args:
        model:          Loaded sklearn RandomForestClassifier
        points_df:      DataFrame with columns: distance_water / dist_to_river,
                        elevation, landcover, latitude, longitude, slope
        today_rainfall: List of today's rainfall values, one per row

    Returns:
        dict with keys:
            'probabilities'  - per-point flood probabilities
            'mean_prob'      - average flood probability across all points
            'prediction'     - True if mean_prob >= 0.5
    """
    df = points_df.copy()

    # Normalise column names (the RF dataset uses 'distance_water')
    if "distance_water" not in df.columns and "dist_to_river" in df.columns:
        df["distance_water"] = df["dist_to_river"]

    # landcover defaults to 0 if not available in the nearby-points dataset
    if "landcover" not in df.columns:
        df["landcover"] = 0

    df["rainfall"] = today_rainfall

    feature_cols = [
        "distance_water", "elevation", "landcover",
        "latitude", "longitude", "rainfall", "slope"
    ]

    # Drop rows where required features are missing
    X = df[feature_cols].fillna(0).values
    probs = model.predict_proba(X)[:, 1].tolist()

    mean_prob = float(np.mean(probs))
    return {
        "probabilities": probs,
        "mean_prob": round(mean_prob, 4),
        "prediction": mean_prob >= 0.5,
    }
