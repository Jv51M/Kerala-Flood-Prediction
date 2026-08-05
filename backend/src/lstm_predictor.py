"""
lstm_predictor.py
Inference wrapper for the trained Hybrid LSTM flood model.

Inputs:
  - Temporal: 15-day daily rainfall sequence  → shape (n, 15, 1)
  - Static:   [elevation, slope, dist_to_river] → shape (n, 3)

Scalers are fitted on training data and must be saved alongside the model.
If no saved scalers exist, we fit approximations from the incoming data
(less ideal, but prevents crashes during development/testing).
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from typing import List

logger = logging.getLogger(__name__)


def load_lstm_model(model_path: str):
    """Load the saved .keras LSTM model."""
    try:
        import tensorflow as tf  # lazy import so FastAPI can boot without TF
        model = tf.keras.models.load_model(model_path)
        logger.info("LSTM model loaded from %s", model_path)
        return model
    except Exception as exc:
        logger.error("Failed to load LSTM model: %s", exc)
        raise


def load_scalers(scaler_dir: str):
    """
    Load StandardScaler objects saved during training.
    Returns (scaler_temporal, scaler_static) or (None, None) if not found.
    """
    temp_path   = os.path.join(scaler_dir, "scaler_temporal.pkl")
    static_path = os.path.join(scaler_dir, "scaler_static.pkl")
    try:
        scaler_temp   = joblib.load(temp_path)
        scaler_static = joblib.load(static_path)
        logger.info("Scalers loaded from %s", scaler_dir)
        return scaler_temp, scaler_static
    except FileNotFoundError:
        logger.warning("Scalers not found in %s. Using identity scaling.", scaler_dir)
        return None, None


def predict_lstm(model, scaler_temp, scaler_static,
                 points_df: pd.DataFrame,
                 rainfall_sequences: List[List[float]]) -> dict:
    """
    Run LSTM prediction on nearby points.

    Args:
        model:               Loaded Keras model
        scaler_temp:         Fitted StandardScaler for temporal data (or None)
        scaler_static:       Fitted StandardScaler for static data (or None)
        points_df:           DataFrame with columns: elevation, slope, dist_to_river
        rainfall_sequences:  List of 15-day rainfall lists, one per point

    Returns:
        dict with 'probabilities', 'mean_prob', 'prediction'
    """
    n = len(points_df)
    if n == 0:
        return {"probabilities": [], "mean_prob": 0.0, "prediction": False}

    # --- Temporal branch ---
    X_temporal = np.array(rainfall_sequences, dtype=np.float32)  # (n, 15)
    if scaler_temp is not None:
        X_temporal_scaled = scaler_temp.transform(
            X_temporal.reshape(-1, 1)
        ).reshape(n, 15, 1)
    else:
        # Simple normalise if no scaler saved
        mean = X_temporal.mean()
        std  = X_temporal.std() + 1e-8
        X_temporal_scaled = ((X_temporal - mean) / std).reshape(n, 15, 1)

    # --- Static branch ---
    df = points_df.copy()
    if "dist_to_river" not in df.columns and "distance_water" in df.columns:
        df["dist_to_river"] = df["distance_water"]
    static_cols = ["elevation", "slope", "dist_to_river"]
    X_static = df[static_cols].fillna(0).values.astype(np.float32)

    if scaler_static is not None:
        X_static_scaled = scaler_static.transform(X_static)
    else:
        mean = X_static.mean(axis=0)
        std  = X_static.std(axis=0) + 1e-8
        X_static_scaled = (X_static - mean) / std

    # --- Inference ---
    preds = model.predict([X_temporal_scaled, X_static_scaled], verbose=0)
    probs = preds.flatten().tolist()
    mean_prob = float(np.mean(probs))

    return {
        "probabilities": [round(p, 4) for p in probs],
        "mean_prob":      round(mean_prob, 4),
        "prediction":     mean_prob >= 0.5,
    }
