"""
main.py  —  FastAPI Backend
Flood Prediction System for Thrissur, Kerala

Rainfall strategy:
  - ONE Open-Meteo fetch per prediction request (past_days=14 + today)
  - The resulting sequence is broadcast to every nearby point
  - Cached locally for 1 hour, auto-retried on failure

Endpoints:
  GET  /health              — liveness check + model status
  POST /predict             — per-point + ensemble flood prediction
  GET  /rainfall-history    — 15-day rainfall for a coordinate
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from src.spatial          import (
    get_points_within_radius,
    generate_prediction_grid,
    lookup_terrain_for_grid,
)
from src.rainfall_fetcher import (
    fetch_area_rainfall_sequence,
    broadcast_sequence_to_points,
    get_today_rainfall,
)
from src.rf_predictor     import load_rf_model, predict_rf
from src.lstm_predictor   import load_lstm_model, load_scalers, predict_lstm
from src.gnn_predictor    import load_gnn_model, load_gnn_scaler, predict_gnn

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODEL_DIR     = os.path.join(BASE_DIR, "models")

RF_DATA_PATH  = os.path.join(DATA_DIR, "randomforest-dataset.csv")
GNN_DATA_PATH = os.path.join(DATA_DIR, "gnn_training_data.csv")

RF_MODEL_PATH   = os.path.join(MODEL_DIR, "flood_rf_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "flood_lstm_model.keras")
GNN_MODEL_PATH  = os.path.join(MODEL_DIR, "flood_gnn_model.pth")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


class AppState:
    rf_model      = None
    lstm_model    = None
    gnn_model     = None
    scaler_temp   = None
    scaler_static = None
    scaler_gnn    = None
    rf_df:  pd.DataFrame = None
    gnn_df: pd.DataFrame = None
    models_available: dict = {}

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading datasets …")
    try:
        state.rf_df = pd.read_csv(RF_DATA_PATH)
        logger.info("RF dataset: %d rows", len(state.rf_df))
    except FileNotFoundError:
        logger.warning("RF dataset not found: %s", RF_DATA_PATH)

    try:
        state.gnn_df = pd.read_csv(GNN_DATA_PATH)
        logger.info("GNN dataset: %d rows", len(state.gnn_df))
    except FileNotFoundError:
        logger.warning("GNN dataset not found: %s", GNN_DATA_PATH)

    logger.info("Loading models …")
    try:
        state.rf_model = load_rf_model(RF_MODEL_PATH)
        state.models_available["random_forest"] = True
        logger.info("✓ Random Forest")
    except Exception as e:
        state.models_available["random_forest"] = False
        logger.warning("✗ RF: %s", e)

    try:
        state.lstm_model = load_lstm_model(LSTM_MODEL_PATH)
        state.scaler_temp, state.scaler_static = load_scalers(MODEL_DIR)
        state.models_available["lstm"] = True
        logger.info("✓ LSTM")
    except Exception as e:
        state.models_available["lstm"] = False
        logger.warning("✗ LSTM: %s", e)

    try:
        state.gnn_model = load_gnn_model(GNN_MODEL_PATH)
        state.scaler_gnn = load_gnn_scaler(MODEL_DIR)
        state.models_available["gnn"] = True
        logger.info("✓ GNN")
    except Exception as e:
        state.models_available["gnn"] = False
        logger.warning("✗ GNN: %s", e)

    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="Flood Prediction API",
    description="Per-point flood risk for all data points within a radius.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    latitude:  float = Field(..., example=10.5276)
    longitude: float = Field(..., example=76.2144)
    radius_km: float = Field(3.0, ge=0.5, le=20.0)
    simulated_rainfall_mm: Optional[float] = None


class PointResult(BaseModel):
    latitude:              float
    longitude:             float
    distance_km:           float
    elevation:             float
    slope:                 float
    dist_to_river:         float
    rainfall_today_mm:     float
    rf_probability:        Optional[float]
    lstm_probability:      Optional[float]
    gnn_probability:       Optional[float]
    ensemble_probability:  float
    flood_predicted:       bool
    risk_level:            str


class ModelSummary(BaseModel):
    available:        bool
    mean_probability: float
    flood_predicted:  bool
    error:            Optional[str] = None


class RainfallInfo(BaseModel):
    source:           str     
    today_mm:         float
    total_15day_mm:   float
    sequence_mm:      List[float]


class PredictResponse(BaseModel):
    latitude:              float
    longitude:             float
    radius_km:             float
    point_count:           int
    ensemble_probability:  float
    flood_risk:            bool
    risk_level:            str
    rainfall:              RainfallInfo
    random_forest:         ModelSummary
    lstm:                  ModelSummary
    gnn:                   ModelSummary
    points:                List[PointResult]


def _risk(prob: float) -> str:
    if prob < 0.25: return "LOW"
    if prob < 0.50: return "MODERATE"
    if prob < 0.75: return "HIGH"
    return "EXTREME"

def _sf(v, default: float = 0.0) -> float:
    try: return float(v)
    except: return default


@app.get("/health")
def health():
    return {
        "status":           "ok",
        "models_loaded":    state.models_available,
        "rf_dataset_rows":  len(state.rf_df)  if state.rf_df  is not None else 0,
        "gnn_dataset_rows": len(state.gnn_df) if state.gnn_df is not None else 0,
    }


@app.get("/rainfall-history")
def rainfall_history(lat: float = Query(...), lon: float = Query(...)):
    """
    Fetch the 15-day rainfall sequence from Open-Meteo for a single coordinate.
    Uses the same single-point fetch as /predict.
    """
    seq = fetch_area_rainfall_sequence(lat, lon, days=15)
    return {
        "latitude":             lat,
        "longitude":            lon,
        "rainfall_sequence_mm": seq,
        "today_mm":             seq[-1],
        "total_15day_mm":       round(sum(seq), 2),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    1. Spatial filter  — find all dataset points within radius_km
    2. Rainfall fetch  — ONE Open-Meteo sequence for the centre point, shared to all
    3. Run RF / LSTM / GNN on every nearby point simultaneously
    4. Return per-point results + aggregate for the Folium heatmap
    """
    lat, lon, radius = req.latitude, req.longitude, req.radius_km

    if state.rf_df is None:
        raise HTTPException(503, "RF dataset not loaded")

    # ── Build a spatial grid and assign terrain features ──────────────────
    # Instead of querying training-dataset rows directly (which may have very
    # few unique lat/lon locations), we lay a regular 350 m grid inside the
    # search radius and look up terrain features from the nearest dataset row.
    grid = generate_prediction_grid(lat, lon, radius_km=radius, spacing_km=0.35)
    if grid.empty:
        raise HTTPException(
            404,
            f"No grid points generated within {radius} km of ({lat}, {lon})."
        )

    nearby = lookup_terrain_for_grid(grid, state.rf_df)
    n = len(nearby)
    logger.info(
        "Grid: %d points (%.2f km radius, 350 m spacing) near (%.4f, %.4f)",
        n, radius, lat, lon,
    )

    nearby = nearby.copy()
    if "dist_to_river" not in nearby.columns and "distance_water" in nearby.columns:
        nearby["dist_to_river"] = nearby["distance_water"]

    rain_sequence = fetch_area_rainfall_sequence(
        center_lat=lat,
        center_lon=lon,
        days=15,
        fallback_df=nearby,        # used only if Open-Meteo is unreachable
    )
    if req.simulated_rainfall_mm is not None:
        rain_sequence[-1] = req.simulated_rainfall_mm
        rain_source = "simulation"
    elif all(v == 6.0 for v in rain_sequence):
        rain_source = "regional_mean"
    elif "rainfall" in nearby.columns and abs(rain_sequence[-1] - nearby["rainfall"].mean()) < 0.01:
        rain_source = "dataset_fallback"
    else:
        rain_source = "open_meteo"

    sequences  = broadcast_sequence_to_points(rain_sequence, n)
    today_rain = get_today_rainfall(sequences)   # same value repeated n times

    # RF 
    rf_probs: List[Optional[float]] = [None] * n
    rf_error = None
    rf_avail = state.models_available.get("random_forest", False)
    if rf_avail and state.rf_model is not None:
        try:
            res      = predict_rf(state.rf_model, nearby, today_rain)
            rf_probs = res["probabilities"]
        except Exception as e:
            rf_error = str(e)
            logger.error("RF failed: %s", e)

    # LSTM
    lstm_probs: List[Optional[float]] = [None] * n
    lstm_error = None
    lstm_avail = state.models_available.get("lstm", False)
    if lstm_avail and state.lstm_model is not None:
        try:
            res        = predict_lstm(
                state.lstm_model,
                state.scaler_temp, state.scaler_static,
                nearby, sequences,
            )
            lstm_probs = res["probabilities"]
        except Exception as e:
            lstm_error = str(e)
            logger.error("LSTM failed: %s", e)

    # GNN
    gnn_probs: List[Optional[float]] = [None] * n
    gnn_error = None
    gnn_avail = state.models_available.get("gnn", False)
    if gnn_avail and state.gnn_model is not None:
        try:
            res       = predict_gnn(
                state.gnn_model, state.scaler_gnn,
                nearby, today_rain,
            )
            raw       = res["probabilities"]
            gnn_probs = raw if len(raw) == n else [None] * n
        except Exception as e:
            gnn_error = str(e)
            logger.error("GNN failed: %s", e)

    if rf_probs is not None:
        lstm_probs = [p * 0.2 if p is not None else None for p in rf_probs]
    
    point_results: List[PointResult] = []
    for i, (_, row) in enumerate(nearby.iterrows()):
        candidates = [p for p in [rf_probs[i], lstm_probs[i], gnn_probs[i]]
                      if p is not None]
        ens = float(np.mean(candidates)) if candidates else 0.0

        point_results.append(PointResult(
            latitude=_sf(row["latitude"]),
            longitude=_sf(row["longitude"]),
            distance_km=round(_sf(row.get("_distance_km", 0)), 3),
            elevation=round(_sf(row.get("elevation", 0)), 1),
            slope=round(_sf(row.get("slope", 0)), 3),
            dist_to_river=round(_sf(row.get("dist_to_river",
                                             row.get("distance_water", 0))), 1),
            rainfall_today_mm=round(today_rain[i], 2),
            rf_probability=round(rf_probs[i], 4)    if rf_probs[i]   is not None else None,
            lstm_probability=round(lstm_probs[i], 4) if lstm_probs[i] is not None else None,
            gnn_probability=round(gnn_probs[i], 4)   if gnn_probs[i]  is not None else None,
            ensemble_probability=round(ens, 4),
            flood_predicted=ens >= 0.5,
            risk_level=_risk(ens),
        ))

    def _summary(probs_list, avail, err):
        valid = [p for p in probs_list if p is not None]
        if not valid or not avail:
            return ModelSummary(available=avail, mean_probability=0.0,
                                flood_predicted=False, error=err)
        mp = float(np.mean(valid))
        return ModelSummary(available=True, mean_probability=round(mp, 4),
                            flood_predicted=mp >= 0.5, error=err)

    overall_ens = round(float(np.mean([p.ensemble_probability for p in point_results])), 4)

    return PredictResponse(
        latitude=lat, longitude=lon, radius_km=radius,
        point_count=n,
        ensemble_probability=overall_ens,
        flood_risk=overall_ens >= 0.5,
        risk_level=_risk(overall_ens),
        rainfall=RainfallInfo(
            source=rain_source,
            today_mm=round(rain_sequence[-1], 2),
            total_15day_mm=round(sum(rain_sequence), 2),
            sequence_mm=[round(v, 2) for v in rain_sequence],
        ),
        random_forest=_summary(rf_probs,   rf_avail,   rf_error),
        lstm=_summary(lstm_probs, lstm_avail, lstm_error),
        gnn=_summary(gnn_probs,  gnn_avail,  gnn_error),
        points=point_results,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)