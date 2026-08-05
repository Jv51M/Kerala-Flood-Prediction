"""
gnn_predictor.py
Inference wrapper for the trained GNN (Graph Convolutional Network) flood model.

At inference time we:
1. Take only the subset of nearby points (not all 1000 training points)
2. Rebuild the spatial k-NN adjacency matrix for that small subgraph
3. Forward-pass through the saved GCN weights

Features: [elevation, slope, dist_to_river, precipitation]
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
import logging
from typing import List

from src.spatial import build_adjacency_matrix

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Model definition (must match gnn_model.py exactly)
# ──────────────────────────────────────────────────────────────
class GCNLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.projection = nn.Linear(in_f, out_f)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.matmul(adj, self.projection(x))


class FloodGNN(nn.Module):
    def __init__(self, n_feat: int = 4):
        super().__init__()
        self.gcn1 = GCNLayer(n_feat, 32)
        self.gcn2 = GCNLayer(32, 16)
        self.fc   = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gcn1(x, adj))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gcn2(x, adj))
        return torch.sigmoid(self.fc(x))


def load_gnn_model(model_path: str) -> FloodGNN:
    """Load the saved GNN state-dict and return an eval-mode model."""
    model = FloodGNN(n_feat=4)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("GNN model loaded from %s", model_path)
    return model


def load_gnn_scaler(scaler_dir: str):
    """Load the GNN StandardScaler saved during training (or None)."""
    scaler_path = os.path.join(scaler_dir, "scaler_gnn.pkl")
    try:
        scaler = joblib.load(scaler_path)
        logger.info("GNN scaler loaded.")
        return scaler
    except FileNotFoundError:
        logger.warning("GNN scaler not found at %s. Using z-score normalisation.", scaler_path)
        return None


def predict_gnn(model: FloodGNN, scaler,
                points_df: pd.DataFrame,
                today_rainfall: List[float]) -> dict:
    """
    Run GNN prediction on nearby points.

    Args:
        model:          Loaded FloodGNN (eval mode)
        scaler:         Fitted StandardScaler for [elev, slope, dist, precip] or None
        points_df:      DataFrame with lat/lon + elevation, slope, dist_to_river
        today_rainfall: Today's rainfall per point

    Returns:
        dict with 'probabilities', 'mean_prob', 'prediction'
    """
    n = len(points_df)
    if n == 0:
        return {"probabilities": [], "mean_prob": 0.0, "prediction": False}

    df = points_df.copy()
    if "dist_to_river" not in df.columns and "distance_water" in df.columns:
        df["dist_to_river"] = df["distance_water"]

    df["rainfall"] = today_rainfall

    feature_cols = ["elevation", "slope", "dist_to_river", "rainfall"]
    X = df[feature_cols].fillna(0).values.astype(np.float32)

    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)
    else:
        mean = X.mean(axis=0)
        std  = X.std(axis=0) + 1e-8
        X    = ((X - mean) / std).astype(np.float32)

    # Rebuild adjacency matrix from geographic coordinates of the subset
    coords = df[["longitude", "latitude"]].values
    adj    = build_adjacency_matrix(coords, n_neighbors=5)

    X_tensor   = torch.FloatTensor(X)
    adj_tensor = torch.FloatTensor(adj)

    with torch.no_grad():
        output = model(X_tensor, adj_tensor)  # (n, 1)

    probs     = output.squeeze().cpu().numpy()
    probs     = np.atleast_1d(probs).tolist()
    mean_prob = float(np.mean(probs))

    return {
        "probabilities": [round(p, 4) for p in probs],
        "mean_prob":      round(mean_prob, 4),
        "prediction":     mean_prob >= 0.5,
    }
