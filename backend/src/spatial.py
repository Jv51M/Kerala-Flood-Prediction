"""
spatial.py
Utilities for finding geographically nearby data points using the Haversine formula.
"""

import math
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph, BallTree


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Returns the great-circle distance in km between two (lat, lon) points.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def get_points_within_radius(df: pd.DataFrame, user_lat: float, user_lon: float,
                              radius_km: float = 3.0,
                              lat_col: str = "latitude",
                              lon_col: str = "longitude") -> pd.DataFrame:
    """
    Returns all rows in df whose (lat, lon) fall within radius_km of (user_lat, user_lon).
    """
    distances = df.apply(
        lambda row: haversine_distance(user_lat, user_lon, row[lat_col], row[lon_col]),
        axis=1
    )
    mask = distances <= radius_km
    result = df[mask].copy()
    result["_distance_km"] = distances[mask].values
    return result.sort_values("_distance_km").reset_index(drop=True)


def generate_prediction_grid(center_lat: float, center_lon: float,
                              radius_km: float, spacing_km: float = 0.35) -> pd.DataFrame:
    """
    Generate a regular rectangular grid of (lat, lon) points covering the
    circle of radius_km around center.  Only points inside the circle are kept.

    spacing_km controls density:
        0.35 km ≈ 350 m  →  ~250 points in a 3 km radius
    """
    spacing_lat = spacing_km / 111.0                                   # deg per km (lat)
    spacing_lon = spacing_km / (111.0 * math.cos(math.radians(center_lat)))  # deg per km (lon)

    lat_range = radius_km / 111.0
    lon_range = radius_km / (111.0 * math.cos(math.radians(center_lat)))

    lats, lons, dists = [], [], []
    lat = center_lat - lat_range
    while lat <= center_lat + lat_range + 1e-9:
        lon = center_lon - lon_range
        while lon <= center_lon + lon_range + 1e-9:
            d = haversine_distance(center_lat, center_lon, lat, lon)
            if d <= radius_km:
                lats.append(round(lat, 6))
                lons.append(round(lon, 6))
                dists.append(round(d, 4))
            lon += spacing_lon
        lat += spacing_lat

    return pd.DataFrame({
        "latitude":     lats,
        "longitude":    lons,
        "_distance_km": dists,
    }).sort_values("_distance_km").reset_index(drop=True)


def lookup_terrain_for_grid(grid: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    For each point in `grid`, find the nearest row in `dataset` (by lat/lon)
    and copy its terrain feature columns:
        elevation, slope, dist_to_river / distance_water, landcover

    Uses sklearn BallTree with haversine metric for efficiency.
    """
    terrain_candidates = ["elevation", "slope", "dist_to_river",
                          "distance_water", "landcover"]
    terrain_cols = [c for c in terrain_candidates if c in dataset.columns]

    # Deduplicate dataset by unique (lat, lon) for the BallTree
    ds_unique = (
        dataset[["latitude", "longitude"] + terrain_cols]
        .drop_duplicates(subset=["latitude", "longitude"])
        .reset_index(drop=True)
    )

    tree = BallTree(
        np.radians(ds_unique[["latitude", "longitude"]].values),
        metric="haversine",
    )
    grid_rad = np.radians(grid[["latitude", "longitude"]].values)
    _, indices = tree.query(grid_rad, k=1)
    indices = indices.flatten()

    result = grid.copy()
    for col in terrain_cols:
        result[col] = ds_unique[col].iloc[indices].values

    # Ensure dist_to_river alias exists
    if "dist_to_river" not in result.columns and "distance_water" in result.columns:
        result["dist_to_river"] = result["distance_water"]
    if "distance_water" not in result.columns and "dist_to_river" in result.columns:
        result["distance_water"] = result["dist_to_river"]

    if "landcover" not in result.columns:
        result["landcover"] = 0

    return result


def build_adjacency_matrix(coords: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """
    Builds a k-NN adjacency matrix from (lon, lat) coordinate array.
    Automatically reduces n_neighbors if fewer points are available.
    """
    n_points = len(coords)
    if n_points <= 1:
        return np.eye(n_points, dtype=np.float32)

    k = min(n_neighbors, n_points - 1)
    adj = kneighbors_graph(coords, n_neighbors=k, mode="connectivity", include_self=True)
    return adj.toarray().astype(np.float32)
