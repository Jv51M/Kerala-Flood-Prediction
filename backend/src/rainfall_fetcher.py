"""
rainfall_fetcher.py
Fetches rainfall using the openmeteo-requests library.

WHY openmeteo-requests INSTEAD OF raw requests:
  The library wraps Open-Meteo with a local disk cache (.cache/) and automatic
  retries with exponential backoff — so transient DNS failures and timeouts
  that caused the original errors are handled transparently.

DESIGN: ONE FETCH PER PREDICTION REQUEST
  Rainfall is spatially uniform within 3-5 km, so we fetch once for the centre
  point and broadcast the same sequence to every nearby point.
  Cost: 1 API call per prediction request, regardless of point count.

SEQUENCE FORMAT:
  past_days=14 + forecast_days=1 → 15 daily values, oldest first.
  Index 0 = 14 days ago, index 14 = today.

FALLBACK CHAIN (if Open-Meteo is still unreachable after retries):
  1. Mean of the `rainfall` column in the nearby-points DataFrame
  2. Thrissur regional mean constant (6.0 mm/day)
"""

import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

_OPEN_METEO_URL          = "https://api.open-meteo.com/v1/forecast"
_CACHE_DIR               = ".cache"
_CACHE_TTL_SECONDS       = 3600          # cache responses for 1 hour
_RETRY_COUNT             = 5
_RETRY_BACKOFF           = 0.2
_THRISSUR_MEAN_MM        = 6.0           # last-resort regional constant


def _make_client() -> openmeteo_requests.Client:
    """Build an Open-Meteo client with caching + retry baked in."""
    cache_session = requests_cache.CachedSession(_CACHE_DIR,
                                                 expire_after=_CACHE_TTL_SECONDS)
    retry_session = retry(cache_session,
                          retries=_RETRY_COUNT,
                          backoff_factor=_RETRY_BACKOFF)
    return openmeteo_requests.Client(session=retry_session)


# Module-level singleton — created once, reused for every request
_client = _make_client()


# ─────────────────────────────────────────────────────────────────────────────

def fetch_area_rainfall_sequence(
    center_lat: float,
    center_lon: float,
    api_key: str = "",          # kept for interface compatibility, not used
    days: int = 15,
    fallback_df: Optional[pd.DataFrame] = None,
) -> List[float]:
    """
    Fetch a `days`-length daily rainfall sequence for the area centre.
    Returns a list of floats (oldest first), length == days.

    The sequence is identical for every nearby point — call
    broadcast_sequence_to_points() to replicate it.

    Args:
        center_lat:  Latitude of the search centre
        center_lon:  Longitude of the search centre
        api_key:     Unused (kept so main.py needs no changes)
        days:        Sequence length — must be 15 to match LSTM input
        fallback_df: Nearby-points DataFrame used as fallback if API fails
    """
    logger.info(
        "Fetching %d-day rainfall from Open-Meteo for (%.4f, %.4f) …",
        days, center_lat, center_lon,
    )

    try:
        params = {
            "latitude":      center_lat,
            "longitude":     center_lon,
            "daily":         "precipitation_sum",
            "past_days":     days - 1,       # 14 past days
            "forecast_days": 1,              # + today = 15
            "timezone":      "Asia/Kolkata",
        }

        responses = _client.weather_api(_OPEN_METEO_URL, params=params)
        response  = responses[0]

        daily      = response.Daily()
        precip_arr = daily.Variables(0).ValuesAsNumpy()   # numpy array, length = days

        # Replace NaN with 0.0 and convert to plain Python floats
        sequence = [
            float(v) if (v is not None and not np.isnan(v)) else 0.0
            for v in precip_arr
        ]

        # Pad or trim to exactly `days` values (safety net)
        if len(sequence) < days:
            sequence = [0.0] * (days - len(sequence)) + sequence
        sequence = sequence[:days]

        logger.info(
            "Open-Meteo OK — today: %.1f mm, 15-day total: %.1f mm",
            sequence[-1], sum(sequence),
        )
        return sequence

    except Exception as exc:
        logger.warning(
            "Open-Meteo fetch failed after retries: %s — using fallback.", exc
        )
        return _fallback_sequence(fallback_df, days)


def _fallback_sequence(
    fallback_df: Optional[pd.DataFrame],
    days: int,
) -> List[float]:
    """
    Build a best-effort sequence when Open-Meteo is unreachable.

    Uses the mean of the dataset's `rainfall` column if available,
    tapering earlier days slightly so the LSTM sees a realistic shape
    rather than a flat line. Falls back to the Thrissur regional mean.
    """
    if fallback_df is not None and "rainfall" in fallback_df.columns:
        mean_rain = float(fallback_df["rainfall"].mean())
        if mean_rain > 0:
            logger.warning(
                "Using dataset rainfall mean (%.2f mm/day) as fallback. "
                "Predictions will be less accurate without live data.", mean_rain,
            )
            # Taper older days: oldest = 40% of mean, today = 100%
            sequence = [
                round(mean_rain * max(0.4, 1.0 - i * 0.04), 2)
                for i in range(days - 1, -1, -1)
            ]
            return sequence

    logger.warning(
        "Using Thrissur regional mean (%.1f mm/day). "
        "Check network connectivity.", _THRISSUR_MEAN_MM,
    )
    return [_THRISSUR_MEAN_MM] * days


# ─────────────────────────────────────────────────────────────────────────────

def broadcast_sequence_to_points(
    sequence: List[float],
    n_points: int,
) -> List[List[float]]:
    """
    Replicate the area rainfall sequence for every nearby point.
    Rainfall within 3-5 km is treated as spatially uniform.
    """
    return [list(sequence) for _ in range(n_points)]


def get_today_rainfall(sequences: List[List[float]]) -> List[float]:
    """Extract today's value (last element) from each sequence."""
    return [seq[-1] for seq in sequences]