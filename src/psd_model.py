"""
psd_model.py
Psychological Safety Distance (PSD) model.

Implements the core formula:
    V(x, p, t, s, n) = exp(-(a*ln(t+1) + b*ln(s+1) + c*ln(n+1))) * p * x

A higher V value indicates stronger perceived risk.
"""

import math
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pytz

from config import (
    PSD_ALPHA, PSD_BETA, PSD_GAMMA, PSD_PROB,
    DISSIPATION_DECAY, DISSIPATION_BASE_LOSS,
    DISSIPATION_DECAY_SLOW, DISSIPATION_DECAY_FAST, DISSIPATION_FLOOR,
    CATEGORY_LOSS_MAP, BASELINE_DATE,
    COOLDOWN_STEPS, RECOVERY_DURATION,
)


# ---------------------------------------------------------------------------
# Geodesic distance
# ---------------------------------------------------------------------------
def calculate_geodesic_distance(lat1: float, lon1: float,
                                lat2: float, lon2: float) -> float:
    """Haversine distance between two (lat, lon) pairs in kilometres."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * math.asin(math.sqrt(a)) * r


# ---------------------------------------------------------------------------
# Temporal distance
# ---------------------------------------------------------------------------
def get_baseline_time() -> datetime:
    """Return baseline time (Hurricane Sandy landfall: 2012-10-30 UTC)."""
    return datetime(2012, 10, 30, 0, 0, 0, tzinfo=pytz.UTC)


def calculate_time_distance(current_time: datetime, baseline_time: datetime) -> float:
    """Return absolute day difference between *current_time* and *baseline_time*."""
    diff = current_time - baseline_time
    return abs(diff.days + diff.seconds / 86400.0)


# ---------------------------------------------------------------------------
# Hurricane loss scalar
# ---------------------------------------------------------------------------
def map_hurricane_category_to_loss(category: str,
                                   current_time: datetime,
                                   dissipated_time: Optional[datetime] = None) -> float:
    """
    Map a hurricane category string to a loss scalar *x*.
    After dissipation, loss decays over time to model residual secondary impacts.
    """
    if pd.isna(category):
        if dissipated_time is None:
            return 100.0
        days_since = (current_time - dissipated_time).days
        if days_since <= 3:
            return DISSIPATION_BASE_LOSS
        elif days_since <= 8:
            return DISSIPATION_BASE_LOSS * (DISSIPATION_DECAY_SLOW ** (days_since - 3))
        else:
            base = DISSIPATION_BASE_LOSS * (DISSIPATION_DECAY_SLOW ** 5)
            return max(DISSIPATION_FLOOR, base * (DISSIPATION_DECAY_FAST ** (days_since - 8)))
    return CATEGORY_LOSS_MAP.get(category, 300)


# ---------------------------------------------------------------------------
# Social distance
# ---------------------------------------------------------------------------
def calculate_social_distance(
        user_id: str,
        following_dict: Dict[str, List[str]],
        followee_count: Dict[str, int],
        panic_history: Dict[str, List[Tuple[int, int]]],
        current_timestep: int) -> Tuple[float, float]:
    """
    Compute social distance *n* and the underlying panic ratio.

    Social distance formula:
        n = 1 - panic_followed / total_followed   (clamped to [0.01, 1.0])

    After a user first panics, *n* enters a cooldown period and then
    gradually recovers to the follower-ratio-based value.

    Returns (social_distance, panic_ratio).
    """
    panic_ratio = 0.0
    total_followed = followee_count.get(user_id, 0)

    if user_id in following_dict and total_followed > 0:
        panic_count = sum(
            1 for fid in following_dict[user_id]
            if fid in panic_history
            and any(e == 1 and step < current_timestep
                    for step, e in panic_history[fid])
        )
        panic_ratio = panic_count / total_followed

    # Post-panic cooldown / recovery
    if user_id in panic_history:
        self_panic_steps = [s for s, e in panic_history[user_id] if e == 1]
        if self_panic_steps:
            steps_since = current_timestep - min(self_panic_steps)
            if steps_since <= COOLDOWN_STEPS:
                return 0.01, panic_ratio
            recovery = min(1.0, (steps_since - COOLDOWN_STEPS) / RECOVERY_DURATION)
            return 0.01 + recovery * 0.99, panic_ratio

    if total_followed == 0:
        return 1.0, panic_ratio

    n = max(0.01, min(1.0, 1.0 - panic_ratio))
    return n, panic_ratio


# ---------------------------------------------------------------------------
# Full PSD calculation
# ---------------------------------------------------------------------------
def calculate_psd(
        user_data: Dict,
        hurricane_row: pd.Series,
        following_dict: Dict[str, List[str]],
        followee_count: Dict[str, int],
        panic_history: Dict[str, List[Tuple[int, int]]],
        current_timestep: int,
        baseline_time: datetime,
        dissipated_time: Optional[datetime] = None,
        last_known_position: Optional[Tuple[float, float]] = None) -> Dict:
    """
    Compute the PSD value V and its component distances.

    Returns a dict with keys:
        psychological_distance, distance_km, social_distance,
        panic_ratio, time_distance
    """
    dissipated = pd.isna(hurricane_row["Category"])

    user_lat = user_data.get("location", [None, None])[0]
    user_lon = user_data.get("location", [None, None])[1]

    # --- Spatial distance s ---
    if dissipated:
        if last_known_position and user_lat is not None and user_lon is not None:
            last_lat, last_lon = last_known_position
            s = calculate_geodesic_distance(user_lat, user_lon, last_lat, last_lon)
            s *= DISSIPATION_DECAY
        else:
            s = 600.0
    elif user_lat is not None and user_lon is not None:
        s = calculate_geodesic_distance(user_lat, user_lon,
                                        hurricane_row["lat_final"],
                                        hurricane_row["lng_final"])
    else:
        s = 5000.0

    # --- Temporal distance t (pre-computed column) ---
    t = hurricane_row["t"]

    # --- Social distance n ---
    n, panic_ratio = calculate_social_distance(
        user_data.get("user_id", ""), following_dict, followee_count,
        panic_history, current_timestep)

    # --- Loss scalar x (pre-computed column) ---
    x = hurricane_row["x"]

    # --- PSD formula ---
    time_disc  = PSD_ALPHA * math.log(t + 1) if t >= 0 else 0.0
    space_disc = PSD_BETA  * math.log(s + 1) if s >= 0 else 0.0
    social_disc = PSD_GAMMA * math.log(n + 1) if n >= 0 else 0.0
    total_disc = time_disc + space_disc + social_disc

    V = math.exp(-total_disc) * PSD_PROB * x

    return {
        "psychological_distance": V,
        "distance_km":            s,
        "social_distance":        n,
        "panic_ratio":            panic_ratio,
        "time_distance":          t,
    }
