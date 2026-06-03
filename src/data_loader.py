"""
data_loader.py
Functions for loading and preprocessing all datasets used by PCP.
"""

import sys
import json
import math
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pytz

from config import (
    HURRICANE_EXCEL_PATH, USER_PROFILE_PATH,
    FOLLOWING_FILE_PATH, BASELINE_DATE,
)
from psd_model import (
    get_baseline_time, calculate_time_distance, map_hurricane_category_to_loss,
)


# ---------------------------------------------------------------------------
# Hurricane data
# ---------------------------------------------------------------------------
def load_hurricane_data() -> Tuple[pd.DataFrame, Optional[datetime],
                                   Optional[Tuple[float, float]]]:
    """
    Load and preprocess hurricane meteorological data.

    Pre-computes:
        - 't'  : temporal distance from baseline (days)
        - 'x'  : loss scalar derived from hurricane category

    Returns
    -------
    hurricane_df       : DataFrame with 115 timesteps
    dissipated_time    : UTC datetime when the hurricane dissipated (or None)
    last_known_position: (lat, lon) of the last active hurricane position
    """
    try:
        df = pd.read_excel(HURRICANE_EXCEL_PATH)
        df = df.rename(columns={
            "wind":         "Wind (km/h)",
            "air_pressure": "Air Pressure (hPa)",
            "category":     "Category",
        })
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)

        if "event_description" not in df.columns:
            df["event_description"] = ""

        baseline_time = get_baseline_time()
        dissipated_time = None
        last_known_position = None

        for _, row in df.iterrows():
            if pd.isna(row["Category"]):
                if dissipated_time is None:
                    dissipated_time = pd.to_datetime(row["time"])
            else:
                last_known_position = (row["lat_final"], row["lng_final"])

        df["t"] = df["time"].apply(
            lambda dt: calculate_time_distance(dt, baseline_time))
        df["x"] = df.apply(
            lambda row: map_hurricane_category_to_loss(
                row["Category"], row["time"], dissipated_time),
            axis=1)

        return df, dissipated_time, last_known_position

    except Exception as exc:
        print(f"Failed to load hurricane data: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# User data
# ---------------------------------------------------------------------------
def load_user_data(start_row: int = 0) -> pd.DataFrame:
    """
    Load preprocessed user profile data.

    Parameters
    ----------
    start_row : skip this many rows from the top (0 = load all users).
    """
    try:
        return pd.read_csv(
            USER_PROFILE_PATH,
            skiprows=lambda i: i > 0 and i <= start_row,
        )
    except Exception as exc:
        print(f"Failed to load user data: {exc}")
        sys.exit(1)


def build_user_profile(row: pd.Series) -> Dict:
    """
    Convert a user data row into the structured profile dict expected
    by the BDEI-CoT pipeline.
    """
    try:
        recent_tweets = json.loads(row.get("recent_tweets", "[]"))
    except Exception:
        recent_tweets = []

    return {
        "user_id":  str(row["user_id"]),
        "location": [row["lat_final"], row["lng_final"]],
        "big_five": {
            "E": row["Extroversion"],
            "N": row["Neuroticism"],
            "A": row["Agreeableness"],
            "C": row["Conscientiousness"],
            "O": row["Openness"],
        },
        "twitter_stats": {
            "followers":    int(row["user_followers_count"]),
            "following":    int(row["user_friends_count"]),
            "daily_tweets": float(row["text_average_count"]),
        },
        "tone_of_voice":      row["tone_of_voice"].split(", ") if pd.notna(row["tone_of_voice"]) else [],
        "emotional_stability": row.get("emotional_stability", "Unknown"),
        "interests":          row["topic"].split(", ") if pd.notna(row["topic"]) else [],
        "recent_tweets":      recent_tweets,
    }


# ---------------------------------------------------------------------------
# Social graph data
# ---------------------------------------------------------------------------
def load_following_data() -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Load the user–user following graph.

    Returns
    -------
    following_dict  : {user_id: [followed_user_id, ...]}
    followee_count  : {user_id: number_of_followed_users}
    """
    try:
        df = pd.read_csv(FOLLOWING_FILE_PATH)
        following_dict: Dict[str, List[str]] = {}
        followee_count: Dict[str, int] = {}

        for _, row in df.iterrows():
            uid = str(row["user_id"])
            raw = row["following_list"]
            if pd.notna(raw) and raw != "":
                flist = [x.strip() for x in raw.split(",")]
            else:
                flist = []
            following_dict[uid] = flist
            followee_count[uid] = len(flist)

        return following_dict, followee_count

    except Exception as exc:
        print(f"Failed to load following data: {exc}")
        return {}, {}
