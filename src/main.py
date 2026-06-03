"""
main.py
Entry point for the PCP (Panic Cognitive Path) framework.

Orchestrates data loading, parallel BDEI-CoT execution across timesteps,
and result persistence.
"""

import os
import sys
import time
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from config import (
    API_BASE_URL, OUTPUT_PATH,
    REQUEST_DELAY_SEC, BATCH_SLEEP_RANGE, START_TIMESTEP,
)
from data_loader import (
    load_hurricane_data, load_user_data,
    load_following_data, build_user_profile,
)
from bdei_cot import execute_user, build_user_profile_text
from psd_model import get_baseline_time


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------
def load_api_configs() -> List[Dict]:
    """
    Build API config list from the environment variable PCP_API_KEYS.
    Expected format: comma-separated list of API key strings.

        export PCP_API_KEYS="key1,key2,key3"

    Falls back to the single key PCP_API_KEY if PCP_API_KEYS is not set.
    """
    multi = os.getenv("PCP_API_KEYS", "")
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
    else:
        single = os.getenv("PCP_API_KEY", "")
        keys = [single] if single else []

    if not keys:
        print("WARNING: No API keys found. Set PCP_API_KEYS or PCP_API_KEY environment variables.")

    base_url = os.getenv("PCP_API_BASE_URL", API_BASE_URL)
    return [{"api_key": k, "base_url": base_url} for k in keys]


# ---------------------------------------------------------------------------
# Batch creation
# ---------------------------------------------------------------------------
def create_user_batches(user_df: pd.DataFrame,
                        api_configs: List[Dict]) -> List[pd.DataFrame]:
    """Distribute users evenly across API configs."""
    n_users = len(user_df)
    n_apis  = len(api_configs)
    if n_apis == 0:
        return [user_df]

    base_size = n_users // n_apis
    remainder = n_users % n_apis
    batches, start = [], 0

    for i in range(n_apis):
        size  = base_size + (1 if i < remainder else 0)
        end   = min(start + size, n_users)
        batch = user_df.iloc[start:end] if start < n_users else pd.DataFrame()
        batches.append(batch)
        start = end

    return batches


# ---------------------------------------------------------------------------
# Per-batch concurrent processing
# ---------------------------------------------------------------------------
def process_batch(user_batch: pd.DataFrame,
                  hurricane_row: pd.Series,
                  following_dict: Dict,
                  followee_count: Dict,
                  panic_history: Dict,
                  timestep_idx: int,
                  dissipated_time,
                  api_config: Dict,
                  user_info_map: Dict,
                  last_known_position=None) -> List:
    """Process all users in *user_batch* for one timestep using a single API config."""
    baseline_time = get_baseline_time()
    results = []

    def _run_user(row):
        uid = str(row["user_id"])
        info = user_info_map.get(uid)
        if info is None:
            return None, None
        try:
            result = execute_user(
                info, hurricane_row,
                following_dict, followee_count, panic_history,
                timestep_idx, baseline_time, dissipated_time,
                api_config, last_known_position,
            )
            lo, hi = BATCH_SLEEP_RANGE
            time.sleep(random.uniform(lo, hi))
            return uid, result
        except Exception as exc:
            print(f"User {uid} error: {exc}")
            return None, None

    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {
            pool.submit(_run_user, user_batch.iloc[j]): j
            for j in range(len(user_batch))
        }
        for fut in as_completed(futures):
            try:
                uid, result = fut.result(timeout=300)
                if result:
                    results.append((uid, result))
            except Exception as exc:
                print(f"Batch future error: {exc}")
                time.sleep(5)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load data
    hurricane_df, dissipated_time, last_known_position = load_hurricane_data()
    user_df = load_user_data(start_row=START_TIMESTEP)

    # Pre-build user info map
    user_info_map: Dict[str, Dict] = {}
    for _, row in user_df.iterrows():
        info = build_user_profile(row)
        info["profile_text"] = build_user_profile_text(info)
        user_info_map[info["user_id"]] = info

    following_dict, followee_count = load_following_data()
    api_configs = load_api_configs()

    panic_history: Dict = {}
    all_results:   List = []

    # Timestep loop
    for ts_idx in tqdm(range(len(hurricane_df)), desc="Timestep"):
        hurricane_row = hurricane_df.iloc[ts_idx]
        batches       = create_user_batches(user_df, api_configs)

        valid_batches = [(b, api_configs[i]) for i, b in enumerate(batches) if not b.empty]

        with ThreadPoolExecutor(max_workers=len(valid_batches)) as pool:
            future_map = {
                pool.submit(
                    process_batch,
                    batch, hurricane_row,
                    following_dict, followee_count, panic_history,
                    ts_idx, dissipated_time, cfg, user_info_map, last_known_position,
                ): i
                for i, (batch, cfg) in enumerate(valid_batches)
            }

            for fut in tqdm(as_completed(future_map),
                            total=len(future_map),
                            desc=f"Ts {ts_idx} batches",
                            leave=False):
                try:
                    batch_results = fut.result(timeout=600)
                    for uid, result in batch_results:
                        if result:
                            panic_history.setdefault(uid, []).append(
                                (ts_idx, result["panic_emotion"]))
                            all_results.append(result)
                except Exception as exc:
                    print(f"Batch error: {exc}")
                    traceback.print_exc()

        time.sleep(REQUEST_DELAY_SEC * 2)

    # Persist results
    print(f"\nTotal records: {len(all_results)}")
    results_df = pd.DataFrame(all_results)

    required_cols = [
        "user_id", "timestep", "time", "psychological_distance",
        "awareness_score", "novelty_score", "uncertainty_score", "coping_score",
        "safety_desire_aroused", "safety_desire_intensity",
        "control_desire_aroused", "control_desire_intensity",
        "certainty_desire_aroused", "certainty_desire_intensity",
        "aroused_desires_count", "avg_desire_intensity",
        "panic_emotion", "desire_details",
    ]
    for col in required_cols:
        if col not in results_df.columns:
            results_df[col] = 0 if "aroused" in col or col in ("panic_emotion",) else (
                0.0 if "intensity" in col or "score" in col else (
                "[]" if col == "desire_details" else 0))

    if len(results_df):
        results_df = results_df.sort_values(["user_id", "timestep"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
