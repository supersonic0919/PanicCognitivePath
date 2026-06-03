"""
config.py
Global configuration for the PCP (Panic Cognitive Path) framework.
Paths, thresholds, and model hyper-parameters are centralised here.
"""

import os

# ---------------------------------------------------------------------------
# Data paths  (override with environment variables or edit here)
# ---------------------------------------------------------------------------
DATA_DIR = os.getenv("PCP_DATA_DIR", "./data")

HURRICANE_EXCEL_PATH   = os.path.join(DATA_DIR, "hurricane_weather_data_115.xlsx")
USER_PROFILE_PATH      = os.path.join(DATA_DIR, "merged_user_files_panic_user_update.csv")
TWEET_DATA_PATH        = os.path.join(DATA_DIR, "cleaned_Formed_data.csv")
FOLLOWING_FILE_PATH    = os.path.join(DATA_DIR, "following_relationships_1.csv")
OUTPUT_PATH            = os.path.join(DATA_DIR, "output", "BDEI_CoT_results.csv")

# Contriever model path (used for semantic tweet retrieval)
CONTRIEVER_MODEL_PATH  = os.getenv("CONTRIEVER_MODEL_PATH", "./contriever")

# ---------------------------------------------------------------------------
# API configuration  (supply via environment variables — never hard-code keys)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("PCP_API_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY      = os.getenv("PCP_API_KEY", "")          # set in environment
LLM_MODEL    = os.getenv("PCP_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# ---------------------------------------------------------------------------
# PSD (Psychological Safety Distance) parameters
# ---------------------------------------------------------------------------
PSD_ALPHA = 1.0    # temporal discount weight  (a)
PSD_BETA  = 1.0    # spatial  discount weight  (b)
PSD_GAMMA = 1.0    # social   discount weight  (c)
PSD_PROB  = 1.0    # occurrence probability    (p) — hurricane is confirmed

SAFETY_THRESHOLD = 0.08   # PSD threshold; below this => no risk perceived

# Dissipation decay: scales perceived spatial distance after hurricane dissipation
DISSIPATION_DECAY = 0.8

# Hurricane loss scalars per category
CATEGORY_LOSS_MAP = {
    "D":  100,
    "S":  200,
    "E":  300,
    "H1": 400,
    "H2": 500,
    "H3": 600,
}

# Post-dissipation loss decay schedule
DISSIPATION_BASE_LOSS   = 300.0
DISSIPATION_DECAY_SLOW  = 0.92   # days 1-3 after dissipation
DISSIPATION_DECAY_FAST  = 0.85   # days 8+  after dissipation
DISSIPATION_FLOOR       = 100.0

# ---------------------------------------------------------------------------
# Social distance parameters
# ---------------------------------------------------------------------------
COOLDOWN_STEPS     = 3    # timesteps with minimal social distance after first panic
RECOVERY_DURATION  = 10   # timesteps to fully recover social distance

# ---------------------------------------------------------------------------
# BDEI-CoT parameters
# ---------------------------------------------------------------------------
DESIRE_THRESHOLD = 3.5    # factor score threshold to arouse a desire (1-5 scale)
LLM_TEMPERATURE  = 0.3
LLM_MAX_TOKENS   = 1500
LLM_MAX_RETRIES  = 10

# Big-Five personality thresholds (used for profile description)
BIG_FIVE_THRESHOLDS = {
    "E": 0.525,   # Extraversion
    "N": 0.537,   # Neuroticism
    "A": 0.449,   # Agreeableness
    "C": 0.304,   # Conscientiousness
    "O": 0.522,   # Openness
}

# Baseline landfall date for temporal distance calculation
BASELINE_DATE = "2012-10-30"

# Processing
START_TIMESTEP     = 0    # first timestep to process (0 = process all)
REQUEST_DELAY_SEC  = 8    # inter-timestep sleep (seconds)
BATCH_SLEEP_RANGE  = (8, 12)   # per-user sleep range inside a batch
