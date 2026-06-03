<div align="center">

# 🌀 PCP — Panic Cognitive Path Framework

> **Predicting delayed panic arousal speed in hurricane disasters via a multi-domain cognitive path model**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Hurricane%20Sandy-orange)](data/samples/)
[![LLM](https://img.shields.io/badge/LLM-Qwen2.5--72B-blueviolet)](https://huggingface.co/Qwen)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-yellow)]()

</div>

---

## 📖 Overview

**PCP (Panic Cognitive Path)** is a framework for predicting panic-arousal timing in social media users during hurricane disasters.  
It couples three cognitive domains into a single reasoning chain:

```
Physical Domain (hurricane meteorology)
        │
        ▼
Psychological Safety Distance (PSD)  ←── Social Domain (follower network)
        │
        ▼  [perceived risk > threshold]
BDEI Chain-of-Thought (LLM role-play)
        │
        ├── Belief  →  risk perception questionnaire (PPDTS 18-item)
        ├── Desire  →  three desire types aroused
        ├── Emotion →  panic probability computation
        └── Intent  →  delayed panic label (timestep of first panic)
```

The framework is validated on the **Hurricane Sandy (2012)** Twitter dataset.

| Metric | Value |
|---|---|
| Accuracy | **71.35 %** |
| Peak-time Error | **11.88 %** |
| Macro F1 | **0.5561** |

---

## 🗂️ Repository Structure

```
PCP-Framework/
├── src/
│   ├── config.py        # All paths, thresholds, hyper-parameters
│   ├── psd_model.py     # Psychological Safety Distance (PSD) model
│   ├── data_loader.py   # Dataset I/O and preprocessing
│   ├── bdei_cot.py      # BDEI Chain-of-Thought pipeline (LLM)
│   └── main.py          # Orchestration entry point
├── data/
│   └── samples/         # Small example rows for each dataset
├── contriever/
│   └── config.json      # Contriever model configuration
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## ⚙️ Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Configure paths and API keys

```bash
cp .env.example .env
# Edit .env — supply your LLM API key(s) and data directory
```

Key environment variables:

| Variable | Description |
|---|---|
| `PCP_API_KEYS` | Comma-separated API keys (e.g. `key1,key2`) |
| `PCP_API_BASE_URL` | LLM API endpoint (default: SiliconFlow) |
| `PCP_LLM_MODEL` | LLM model name (default: `Qwen/Qwen2.5-72B-Instruct`) |
| `PCP_DATA_DIR` | Path to your local data directory |
| `CONTRIEVER_MODEL_PATH` | Path to the Contriever model weights |

### 3 · Run

```bash
cd src
python main.py
```

Results are written to `data/output/BDEI_CoT_results.csv`.

---

## 📊 Datasets

All experiments use data collected from Twitter during **Hurricane Sandy (Oct–Nov 2012)**.  
Full datasets are **not redistributed** in this repository for privacy and size reasons.  
Sample rows for each dataset are available under `data/samples/`.

---

### 🌪️ `hurricane_weather_data_115.xlsx`  — Hurricane Meteorological Data

Tracks the lifecycle of Hurricane Sandy across **115 timesteps** (6-hour intervals).

| Field | Type | Description |
|---|---|---|
| `time` | datetime (UTC) | Observation timestamp |
| `category` | str / NaN | Hurricane category: `D`, `S`, `E`, `H1`, `H2`, `H3`, or NaN (dissipated) |
| `wind` | float | Maximum sustained wind speed (km/h) |
| `air_pressure` | float | Central air pressure (hPa) |
| `lat_final` | float | Hurricane centre latitude (°N) |
| `lng_final` | float | Hurricane centre longitude (°W, negative) |
| `event_description` | str | Narrative description of notable events at this timestep |

**Statistics**

| Item | Value |
|---|---|
| Total timesteps | 115 |
| Date range | 2012-10-23 → 2012-11-10 |
| Active hurricane timesteps | ~90 |
| Post-dissipation timesteps | ~25 |
| Category distribution | S (Tropical Storm) · E (Extratropical) · H1–H3 |

---

### 👤 `merged_user_files_panic_user_update.csv`  — User Profile Data

Preprocessed profile for each Twitter user, including Big-Five personality scores,  
geo-location, topic interests, and ground-truth panic labels.

| Field | Type | Description |
|---|---|---|
| `user_id` | int | Unique Twitter user identifier |
| `user_followers_count` | int | Number of followers |
| `user_friends_count` | int | Number of accounts the user follows |
| `lat_final` | float | Inferred user latitude |
| `lng_final` | float | Inferred user longitude |
| `user_event_attitude` | float | Attitude score toward the disaster event |
| `sentiment_trend` | float | Overall sentiment trend during the event |
| `topic` | str | Comma-separated topic interests |
| `Extroversion` | float | Big-Five E score (0–1) |
| `Neuroticism` | float | Big-Five N score (0–1) |
| `Agreeableness` | float | Big-Five A score (0–1) |
| `Conscientiousness` | float | Big-Five C score (0–1) |
| `Openness` | float | Big-Five O score (0–1) |
| `tone_of_voice` | str | Comma-separated writing style tags |
| `text_average_count` | float | Average daily tweet count |
| `panic_user` | int | Ground-truth label: `0` = non-panic, `1` = immediate panic, `2` = delayed panic |
| `text` | str | Concatenated representative tweets |
| `recent_tweets` | JSON str | List of Sandy-relevant tweets (used in LLM context) |
| `emotional_stability` | str | Natural-language emotional stability description |

**Statistics**

| Item | Value |
|---|---|
| Total users | 9,065 |
| Non-panic (`0`) | 5,825 (64.3 %) |
| Immediate-panic (`1`) | 2,164 (23.9 %) |
| Delayed-panic (`2`) | 1,076 (11.9 %) |

---

### 🐦 `cleaned_Formed_data.csv`  — Tweet Data

All Sandy-period tweets from the user cohort, cleaned and geo-tagged.

| Field | Type | Description |
|---|---|---|
| `tweet_id` | int | Unique tweet identifier |
| `text` | str | Tweet content |
| `created_at` | datetime | Tweet creation timestamp |
| `user_id` | int | Author user ID |
| `user_followers_count` | int | Follower count at tweet time |
| `user_friends_count` | int | Following count at tweet time |
| `sentiment_ground_truth` | int | Sentiment label: `-1` neg / `0` neutral / `1` pos |
| `lat_final` | float | Author latitude |
| `lng_final` | float | Author longitude |
| `emoji` | str | Emoji characters present in tweet (or NaN) |

**Statistics**

| Item | Value |
|---|---|
| Total tweets | 242,363 |
| Date range | Oct 2012 – Nov 2012 |
| Unique users | ~9,065 |

---

### 🔗 `following_relationships_1.csv`  — Social Graph

User-level following relationships among the study cohort.

| Field | Type | Description |
|---|---|---|
| `user_id` | int | User identifier |
| `following_count` | int | Total number of users followed |
| `following_list` | str | Comma-separated list of followed user IDs |

**Statistics**

| Item | Value |
|---|---|
| Nodes (users) | 8,833 |
| Avg. following per user | varies |

---

### 🚨 `panic_disaster_data.csv`  — Panic-Labelled Tweet Data

Extended tweet dataset with an additional panic-relevance label.

| Field | Type | Description |
|---|---|---|
| `tweet_id` – `emoji` | — | Same as `cleaned_Formed_data.csv` |
| `panic_related` | int | `1` if tweet content is panic-related, `0` otherwise |

**Statistics**

| Item | Value |
|---|---|
| Total records | 1,142,626 |

---

## 🧠 Model Details

### Psychological Safety Distance (PSD)

The PSD value **V** quantifies perceived risk using four dimensions:

```
V(x, p, t, s, n) = exp(−[a·ln(t+1) + b·ln(s+1) + c·ln(n+1)]) · p · x
```

| Symbol | Meaning |
|---|---|
| `t` | Temporal distance (days from landfall baseline) |
| `s` | Spatial distance (km between user and hurricane) |
| `n` | Social distance (1 − panic ratio among followed users) |
| `x` | Loss scalar derived from hurricane category |
| `p` | Occurrence probability (= 1 for confirmed hurricane) |
| `a, b, c` | Discount weights (all = 1.0 by default) |

Users with `V ≥ SAFETY_THRESHOLD (0.08)` enter the BDEI-CoT stage.

### BDEI Chain-of-Thought

The LLM (Qwen2.5-72B) role-plays as each user and answers 18 PPDTS items.  
Responses are mapped to four cognitive factors:

| Factor | Items (1-indexed) | Scoring |
|---|---|---|
| Awareness | Q4, Q8, Q9 | Forward |
| Novelty | Q1, Q5, Q10 | Reverse |
| Uncertainty | Q2, Q3, Q6, Q7 | Reverse |
| Coping | Q11–Q18 | Forward |

Three desires are then assessed against a threshold (default **3.5 / 5**):

- **Desire for Safety** — average of all four factors  
- **Desire for Control** — average of Awareness + (5 − Coping)  
- **Desire for Certainty** — average of Uncertainty + Novelty  

Final panic probability:

```
P(panic) = (n_desires / 3) × avg_intensity × safety_factor
```

---

## 📝 Citation

> **Note:** The associated paper is currently under review. Citation information will be added upon publication.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
<sub>Built for research on extreme emotion propagation and panic prediction in social media during disasters.</sub>
</div>
