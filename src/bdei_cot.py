"""
bdei_cot.py
BDEI Chain-of-Thought reasoning pipeline.

Implements the three-stage cognitive reasoning chain:
    Stage 1 — PSD filtering:  check whether a user perceives risk
    Stage 2 — Desire arousal: role-play 18 PPDTS questions via LLM
    Stage 3 — Panic emotion:  map aroused desires to a panic outcome
"""

import json
import re
import random
import time
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd

from config import (
    API_BASE_URL, API_KEY, LLM_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MAX_RETRIES,
    SAFETY_THRESHOLD, DESIRE_THRESHOLD,
    BIG_FIVE_THRESHOLDS,
)
from psd_model import calculate_psd, get_baseline_time


# ---------------------------------------------------------------------------
# Psychology knowledge base (injected into LLM system context)
# ---------------------------------------------------------------------------
PSYCHOLOGY_KNOWLEDGE = """
1. Public Risk Perception Formation:
   Risk perception is shaped by two factors and their interaction:
   (a) Characteristics of the risk event itself
   (b) Personal characteristics of the audience

2. Personality Traits and Risk Response:
   Psychoticism: Users with higher psychoticism may overestimate their ability
     to control events, potentially leading to riskier behaviours.
   Neuroticism: Neuroticism affects emergency comprehension and fear levels.
     Users above 0.537 may experience higher fear and prefer passive coping.
   Extraversion: Users above 0.525 tend to adopt proactive measures.
   Agreeableness: Users above 0.449 may seek harmony and follow recommended actions.
   Conscientiousness: Users above 0.304 may adhere strictly to safety protocols.
   Openness: Users above 0.522 may explore innovative solutions.

3. Social Media Language Style Effects:
   Sarcasm/irony may amplify anxiety in crisis contexts.

4. Content Type Emotional Impacts:
   Disaster-related serious news increases situational awareness but may elevate stress.

5. Emotional Stability Mechanisms:
   Regular use of cognitive reappraisal strategies buffers acute stress during disasters.

6. Social Media Network Characteristics and Panic Formation:
   Users with more follows/followers are exposed to diverse information,
     increasing cognitive load and anxiety.
   Dense social networks can lead to group polarisation and echo-chamber effects.
   Social comparison can weaken self-efficacy when others display superior coping resources.
"""

# ---------------------------------------------------------------------------
# Hurricane event taxonomy
# ---------------------------------------------------------------------------
HURRICANE_DESCRIPTION = {
    "core":        "Hurricane Sandy, tropical storm, extreme weather",
    "response":    "rescue operations, emergency response, disaster relief",
    "impact":      "government actions, evacuation procedures, infrastructure damage",
    "preparation": "supply distribution, supply preparation, power restoration, medical aid",
    "secondary":   "flooding, power outages, transportation disruption",
}


# ---------------------------------------------------------------------------
# Helper: user-profile text block
# ---------------------------------------------------------------------------
def build_user_profile_text(user_info: Dict) -> str:
    bf = user_info["big_five"]
    t = BIG_FIVE_THRESHOLDS
    return (
        "User Profile:\n"
        "- Personality Traits (Big Five, scaled 0-1):\n"
        f"  * Extraversion:     {bf['E']:.3f} ({'Above average' if bf['E'] > t['E'] else 'Below average'})\n"
        f"  * Neuroticism:      {bf['N']:.3f} ({'High emotional sensitivity' if bf['N'] > t['N'] else 'Emotionally stable'})\n"
        f"  * Agreeableness:    {bf['A']:.3f} ({'Cooperative and trusting' if bf['A'] > t['A'] else 'More independent'})\n"
        f"  * Conscientiousness:{bf['C']:.3f} ({'Organised and disciplined' if bf['C'] > t['C'] else 'Flexible and adaptable'})\n"
        f"  * Openness:         {bf['O']:.3f} ({'Open to new experiences' if bf['O'] > t['O'] else 'Prefer familiar routines'})\n"
        f"- Emotional Stability: {user_info['emotional_stability']}\n"
        f"- Typical Interests: {', '.join(user_info['interests'][:3]) or 'General topics'}\n"
        f"- Recent Relevant Tweets: {user_info['recent_tweets'][:2] or 'None'}"
    ).strip()


# ---------------------------------------------------------------------------
# PPDTS 18-item questionnaire
# ---------------------------------------------------------------------------
PPDTS_QUESTIONS = [
    "I am familiar with the natural hazard/disaster preparedness materials relevant to my area",
    "I know how to adequately prepare my home for the forthcoming fire/flood/cyclone season",
    "I know which household preparedness measures are needed to stay safe in a natural hazard/disaster",
    "I know what to look out for in my home and workplace if an emergency weather situation should develop",
    "I am familiar with the disaster warning system messages used for extreme weather events",
    "I am confident that I know what to do and what actions to take in a severe weather situation",
    "I would be able to locate the natural hazard/disaster preparedness materials in a warning situation easily",
    "I am knowledgeable about the impact that a natural hazard/disaster can have on my home",
    "I know what the difference is between a disaster warning and a disaster watch situation",
    "I am familiar with the weather signs of an approaching fire/flood/cyclone",
    "I think I am able to manage my feelings pretty well in difficult and challenging situations",
    "In a natural hazard/disaster situation I would be able to cope with my anxiety and fear",
    "I seem to be able to stay cool and calm in most difficult situations",
    "I feel reasonably confident in my own ability to deal with stressful situations that I might find myself in",
    "When necessary, I can talk myself through challenging situations",
    "If I found myself in a natural hazard/disaster situation I would know how to manage my own response",
    "I know which strategies I could use to calm myself in a natural hazard/disaster situation",
    "I have a good idea of how I would likely respond in an emergency situation",
]

# Factor–question mapping (1-indexed)
FACTORS_MAPPING = {
    "awareness":   {"indices": [4, 8, 9],          "reverse": False},
    "novelty":     {"indices": [1, 5, 10],          "reverse": True},
    "uncertainty": {"indices": [2, 3, 6, 7],        "reverse": True},
    "coping":      {"indices": [11,12,13,14,15,16,17,18], "reverse": False},
}


def parse_answers(response_text: str) -> Dict:
    pattern = r"Q(\d+)\s*:\s*(\d+)\s*\(([^)]+?)\)"
    matches = re.findall(pattern, response_text, re.MULTILINE)
    if not matches:
        raise ValueError("No Q-format answers found in LLM response.")
    answers = {}
    for m in matches:
        q_num = int(m[0])
        q_text = PPDTS_QUESTIONS[q_num - 1] if q_num - 1 < len(PPDTS_QUESTIONS) else f"Q{q_num}"
        answers[f"Q{q_num}"] = {
            "score":  int(m[1]),
            "reason": m[2].strip(),
            "text":   q_text,
        }
    if len(answers) != 18:
        raise ValueError(f"Expected 18 answers, got {len(answers)}")
    return answers


def calculate_four_factors(answers: Dict) -> Dict[str, float]:
    """Convert 18 PPDTS answers into four factor scores on a 1-5 scale."""
    results = {}
    for factor, meta in FACTORS_MAPPING.items():
        scores = []
        for idx in meta["indices"]:
            key = f"Q{idx}"
            if key in answers:
                s = answers[key]["score"]
                scores.append(5 - s if meta["reverse"] else s)
        if scores:
            raw_avg = sum(scores) / len(scores)
            results[factor] = round(1 + (4 / 3) * (raw_avg - 1), 2)
        else:
            results[factor] = 3.0
    return results


# ---------------------------------------------------------------------------
# Hurricane category description helper
# ---------------------------------------------------------------------------
CATEGORY_DESCRIPTIONS = {
    "D":  "Tropical Depression (minimal damage: road flooding, branch breakage)",
    "S":  "Tropical Storm (moderate damage: window breakage, temporary structure collapse)",
    "E":  "Extratropical (variable conditions: combined wind-rain-snow disasters)",
    "H1": "Category 1 Hurricane (moderate damage: partial roof damage, prolonged power outages)",
    "H2": "Category 2 Hurricane (extensive damage: structural damage, widespread tree falls)",
    "H3": "Category 3 Hurricane (devastating damage: building frame exposure, coastal flooding)",
}


def get_category_description(category, event_description="") -> str:
    if pd.isna(category):
        base = "Hurricane has dissipated - minimal direct threat remaining"
    else:
        base = CATEGORY_DESCRIPTIONS.get(category, "Unknown category")
    if event_description and pd.notna(event_description):
        return f"{base}\nRecent Event: {event_description}"
    return base


# ---------------------------------------------------------------------------
# LLM call (raw httpx, bypasses SDK timeout quirks)
# ---------------------------------------------------------------------------
def _call_llm(messages: List[Dict], api_config: Dict,
              temperature: float = LLM_TEMPERATURE) -> str:
    url = f"{api_config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_config['api_key']}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       LLM_MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  LLM_MAX_TOKENS,
    }
    timeout = httpx.Timeout(connect=180.0, read=300.0, write=30.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _build_assessment_messages(profile_text: str, situation_text: str) -> List[Dict]:
    system_msg = {
        "role":    "system",
        "content": (
            "You are role-playing as a Twitter user during Hurricane Sandy. "
            "Answer questions from this user's perspective, considering their personality, "
            "experiences, and the current situation. "
            "Do NOT mention that you are an AI or language model."
        ),
    }
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(PPDTS_QUESTIONS))
    user_content = (
        f"Based on this user profile and current situation, answer the following "
        f"18 questions as if you ARE this user during Hurricane Sandy.\n"
        f"{profile_text}\n{situation_text}\n\n"
        f"Answer these questions from the user's perspective:\n{questions_text}\n\n"
        "Response Requirements:\n"
        "1. Answer ALL 18 questions sequentially\n"
        "2. Format: 'Q[number]: [score] (reason)'\n"
        "   Example: Q1: 3 (I've seen hurricane materials before)\n"
        "3. Use 1-4 rating scale:\n"
        "   1: Completely Disagree, 2: Somewhat Agree, "
        "   3: Mostly Agree, 4: Completely Agree\n"
        "4. Do NOT include any internal thinking process\n"
        "5. Provide answers in direct format without any additional text"
    )
    return [system_msg, {"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# Single-user BDEI-CoT execution
# ---------------------------------------------------------------------------
def execute_user(
        user_info: Dict,
        hurricane_row: pd.Series,
        following_dict: Dict,
        followee_count: Dict,
        panic_history: Dict,
        current_timestep: int,
        baseline_time,
        dissipated_time,
        api_config: Dict,
        last_known_position=None) -> Optional[Dict]:
    """
    Run the full three-stage BDEI-CoT pipeline for a single user at one timestep.

    Returns a result dict, or None if an unrecoverable error occurs.
    """
    # --- Stage 1: PSD filtering ---
    psd = calculate_psd(
        user_info, hurricane_row,
        following_dict, followee_count, panic_history,
        current_timestep, baseline_time, dissipated_time, last_known_position,
    )
    V              = psd["psychological_distance"]
    distance_km    = psd["distance_km"]
    panic_ratio    = psd["panic_ratio"]
    dissipated     = pd.isna(hurricane_row["Category"])

    null_result = {
        "user_id":                  user_info["user_id"],
        "timestep":                 current_timestep,
        "time":                     hurricane_row["time"],
        "psychological_distance":   V,
        "awareness_score":          0.0,
        "novelty_score":            0.0,
        "uncertainty_score":        0.0,
        "coping_score":             0.0,
        "safety_desire_aroused":    0,
        "safety_desire_intensity":  0.0,
        "control_desire_aroused":   0,
        "control_desire_intensity": 0.0,
        "certainty_desire_aroused": 0,
        "certainty_desire_intensity": 0.0,
        "aroused_desires_count":    0,
        "avg_desire_intensity":     0.0,
        "panic_emotion":            0,
        "desire_details":           "[]",
        "llm_failed":               0,
    }

    if V < SAFETY_THRESHOLD:
        return null_result

    # --- Build situation context ---
    days_delta = abs(
        (pd.Timestamp("2012-10-30", tz="UTC") - pd.to_datetime(hurricane_row["time"])).days
    )
    ev_desc = hurricane_row.get("event_description", "") or ""

    if dissipated:
        situation_text = (
            f"Current Situation:\n"
            f"- Hurricane Status: DISSIPATED\n"
            f"- Time Since Landfall: {days_delta} days\n"
            f"- Wind Speed: Negligible\n"
            f"- Recent Events: {ev_desc or 'No major events reported'}\n"
            f"- Social Context: {panic_ratio*100:.1f}% of people I follow showed panic earlier"
        )
    else:
        situation_text = (
            f"Current Hurricane Situation:\n"
            f"- Time to Landfall: {days_delta} days\n"
            f"- Distance from Hurricane: {distance_km:.0f} km\n"
            f"- Hurricane Intensity: Category {hurricane_row['Category']} "
            f"({get_category_description(hurricane_row['Category'], ev_desc)})\n"
            f"- Wind Speed: {hurricane_row['Wind (km/h)']} km/h\n"
            f"- Recent Events: {ev_desc or 'No major events reported'}\n"
            f"- Social Context: {panic_ratio*100:.1f}% of people I follow have shown panic"
        )

    # --- Stage 2: LLM desire arousal with retry ---
    messages = _build_assessment_messages(user_info["profile_text"], situation_text)
    llm_ok = False
    factor_scores = {}
    response_text = ""

    for attempt in range(LLM_MAX_RETRIES):
        current_msgs = messages.copy()
        if attempt > 0:
            current_msgs.append({
                "role":    "system",
                "content": (
                    "IMPORTANT: Answer strictly as: 'Q[number]: [score] (reason)'. "
                    "No additional text."
                ),
            })
        try:
            temp = 0.5 if attempt > 0 else LLM_TEMPERATURE
            response_text = _call_llm(current_msgs, api_config, temp)
            answers       = parse_answers(response_text)
            factor_scores = calculate_four_factors(answers)
            llm_ok        = True
            break
        except Exception as exc:
            msg = str(exc)
            if "connection" in msg.lower():
                wait = min(30, 3 * (attempt + 1))
            elif "timeout" in msg.lower() or "timed out" in msg.lower():
                wait = min(60, 10 * (attempt + 1))
            else:
                wait = min(60, 5 * (2 ** attempt))
            time.sleep(wait)

    if not llm_ok:
        result = null_result.copy()
        result["llm_failed"] = 1
        return result

    # --- Stage 3: desire → panic mapping ---
    aw = factor_scores["awareness"]
    nv = factor_scores["novelty"]
    un = factor_scores["uncertainty"]
    cp = 5 - factor_scores["coping"]   # invert: high coping => low uncontrollability

    THRESH = DESIRE_THRESHOLD
    safety_avg    = (aw + nv + un + cp) / 4
    control_avg   = (aw + cp) / 2
    certainty_avg = (un + nv) / 2

    desires = []
    desire_details = []

    for tag, name, avg, factors, scores in [
        ("safety",    "desire_for_safety",    safety_avg,
         ["awareness","novelty","uncertainty","coping"], [aw,nv,un,cp]),
        ("control",   "desire_for_control",   control_avg,
         ["awareness","coping"], [aw,cp]),
        ("certainty", "desire_for_certainty", certainty_avg,
         ["uncertainty","novelty"], [un,nv]),
    ]:
        if avg > THRESH:
            desires.append((name, avg))
            desire_details.append({
                "type":          tag,
                "name":          name,
                "intensity":     avg,
                "factors":       factors,
                "factor_scores": scores,
                "avg_score":     avg,
            })

    if desires:
        n_d = len(desires)
        avg_intensity  = sum(d[1] for d in desires) / n_d
        safety_factor  = min(1.0, V / SAFETY_THRESHOLD)
        panic_prob     = (n_d / 3) * avg_intensity * safety_factor
        panic_emotion  = 1 if random.random() < panic_prob else 0
    else:
        n_d           = 0
        avg_intensity = 0.0
        panic_emotion = 0

    def _get(desires, tag, key):
        for d in desire_details:
            if d["type"] == tag:
                return d[key]
        return 0.0

    return {
        "user_id":                    user_info["user_id"],
        "timestep":                   current_timestep,
        "time":                       hurricane_row["time"],
        "psychological_distance":     V,
        "awareness_score":            factor_scores.get("awareness",    0.0),
        "novelty_score":              factor_scores.get("novelty",      0.0),
        "uncertainty_score":          factor_scores.get("uncertainty",  0.0),
        "coping_score":               factor_scores.get("coping",       0.0),
        "safety_desire_aroused":      1 if _get(desires,"safety","intensity") > 0 else 0,
        "safety_desire_intensity":    _get(desires,"safety","intensity"),
        "control_desire_aroused":     1 if _get(desires,"control","intensity") > 0 else 0,
        "control_desire_intensity":   _get(desires,"control","intensity"),
        "certainty_desire_aroused":   1 if _get(desires,"certainty","intensity") > 0 else 0,
        "certainty_desire_intensity": _get(desires,"certainty","intensity"),
        "aroused_desires_count":      n_d,
        "avg_desire_intensity":       avg_intensity,
        "panic_emotion":              panic_emotion,
        "desire_details":             json.dumps(desire_details),
        "llm_failed":                 0,
    }
