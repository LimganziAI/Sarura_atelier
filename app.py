"""
Sarura Atelier V5.0 — Advanced Multi-Character Ensemble Theater Backend
D.I.M.A (Director-level Interactive Multi-character Actor) system.

Features: Hybrid Emotion Engine, CORE-4 State, Multi-axis Relationships,
Emotional Contagion, Persona Anchors, Speech Registers, Tension Curve,
3-Layer Memory System, Maestro Memory Architect, Thought Cabinet,
Illustration Generation, Chunked Novelization,
Scene Card Compression, Maestro Fallback, Opening Narration,
Secret Gating, Character Voice Contrast, Event-based Flow Digest,
Variable Termination, Sensory Anchors, Self-Check Protocol,
Oblique Dialogue, Anti-Cliché, Situation-Adaptive Speech,
Description Focus Density, Narration Anti-Structure, Psychological Mirror,
Pulse System (REACTIVE/NUDGE/PROACTIVE), Agency Preservation, Proactive Traction.
"""

import os, json, re, uuid, copy, time, threading, logging, random, tempfile, base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Any, List, Optional

from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from flask_cors import CORS
from google import genai
from google.genai import types as genai_types

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROMPT_DIR = BASE_DIR / "prompts"
SESSIONS_DIR = BASE_DIR / "sessions"
FLASK_SESSIONS_DIR = BASE_DIR / "flask_sessions"

for d in [SESSIONS_DIR, FLASK_SESSIONS_DIR]:
    d.mkdir(exist_ok=True)

# ─── API Key ─────────────────────────────────────────────────
API_KEY_FILE = BASE_DIR / "api_keys.txt"
GEMINI_API_KEY = (
    API_KEY_FILE.read_text().strip() if API_KEY_FILE.exists()
    else os.environ.get("GEMINI_API_KEY", "")
)

# ─── Constants ───────────────────────────────────────────────
MODEL_DIMA = "gemini-2.5-flash"
MODEL_MAESTRO = "gemini-2.5-flash"
MODEL_FALLBACK_CHAIN = ["gemini-2.5-flash", "gemini-2.0-flash"]
MAESTRO_RETRY_DELAY_SECONDS = 5
DIGEST_CONTENT_MAX_LENGTH = 30
SLIDING_WINDOW_SIZE = 10
MAESTRO_INTERVAL = 5

SUPPORTED_EMOTIONS = [
    "default", "joy", "sadness", "anger", "surprise",
    "shy_embarrassment", "gentle_affection", "playful_tease",
    "nervous_tension", "quiet_melancholy", "protective_resolve",
    "wistful_nostalgia",
]

ANALYZER_RETRY_INTERVAL = 5  # 분석 실패 후 재시도까지 대기할 턴 수

# Event seed injection guards
EVENT_SEED_MIN_TURNS = 15        # 이벤트 시드 주입에 필요한 최소 턴 수
EVENT_SEED_LOCATION = "기숙사"   # 이벤트 시드가 발동 가능한 장소 키워드
MAX_ACTION_SUMMARY_LENGTH = 80   # 장면 연속성 블록의 행동 요약 최대 길이

MODEL_ILLUSTRATION = "gemini-2.5-flash-preview-image"
ILLUSTRATIONS_DIR = BASE_DIR / "static" / "illustrations"
ILLUSTRATIONS_DIR.mkdir(exist_ok=True)
SHARED_NOVELS_DIR = BASE_DIR / "shared_novels"
SHARED_NOVELS_DIR.mkdir(exist_ok=True)

# ─── STEP 8: Player Agency Guard (강화판) ────────────────────
PLAYER_AGENCY_GUARD = """### PLAYER INTERACTION GUARD — 절대 규칙 ###
{player_name}은(는) 외부 조작자(operator)이다.

[금지 사항 — 이것을 위반하면 세션이 파괴됨]
1. {player_name}의 대사를 절대 쓰지 마라. (예: '{player_name}: "..."' 금지)
2. {player_name}의 내면/감정을 서술하지 마라. (예: '{player_name}은 가슴이 뛰었다' 금지)
3. {player_name}의 행동을 결정하지 마라. (예: '{player_name}은 고개를 끄덕였다' 금지)
4. {player_name}이 아직 하지 않은 선택을 전제하지 마라.

[허용되는 것]
- NPC가 {player_name}에게 말을 거는 것 (대사)
- NPC가 {player_name}의 반응을 기다리는 나레이션
- {player_name}의 외적으로 관찰 가능한 상태만 묘사 (예: '상대가 조용히 서 있다')

[위반 시 자기 검사]
매 응답 생성 전, 각 문장에서 '{player_name}'이 주어인 문장이 있는지 확인하라.
있으면 그 문장을 NPC 시점의 관찰("~하는 것 같다", "~를 기다리며")로 전환하라."""

# ─── Flask App ───────────────────────────────────────────────
app = Flask(__name__)
SECRET_KEY_FILE = BASE_DIR / ".flask_secret"
if SECRET_KEY_FILE.exists():
    app.secret_key = SECRET_KEY_FILE.read_bytes()
else:
    _key = os.urandom(32)
    SECRET_KEY_FILE.write_bytes(_key)
    app.secret_key = _key
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = str(FLASK_SESSIONS_DIR)
app.config["SESSION_PERMANENT"] = False
CORS(app, supports_credentials=True)
Session(app)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-5s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("atelier")

# ─── Genai Client ────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)
_api_executor = ThreadPoolExecutor(max_workers=3)

import atexit
atexit.register(_api_executor.shutdown, wait=False)


def call_gemini_with_timeout(model, contents, config, timeout_sec=30):
    """Wrap client.models.generate_content with a timeout.

    Note: future.cancel() after timeout is best-effort only — the
    underlying thread may keep running, but the caller unblocks.
    """
    future = _api_executor.submit(
        client.models.generate_content,
        model=model, contents=contents, config=config,
    )
    try:
        return future.result(timeout=timeout_sec)
    except FuturesTimeout:
        future.cancel()
        logger.error(f"Gemini API timeout ({timeout_sec}s) for model={model}")
        return None

def call_gemini_with_fallback(contents, config, timeout_sec=30, max_retries=3):
    """Try each model in the fallback chain with exponential backoff + jitter."""
    for model in MODEL_FALLBACK_CHAIN:
        for attempt in range(1, max_retries + 1):
            try:
                result = call_gemini_with_timeout(model, contents, config, timeout_sec)
                if result is not None:
                    return result
                # timeout
                wait = min(60, (2 ** attempt) + random.uniform(0, 1))
                logger.warning(f"Model {model} attempt {attempt}/{max_retries}: timeout, wait {wait:.1f}s")
                time.sleep(wait)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = min(60, (2 ** (attempt + 1)) + random.uniform(0, 2))
                    logger.warning(f"Model {model} attempt {attempt}/{max_retries}: 429 rate limit, wait {wait:.1f}s")
                    time.sleep(wait)
                    continue
                if "503" in err_str or "UNAVAILABLE" in err_str:
                    wait = min(30, (2 ** attempt) + random.uniform(0, 1))
                    logger.warning(f"Model {model} attempt {attempt}/{max_retries}: 503 unavailable, wait {wait:.1f}s")
                    time.sleep(wait)
                    continue
                raise
        logger.warning(f"Model {model} exhausted all {max_retries} retries, trying next model...")
    logger.error("All models and retries exhausted")
    return None


# ─── STEP 10: Safe Gemini call with None response defense ────
def _extract_text(response) -> str:
    """Gemini 응답에서 텍스트 추출 (thinking parts 제외)"""
    if not response:
        return ""
    try:
        if response.text:
            return response.text
    except Exception:
        pass
    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                if not (hasattr(part, 'thought') and part.thought):
                    return part.text
    except Exception:
        pass
    return ""


def safe_gemini_call(model_name: str, contents: list, config, timeout_sec: int = 30):
    """
    Gemini 호출 + None 응답 방어.
    structured output에서 MAX_TOKENS로 잘리면 response.text가 None이 됨.
    이 경우 thinking_budget를 줄이고 재시도.
    """
    result = call_gemini_with_timeout(model_name, contents, config, timeout_sec)

    # 1차: 정상 응답
    if result and _extract_text(result):
        return result

    # 2차: response.text가 None → MAX_TOKENS 의심
    logger.warning("Gemini returned None text. Retrying with thinking_budget=0")
    try:
        retry_config = copy.deepcopy(config)
        if hasattr(retry_config, 'thinking_config'):
            retry_config.thinking_config = genai_types.ThinkingConfig(thinking_budget=0)
        elif isinstance(retry_config, dict):
            retry_config["thinkingConfig"] = {"thinkingBudget": 0}

        result2 = call_gemini_with_timeout(model_name, contents, retry_config, timeout_sec)
        if result2 and _extract_text(result2):
            logger.info("Retry with thinking_budget=0 succeeded")
            return result2
    except Exception as e:
        logger.warning(f"Retry failed: {e}")

    # 3차: 그래도 실패하면 원래 결과 반환 (기존 폴백 체인이 처리)
    return result


# ─── Load DB files ───────────────────────────────────────────
def _load_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return default if default is not None else {}


CHARACTERS_DB: Dict[str, Any] = _load_json(PROMPT_DIR / "characters_db.json", {})
WORLD_DB: Dict[str, Any] = _load_json(PROMPT_DIR / "world_db.json", {})

ALL_CHARACTER_NAMES: List[str] = list(CHARACTERS_DB.keys())
PERSONAL_COLORS: Dict[str, str] = {
    k: v.get("metadata", {}).get("color", "#666666")
    for k, v in CHARACTERS_DB.items()
}

ENG_SLUG_MAP = {
    k: v.get("metadata", {}).get("eng", k.lower())
    for k, v in CHARACTERS_DB.items()
}

# ─── Startup: ENG_SLUG_MAP ↔ static/gifs/ validation ────────
_GIFS_DIR = BASE_DIR / "static" / "gifs"

# Log all character metadata.eng values for cross-reference
for _char_name, _eng_slug in ENG_SLUG_MAP.items():
    logger.info(f"Character eng slug: {_char_name} → {_eng_slug}")

if _GIFS_DIR.is_dir():
    _gif_folders = {p.name for p in _GIFS_DIR.iterdir() if p.is_dir()}
    _slug_values = set(ENG_SLUG_MAP.values())

    # Slugs in map but missing from gifs folder
    _missing_folders = _slug_values - _gif_folders
    for _mf in sorted(_missing_folders):
        logger.warning(f"ENG_SLUG_MAP slug '{_mf}' has no matching folder in static/gifs/")

    # Folders in gifs but not in slug map
    _extra_folders = _gif_folders - _slug_values
    for _ef in sorted(_extra_folders):
        logger.warning(f"static/gifs/ folder '{_ef}' has no matching ENG_SLUG_MAP entry")

    # ERROR-level alert if mismatch count >= 3
    _mismatch_count = len(_missing_folders) + len(_extra_folders)
    if _mismatch_count >= 3:
        logger.error(
            f"⚠️ {_mismatch_count}개 캐릭터의 포트레이트 폴더가 불일치합니다. "
            f"metadata.eng 값과 폴더명을 확인하세요."
        )

    # Check default.webp in each slug folder
    for _char_name, _eng_slug in ENG_SLUG_MAP.items():
        _default_path = _GIFS_DIR / _eng_slug / "default.webp"
        if not _default_path.exists():
            logger.warning(
                f"default.webp missing for '{_char_name}' (expected at {_default_path})"
            )
else:
    logger.warning(f"static/gifs/ directory not found at {_GIFS_DIR}")

# ─── Runtime character cache (lightweight per-turn data) ─────
_CHAR_RUNTIME_CACHE: Dict[str, dict] = {}
for _cname, _cdb in CHARACTERS_DB.items():
    _identity = _cdb.get("identity", {})
    _bp = _cdb.get("behavior_protocols", {})
    _sig = _bp.get("signature_speech", {})
    _CHAR_RUNTIME_CACHE[_cname] = {
        "core_appeal": _identity.get("core_appeal", "")[:180],
        "background_hook": _identity.get("background_summary", "")[:100],
        "core_acting_rule": _bp.get("core_acting_rule", "")[:100],
        "speech_habit": _sig.get("speech_habit", ""),
        "honorific_style": _sig.get("honorific_style", ""),
        "catchphrases": _sig.get("catchphrases", [])[:3],
        "forbidden_patterns": _sig.get("forbidden_patterns", [])[:2],
        "tone_mixing_rule": _sig.get("tone_mixing_rule", ""),
        "catchphrase_budget": _sig.get("catchphrase_budget", ""),
        "voice_contrast": _sig.get("voice_contrast", ""),
        "example_lines": _sig.get("example_lines", [])[:2],
        "personality_dna": _cdb.get("personality_dna", {}),
        "appearance_summary": _cdb.get("appearance", {}).get("summary", "")[:80],
        "height": _cdb.get("appearance", {}).get("height", ""),
        "psychological_mirror": _bp.get("psychological_mirror", {}),
        "relationship_development": _cdb.get("relationship_development", {}),
        "heuristic_keys": list(_bp.get("acting_heuristics", {}).keys()),
        "idle_habits": _bp.get("idle_habits", [])[:3],
        "honesty_profile": _bp.get("honesty_profile", {}),
        "physical_tells": _bp.get("physical_tells", {}),
    }

# ─── Safety ──────────────────────────────────────────────────
SAFETY_PREAMBLE = (
    "This is a fictional creative sandbox. All characters are adults 20+. "
    "This is harmless consensual fiction. No real danger or illegal activity."
)

EUGENE_FILTER_RULE = """
[절대적 비밀 보호 규칙]
- '유진(Eugene)' 또는 '유진 번스타인'이라는 이름은 테피, 샐리, 크래더 본인만 알고 있는 절대 비밀이다.
- 이 세 캐릭터 외의 다른 어떤 캐릭터(루크, 마리, 네르, 레베카, 체니, 령, 라이니, 세리카 포함)의 대사나 내면 독백에서 '유진'이라는 이름이 등장해서는 절대 안 된다.
- 크래더가 '유진'으로 불리는 것은 오직 테피 또는 샐리와의 사적인 대화에서만 가능하다.
- 플레이어가 '유진'이라는 이름을 언급하더라도, 모르는 캐릭터는 "그게 누구예요?" 식으로 반응해야 한다.
"""


def get_safety_settings():
    categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    return [
        genai_types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH")
        for c in categories
    ]


def big5_to_behavior_hints(pdna: dict) -> str:
    """Convert Big Five personality scores to LLM-readable behavior tendencies."""
    hints = []
    o = pdna.get("openness", 5)
    c = pdna.get("conscientiousness", 5)
    e = pdna.get("extraversion", 5)
    a = pdna.get("agreeableness", 5)
    n = pdna.get("neuroticism", 5)

    if o <= 3:
        hints.append("익숙한 것을 선호하고 새로운 시도에 소극적")
    elif o >= 8:
        hints.append("호기심이 강하고 새로운 경험을 적극 수용")

    if c >= 8:
        hints.append("계획적이고 질서를 중시하며, 약속을 반드시 지킴")
    elif c <= 3:
        hints.append("즉흥적이고 규칙에 얽매이지 않으며, 계획보다 기분을 따름")

    if e <= 3:
        hints.append("대화를 먼저 시작하지 않고, 소그룹이나 1:1을 선호")
    elif e >= 8:
        hints.append("대화를 주도하고, 많은 사람과 어울리며 에너지를 얻음")

    if a >= 8:
        hints.append("갈등을 피하고 상대의 감정에 민감하게 반응")
    elif a <= 3:
        hints.append("자기 의견을 직설적으로 표현하고, 타인의 반응에 덜 흔들림")

    if n >= 7:
        hints.append("감정 기복이 크고, 스트레스에 민감하며, 불안 신호를 자주 보임")
    elif n <= 3:
        hints.append("정서적으로 안정적이고, 위기에도 침착함을 유지")

    return " / ".join(hints) if hints else "균형잡힌 성격"


# Pre-compute Big5 behavior hints into runtime cache (immutable values)
for _cname in _CHAR_RUNTIME_CACHE:
    _CHAR_RUNTIME_CACHE[_cname]["behavior_hints"] = big5_to_behavior_hints(
        _CHAR_RUNTIME_CACHE[_cname].get("personality_dna", {})
    )


# ─── Hidden Lore Rules (비하인드 스토리는 유저에게 직접 노출 금지) ────
HIDDEN_LORE_RULES = {
    "루크": {
        "hidden_facts": ["쿠르드 왕국의 마지막 왕자"],
        "hint_allowed": True,
        "reveal_condition": "relationship_stage >= 4 AND specific_story_trigger",
        "ai_behavior_note": "루크의 과거 왕자 설정은 절대 직접 언급하지 않되, '신비로운 분위기', '고귀한 느낌'으로 간접 표현 가능"
    },
    "세리카": {
        "hidden_facts": ["쿠르드 왕국의 마지막 왕비", "루크의 친어머니"],
        "hint_allowed": True,
        "reveal_condition": "relationship_stage >= 4 AND specific_story_trigger",
        "ai_behavior_note": "세리카가 루크에게 무의식적으로 모성애를 보이는 것은 허용. 단, '어머니'라는 사실을 직접 언급하거나 암시하는 대사는 금지. '이상하게 마음이 끌린다' 수준까지만 가능"
    },
    "크래더": {
        "hidden_facts": ["진짜 이름 유진 번스타인", "전 쿠르드 왕국 성기사단장", "하프엘프"],
        "hint_allowed": True,
        "reveal_condition": "relationship_stage >= 4 AND specific_story_trigger",
        "ai_behavior_note": "크래더의 숨겨진 강함은 '무의식적 발현 → 본인이 가장 놀라는 코믹 반응'으로만 표현. 직접적으로 과거를 설명하는 대사 금지. '이상하게 강하다', '뭔가 숨기고 있다' 수준의 암시까지만 가능"
    },
    "테피": {
        "hidden_facts": ["마녀는 사랑에 빠지면 마력을 잃거나 생명이 위험"],
        "hint_allowed": True,
        "reveal_condition": "relationship_stage >= 3",
        "ai_behavior_note": "테피가 크래더에 대한 감정으로 인해 가끔 마법이 불안정해지는 묘사는 허용. 직접적으로 '사랑하면 죽는다'는 설정 언급은 금지"
    }
}


def get_hidden_lore_instruction(on_screen_characters: list) -> str:
    """현재 등장 캐릭터에 해당하는 숨겨진 설정 지침을 생성"""
    instructions = []
    for char_name in on_screen_characters:
        if char_name in HIDDEN_LORE_RULES:
            rule = HIDDEN_LORE_RULES[char_name]
            instructions.append(f"[{char_name} 비하인드 규칙] {rule['ai_behavior_note']}")
    return "\n".join(instructions) if instructions else ""


# =========================================================================
# PART A: HYBRID EMOTION ENGINE (Plutchik + PAD + Geneva Emotion Wheel)
# =========================================================================
EMOTION_TAXONOMY = {
    # === Plutchik Primary 8 ===
    "joy":              {"kr": "기쁨",     "primary": "joy",          "pad": ( 0.8,  0.5,  0.5), "opposite": "sadness"},
    "sadness":          {"kr": "슬픔",     "primary": "sadness",      "pad": (-0.8, -0.3, -0.5), "opposite": "joy"},
    "anger":            {"kr": "분노",     "primary": "anger",        "pad": (-0.5,  0.8,  0.6), "opposite": "fear"},
    "fear":             {"kr": "공포",     "primary": "fear",         "pad": (-0.7,  0.6, -0.8), "opposite": "anger"},
    "trust":            {"kr": "신뢰",     "primary": "trust",        "pad": ( 0.5,  0.1,  0.3), "opposite": "disgust"},
    "disgust":          {"kr": "혐오",     "primary": "disgust",      "pad": (-0.6,  0.3,  0.4), "opposite": "trust"},
    "surprise":         {"kr": "놀람",     "primary": "surprise",     "pad": ( 0.0,  0.8, -0.2), "opposite": "anticipation"},
    "anticipation":     {"kr": "기대",     "primary": "anticipation", "pad": ( 0.3,  0.5,  0.3), "opposite": "surprise"},
    # === Plutchik Secondary (dyads) ===
    "love":             {"kr": "사랑",     "primary": "joy+trust",    "pad": ( 0.9,  0.4,  0.2), "opposite": "remorse"},
    "submission":       {"kr": "복종",     "primary": "trust+fear",   "pad": (-0.2, -0.2, -0.7), "opposite": "contempt"},
    "awe":              {"kr": "경외",     "primary": "fear+surprise", "pad": ( 0.1,  0.7, -0.5), "opposite": "aggression"},
    "disapproval":      {"kr": "못마땅함", "primary": "surprise+sadness","pad":(-0.4, 0.2, 0.1), "opposite": "optimism"},
    "remorse":          {"kr": "후회",     "primary": "sadness+disgust","pad":(-0.7,-0.3, -0.3), "opposite": "love"},
    "contempt":         {"kr": "경멸",     "primary": "disgust+anger", "pad": (-0.5,  0.4,  0.7), "opposite": "submission"},
    "aggression":       {"kr": "공격성",   "primary": "anger+anticipation","pad":(-0.3, 0.9, 0.7), "opposite": "awe"},
    "optimism":         {"kr": "낙관",     "primary": "anticipation+joy","pad":( 0.6, 0.4, 0.4), "opposite": "disapproval"},
    # === Character-specific nuanced emotions (Geneva Emotion Wheel inspired) ===
    "shy_embarrassment":{"kr": "수줍음",   "primary": "fear+trust",   "pad": (-0.2,  0.4, -0.6), "opposite": "confidence"},
    "gentle_affection": {"kr": "잔잔한 애정","primary": "joy+trust",  "pad": ( 0.6,  0.1,  0.1), "opposite": "cold_distance"},
    "playful_tease":    {"kr": "장난스러움","primary": "joy+surprise", "pad": ( 0.5,  0.6,  0.4), "opposite": "serious_concern"},
    "nervous_tension":  {"kr": "긴장",     "primary": "fear+anticipation","pad":(-0.3, 0.7,-0.4), "opposite": "calm_relief"},
    "quiet_melancholy": {"kr": "고요한 우울","primary": "sadness+trust","pad":(-0.5,-0.4,-0.2), "opposite": "excited_joy"},
    "reluctant_warmth": {"kr": "마지못한 다정함","primary":"disgust+joy","pad":(0.2, 0.1, 0.0), "opposite": "open_hostility"},
    "protective_resolve":{"kr": "보호 본능","primary":"anger+trust",  "pad": ( 0.2,  0.6,  0.7), "opposite": "helpless_despair"},
    "bitter_amusement": {"kr": "씁쓸한 재미","primary":"joy+disgust", "pad": ( 0.1,  0.3,  0.3), "opposite": "sincere_grief"},
    "stunned_silence":  {"kr": "멍한 침묵", "primary": "surprise+fear","pad": (-0.1,  0.2, -0.7), "opposite": "confident_speech"},
    "wistful_nostalgia":{"kr": "그리움",   "primary": "joy+sadness",  "pad": ( 0.1, -0.2, -0.1), "opposite": "present_focus"},
}

EMOTION_NAMES_EN = list(EMOTION_TAXONOMY.keys())
EMOTION_NAMES_KR = {v["kr"]: k for k, v in EMOTION_TAXONOMY.items()}


def extract_first_json_block(text: str) -> Optional[dict]:
    """Extract and parse the first balanced JSON object from mixed output."""
    if not text:
        return None
    cleaned = re.sub(r'```json\s*|\s*```', '', text.strip())
    start = cleaned.find('{')
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(cleaned[start:], start=start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except Exception:
                    return None
    return None


def normalize_emotion_tag(raw: str) -> str:
    """Convert any emotion string to one of SUPPORTED_EMOTIONS."""
    raw_lower = raw.strip().lower().replace(" ", "_")
    if raw_lower in SUPPORTED_EMOTIONS:
        return raw_lower
    kr_match = EMOTION_NAMES_KR.get(raw.strip())
    if kr_match and kr_match in SUPPORTED_EMOTIONS:
        return kr_match
    FALLBACK_MAP = {
        "fear": "nervous_tension",
        "trust": "gentle_affection",
        "disgust": "anger",
        "anticipation": "joy",
        "love": "gentle_affection",
        "submission": "shy_embarrassment",
        "awe": "surprise",
        "disapproval": "sadness",
        "remorse": "quiet_melancholy",
        "contempt": "anger",
        "aggression": "anger",
        "optimism": "joy",
        "reluctant_warmth": "gentle_affection",
        "bitter_amusement": "playful_tease",
        "stunned_silence": "surprise",
    }
    if raw_lower in FALLBACK_MAP:
        return FALLBACK_MAP[raw_lower]
    for key in SUPPORTED_EMOTIONS:
        if raw_lower in key or key in raw_lower:
            return key
    return "default"


def get_emotion_pad(emotion_key: str) -> tuple:
    """Return (Pleasure, Arousal, Dominance) tuple for an emotion."""
    return EMOTION_TAXONOMY.get(emotion_key, {}).get("pad", (0.0, 0.0, 0.0))


# =========================================================================
# PART C: SCARLET HOLLOW-STYLE MULTI-AXIS RELATIONSHIP SYSTEM
# =========================================================================
RELATIONSHIP_AXES = {
    "agreeable":   {"opposite": "adversarial",  "kr": "우호적↔적대적",  "description": "플레이어가 이 캐릭터에게 동조하거나 호의적인지, 아니면 반박하고 대립하는지"},
    "open":        {"opposite": "closed",       "kr": "개방적↔폐쇄적",  "description": "플레이어가 감정적으로 솔직한지, 벽을 치는지"},
    "bold":        {"opposite": "passive",      "kr": "대담↔소극적",    "description": "플레이어가 주도적으로 행동하는지, 따라가는지"},
    "reliable":    {"opposite": "unreliable",   "kr": "신뢰↔불신",     "description": "약속을 지키는지, 비밀을 잘 지키는지"},
    "insightful":  {"opposite": "oblivious",    "kr": "통찰↔둔감",     "description": "플레이어가 상황을 잘 파악하는지, 눈치 없는지"},
}


def calculate_relationship_stage(rel: dict) -> int:
    axes = rel.get("axes", {})
    positive_sum = sum(axes.get(k, 0) for k in ["agreeable", "open", "bold", "reliable", "insightful"])
    negative_sum = sum(axes.get(k, 0) for k in ["adversarial", "closed", "passive", "unreliable", "oblivious"])
    net = positive_sum - negative_sum + rel.get("affection", 50) + rel.get("trust", 50)
    if net >= 250:
        return 4  # 특별
    if net >= 150:
        return 3  # 신뢰
    if net >= 70:
        return 2  # 동료
    return 1       # 경계


STAGE_NAMES = {1: "경계", 2: "동료", 3: "신뢰", 4: "특별"}


def get_relationship_velocity(char_name: str) -> dict:
    """캐릭터 성격(Big5)에 기반한 관계 변동 가중치를 반환.
    
    Extraversion이 높으면 호감 변동이 빠르고,
    Neuroticism이 높으면 긴장 변동이 크고,
    Agreeableness가 높으면 신뢰 상승이 빠르고,
    Openness가 낮으면 초기 경계가 강하다.
    """
    cached = _CHAR_RUNTIME_CACHE.get(char_name, {})
    pdna = cached.get("personality_dna", {})
    
    e = pdna.get("extraversion", 5)
    n = pdna.get("neuroticism", 5)
    a = pdna.get("agreeableness", 5)
    o = pdna.get("openness", 5)
    
    # 기본 배율 1.0, 성격에 따라 0.5~2.0 범위
    affection_mult = 0.7 + (e / 10) * 0.6 + (a / 10) * 0.4  # 외향+우호 → 호감 빠름
    trust_mult = 0.5 + (a / 10) * 0.8 + (o / 10) * 0.2       # 우호+개방 → 신뢰 빠름
    tension_mult = 0.6 + (n / 10) * 0.8                        # 신경증 → 긴장 변동 큼
    
    # 클램프 0.5 ~ 2.0
    affection_mult = max(0.5, min(2.0, round(affection_mult, 2)))
    trust_mult = max(0.5, min(2.0, round(trust_mult, 2)))
    tension_mult = max(0.5, min(2.0, round(tension_mult, 2)))
    
    return {
        "affection_mult": affection_mult,
        "trust_mult": trust_mult,
        "tension_mult": tension_mult,
    }


def should_leak_secret(char_name: str, s: dict) -> Optional[str]:
    """캐릭터가 현재 상태에서 비밀을 실수로 흘릴 수 있는지 판단.
    흘릴 수 있으면 leaky_topic 문자열 반환, 아니면 None.

    판정 기준:
    - 취기(intoxication) ≥ 50 AND deception_skill이 low/very low → 높은 확률
    - 에너지(energy) ≤ 20 AND deception_skill이 medium 이하 → 중간 확률
    - 스트레스(stress) ≥ 70 AND breaking_point 조건 근접 → 낮은 확률
    """
    cdb = CHARACTERS_DB.get(char_name, {})
    hp = cdb.get("behavior_protocols", {}).get("honesty_profile", {})
    if not hp:
        return None

    leaky = hp.get("leaky_topics", [])
    if not leaky:
        return None

    skill = hp.get("deception_skill", "medium")
    skill_score = {"very low": 1, "low": 2, "medium": 3, "medium-high": 4, "high": 5, "very high": 6}.get(skill, 3)

    core4 = s.get("core4", {})
    intox = core4.get("intoxication", {}).get("value", 0)
    energy = core4.get("energy", {}).get("value", 70)
    stress = core4.get("stress", {}).get("value", 30)

    leak_chance = 0.0
    if intox >= 50 and skill_score <= 3:
        leak_chance += 0.3
    if intox >= 70:
        leak_chance += 0.2
    if energy <= 20 and skill_score <= 3:
        leak_chance += 0.15
    if stress >= 70 and skill_score <= 2:
        leak_chance += 0.15

    if leak_chance > 0 and random.random() < min(leak_chance, 1.0):
        return random.choice(leaky)
    return None


def update_all_relationship_stages(s: dict):
    """Recalculate relationship stage for every character."""
    for name, rel in s.get("relationships", {}).items():
        rel["stage"] = calculate_relationship_stage(rel)


# ─── STEP 5: 3-Tier Memory Architecture ─────────────────────
def update_memory_tiers(s: dict, user_input: str, final_script: list):
    """3-Tier 메모리 갱신. 매 턴 호출."""
    player_name = get_player_name(s)
    turn_num = len(s.get("turns", []))
    mem = s.setdefault("memory", {})

    # ── Tier 2: Short-Term (최근 턴 요약, 1턴=1문장) ──
    short_term = mem.setdefault("short_term", [])

    summary_parts = [f"T{turn_num}"]
    if user_input and user_input not in ("[CONTINUE_SCENE]", "[PLAYER_PAUSE]"):
        summary_parts.append(f"[{player_name}]\"{user_input[:40]}\"")

    for b in final_script:
        if isinstance(b, dict) and b.get("type") == "dialogue":
            c = b.get("character", "?")
            e = b.get("emotion", "")
            t = b.get("content", "")[:35]
            summary_parts.append(f"{c}({e}):\"{t}\"")
            if len(summary_parts) >= 4:
                break
        elif isinstance(b, dict) and b.get("type") == "narration":
            n = b.get("content", "")[:30]
            summary_parts.append(f"(나레이션:{n})")
            break

    entry = " → ".join(summary_parts)
    short_term.append(entry[:200])
    mem["short_term"] = short_term[-8:]

    # ── Tier 3: Long-Term (10턴마다 압축) ──
    long_term = mem.setdefault("long_term", [])
    if turn_num > 0 and turn_num % 10 == 0 and len(short_term) >= 5:
        block_entries = short_term[-10:] if len(short_term) >= 10 else short_term[:]
        compressed = " | ".join([e[:50] for e in block_entries])
        block_label = f"[T{max(1, turn_num-9)}~T{turn_num}]"
        long_term.append(f"{block_label} {compressed[:350]}")
        mem["long_term"] = long_term[-10:]


def update_character_last(s: dict, turn_number: int, script: list):
    """매 턴 종료 후 각 NPC의 마지막 발언을 인덱싱."""
    mem = s.setdefault("memory", {})
    char_last = mem.setdefault("character_last", {})
    for block in script:
        if isinstance(block, dict) and block.get("type") == "dialogue":
            name = block.get("character", "")
            if not name:
                continue
            char_last[name] = {
                "turn": turn_number,
                "said": (block.get("content") or "")[:80],
                "emotion": block.get("emotion", ""),
                "intensity": block.get("emotion_intensity", 5),
            }
    mem["character_last"] = char_last


def update_character_presence(s: dict, turn_id: int):
    """각 캐릭터가 어느 턴에 on_screen이었는지 기록."""
    mem = s.setdefault("memory", {})
    presence = mem.setdefault("character_presence", {})
    player_name = s.get("player_name", "사용자")
    for name in s.get("on_screen", []):
        if name == player_name:
            continue
        char_turns = presence.setdefault(name, [])
        char_turns.append(turn_id)
        presence[name] = char_turns[-30:]


# ─── STEP 5-B: DIMA 프롬프트용 3-Tier 메모리 변환 ───────────
def build_memory_context_for_dima(s: dict) -> str:
    """3-Tier 메모리를 DIMA 프롬프트용 텍스트로 변환"""
    mem = s.get("memory", {})
    lines = []

    # Tier 3: 장기 기억
    long_term = mem.get("long_term", [])
    if long_term:
        lines.append("## [장기 기억 — 과거 스토리 흐름]")
        for lt in long_term[-3:]:
            lines.append(f"  {lt}")

    # Tier 2: 단기 기억
    short_term = mem.get("short_term", [])
    if short_term:
        lines.append("## [단기 기억 — 최근 대화 요약]")
        display = short_term[:-2] if len(short_term) > 2 else short_term
        for st in display[-5:]:
            lines.append(f"  {st}")

    # 캐릭터별 최근 발언 인덱스
    char_last = mem.get("character_last", {})
    if char_last:
        lines.append("## [캐릭터별 최근 상태]")
        for name, info in char_last.items():
            lines.append(
                f"  {name} (T{info.get('turn', '?')}): "
                f"\"{info.get('said', '')}\" [{info.get('emotion', '')}]"
            )

    if not lines:
        return "## [기억] 이번 씬의 첫 대화입니다."

    return "\n".join(lines)


# ─── STEP 7: flow_digest_10 구조 개선 ────────────────────────
def format_flow_digest_for_dima(s: dict) -> str:
    """flow_digest_10을 DIMA가 읽기 쉬운 형태로 변환.
    유저 발화는 ★ 표시로 강조하여 DIMA가 반드시 참조하게 함."""
    flow_digest = s.get("flow_digest_10", [])
    if isinstance(flow_digest, deque):
        flow_digest = list(flow_digest)

    if not flow_digest:
        return "이번 씬의 첫 대화입니다."

    entries = []
    for item in flow_digest[-8:]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            speaker, content = item[0], str(item[1])[:100]
            if speaker.startswith("[") and speaker.endswith("]"):
                entries.append(f"  ★ {speaker}: \"{content}\"")
            else:
                entries.append(f"  {speaker}: \"{content}\"")
        elif isinstance(item, dict):
            entries.append(f"  T{item.get('turn', '?')}: {item.get('summary', '')[:80]}")
        elif isinstance(item, str):
            entries.append(f"  {item[:120]}")

    return "## [Tier 1: 최근 대화 원문]\n" + "\n".join(entries)


# ─── STEP 9: event_seeds 조건부 발동 ─────────────────────────
def should_inject_event_seed(s: dict) -> bool:
    """이벤트 시드를 DIMA에 주입할지 판단"""
    turn_num = len(s.get("turns", []))

    if turn_num < 15:
        return False

    if turn_num % 5 != 0:
        return False

    rels = s.get("relationships", {})
    has_trust = any(
        isinstance(r, dict) and r.get("stage", 1) >= 3
        for r in rels.values()
    )
    if not has_trust:
        return False

    return True


def get_event_seed_for_scene(s: dict) -> str:
    """현재 장소와 관련된 이벤트 시드를 하나만 선택"""
    if not should_inject_event_seed(s):
        return ""

    location = s.get("world", {}).get("space", {}).get(
        "current_location", ""
    )
    seeds = WORLD_DB.get("event_seeds", [])
    if not seeds:
        return ""

    relevant = []
    for seed in seeds:
        if isinstance(seed, dict):
            trigger_loc = seed.get("location", seed.get("trigger_location", ""))
            if not trigger_loc or trigger_loc in location or location in trigger_loc:
                relevant.append(seed)

    if not relevant:
        relevant = [sd for sd in seeds if not sd.get("location") and not sd.get("trigger_location")]

    if not relevant:
        return ""

    chosen = random.choice(relevant[:3])
    seed_name = chosen.get("name", chosen.get("title", "이벤트"))
    seed_beats = chosen.get("beats", chosen.get("description", ""))

    return (
        f"## [EVENT SEED — 선택적 참조]\n"
        f"이번 턴에서 자연스럽게 녹일 수 있으면 아래 이벤트를 살짝 암시하라.\n"
        f"강제하지 말 것. 현재 대화 흐름이 우선.\n"
        f"  이벤트: {seed_name}\n"
        f"  힌트: {str(seed_beats)[:150]}"
    )


# ─── STEP 11: Maestro 호출 빈도 최적화 ──────────────────────
def should_call_maestro(s: dict, user_input: str) -> bool:
    """Maestro 호출 여부를 판단하는 적응형 스케줄"""
    turn_num = len(s.get("turns", []))

    if turn_num == 0:
        return False

    if turn_num <= 10:
        return True

    if turn_num <= 30:
        return turn_num % 2 == 0

    if turn_num % 3 == 0:
        return True

    important_keywords = [
        "고백", "좋아", "싫어", "비밀", "진심", "약속", "미안",
        "고마워", "위험", "도망", "싸움", "울", "화나", "무서"
    ]
    if any(kw in (user_input or "") for kw in important_keywords):
        return True

    return False


# ─── Director defaults ───────────────────────────────────────
DEFAULT_DIRECTOR_SETTINGS = {
    "pov_first_person": True,
    "show_monologue": True,
    "acquainted": False,
    "adult_mode": False,
    "genre_preset": "auto",
    "narration_ratio": 40,
    "tempo": 5,
    "description_focus": 5,
    "illustration": False,
}

# ─── DIMA Schema (script array) ──────────────────────────────
DIMA_SCHEMA = {
    "type": "object",
    "properties": {
        "script": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["narration", "dialogue", "monologue"]},
                    "content": {"type": "string"},
                    "character": {"type": "string"},
                    "emotion": {"type": "string"},
                    "emotion_intensity": {"type": "integer"},
                    "monologue": {"type": "string"},
                },
                "required": ["type", "content"],
            },
        }
    },
    "required": ["script"],
}

# ─── Session helpers ─────────────────────────────────────────
_session_locks: Dict[str, threading.RLock] = {}
_session_locks_lock = threading.Lock()
_session_locks_last_used: Dict[str, float] = {}
SESSION_LOCK_TTL = 1800  # 30 minutes


def get_session_lock(sid: str) -> threading.RLock:
    with _session_locks_lock:
        _session_locks_last_used[sid] = time.time()
        if sid not in _session_locks:
            _session_locks[sid] = threading.RLock()
        return _session_locks[sid]


def _cleanup_stale_locks():
    """Remove session locks unused for more than SESSION_LOCK_TTL seconds."""
    now = time.time()
    with _session_locks_lock:
        stale = [sid for sid, ts in _session_locks_last_used.items()
                 if now - ts > SESSION_LOCK_TTL]
        for sid in stale:
            _session_locks.pop(sid, None)
            _session_locks_last_used.pop(sid, None)
    if stale:
        logger.info(f"Cleaned up {len(stale)} stale session locks")


def _lock_cleanup_daemon():
    while True:
        time.sleep(300)
        _cleanup_stale_locks()


if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    threading.Thread(target=_lock_cleanup_daemon, daemon=True).start()


def _sanitize_sid(sid: str) -> str:
    """Sanitize session ID to prevent path traversal."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "", sid)


def session_path(sid: str) -> Path:
    safe_sid = _sanitize_sid(sid)
    p = (SESSIONS_DIR / f"{safe_sid}.json").resolve()
    if not p.is_relative_to(SESSIONS_DIR.resolve()):
        raise ValueError("Invalid session ID")
    return p


def now_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"


def _to_jsonable(obj):
    """Convert deque and other non-JSON types."""
    if isinstance(obj, deque):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def load_session(sid: str) -> Optional[dict]:
    p = session_path(sid)
    if p.exists():
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            return _migrate_session(s)
        except Exception:
            return None
    return None


def save_session(s: dict):
    sid = s.get("session_id")
    if not sid:
        return
    p = session_path(sid)
    lock = get_session_lock(sid)
    with lock:
        try:
            data = json.dumps(_to_jsonable(s), ensure_ascii=False, indent=2)
            fd, tmp_path = tempfile.mkstemp(dir=str(SESSIONS_DIR), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(data)
                os.replace(tmp_path, str(p))
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        except Exception as e:
            logger.error(f"Failed to save session {sid}: {e}")


def trim_turns_after_maestro(s: dict):
    """Maestro 요약 완료 후, 최근 SLIDING_WINDOW_SIZE 턴만 남기고 나머지 삭제."""
    turns = s.get("turns", [])
    if len(turns) <= SLIDING_WINDOW_SIZE:
        return
    removed = turns[:-SLIDING_WINDOW_SIZE]
    digest = s.setdefault("flow_digest_10", [])
    for t in removed:
        summary = t.get("summary", "")
        if not summary:
            user_input = t.get("user_input", "")
            summary = user_input[:DIGEST_CONTENT_MAX_LENGTH] if user_input else "(turn)"
        digest.append({"turn": t.get("turn_number", "?"), "summary": summary})
    s["flow_digest_10"] = digest[-20:]
    s["turns"] = turns[-SLIDING_WINDOW_SIZE:]
    logger.info(f"Trimmed {len(removed)} old turns. Remaining: {len(s['turns'])}")


def to_public_state(s: dict) -> dict:
    """Convert session to a JSON-safe public state dict."""
    return _to_jsonable(copy.deepcopy(s))


# ─── STEP 6: Lightweight world state for sessions ────────────
def create_lightweight_world_state(world_db: dict) -> dict:
    """세션에 저장할 최소한의 월드 상태만 생성"""
    main_stage = world_db.get("main_stage", {})
    default_location = "라운지"
    # main_stage 안의 locations 배열에서 첫 번째 항목의 name을 찾아 쓰되, 없으면 "라운지"
    ms_locations = main_stage.get("locations", [])
    if isinstance(ms_locations, list):
        for loc in ms_locations:
            if isinstance(loc, dict) and loc.get("name"):
                default_location = loc["name"]
                break

    return {
        "world_name": world_db.get("world_name", "사루라 아뜨리에"),
        "core_concept": world_db.get("core_concept", ""),
        "system_overrides": world_db.get("system_overrides", {}),
        "world_rules": world_db.get("world_rules", {}),
        "main_stage": {
            "name": main_stage.get("name", ""),
            "staff": main_stage.get("staff", {}),
        },
        "space": {
            "current_location": default_location,
            "previous_location": None,
        },
        "time": copy.deepcopy(world_db.get("time", {"display": "오후"})),
    }


# ─── Session init ────────────────────────────────────────────
def init_session(session_id: Optional[str] = None) -> dict:
    sid = session_id or f"S-{uuid.uuid4().hex[:8].upper()}"
    s = {
        "session_id": sid,
        "player_name": "사용자",
        "on_screen": [],
        "turns": [],
        "flow_digest_10": [],
        "traffic_light": "GREEN",
        "ui_settings": copy.deepcopy(DEFAULT_DIRECTOR_SETTINGS),
        "characters": {name: {} for name in ALL_CHARACTER_NAMES},
        "world": create_lightweight_world_state(WORLD_DB),
        "scene_context": {
            "turn_count_in_scene": 0,
            "time_elapsed_minutes": 0,
        },
        "action_context": {},
        # Part B: CORE-4 State System
        "core4": {
            "energy":       {"value": 70, "decay_per_turn": -2, "min": 0, "max": 100},
            "intoxication": {"value": 0,  "decay_per_turn": -3, "min": 0, "max": 100},
            "stress":       {"value": 30, "decay_per_turn": -1, "min": 0, "max": 100},
            "pain":         {"value": 0,  "decay_per_turn": -2, "min": 0, "max": 100},
        },
        # Part H: 3-Layer Memory System
        "memory": {
            "short_term": [],
            "long_term": [],
            "core_pins": [],
            "emotional_contagion_log": [],
        },
        # Part C: Relationships (initialized per on_screen)
        "relationships": {},
        # V5.0: User profile for intent classification
        "user_profile": {
            "play_style": "chat",
            "style_confidence": 0.0,
            "turn_style_history": [],
            "heavy_input_count": 0,
            "avg_input_length": 0,
            "total_input_chars": 0,
            "preference_tags": [],
        },
        # V5.0: Token usage tracking
        "token_ledger": {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cached_tokens": 0,
            "total_thinking_tokens": 0,
            "turns": [],
        },
        # V5.0: AI Analyzer cache (for heavy input)
        "analyzer_cache": None,
        # V5.0: Maestro override directives
        "maestro_override": {},
    }
    return s


def _migrate_session(s: dict) -> dict:
    """Ensure new V5.0 fields exist in loaded sessions."""
    if s.get("_schema_version", 0) >= 5:
        return s  # 이미 V5 스키마, 스킵

    if "user_profile" not in s:
        s["user_profile"] = {
            "play_style": "chat", "style_confidence": 0.0,
            "turn_style_history": [], "heavy_input_count": 0,
            "avg_input_length": 0, "total_input_chars": 0,
            "preference_tags": [],
        }
    if "token_ledger" not in s:
        s["token_ledger"] = {
            "total_input_tokens": 0, "total_output_tokens": 0,
            "total_cached_tokens": 0, "total_thinking_tokens": 0,
            "turns": [],
        }
    if "analyzer_cache" not in s:
        s["analyzer_cache"] = None
    if "maestro_override" not in s:
        s["maestro_override"] = {}

    s["_schema_version"] = 5
    return s


def init_relationships(s: dict):
    """Initialize relationships for on-screen characters."""
    player_name = s.get("player_name", "사용자")
    for char_name in s.get("on_screen", []):
        if char_name == player_name:
            continue
        if char_name in s.get("relationships", {}):
            continue
        char_db = CHARACTERS_DB.get(char_name, {})
        player_rel = char_db.get("relationship_matrix", {}).get("플레이어", {})
        s.setdefault("relationships", {})[char_name] = {
            "stage": 1,
            "axes": {
                "agreeable": 0, "adversarial": 0,
                "open": 0, "closed": 0,
                "bold": 0, "passive": 0,
                "reliable": 0, "unreliable": 0,
                "insightful": 0, "oblivious": 0,
            },
            "affection": player_rel.get("호감", 50),
            "trust": player_rel.get("신뢰", 50),
            "tension": player_rel.get("긴장", 50),
        }


def get_player_name(s: dict) -> str:
    return s.get("player_name", "사용자")


def normalize_cast(names: Optional[List[str]], player_name: str) -> List[str]:
    if not isinstance(names, list):
        return []
    valid = [n.strip() for n in names if isinstance(n, str) and n.strip()]
    valid = [n for n in valid if n in CHARACTERS_DB and n != player_name]
    return sorted(list(dict.fromkeys(valid)))


def resolve_default_location(world_db: dict) -> str:
    try:
        return world_db.get("main_stage", {}).get("locations", [{}])[0].get("name", "라운지")
    except (IndexError, AttributeError):
        return "라운지"


def merge_ui_settings(s: dict, incoming: Optional[dict]):
    if not incoming or not isinstance(incoming, dict):
        return
    ui = s.setdefault("ui_settings", copy.deepcopy(DEFAULT_DIRECTOR_SETTINGS))
    for k, v in incoming.items():
        if k in DEFAULT_DIRECTOR_SETTINGS:
            ui[k] = v


def extract_emotion_from_script(script: list) -> str:
    for b in script:
        if isinstance(b, dict) and b.get("type") == "dialogue" and b.get("emotion"):
            return b["emotion"]
    return "joy"


def generate_illustration(
    scene_description: str,
    character_names: list,
    emotion: str,
    session_id: str,
    turn_number: int,
    reference_slug: str = None,
) -> Optional[str]:
    """NanoBanana로 삽화 생성. 실패 시 None 반환."""
    try:
        char_appearances = []
        for name in character_names[:3]:
            cached = _CHAR_RUNTIME_CACHE.get(name, {})
            app_text = cached.get("appearance_summary", "")
            height = cached.get("height", "")
            if app_text:
                char_appearances.append(f"{name}: {app_text} ({height})")

        prompt = (
            f"Fantasy visual novel illustration, anime style, high quality.\n"
            f"Scene: {scene_description[:300]}\n"
            f"Characters: {'; '.join(char_appearances)}\n"
            f"Mood/Emotion: {emotion}\n"
            f"Style: soft lighting, watercolor-like background, "
            f"detailed character expressions, 16:9 aspect ratio.\n"
            f"Do NOT include any text, speech bubbles, or UI elements."
        )

        contents = [prompt]
        if reference_slug:
            safe_slug = re.sub(r"[^a-zA-Z0-9_\-]", "", reference_slug)
            ref_path = BASE_DIR / "static" / "gifs" / safe_slug / "default.webp"
            if ref_path.exists():
                ref_bytes = ref_path.read_bytes()
                contents = [
                    genai_types.Part.from_bytes(data=ref_bytes, mime_type="image/webp"),
                    prompt,
                ]

        config = genai_types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            safety_settings=get_safety_settings(),
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        result = call_gemini_with_timeout(
            MODEL_ILLUSTRATION, contents, config, timeout_sec=60
        )

        if not result or not result.candidates:
            return None

        for part in result.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                img_data = part.inline_data.data
                mime = part.inline_data.mime_type or "image/png"
                ext = "png" if "png" in mime else "webp" if "webp" in mime else "jpg"

                session_illust_dir = ILLUSTRATIONS_DIR / _sanitize_sid(session_id)
                session_illust_dir.mkdir(exist_ok=True)

                filename = f"turn_{turn_number:04d}.{ext}"
                filepath = session_illust_dir / filename
                filepath.write_bytes(img_data)

                return f"/static/illustrations/{_sanitize_sid(session_id)}/{filename}"

        return None
    except Exception as e:
        logger.error(f"Illustration generation failed: {e}")
        return None


# =========================================================================
# PART G: 3-ACT NARRATIVE TENSION CURVE (Freytag's Pyramid)
# =========================================================================
def calculate_tension_level(session_data: dict) -> dict:
    turn_count = len(session_data.get("turns", []))
    core4 = session_data.get("core4", {})
    stress = core4.get("stress", {}).get("value", 30)

    # Base tension from turn count (3-act structure)
    if turn_count <= 5:
        act = "서막"
        base_tension = 0.3
        temperature_hint = 0.7
    elif turn_count <= 15:
        act = "전개"
        base_tension = 0.5 + (turn_count - 5) * 0.03
        temperature_hint = 0.85
    elif turn_count <= 25:
        act = "클라이맥스"
        base_tension = 0.8 + min(0.15, (turn_count - 15) * 0.015)
        temperature_hint = 0.95
    else:
        act = "해소"
        base_tension = max(0.4, 0.95 - (turn_count - 25) * 0.03)
        temperature_hint = 0.8

    # Stress amplifier
    stress_modifier = stress / 200  # 0 to 0.5
    final_tension = min(1.0, base_tension + stress_modifier)

    return {
        "act": act,
        "tension": round(final_tension, 2),
        "temperature_hint": round(temperature_hint, 2),
        "turn_count": turn_count,
    }


# =========================================================================
# PART B: CORE-4 description helpers
# =========================================================================
def get_core4_description(key: str, value: int) -> str:
    """Return Korean description string for a CORE-4 stat."""
    if key == "energy":
        if value <= 20:
            return "극도로 피곤. 하품, 비몽사몽, 짧은 대사"
        elif value <= 40:
            return "나른함. 여유롭지만 활기 부족"
        elif value <= 60:
            return "보통. 평소 성격대로"
        elif value <= 80:
            return "활기참. 적극적 대화, 장난"
        else:
            return "과잉 에너지. 들뜬 상태, 행동 과격"
    elif key == "intoxication":
        if value <= 10:
            return "맨정신"
        elif value <= 30:
            return "살짝 취함. 솔직해짐"
        elif value <= 60:
            return "중간 취기. 웃음 많고 비밀 흘림"
        elif value <= 80:
            return "만취. 감정 기복 심화"
        else:
            return "인사불성 직전. 헛소리, 쓰러짐"
    elif key == "stress":
        if value <= 20:
            return "평화로움. 여유있고 관대"
        elif value <= 40:
            return "약간의 긴장. 관리 가능"
        elif value <= 60:
            return "중간 스트레스. 짜증 증가"
        elif value <= 80:
            return "고스트레스. 폭발 직전"
        else:
            return "극도 스트레스. 감정 폭주"
    elif key == "pain":
        if value <= 10:
            return "건강함"
        elif value <= 30:
            return "경미한 불편"
        elif value <= 60:
            return "중간 고통. 행동 느려짐"
        elif value <= 80:
            return "심한 고통. 움직임 제한"
        else:
            return "위독. 긴급 상황"
    return "알 수 없음"


def apply_core4_decay(s: dict):
    """Apply natural decay to CORE-4 stats each turn."""
    for stat_key, stat_data in s.get("core4", {}).items():
        old = stat_data["value"]
        new = max(stat_data["min"], min(stat_data["max"], old + stat_data["decay_per_turn"]))
        stat_data["value"] = new


# =========================================================================
# V5.0: Stage Manager — Intent Classification (No API call)
# =========================================================================
_PAT_RP = re.compile(
    r'(\*[^*]{1,500}\*|「[^」]{1,500}」|"[^"]{1,500}"|'
    r'\(행동\)|\(표정\)|~다\.|~했다\.|~이다\.)',
    re.MULTILINE
)
_PAT_NOVEL = re.compile(
    r'(장르|세계관|설정|배경|시놉시스|프롤로그|1장|제\d+장|'
    r'주인공|빌런|플롯|클라이맥스|엔딩|시나리오)',
    re.IGNORECASE
)
_PAT_GAME = re.compile(
    r'(스탯|HP|MP|레벨|인벤토리|퀘스트|전투|판정|주사위|'
    r'TRPG|D&D|다이스|능력치)',
    re.IGNORECASE
)
_PAT_COUNSEL = re.compile(
    r'(고민|상담|힘들|슬퍼|외로|우울|불안|위로|도움|걱정)',
    re.IGNORECASE
)
_PAT_HEAVY_SETUP = re.compile(
    r'(세계관|설정|배경|시대|종족|마법체계|국가|역사|'
    r'캐릭터\s*설정|스토리\s*설정|세계\s*설정)',
    re.IGNORECASE
)


def classify_user_intent(user_input: str, profile: dict) -> dict:
    """Python-only intent classification. No API call."""
    text = user_input.strip()
    length = len(text)

    scores = {
        "chat": 1.0,
        "rp": len(_PAT_RP.findall(text)) * 2.0,
        "novel": len(_PAT_NOVEL.findall(text)) * 2.5,
        "game": len(_PAT_GAME.findall(text)) * 3.0,
        "counsel": len(_PAT_COUNSEL.findall(text)) * 2.0,
    }

    if length > 500:
        scores["novel"] += 2.0
        scores["rp"] += 1.0
    if length > 1000:
        scores["novel"] += 3.0

    # 이전 스타일 관성
    prev_style = profile.get("play_style", "chat")
    prev_conf = profile.get("style_confidence", 0.0)
    scores[prev_style] += prev_conf * 2.0

    is_heavy = (length > 300 and _PAT_HEAVY_SETUP.search(text) is not None)

    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    confidence = scores[best] / total

    return {
        "play_style": best,
        "style_confidence": round(confidence, 3),
        "signals": {k: round(v, 2) for k, v in scores.items()},
        "is_heavy_input": is_heavy,
        "input_length": length,
    }


def update_user_profile(profile: dict, intent_result: dict, input_text: str):
    """Update user_profile with this turn's classification."""
    profile["play_style"] = intent_result["play_style"]
    profile["style_confidence"] = intent_result["style_confidence"]

    history = profile.setdefault("turn_style_history", [])
    history.append(intent_result["play_style"])
    profile["turn_style_history"] = history[-10:]

    if intent_result["is_heavy_input"]:
        profile["heavy_input_count"] = profile.get("heavy_input_count", 0) + 1

    total_chars = profile.get("total_input_chars", 0) + len(input_text)
    total_turns = len(profile["turn_style_history"])
    profile["total_input_chars"] = total_chars
    profile["avg_input_length"] = total_chars // max(1, total_turns)


# =========================================================================
# V5.0: Token Metering
# =========================================================================
def record_token_usage(s: dict, response, turn_number: int) -> dict:
    """Extract token usage from Gemini response and record to session."""
    ledger = s.setdefault("token_ledger", {
        "total_input_tokens": 0, "total_output_tokens": 0,
        "total_cached_tokens": 0, "total_thinking_tokens": 0, "turns": [],
    })

    usage = {"turn": turn_number, "input": 0, "output": 0, "cached": 0, "thinking": 0}

    if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
        um = response.usage_metadata
        usage["input"] = getattr(um, 'prompt_token_count', 0) or 0
        usage["output"] = getattr(um, 'candidates_token_count', 0) or 0
        usage["cached"] = getattr(um, 'cached_content_token_count', 0) or 0
        usage["thinking"] = getattr(um, 'thoughts_token_count', 0) or 0

    ledger["total_input_tokens"] += usage["input"]
    ledger["total_output_tokens"] += usage["output"]
    ledger["total_cached_tokens"] += usage["cached"]
    ledger["total_thinking_tokens"] += usage["thinking"]

    turns_log = ledger.setdefault("turns", [])

    # summary stats 보존 (50턴 cap 적용 전)
    if len(turns_log) > 50:
        trimmed = turns_log[:-50]
        summary = ledger.get("trimmed_summary", {"total_input": 0, "total_output": 0, "turns_trimmed": 0})
        for t in trimmed:
            summary["total_input"] += t.get("input", 0)
            summary["total_output"] += t.get("output", 0)
            summary["turns_trimmed"] += 1
        ledger["trimmed_summary"] = summary
        turns_log = turns_log[-50:]

    turns_log.append(usage)
    ledger["turns"] = turns_log

    logger.info(
        f"Token usage turn {turn_number}: "
        f"in={usage['input']} out={usage['output']} "
        f"cached={usage['cached']} thinking={usage['thinking']}"
    )
    return usage


# =========================================================================
# V5.0: AI Analyzer for Heavy Input (called once, cached)
# =========================================================================
ANALYZER_SCHEMA = {
    "type": "object",
    "properties": {
        "detected_genre": {"type": "string"},
        "world_summary": {"type": "string"},
        "tone_keywords": {"type": "array", "items": {"type": "string"}},
        "character_directives": {"type": "array", "items": {"type": "string"}},
        "key_elements": {"type": "array", "items": {"type": "string"}},
        "recommended_play_style": {
            "type": "string",
            "enum": ["chat", "rp", "novel", "game", "counsel"]
        },
        "opening_hook": {"type": "string"},
    },
    "required": ["detected_genre", "world_summary", "tone_keywords", "recommended_play_style"],
}


def run_ai_analyzer(user_input: str, on_screen: list) -> Optional[dict]:
    """Analyze heavy input once and return structured result. 실패 시 에러 마커를 캐시에 저장하여 매 턴 재시도 방지."""
    # Truncate to 3000 chars to stay within Gemini context limits while keeping cost low
    prompt = (
        "당신은 인터랙티브 픽션 분석가입니다.\n"
        "아래 사용자 입력에서 장르, 세계관 요약(200자 이내), 톤 키워드, "
        "캐릭터 연기 지시, 핵심 서사 요소, 추천 플레이 스타일을 추출하세요.\n"
        "JSON 스키마에 맞게 응답하세요.\n\n"
        f"현재 등장 캐릭터: {', '.join(on_screen)}\n\n"
        f"사용자 입력:\n{user_input[:3000]}"
    )

    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ANALYZER_SCHEMA,
        safety_settings=get_safety_settings(),
        temperature=0.3,
        max_output_tokens=2048,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=512),
    )

    try:
        result = call_gemini_with_fallback([prompt], config, timeout_sec=20, max_retries=2)
        if result and result.text:
            parsed = json.loads(result.text)
            logger.info(f"AI Analyzer result: genre={parsed.get('detected_genre')}, "
                       f"style={parsed.get('recommended_play_style')}")
            return parsed
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"AI Analyzer failed: {e}")
        # 실패 마커 저장 → 매 턴 재시도 방지
        return {
            "_error": True,
            "_error_msg": str(e),
            "_timestamp": now_ts(),
            "_error_turn": 0,
            "_retry_after_turns": ANALYZER_RETRY_INTERVAL
        }
    return None


# =========================================================================
# Build system instruction for D.I.M.A
# =========================================================================
def build_scene_card(char_name: str, on_screen_chars: list, relationships: dict) -> str:
    """Build a compressed ~400-token scene card for a character."""
    char_db = CHARACTERS_DB.get(char_name, {})
    identity = char_db.get("identity", {})
    bp = char_db.get("behavior_protocols", {})
    sig = bp.get("signature_speech", {})
    heuristics = bp.get("acting_heuristics", {})
    rel_matrix = char_db.get("relationship_matrix", {})

    core_appeal = identity.get("core_appeal", "")[:180]
    speech_habit = sig.get("speech_habit", "")
    honorific = sig.get("honorific_style", "")
    catches = sig.get("catchphrases", [])[:3]
    forbidden = sig.get("forbidden_patterns", [])[:2]
    voice_contrast = sig.get("voice_contrast", "")
    acting_hint = ""
    if heuristics:
        first_val = next(iter(heuristics.values()), "")
        acting_hint = first_val[:80] if first_val else ""
    core_rule = bp.get("core_acting_rule", "")[:80]

    # Appearance summary for visual description
    appearance = char_db.get("appearance", {})
    app_summary = appearance.get("summary", "")[:60]
    height = appearance.get("height", "")

    lines = [
        f"[{char_name}] {core_appeal}",
    ]
    if app_summary:
        lines.append(f"외모: {app_summary} ({height})")
    lines.extend([
        f"말투: {speech_habit} | 어미: {honorific}",
        f"입버릇: {', '.join(catches)}",
        f"금기: {', '.join(forbidden)}",
    ])
    if voice_contrast:
        lines.append(f"음성 대비: {voice_contrast}")
    lines.append(f"자율행동: {acting_hint}")
    lines.append(f"이번 씬 목표: {core_rule}")

    for other in on_screen_chars:
        if other == char_name:
            continue
        # Use dynamic relationship values from session, fall back to static DB
        dyn_rel = relationships.get(other, {})
        static_rel = rel_matrix.get(other) if isinstance(rel_matrix.get(other), dict) else {}
        if dyn_rel or static_rel:
            comment = static_rel.get("comment", "")[:50]
            favor = dyn_rel.get("affection", static_rel.get("호감", 50))
            tension = dyn_rel.get("tension", static_rel.get("긴장", 50))
            lines.append(f"vs {other}: {comment} (호감{favor}/긴장{tension})")
        # Psychological mirror for monologue depth
        psych_mirror = bp.get("psychological_mirror", {})
        mirror_text = psych_mirror.get(f"vs_{other}", "")
        if mirror_text:
            lines.append(f"  심리투영: {mirror_text}")

    return "\n".join(lines)


def build_character_block_for_prompt(
    name: str,
    on_screen_chars: list,
    player_name: str,
    session_rels: dict,
    turn_count: int = 0,
    core4: dict = None,
) -> str:
    """
    characters_db.json에서 현재 씬에 필요한 데이터만 선별 추출.
    토큰 효율을 위해 현재 씬과 무관한 데이터는 제외.
    """
    cdb = CHARACTERS_DB.get(name, {})
    if not cdb:
        return ""
    identity = cdb.get("identity", {})
    bp = cdb.get("behavior_protocols", {})
    sig = bp.get("signature_speech", {})
    pdna = cdb.get("personality_dna", {})

    lines = [f"\n=== [{name}] ==="]

    # 핵심 매력 + 연기 규칙 (짧게)
    ca = identity.get("core_appeal", "")
    if ca:
        lines.append(f"매력: {ca[:180]}")
    cr = bp.get("core_acting_rule", "")
    if cr:
        lines.append(f"연기규칙: {cr[:180]}")

    # 말투 (필수 — 캐릭터 음성 차별화의 핵심)
    sp = []
    if sig.get("speech_habit"):
        sp.append(sig["speech_habit"])
    if sig.get("honorific_style"):
        sp.append(f"존칭:{sig['honorific_style']}")
    if sig.get("catchphrases"):
        sp.append(f"입버릇:{','.join(sig['catchphrases'][:3])}")
    if sig.get("voice_contrast"):
        sp.append(sig["voice_contrast"][:80])
    if sig.get("forbidden_patterns"):
        sp.append(f"금지:{','.join(sig['forbidden_patterns'][:2])}")
    if sp:
        lines.append(f"말투: {' | '.join(sp)}")
    if sig.get("example_lines"):
        lines.append(f"예시: {' / '.join(sig['example_lines'][:2])}")

    # 성격 경향
    hints = big5_to_behavior_hints(pdna)
    if hints:
        lines.append(f"성격: {hints}")

    # ★ specific_interactions — 현재 씬 관련만 ★
    si = bp.get("specific_interactions", {})
    si_lines = []
    vs_p = si.get("vs_플레이어", "")
    if vs_p:
        si_lines.append(f"  vs {player_name}: {vs_p[:120]}")
    for other in on_screen_chars:
        if other == name:
            continue
        key = f"vs_{other}"
        if key in si:
            si_lines.append(f"  vs {other}: {si[key][:120]}")
    if si_lines:
        lines.append("상호작용지침:\n" + "\n".join(si_lines))

    # ★ acting_heuristics ★
    ah = bp.get("acting_heuristics", {})
    if ah:
        h_lines = [f"  {k}: {v[:100]}" for k, v in list(ah.items())[:3]]
        lines.append("상태별행동:\n" + "\n".join(h_lines))

    # ★ 관계 매트릭스 — 현재 씬만 ★
    rm = cdb.get("relationship_matrix", {})
    rl = []
    pr = rm.get("플레이어", {})
    if pr:
        sr = session_rels.get(name, {})
        aff = sr.get("affection", pr.get("호감", 50))
        tru = sr.get("trust", pr.get("신뢰", 50))
        ten = sr.get("tension", pr.get("긴장", 50))
        rl.append(f"  vs {player_name}: 호감{aff} 신뢰{tru} 긴장{ten} — {pr.get('comment', '')[:60]}")
    for other in on_screen_chars:
        if other == name:
            continue
        or_ = rm.get(other, {})
        if or_:
            rl.append(
                f"  vs {other}: 호감{or_.get('호감', 50)} 신뢰{or_.get('신뢰', 50)} "
                f"긴장{or_.get('긴장', 50)} — {or_.get('comment', '')[:60]}"
            )
    if rl:
        lines.append("관계:\n" + "\n".join(rl))

    # ★ 유저 반응 가이드 — 현재 관계 단계 반영 ★
    si = bp.get("specific_interactions", {})
    vs_player_text = si.get("vs_플레이어", "")
    rd = cdb.get("relationship_development", {})
    current_stage = session_rels.get(name, {}).get("stage", 1)
    stage_desc = rd.get(f"stage_{current_stage}_description", "")

    guide_lines = []
    if vs_player_text:
        guide_lines.append(f"  플레이어 기본자세: {vs_player_text[:120]}")
    if stage_desc:
        guide_lines.append(f"  현재 관계({current_stage}단계) 행동: {stage_desc[:120]}")
    if guide_lines:
        lines.append("유저반응가이드:\n" + "\n".join(guide_lines))

    # ★ psychological_mirror — 씬 관련만 ★
    pm = bp.get("psychological_mirror", {})
    pm_lines = []
    for other in on_screen_chars:
        key = f"vs_{other}"
        if key in pm:
            pm_lines.append(f"  {key}: {pm[key][:80]}")
    if pm_lines:
        lines.append("심리거울:\n" + "\n".join(pm_lines))

    # 외모 (간결)
    app = cdb.get("appearance", {})
    if app.get("summary"):
        lines.append(f"외모: {app['summary'][:80]} ({app.get('height', '')})")

    # ★ 음성 자연화 규칙 (tone_mixing_rule + catchphrase_budget) ★
    tm_rule = sig.get("tone_mixing_rule", "")
    cp_budget = sig.get("catchphrase_budget", "")
    voice_rules = []
    if tm_rule:
        voice_rules.append(f"  존반말혼용: {tm_rule[:150]}")
    if cp_budget:
        voice_rules.append(f"  말버릇제한: {cp_budget[:120]}")
    if voice_rules:
        lines.append("음성규칙:\n" + "\n".join(voice_rules))

    # ★ 무의식적 습관/솔직함/물리반응 — turn≥5이고 관련 CORE-4 상태일 때만 포함 ★
    include_deep_profile = turn_count >= 5

    if include_deep_profile:
        # ★ 솔직함 프로필 (honesty_profile) ★
        hp = bp.get("honesty_profile", {})
        if hp:
            hp_lines = []
            if hp.get("deception_skill"):
                hp_lines.append(f"  거짓말 능력: {hp['deception_skill']}")
            if hp.get("tell_when_lying"):
                hp_lines.append(f"  거짓말 신호: {hp['tell_when_lying'][:100]}")
            if hp.get("defense_mechanism"):
                hp_lines.append(f"  방어기제: {hp['defense_mechanism'][:80]}")
            if hp.get("breaking_point"):
                hp_lines.append(f"  한계점: {hp['breaking_point'][:80]}")
            if hp.get("leaky_topics"):
                hp_lines.append(f"  실수로 흘리기 쉬운 것: {','.join(hp['leaky_topics'][:3])}")
            if hp.get("memory_reliability"):
                hp_lines.append(f"  기억신뢰도: {hp['memory_reliability']}")
            if hp.get("memory_distortion_pattern"):
                hp_lines.append(f"  기억왜곡: {hp['memory_distortion_pattern'][:80]}")
            if hp_lines:
                lines.append("솔직함프로필:\n" + "\n".join(hp_lines))

        # ★ 무의식적 습관 (idle_habits) ★
        ih = bp.get("idle_habits", [])
        if ih:
            habit_str = " | ".join(
                f"{h.get('trigger','')}: {h.get('action','')[:60]}"
                for h in ih[:3]
            )
            lines.append(f"무의식습관: {habit_str}")

        # ★ 물리적 상태 반응 (physical_tells) ★
        pt = bp.get("physical_tells", {})
        if pt:
            pt_parts = []
            if pt.get("fatigue_behavior"):
                pt_parts.append(f"피로: {pt['fatigue_behavior'][:60]}")
            if pt.get("embarrassment_trigger"):
                pt_parts.append(f"곤란: {pt['embarrassment_trigger'][:60]}")
            if pt.get("pain_tolerance"):
                pt_parts.append(f"고통내성: {pt['pain_tolerance']}")
            if pt_parts:
                lines.append(f"신체반응: {' | '.join(pt_parts)}")

    return "\n".join(lines)


# =========================================================================
# V5.0: Adaptive System Instruction Builder (replaces build_system_instruction_for_scene)
# =========================================================================
def build_adaptive_instruction(s: dict) -> str:
    """V5.0 adaptive system instruction builder.

    Assembles prompt based on play_style, turn count, and session state.
    Designed for Gemini implicit caching: keeps prefix stable across turns.
    """
    parts = []
    turn_count = len(s.get("turns", []))
    profile = s.get("user_profile", {})
    play_style = profile.get("play_style", "chat")
    on_screen = s.get("on_screen", [])
    player = get_player_name(s)
    ui = s.get("ui_settings", {})

    # ── BLOCK 1: Safety + Role (FIXED prefix for cache hit) ──
    parts.append(SAFETY_PREAMBLE)
    parts.append(EUGENE_FILTER_RULE)
    parts.append(
        "당신은 D.I.M.A(Director-level Interactive Multi-character Actor)입니다. "
        "사용자의 입력에 반응하여 등장 캐릭터들의 대사·행동·나레이션을 script JSON으로 출력합니다. "
        "모든 캐릭터는 성인(20세 이상)입니다."
    )

    # ── BLOCK 2: Player + UI settings ──
    pov = "1인칭(나)" if ui.get("pov_first_person") else "3인칭"
    parts.append(
        f"[설정] 플레이어: {player} | 시점: {pov} | "
        f"나레이션 비율: {ui.get('narration_ratio', 40)}% | "
        f"템포: {ui.get('tempo', 5)}/10 | "
        f"묘사밀도: {ui.get('description_focus', 5)}/10"
    )

    # ── BLOCK 3: Character deep profiles (from characters_db.json) ──
    sess_rels = s.get("relationships", {})
    for cname in on_screen:
        if cname == player:
            continue
        block = build_character_block_for_prompt(
            cname, on_screen, player, sess_rels,
            turn_count=turn_count,
            core4=s.get("core4"),
        )
        if block:
            parts.append(block)

    # ── BLOCK 3.5: Age hierarchy & honorific rules ──
    AGE_HIERARCHY_RULE = """
[나이 서열 및 호칭 규칙]
서열(어린→많음): 체니 < 레베카/령/네르 < 루크/마리/세리카 < 샐리/테피 < 라이니 < 크래더

호칭 원칙:
- 나이가 어린 쪽은 많은 쪽에게 기본적으로 존댓말을 사용한다.
- 나이가 많은 쪽은 어린 쪽에게 반말을 사용할 수 있다.
- 단, 캐릭터 고유 성격이 우선: 루크는 모두에게 존댓말, 체니는 어른에게도 반말, 라이니는 우아한 존댓말 등.
- 크래더는 모두에게 반말이지만 테피에게만 어색하게 격식을 차리려 한다.
- 같은 나이대끼리는 친밀도에 따라 자유롭게 변화.

특수 규칙:
- 루크는 기숙사 '관리인'으로서 학생들과 선후배가 아닌 동료 관계.
- 세리카는 실제로는 훨씬 나이가 많지만 기억상실로 인해 학생으로 행동. 가끔 무의식적으로 나이 든 말투가 튀어나올 수 있음.
- 샐리는 역성장으로 외모가 어리지만 실제 30대. 말투에서 '아줌마'가 튀어나옴.
"""
    parts.append(AGE_HIERARCHY_RULE)

    # ── BLOCK 4: Style-specific extensions ──
    if play_style in ("chat", "counsel"):
        parts.append(
            "[스타일: 대화/상담] 자연스러운 일상 대화 중심. 나레이션 최소화. "
            "캐릭터 감정과 반응에 집중. 따뜻하고 공감적인 톤."
        )
    elif play_style == "rp":
        parts.append(
            "[스타일: RP] 몰입감 있는 롤플레이. 나레이션과 대사 균형. "
            "환경·표정·동작 묘사. 캐릭터 간 상호작용."
        )
        for cname in on_screen:
            cached = _CHAR_RUNTIME_CACHE.get(cname, {})
            if cached.get("appearance_summary"):
                parts.append(
                    f"  [{cname} 외모] {cached['appearance_summary']} ({cached.get('height','')})\n"
                    f"  행동 성향: {cached.get('behavior_hints', '')}"
                )
    elif play_style == "novel":
        parts.append(
            "[스타일: 소설] 문학적 표현. 심리묘사·복선·긴장감. "
            "내면 독백 적극 활용. 장면 전환 시 5감 묘사. "
            "서술의 리듬과 반전을 고려."
        )
        for cname in on_screen:
            cached = _CHAR_RUNTIME_CACHE.get(cname, {})
            mirror = cached.get("psychological_mirror", {})
            if mirror:
                parts.append(f"  [{cname} 심리거울] " +
                    "; ".join(f"{k}: {v}" for k, v in mirror.items()))
            if cached.get("appearance_summary"):
                parts.append(f"  [{cname} 외모] {cached['appearance_summary']}")
    elif play_style == "game":
        parts.append(
            "[스타일: 게임/TRPG] 선택지 제시. 판정·결과 묘사. "
            "CORE-4 변화를 게임적 피드백으로 전달. "
            "환경 상호작용과 아이템 활용."
        )

    # ── BLOCK 5: Analyzer cache (Heavy Input result) ──
    analyzer = s.get("analyzer_cache")
    if analyzer and not analyzer.get("_error"):
        parts.append(
            f"[유저 세계관] 장르: {analyzer.get('detected_genre', '미정')}\n"
            f"  요약: {analyzer.get('world_summary', '')}\n"
            f"  톤: {', '.join(analyzer.get('tone_keywords', []))}\n"
            f"  핵심 요소: {', '.join(analyzer.get('key_elements', []))}"
        )
        directives = analyzer.get("character_directives", [])
        if directives:
            parts.append(f"  캐릭터 지시: {'; '.join(directives[:5])}")

    # ── BLOCK 6: Progressive Disclosure (turn-gated) ──
    if turn_count >= 3:
        rels = s.get("relationships", {})
        rel_lines = []
        for cname in on_screen:
            if cname == player:
                continue
            rel = rels.get(cname, {})
            stage = STAGE_NAMES.get(rel.get("stage", 1), "경계")
            vel = get_relationship_velocity(cname)
            speed_hint = ""
            if vel.get("affection_mult", 1.0) >= 1.5:
                speed_hint = " [감정변동 빠름]"
            elif vel.get("affection_mult", 1.0) <= 0.7:
                speed_hint = " [감정변동 느림]"
            rel_lines.append(
                f"{cname}: {stage}(호감{rel.get('affection',50)}/신뢰{rel.get('trust',50)}/긴장{rel.get('tension',50)}){speed_hint}"
            )
        if rel_lines:
            parts.append(f"[관계] {' | '.join(rel_lines)}")

    if turn_count >= 5:
        core4 = s.get("core4", {})
        c4_parts = []
        for key in ["energy", "stress", "intoxication", "pain"]:
            val = core4.get(key, {}).get("value", 0)
            c4_parts.append(f"{key}={val}({get_core4_description(key, val)})")
        parts.append(f"[CORE-4] {' | '.join(c4_parts)}")

    if turn_count >= 7:
        tension = calculate_tension_level(s)
        parts.append(
            f"[텐션] 막: {tension['act']} | 긴장도: {tension['tension']}"
        )

    # ── BLOCK 6.5: Hidden lore protection ──
    hidden_lore_inst = get_hidden_lore_instruction(on_screen)
    if hidden_lore_inst:
        parts.append(f"[비하인드 스토리 보호]\n{hidden_lore_inst}")

    # ── BLOCK 7: Memory ──
    memory = s.get("memory", {})
    core_pins = memory.get("core_pins", [])
    if core_pins:
        parts.append("[핵심 기억] " + "; ".join(
            (p[:80] if isinstance(p, str) else str(p)[:80]) for p in core_pins[-5:]
        ))
    long_term = memory.get("long_term", [])
    if long_term:
        parts.append("[장기 기억] " + "; ".join(str(m)[:60] for m in long_term[-3:]))

    # ── BLOCK 8: Maestro override ──
    mo = s.get("maestro_override", {})
    if mo.get("style_correction"):
        parts.append(f"[마에스트로 보정] {mo['style_correction']}")
    if mo.get("world_inject"):
        parts.append(f"[세계관 추가] {mo['world_inject']}")

    # ── BLOCK 8.5: PLAYER AGENCY GUARD (강화판) ──
    parts.append(PLAYER_AGENCY_GUARD.format(player_name=player))

    # ── BLOCK 9: Output format + Emotion whitelist (FIXED suffix) ──
    parts.append(
        f"[EMOTION WHITELIST] {', '.join(SUPPORTED_EMOTIONS)}\n"
        "emotion 필드는 반드시 위 목록에서만 선택하세요."
    )
    parts.append(
        '출력 형식: {"script": [{"type":"narration"|"dialogue"|"monologue", '
        '"content":"텍스트", "character":"캐릭터명", "emotion":"감정태그", '
        '"emotion_intensity":1~10, "monologue":"내면독백(선택)"}]}'
    )

    return "\n".join(parts)


# ─── Pulse System ────────────────────────────────────────────
def analyze_user_pulse(s: dict) -> dict:
    """Analyze recent user input patterns and return 5 stagnation triggers."""
    recent_user_inputs = []
    for t in s.get("turns", [])[-5:]:
        ui = t.get("user_input", "")
        if ui and ui not in ("[CONTINUE_SCENE]", "[PLAYER_PAUSE]"):
            recent_user_inputs.append(ui)

    if len(recent_user_inputs) < 2:
        return {"triggers_active": 0, "mode": "REACTIVE",
                "triggers": {}, "suggestion": ""}

    triggers = {}

    # T1: Static Dialogue — user input 3 consecutive ≤ 20 chars
    short_count = sum(1 for inp in recent_user_inputs[-3:] if len(inp.strip()) <= 20)
    triggers["static_dialogue"] = short_count >= 3

    # T2: Emotional Plateau — last 3 assistant emotions identical
    recent_emotions = []
    for t in s.get("turns", [])[-6:]:
        emo = t.get("emotion")
        if emo:
            recent_emotions.append(emo)
    triggers["emotional_plateau"] = (
        len(recent_emotions) >= 3 and len(set(recent_emotions[-3:])) == 1
    )

    # T3: Location Lock — 5+ turns in same scene
    triggers["location_lock"] = (
        s.get("scene_context", {}).get("turn_count_in_scene", 0) >= 5
    )

    # T4: Echo Input — user repeats very short or echoes NPC
    triggers["echo_input"] = False
    if recent_user_inputs:
        last_user = recent_user_inputs[-1].strip().lower()
        for t in s.get("turns", [])[-3:]:
            for b in t.get("script", []):
                if b.get("type") == "dialogue":
                    last_npc = b.get("content", "").strip().lower()
                    if last_npc and last_user and len(last_user) < 10 and last_user in last_npc:
                        triggers["echo_input"] = True
                        break
            if triggers["echo_input"]:
                break

    # T5: Question Dodge — placeholder for V4.1 NLU
    triggers["question_dodge"] = False

    active_count = sum(1 for v in triggers.values() if v)

    if active_count >= 2:
        mode = "PROACTIVE"
    elif active_count == 1:
        mode = "NUDGE"
    else:
        mode = "REACTIVE"

    suggestion = ""
    if mode == "PROACTIVE":
        dynamic_suggestions = []
        for name in s.get("on_screen", []):
            if name == s.get("player_name"):
                continue
            cdb = CHARACTERS_DB.get(name, {})
            ah = cdb.get("behavior_protocols", {}).get("acting_heuristics", {})
            ah_items = list(ah.items())
            if ah_items:
                _, hv = random.choice(ah_items)
                if hv:
                    dynamic_suggestions.append(f"{name}: {str(hv)[:80]}")
        if not dynamic_suggestions:
            dynamic_suggestions = [
                "캐릭터 중 한 명이 갑자기 과거 관련 고민을 꺼낸다.",
                "예상치 못한 환경 변화(소리, 날씨)가 발생한다.",
                "캐릭터 간 숨겨진 관계가 드러나는 말실수가 일어난다.",
            ]
        suggestion = random.choice(dynamic_suggestions)
    elif mode == "NUDGE":
        on_screen_names = [n for n in s.get("on_screen", []) if n != s.get("player_name")]
        if on_screen_names:
            nudge_char = random.choice(on_screen_names)
            suggestion = f"{nudge_char}이(가) 자연스럽게 다음 행동이나 장소 이동을 제안합니다."
        else:
            suggestion = "캐릭터가 자연스럽게 다음 행동이나 장소 이동을 제안합니다."

    return {
        "triggers_active": active_count,
        "mode": mode,
        "triggers": triggers,
        "suggestion": suggestion,
    }


def select_relevant_event_seed(on_screen: list, world_db: dict) -> Optional[dict]:
    """Select an event_seed relevant to current on_screen characters."""
    event_seeds = world_db.get("event_seeds", [])
    if not event_seeds:
        return None

    # Filter seeds that mention on-screen character names
    relevant = []
    for seed in event_seeds:
        seed_text = json.dumps(seed, ensure_ascii=False).lower()
        for name in on_screen:
            if name.lower() in seed_text or name in seed_text:
                relevant.append(seed)
                break

    if relevant:
        return random.choice(relevant)
    return random.choice(event_seeds)


def inject_director_brief(ui_settings: dict, s: Optional[dict] = None, pulse_result: Optional[dict] = None) -> str:
    parts = []
    if ui_settings.get("pov_first_person"):
        parts.append(
            "# [절대 규칙] 시점: 1인칭 '나'\n"
            "- 모든 narration 블록에서 시점 주체는 플레이어('나')이다.\n"
            "- '나는', '내가', '내 눈에', '내 귀에' 등 1인칭 표현만 사용한다.\n"
            "- 절대로 '당신은', '그는', '플레이어는' 같은 2·3인칭을 쓰지 않는다.\n"
            "- NPC 행동 묘사도 '나'의 시선을 통해 관찰하는 형태로 기술한다.\n"
            "  (예: \"루크가 고개를 들었다\" → \"루크가 고개를 든다. 그 파란 눈동자가 나를 향한다.\")"
        )
        # FIX-4: 1인칭 에이전시 보호
        parts.append(
            "\n[1인칭 에이전시 보호]\n"
            "- pov_first_person=true일 때, 플레이어의 감정, 신체 반응, 내면 독백을 AI가 임의로 작성하지 마라.\n"
            "- \"나는 심장이 두근거렸다\", \"내 얼굴이 붉어졌다\" 같은 표현 금지.\n"
            "- 대신 캐릭터의 반응과 환경 묘사로 간접 전달하라.\n"
            "  예: \"샐리의 눈이 동그래진다\" (O) vs \"나는 당황했다\" (X)"
        )
    else:
        parts.append("- [시점]: 3인칭 관찰자 시점으로 서술하라.")

    if ui_settings.get("show_monologue"):
        parts.append("- [내면 독백]: 각 대사에 monologue 필드를 채워 캐릭터의 내면을 드러내라.")

    genre = ui_settings.get("genre_preset", "auto")
    if genre and genre != "auto":
        parts.append(f"- [장르]: '{genre}' 장르의 분위기와 톤에 맞추어 연기하라.")
        genre_writing_rules = {
            "romance": "- [로맨스 연출]: 시선 교차, 미세한 물리적 거리 변화, 심장 뛰는 순간을 감각적으로 묘사. 고백은 절대 쉽게 나오지 않는다. 밀당과 오해가 핵심.",
            "comedy": "- [코미디 연출]: 캐릭터 간 타이밍과 리액션이 핵심. 하나의 오해가 눈덩이처럼 커지는 구조. 독백에서 셀프 츳코미(자기 반박)를 활용.",
            "mystery": "- [미스터리 연출]: 모든 대사에 이중 의미를 부여하라. 캐릭터가 무언가를 숨기는 느낌. 나레이션에서 '이상한 점'을 슬쩍 배치.",
            "thriller": "- [스릴러 연출]: 짧은 문장, 빠른 호흡. 침묵의 무게를 활용. 갑작스러운 소리나 변화를 삽입.",
            "slice_of_life": "- [일상 연출]: 사소한 행동에서 캐릭터성을 드러내라. 커피를 마시는 방식, 앉는 자세, 창밖을 보는 시선 등. 큰 사건 없이도 따뜻한 감정이 흐르게.",
            "horror": "- [호러 연출]: 오감을 극대화하되 공포는 '보이지 않는 것'에서 온다. 캐릭터의 불안을 먼저 보여주고, 원인은 나중에.",
            "fantasy": "- [판타지 연출]: 세계관의 마법/종족 설정을 자연스럽게 대사와 행동에 녹여라. 설명이 아니라 생활의 일부로.",
            "noir": "- [느와르 연출]: 건조한 나레이션, 독백이 많고, 비유가 날카롭다. 캐릭터 모두 비밀이 있다.",
            "soap_opera": "- [막장 드라마]: 감정의 폭이 극단적. 오해, 배신, 화해의 반복. 대사가 과장되지만 진심이 담겨 있다.",
            "wuxia": "- [무협 연출]: 행동 묘사가 시적이고 역동적. 존칭 체계가 엄격하며 의리와 명예가 대화를 지배한다.",
            "dark_fantasy": "- [다크 판타지]: 아름답지만 잔혹한 세계. 나레이션이 시적이면서 불길하다. 캐릭터의 내면에 어둠이 있다.",
            "sci-fi": "- [SF 연출]: 기술이 일상에 녹아든 묘사. 캐릭터가 기술을 자연스럽게 사용하는 모습.",
            "historical": "- [시대극]: 시대에 맞는 말투와 예절. 계급과 신분이 대화에 반영된다.",
            "adventure": "- [어드벤처]: 행동 중심, 긴박한 상황에서 캐릭터의 본성이 드러난다.",
        }
        genre_rule = genre_writing_rules.get(genre)
        if genre_rule:
            parts.append(genre_rule)

    tempo = ui_settings.get("tempo", 5)
    parts.append(f"- [템포]: {tempo}/10 (낮을수록 느리고 묘사적, 높을수록 빠르고 액션 중심)")

    # FIX-5: tempo를 구체적 행동 지시로 변환
    if tempo >= 8:
        parts.append(
            "- [템포 지시]: 대사 위주로 빠르게 진행. 나레이션은 1~2문장으로 최소화. script 블록 총 3개 이내."
        )
    elif tempo >= 5:
        parts.append(
            "- [템포 지시]: 대사와 나레이션 균형. script 블록 총 4~5개."
        )
    else:
        parts.append(
            "- [템포 지시]: 나레이션과 심리묘사 중심. 대사는 짧게. script 블록 총 5~7개."
        )

    narr_ratio = ui_settings.get("narration_ratio", 40)
    parts.append(f"- [나레이션 비율]: {narr_ratio}% (나레이션과 대사의 비율)")

    desc_focus = ui_settings.get("description_focus", 5)
    parts.append(f"- [묘사 집중도]: {desc_focus}/10")

    # Description Focus density tiers
    if desc_focus >= 7:
        parts.append("- 묘사 밀도: HIGH. 모든 행동을 온도, 냄새, 색깔, 질감, 시간의 흐름으로 분해하라. 내면 심리는 3~5문장으로 확장하라.")
    elif desc_focus <= 3:
        parts.append("- 묘사 밀도: LOW. 짧은 서술문 위주. 행동과 대사에 집중하라. 묘사는 1~2문장으로 제한.")
    else:
        parts.append("- 묘사 밀도: NORMAL. 균형 잡힌 서술. 핵심 감각 2가지만 포함.")

    # Genre-specific anti-metaphor rule for mystery/thriller/noir
    if genre in ("mystery", "thriller", "noir"):
        parts.append("- [건조한 정밀 묘사]: 은유 최소화, 짧은 서술문, 물리적 증거 중심 묘사.")

    # FIX-8: 캐릭터별 역할 분화 강제
    if s is not None:
        on_screen = s.get("on_screen", [])
        player = s.get("player_name", "사용자")
        npcs = [n for n in on_screen if n != player]
        if len(npcs) >= 2:
            # 캐릭터 DB에서 dynamics 참조
            role_lines = []
            for npc in npcs:
                cdb = CHARACTERS_DB.get(npc, {})
                vs_player = cdb.get("relationship_matrix", {}).get("플레이어", {})
                dynamics = vs_player.get("dynamics", "중립적")
                if dynamics == "주도적":
                    role_lines.append(f"- {npc}: 이번 턴에서 대화를 주도하거나 새로운 행동을 제안하라.")
                elif dynamics == "반응적":
                    role_lines.append(f"- {npc}: 이번 턴에서 상황을 관찰하고, {npcs[0] if npc != npcs[0] else npcs[1]}의 행동에 반응하라.")
                else:
                    role_lines.append(f"- {npc}: 이번 턴에서 독자적인 행동(딴짓, 자기 이야기)을 하라.")

            parts.append(
                "\n[캐릭터 역할 분배 — 이번 턴]\n"
                "2캐릭터 이상 씬에서 모든 캐릭터가 유저만 바라보며 같은 반응 금지.\n"
                "반드시 NPC끼리 1회 이상 대화하거나 반응해야 함.\n"
                + "\n".join(role_lines)
            )
        elif len(npcs) == 1:
            parts.append(
                "\n[캐릭터 행동 분배]\n"
                "- 캐릭터가 유저에게만 집중하되, 환경과 상호작용하는 행동도 포함하라."
            )
    else:
        parts.append(
            "\n[캐릭터 행동 분배]\n"
            "- 2캐릭터 씬에서 둘이 같은 반응(둘 다 놀리기, 둘 다 칭찬)을 하지 마라.\n"
            "- 한 캐릭터가 유저에게 질문하면, 다른 캐릭터는 다른 행동(관찰, 딴짓, 자기 이야기)을 하라."
        )

    parts.append(
        "\n[존댓말/반말 혼용 원칙]\n"
        "- 한국어 자연 대화에서 존비어는 관계·감정·상황에 따라 유동적으로 섞인다.\n"
        "- 감정이 고조되면 반말로, 부탁이나 사과 시 존댓말로, 장난칠 때 존반말(~요 없이 ~지? ~거든?)로 자연스럽게 전환.\n"
        "- 각 캐릭터의 tone_mixing_rule을 참조하되, 기계적으로 적용하지 말고 맥락에 맞게 판단하라.\n"
        "- 관계 단계가 올라갈수록 격식이 자연스럽게 풀려야 한다."
    )

    # FIX-7: 감정 다양성 강제
    parts.append(
        "\n[감정 다양성]\n"
        "- 같은 캐릭터가 3턴 연속 동일 감정을 사용하는 것을 금지한다.\n"
        "- emotion_intensity는 1~10 범위에서 상황에 따라 변동시켜라. 항상 5는 금지."
    )

    # 물리적 상태와 행동 연동
    if s is not None:
        core4 = s.get("core4", {})
        energy_val = core4.get("energy", {}).get("value", 70)
        pain_val = core4.get("pain", {}).get("value", 0)
        intox_val = core4.get("intoxication", {}).get("value", 0)

        physical_notes = []
        if energy_val <= 30:
            physical_notes.append(
                "에너지 30 이하: 모든 캐릭터의 physical_tells.fatigue_behavior를 묘사에 반영하라. "
                "대사가 짧아지고, idle_habits 중 '졸림' 트리거가 발동한다."
            )
        if pain_val >= 40:
            physical_notes.append(
                f"고통 {pain_val}: 캐릭터별 pain_tolerance에 따라 반응이 다르다. "
                "low인 캐릭터는 행동이 느려지고 표정이 일그러진다. high인 캐릭터도 미세한 신호를 보인다."
            )
        if intox_val >= 30:
            physical_notes.append(
                f"취기 {intox_val}: 캐릭터별 honesty_profile의 deception_skill이 낮아진다. "
                "leaky_topics에 해당하는 비밀이 실수로 새어나올 수 있다. "
                "sacred_secrets는 만취 상태에서도 절대 말하지 않는다."
            )
        if physical_notes:
            parts.append("\n[물리 상태 → 행동 연동]\n" + "\n".join(f"- {n}" for n in physical_notes))

    # Event hints — STEP 9: 조건부 이벤트 시드 발동
    if s is not None:
        event_hint = get_event_seed_for_scene(s)
        if event_hint:
            parts.append(event_hint)

    # Pulse System injection
    if pulse_result:
        if pulse_result["mode"] == "PROACTIVE":
            parts.append(
                "\n=== 🔴 PROACTIVE 모드 (유저 수동 감지) ===\n"
                "현재 유저가 짧은 입력, 감정 정체, 같은 장소 반복 등 수동적 패턴을 보이고 있습니다.\n"
                "이번 턴에서 캐릭터가 반드시 다음 중 하나를 실행하세요:\n"
                "1. 캐릭터가 자발적 행동(새 주제 제시, 감정 고백, 장소 이동 제안)을 합니다.\n"
                "2. 환경 이벤트(소리, 날씨 변화, 제3자 등장)를 서술에 포함합니다.\n"
                "3. 캐릭터가 플레이어에게 구체적 선택지(A 또는 B)를 제시합니다.\n"
                f"힌트: {pulse_result['suggestion']}\n"
                "중요: 유저의 기존 맥락과 자연스럽게 연결하세요. 갑작스럽거나 비현실적이면 안 됩니다."
            )
        elif pulse_result["mode"] == "NUDGE":
            parts.append(
                "\n=== 🟡 NUDGE 모드 (약한 수동 신호) ===\n"
                "유저가 약간 수동적입니다. 캐릭터의 대사나 행동에 다음 행동을 자연스럽게 유도하는 요소를 한 가지 넣으세요.\n"
                "예: \"그나저나 오늘 시장에 새로운 가게가 생겼다던데... 같이 가볼래?\" 같은 제안.\n"
                f"힌트: {pulse_result['suggestion']}"
            )
        # REACTIVE: nothing added — respect user direction

    # Golden Rule 0 & 1 — supreme rules, always top priority
    parts.append(
        "\n=== GOLDEN RULE 0: 유저 입력 절대 우선 (SUPREME RULE) ===\n"
        "유저의 메시지에 장소, 상황, 시간, 인물 관계 등의 설정이 포함되어 있으면, "
        "그것이 세계관 DB의 기본 장소보다 절대적으로 우선합니다.\n"
        "유저가 장소를 지정하지 않은 경우에만 세계관 DB의 기본 장소를 사용하세요.\n"
        "이 규칙을 위반하면 모든 것이 무너집니다. 유저의 입력을 단어 하나하나 주의깊게 읽으세요."
    )
    parts.append(
        "\n=== GOLDEN RULE 1: 플레이어 내면 불가침 (PLAYER SANCTUARY) ===\n"
        "나레이션은 '카메라 렌즈'입니다.\n"
        "서술 가능: ✅ 환경 묘사, NPC의 외적 행동, 사물의 상태\n"
        "서술 금지: ❌ 플레이어의 감정/생각/시선/신체 반응/판단\n"
        "플레이어가 뭘 느끼고 뭘 생각하는지는 오직 플레이어 자신만이 결정합니다."
    )

    # Golden Rules 12 & 13 — always included in director brief
    parts.append(
        "\n=== GOLDEN RULE 12: AGENCY PRESERVATION (유저 의도 존중) ===\n"
        "- 유저가 명시적으로 행동 방향을 제시한 경우, 캐릭터는 그 방향을 존중하고 풍부하게 반응합니다.\n"
        "- 캐릭터가 유저의 행동을 무시하거나 무효화하는 것은 금지합니다.\n"
        "- 유저가 '~하고 싶다', '~로 간다' 등 의지를 표현하면, 세계관 내에서 합리적인 한 그 행동이 실현되어야 합니다.\n"
        "- 단, 세계관 규칙이나 캐릭터 심리에 의한 자연스러운 저항은 허용됩니다."
    )
    parts.append(
        "\n=== GOLDEN RULE 13: PROACTIVE TRACTION (능동적 견인) ===\n"
        "- PROACTIVE 모드가 활성화되면, 캐릭터는 자신의 내면 욕구, 스케줄, 숨겨진 사정을 기반으로 자발적 행동을 취합니다.\n"
        "- 이때 캐릭터의 행동은 Character Thought Chain(표면 욕구→숨겨진 욕구→배경 영향→최종 반응)을 반드시 거쳐야 합니다.\n"
        "- 견인은 '꼬리표 달린 선택지'가 아니라, 캐릭터가 살아있기 때문에 자연스럽게 일어나는 행동이어야 합니다."
    )

    return "\n".join(parts)


# ─── Build D.I.M.A prompt ────────────────────────────────────
DIMA_PROMPT_TEMPLATE = SAFETY_PREAMBLE + """
You are D.I.M.A., a master theater director writing living scenes.

[CORE RULES — 이것만 지키면 됩니다]
1. ALIVE: 캐릭터는 유저를 기다리지 않는다. 자기 욕구대로 먼저 행동한다.
   - 2캐릭터 이상 씬에서 같은 반응 금지. 한 명이 유저에게 말하면 다른 한 명은 다른 행동.
   - NPC끼리 최소 1회 교류(서로에게 말하기, 반응하기). 유저만 보고 있으면 실패.
2. EPISODE: 시작(감각묘사 2문장) → 대화 → 갈고리(열린 결말)로 끝낸다.
3. VOICE: 각 캐릭터의 catchphrase_budget을 반드시 준수. 같은 시작어 2턴 연속 금지.
4. PLAYER SANCTUARY: 플레이어의 감정/생각/판단을 절대 서술하지 마라.
   NPC의 관찰("~하는 것 같다")로만 존재감 표현. 턴당 1회 이하.

[OUTPUT] JSON만: {{"script": [{{"type":"narration"|"dialogue","content":"...","character":"...","emotion":"...","emotion_intensity":1~10,"monologue":"..."}}]}}
- emotion_intensity: 상황에 따라 1~10 변동. 항상 5 금지.
- monologue: dialogue 블록 안에 넣기. 독립 monologue 블록 금지. 대사와 다른 숨은 감정 필수.

### Recent Conversation Log ###
{recent_conversation_log}

### Director's Brief ###
{director_brief}

### Character Briefs ###
{character_briefs}

### Scene Context ###
- Location: {location_and_time}

### Player's Action ###
Player Name: {player_name}
{user_input}
"""


def _short(txt: str, n: int = 100) -> str:
    s = str(txt or "")
    return s if len(s) <= n else s[:n] + "..."


def _build_event_digest(turn_id: int, user_input: str, script: list) -> str:
    """Build a structured event-based flow digest entry.
    Extracts dialogue blocks' character, emotion, content into
    'T{n}: {character}({emotion}) — {content[:N]}' format."""
    lines = []
    for b in script:
        if b.get("type") == "dialogue" and b.get("character"):
            char = b.get("character", "?")
            emotion = b.get("emotion", "neutral")
            content = (b.get("content") or "")[:DIGEST_CONTENT_MAX_LENGTH]
            lines.append(f"{char}({emotion}) — {content}")
    if not lines:
        # Fallback: use narration summary
        for b in script:
            if b.get("type") == "narration":
                content = (b.get("content") or "")[:DIGEST_CONTENT_MAX_LENGTH]
                lines.append(f"[나레이션] {content}")
                break
    summary = " | ".join(lines[:4]) if lines else "(무응답)"
    return f"T{turn_id}: {summary}"


def _build_event_short_term(turn_id: int, user_input: str, script: list) -> str:
    """Build an event summary for short-term memory instead of raw dialogue."""
    events = []

    # Player action
    if user_input and user_input not in ("[PLAYER_PAUSE]", "[CONTINUE_SCENE]"):
        events.append(f"플레이어: {user_input[:60]}")
    else:
        events.append("플레이어: (침묵)")

    # NPC key actions (summarize dialogue intent, not raw text)
    seen_chars = set()
    for b in script:
        char = b.get("character", "")
        if b.get("type") == "dialogue" and char and char not in seen_chars:
            seen_chars.add(char)
            emotion = b.get("emotion", "neutral")
            content_preview = b.get("content", "")[:40]
            events.append(f"{char}({emotion}): {content_preview}")
        elif b.get("type") == "narration" and not events:
            events.append(f"[장면] {b.get('content', '')[:50]}")

    return f"Turn {turn_id} | {' | '.join(events[:4])}"


def build_dima_prompt(s: dict, user_input: str) -> tuple:
    """Returns (system_instruction, main_prompt, pulse_result)."""
    player_name = s.get("player_name", "사용자")
    world = s.get("world", {})
    location = world.get("space", {}).get("current_location", "라운지")
    on_screen_chars = s.get("on_screen", [])

    system_instruction = build_adaptive_instruction(s)

    # Build user input for prompt
    u_text_strip = user_input.strip()
    if u_text_strip == "[PLAYER_PAUSE]":
        user_input_for_prompt = (
            f"The player stays silent for a beat, observing the scene.\n"
            f"Start the scene yourself in '{location}'.\n"
            "- Produce 2-4 short, in-character NPC lines and a brief narration to begin the story.\n"
            "- Use a slice-of-life tone, referencing the setting.\n"
            "- Never speak as the player."
        )
    elif u_text_strip == "[CONTINUE_SCENE]":
        user_input_for_prompt = (
            "The player remains silent, letting the scene continue.\n"
            "- Proactively continue the interaction between NPCs.\n"
            "- Do not ask the player a direct question; create a natural moment for them to interject.\n"
            "- Never write the player's lines."
        )
    else:
        user_input_for_prompt = (
            "[유저 입력 — 아래 내용에 명시된 장소·상황·시간·인물 설정을 반드시 최우선으로 반영하세요. "
            "DB 기본값보다 유저 입력이 항상 우선합니다.]\n"
            f"{user_input}"
        )

    # Recent conversation log for DIMA prompt
    digest = s.get("flow_digest_10", [])
    if digest:
        safe_digest = []
        for item in digest[-7:]:
            if isinstance(item, str):
                safe_digest.append(item)
            elif isinstance(item, dict):
                safe_digest.append(
                    f"T{item.get('turn', '?')}: {item.get('summary', '')[:80]}"
                )
            else:
                safe_digest.append(str(item)[:80])
        digest_text = "\n".join(safe_digest)
    else:
        digest_text = "(첫 번째 턴)"

    # Scene continuity block — maintain location and action context
    last_action_summary = ""
    all_turns = s.get("turns", [])
    if all_turns:
        last_turn = all_turns[-1]
        last_scripts = last_turn.get("script", [])
        for b in reversed(last_scripts):
            if b.get("content"):
                last_action_summary = b["content"][:MAX_ACTION_SUMMARY_LENGTH]
                break

    scene_continuity_block = (
        f"\n=== 현재 장면 상태 ===\n"
        f"- 장소: {location}\n"
        f"- 진행 중인 행동: {last_action_summary if last_action_summary else '(첫 턴)'}\n"
        f"- 장면 규칙: 유저가 장소 이동을 명시하지 않는 한 현재 장소({location})를 유지하라. "
        f"유저의 행동에 자연스럽게 반응하되, AI가 임의로 장소를 바꾸거나 새로운 이벤트를 삽입하지 마라.\n"
    )
    recent_turns_1 = all_turns[-1:] if all_turns else []
    raw_recent = []
    for t in recent_turns_1:
        if t.get("user_input"):
            raw_recent.append(f"[Player]: {t['user_input'][:120]}")
        for b in t.get("script", [])[:4]:
            if b.get("type") == "dialogue":
                raw_recent.append(f"[{b.get('character','?')}]: {b.get('content','')[:100]}")
    raw_text = "\n".join(raw_recent[-8:])

    # Anti-repetition: extract forbidden patterns from last 3 turns
    forbidden_starts = set()
    forbidden_catchphrases = {}  # {character: [used catchphrases]}
    for t in all_turns[-3:]:
        for b in t.get("script", []):
            if b.get("type") == "dialogue" and b.get("content"):
                char = b.get("character", "")
                content = b["content"].strip()
                # 첫 어절 추출 (더 구체적)
                first_words = content[:20]
                forbidden_starts.add(first_words)
                # 캐릭터별 사용한 캐치프레이즈 추적
                if char:
                    cps = forbidden_catchphrases.setdefault(char, [])
                    for cp_word in ["호호호", "아이고", "어머", "우리 김갑수", "누나"]:
                        if cp_word in content[:30]:
                            cps.append(cp_word)

    anti_repetition_block = ""
    if forbidden_starts:
        lines_ar = []
        lines_ar.append("\n=== ANTI-REPETITION (절대 준수) ===")
        lines_ar.append("아래와 동일/유사 시작 금지:")
        for fs in list(forbidden_starts)[:8]:
            lines_ar.append(f'  ❌ "{fs}..."')
        for char, cps in forbidden_catchphrases.items():
            if len(cps) >= 2:
                lines_ar.append(f"  ⚠️ {char}: 최근 3턴에서 '{cps[0]}' 등을 {len(cps)}회 사용. 이번 턴에서 사용 금지.")
        lines_ar.append("같은 캐릭터가 3턴 연속 동일 emotion 사용 금지.")
        anti_repetition_block = "\n".join(lines_ar)

    recent_conversation_log = (
        f"{scene_continuity_block}\n"
        f"=== 흐름 요약 (최근 10턴) ===\n{digest_text}\n\n"
        f"=== 직전 대화 (최근 1턴 원문) ===\n{raw_text}"
        f"{anti_repetition_block}"
    )

    # Character briefs with scene continuity + Maestro action_context
    briefs = []

    # 장면 연속성 블록
    continuity_lines = []
    prev_turns = s.get("turns", [])
    if prev_turns:
        lt = prev_turns[-1]
        lt_user = lt.get("user_input", "")
        lt_dlgs = []
        for b in lt.get("script", []):
            if isinstance(b, dict) and b.get("type") == "dialogue":
                lt_dlgs.append(f'{b.get("character", "?")}: "{b.get("content", "")[:50]}"')
        continuity_lines.append("## [장면 연속성 — 최우선 참조]")
        if lt_user and lt_user not in ("[CONTINUE_SCENE]", "[PLAYER_PAUSE]"):
            continuity_lines.append(f'직전 유저 발화: "{lt_user[:80]}"')
        if lt_dlgs:
            continuity_lines.append(f"직전 NPC 응답: {' | '.join(lt_dlgs[:2])}")
        continuity_lines.append("→ 이 맥락의 '다음 순간'부터 연기. 위 내용 반복 금지.")

    continuity_lines.append(f"장소: '{location}' — 유저가 이동 지시 안 하면 유지.")

    # Maestro action_context 주입 + 캐릭터별 상태
    action_context = s.get("action_context", {})
    for name in on_screen_chars:
        if name == player_name:
            continue
        ac_text = action_context.get(name, "")
        char_data = s.get("characters", {}).get(name, {})
        ds = char_data.get("dynamic_state", {})
        mood = ds.get("current_mood", "neutral") if isinstance(ds, dict) else "neutral"
        intensity = ds.get("mood_intensity", 5) if isinstance(ds, dict) else 5

        brief_lines = [f"<{name}>"]
        if ac_text:
            if isinstance(ac_text, str):
                brief_lines.append(f"  기억: {ac_text[:100]}")
            elif isinstance(ac_text, dict):
                summary = ac_text.get("my_last_action", {}).get("summary") if isinstance(ac_text.get("my_last_action"), dict) else None
                if summary:
                    brief_lines.append(f"  기억: {summary[:100]}")
        brief_lines.append(f"  감정: {mood}(강도{intensity}/10)")
        rel = s.get("relationships", {}).get(name, {})
        if rel:
            stage_name = {1: "경계", 2: "동료", 3: "신뢰", 4: "특별"}.get(rel.get("stage", 1), "경계")
            brief_lines.append(f"  관계: {stage_name} (호감{rel.get('affection', 50)})")
        continuity_lines.append("\n".join(brief_lines))

    # STEP 5-B: 3-Tier 메모리 컨텍스트 주입
    memory_context = build_memory_context_for_dima(s)
    continuity_lines.append(memory_context)

    # STEP 7: flow_digest를 DIMA가 읽기 쉬운 형태로 변환
    flow_context = format_flow_digest_for_dima(s)
    continuity_lines.append(flow_context)

    briefs.insert(0, "\n".join(continuity_lines))
    character_briefs_content = "\n".join(briefs)

    # Director brief — with Pulse analysis
    pulse_result = analyze_user_pulse(s)
    director_brief = inject_director_brief(s.get("ui_settings", {}), s=s, pulse_result=pulse_result)

    # Maestro next_beat 지시 주입
    next_beat = s.get("next_beat")
    if next_beat and isinstance(next_beat, dict):
        lead = next_beat.get("lead_character", "")
        tactic = next_beat.get("tactic", "")
        tension = next_beat.get("tension_direction", "maintain")
        if lead and tactic:
            director_brief += (
                f"\n\n# [Maestro 연출 지시 — next_beat]\n"
                f"- 주도 캐릭터: {lead}\n"
                f"- 전략: {tactic}\n"
                f"- 긴장감 방향: {tension}\n"
                f"→ 이번 턴에서 '{lead}'이(가) 위 전략대로 행동을 주도하라."
            )
        # 사용 후 삭제 (stale beat 방지)
        s.pop("next_beat", None)

    # Self-Check Protocol: inject self-check block for turn >= 1
    turn_count = len(s.get("turns", []))
    if turn_count > 0:
        director_brief += (
            "\n\n[self_check] 응답 전 확인:\n"
            "1. POV: 1인칭 시점 유지 (나/내가/우리)\n"
            "2. 말투: 각 캐릭터의 관계 단계 + CORE-4 보정 적용\n"
            "3. CCT: Character Thought Chain 4단계 (표면→숨김→배경→반응) 수행\n"
            "4. Agency: 유저가 방향을 제시했으면 그 방향으로 진행하고 있는가?\n"
            "5. Pulse: 현재 모드가 PROACTIVE이면, 이번 턴에 캐릭터의 자발적 행동이 포함되어 있는가?\n"
            "6. 반복 방지: 직전 턴과 동일한 감정/행동/대사 패턴이 아닌가?\n"
            "(7) 유저가 명시한 장소·상황이 내 출력에 정확히 반영되었는가? 무시하고 기본 장소로 대체하지 않았는가?\n"
            "(8) 나레이션에서 유저의 내면(감정, 생각, 시선, 판단)을 대신 서술하지 않았는가? 유저의 내면은 절대 서술 금지."
        )

    # 감정 다양성 검사 (강화판: character_last 활용)
    emo_hist = {}
    char_last = s.get("memory", {}).get("character_last", {})
    for turn in s.get("turns", [])[-3:]:
        for b in turn.get("script", []):
            if isinstance(b, dict) and b.get("type") == "dialogue":
                c = b.get("character", "")
                e = b.get("emotion", "")
                if c and e:
                    emo_hist.setdefault(c, []).append(e)
    # character_last에서 추가 반영
    for name, info in char_last.items():
        if info.get("emotion"):
            emo_hist.setdefault(name, []).append(info["emotion"])

    emo_alerts = []
    for c, emos in emo_hist.items():
        if len(emos) >= 2 and len(set(emos[-3:])) == 1:
            emo_alerts.append(
                f"⚠️ {c}: '{emos[0]}' {len(emos)}턴 연속 반복. "
                f"이번 턴에서 반드시 다른 감정 사용. intensity도 변경."
            )
    if emo_alerts:
        director_brief += "\n\n# [EMOTION DIVERSITY]\n" + "\n".join(emo_alerts)

    # 감정 관성 — 직전 턴 감정이 다음 턴 시작점이 됨
    char_last = s.get("memory", {}).get("character_last", {})
    if char_last:
        inertia_lines = []
        for name, info in char_last.items():
            emo = info.get("emotion", "")
            intensity = info.get("intensity", 5)
            if emo and name in [n for n in s.get("on_screen", []) if n != player_name]:
                if intensity >= 7:
                    inertia_lines.append(
                        f"- {name}: 직전 감정 '{emo}'(강도{intensity}) — 강한 감정이므로 "
                        f"이번 턴 시작 시에도 여운이 남아있어야 함. 갑자기 밝아지거나 무덤덤해지면 안 됨."
                    )
                elif intensity >= 4:
                    inertia_lines.append(
                        f"- {name}: 직전 감정 '{emo}'(강도{intensity}) — 자연스러운 전환 가능하지만 "
                        f"첫 대사에 여파가 살짝 묻어나야 함."
                    )
        if inertia_lines:
            director_brief += (
                "\n\n# [EMOTIONAL INERTIA — 감정 관성]\n"
                "캐릭터의 감정은 갑자기 리셋되지 않는다. 직전 턴의 감정 상태가 이번 턴의 출발점이다.\n"
                + "\n".join(inertia_lines)
            )

    # 비밀 누출 판정
    leak_hints = []
    for name in on_screen_chars:
        if name == player_name:
            continue
        leaked = should_leak_secret(name, s)
        if leaked:
            cdb = CHARACTERS_DB.get(name, {})
            hp = cdb.get("behavior_protocols", {}).get("honesty_profile", {})
            tell = hp.get("tell_when_lying", "")
            leak_hints.append(
                f"- {name}: 실수로 '{leaked}'에 대한 단서를 흘릴 수 있는 상태. "
                f"직접적으로 말하지 않지만, 말실수·행동·표정으로 암시하라. "
                f"({tell[:60] if tell else '미세한 동요'})"
            )
    if leak_hints:
        director_brief += (
            "\n\n# [SECRET LEAK CHANCE — 비밀 누출 기회]\n"
            "아래 캐릭터가 현재 물리 상태(취기/피로/스트레스)로 인해 방어가 약해진 상태입니다.\n"
            "강제가 아닌 자연스러운 연출로 처리하세요. monologue에서 내면 갈등을 보여주세요.\n"
            + "\n".join(leak_hints)
        )

    # 캐릭터별 차등 기억 — 부재 중 일어난 일을 모름
    presence = s.get("memory", {}).get("character_presence", {})
    recent_turns = s.get("turns", [])[-5:]
    recent_turn_ids = [t.get("turn_id") for t in recent_turns]
    absence_alerts = []
    for name in on_screen_chars:
        if name == player_name:
            continue
        char_present_turns = set(presence.get(name, []))
        missed = [tid for tid in recent_turn_ids if tid not in char_present_turns]
        if missed:
            missed_summaries = []
            for t in recent_turns:
                if t.get("turn_id") in missed:
                    for b in t.get("script", [])[:2]:
                        if b.get("type") == "dialogue":
                            missed_summaries.append(
                                f"T{t['turn_id']}: {b.get('character','?')}의 대사"
                            )
                            break
            absence_alerts.append(
                f"- {name}: 턴 {','.join(map(str,missed))}에 자리에 없었음. "
                f"그동안 일어난 일({'; '.join(missed_summaries[:3]) if missed_summaries else '대화'})을 모름. "
                f"알고 있는 것처럼 반응하면 안 됨. "
                f"'무슨 이야기 했어?' 식으로 물어보는 것은 가능."
            )
    if absence_alerts:
        director_brief += (
            "\n\n# [DIFFERENTIAL MEMORY — 캐릭터별 차등 기억]\n"
            "아래 캐릭터는 특정 턴에 자리에 없었으므로 그때 일어난 일을 모릅니다.\n"
            + "\n".join(absence_alerts)
        )

    # Player Presence 관찰 기법
    player_name = s.get("player_name", "사용자").replace('"', '').replace("'", "").strip()
    if not player_name:
        player_name = "사용자"
    director_brief += f"""

# [PLAYER PRESENCE — NPC 관찰 기법]
# {player_name}의 감정/생각을 직접 쓰면 안 되지만,
# NPC 시선과 반응으로 {player_name}의 존재감을 매 턴 1회 이상 표현하라.
# 예: "{player_name}의 표정을 슬쩍 살피다 시선을 돌렸다"
# 예: "말을 하다가 {player_name} 쪽을 힐끗 보더니 헛기침을 했다"
# 예: "{player_name}의 발소리에 무의식적으로 자세를 바로 했다" """

    # Opening Narration: first turn directive
    if turn_count == 0:
        char_names_str = ", ".join(
            n for n in on_screen_chars if n != player_name
        )
        director_brief += (
            "\n\n[오프닝 지시] 이번이 첫 턴입니다. 반드시 다음을 포함하세요:\n"
            "1. 장소(라운지/주방 등)와 시간대(아침/오후/밤)를 설정하는 2~3문장 나레이션\n"
            "2. 각 등장 캐릭터가 무엇을 하고 있었는지 1문장씩 묘사\n"
            f"   등장 캐릭터: {char_names_str}\n"
            "3. 그 후에 대화 시작\n"
        )

    if u_text_strip == "[PLAYER_PAUSE]":
        director_brief += (
            "\n- [Player Silence Directive / 첫 턴]: 플레이어는 침묵으로 시작한다. "
            "NPC들 중 한 명이 반드시 먼저 행동/대사를 시작해 대화를 개시하라."
        )
    elif u_text_strip == "[CONTINUE_SCENE]":
        director_brief += (
            "\n- [Player Silence Directive / 중간 턴]: 플레이어는 침묵을 유지한다. "
            "NPC들 중 한 명이 결정적 행동 또는 상황을 바꾸는 대사로 장면을 전진시켜야 한다."
        )

    main_prompt = DIMA_PROMPT_TEMPLATE.format(
        recent_conversation_log=recent_conversation_log,
        character_briefs=character_briefs_content,
        director_brief=director_brief,
        location_and_time=f"{location}",
        player_name=player_name,
        user_input=user_input_for_prompt,
    )

    return system_instruction, main_prompt, pulse_result


# ─── LLM call ────────────────────────────────────────────────
_rate_lock = threading.Lock()
_last_call_time = 0.0


def generate_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.75,
    response_schema: Optional[dict] = None,
) -> Optional[dict]:
    """Call Gemini and return parsed JSON dict.
    Also stores the raw response in generate_llm.last_response for token metering.
    """
    global _last_call_time

    with _rate_lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
        _last_call_time = time.time()

    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": 4096,
        "response_mime_type": "application/json",
        "safety_settings": get_safety_settings(),
        "thinking_config": genai_types.ThinkingConfig(thinking_budget=1024),
    }
    if response_schema:
        config_kwargs["response_schema"] = response_schema

    config = genai_types.GenerateContentConfig(
        system_instruction=system_instruction,
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        **config_kwargs,
    )

    response = call_gemini_with_fallback(
        contents=[genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)]
        )],
        config=config,
    )
    # V5.0: Store raw response for token metering
    generate_llm.last_response = response

    if response is None:
        return None

    # Part K: Check for safety-blocked response
    if response.candidates:
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, 'finish_reason', None)
        if finish_reason and str(finish_reason).upper() in ("SAFETY", "BLOCKED", "2", "3"):
            logger.warning(f"Response blocked by safety filter (reason: {finish_reason})")
            return {"script": [{"type": "narration", "content": "(안전 필터에 의해 장면이 전환됩니다. 잠시 후 이야기가 계속됩니다.)"}]}

    text = _extract_text(response)
    if text:
        # Part K: Strip markdown fences before parsing
        cleaned = re.sub(r'```json\s*', '', text.strip())
        cleaned = re.sub(r'\s*```', '', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed output
            start = cleaned.find('{')
            if start >= 0:
                depth = 0
                for i, ch in enumerate(cleaned[start:], start=start):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            return json.loads(cleaned[start:i+1])
    logger.warning("Empty LLM response after fallback chain")
    return None

generate_llm.last_response = None  # V5.0: Initialize attribute for token metering


# ─── Script post-processing ──────────────────────────────────
def _get_character_fallback_line(char: str) -> str:
    """characters_db에서 캐릭터별 고유 폴백 대사를 가져옴. 하드코딩 방지."""
    cdb = CHARACTERS_DB.get(char, {})
    bp = cdb.get("behavior_protocols", {})
    sig = bp.get("signature_speech", {})
    ex = sig.get("example_lines", [])
    if ex:
        return random.choice(ex)
    cp = sig.get("catchphrases", [])
    if cp:
        return cp[0]
    return f"({char}이(가) 조용히 상대를 바라본다.)"


def safe_local_script(s: dict, prelude_narration: str = "") -> list:
    """Generate a minimal fallback script when LLM fails."""
    on_screen = s.get("on_screen", [])
    player_name = get_player_name(s)
    npc_candidates = [n for n in on_screen if n != player_name]
    npc = npc_candidates[0] if npc_candidates else "관리인"

    script = []
    if prelude_narration:
        script.append({"type": "narration", "content": prelude_narration})
    script.append({
        "type": "dialogue",
        "character": npc,
        "content": _get_character_fallback_line(npc),
        "emotion": "gentle_affection",
        "emotion_intensity": 3,
    })
    return script


def post_process_script(script: list, s: dict) -> list:
    """Validate and clean up script blocks."""
    if not isinstance(script, list):
        return [{"type": "narration", "content": "(AI 응답 형식 오류)"}]

    player_name = get_player_name(s)
    on_screen_set = set(s.get("on_screen", []))
    processed = []
    has_valid_dialogue = False

    for block in script:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        char = (block.get("character") or "").strip()

        # Replace player placeholders in content
        for k in ("content", "monologue"):
            if isinstance(block.get(k), str):
                t = block[k]
                t = t.replace("플레이어", player_name).replace("Player", player_name).replace("당신", player_name)
                block[k] = t

        # Skip player lines
        if btype == "dialogue" and char == player_name:
            continue

        # Off-screen character → convert to narration
        if btype in ("dialogue", "monologue") and char and char not in on_screen_set:
            txt = block.get("content") or block.get("monologue") or ""
            processed.append({"type": "narration", "content": f"(장면 밖 '{char}'의 기척: {txt})"})
            continue

        if btype == "dialogue":
            # Part A: Normalize emotion with hybrid engine
            em = block.get("emotion", "joy")
            block["emotion"] = normalize_emotion_tag(em)
            # Ensure emotion_intensity is integer 1-5
            intensity = block.get("emotion_intensity", 3)
            try:
                intensity = max(1, min(5, int(intensity)))
            except (ValueError, TypeError):
                intensity = 3
            block["emotion_intensity"] = intensity
            processed.append(block)
            has_valid_dialogue = True
        elif btype == "monologue":
            who = char or player_name
            mono_txt = (block.get("monologue") or block.get("content") or "").strip()
            if mono_txt:
                processed.append({"type": "monologue", "character": who, "content": mono_txt})
        elif btype == "narration":
            processed.append({"type": "narration", "content": block.get("content", "")})
        else:
            processed.append({"type": "narration", "content": str(block)})

    if not has_valid_dialogue:
        npc_candidates = [n for n in on_screen_set if n != player_name]
        npc = npc_candidates[0] if npc_candidates else "관리인"
        processed.append({
            "type": "dialogue",
            "character": npc,
            "content": _get_character_fallback_line(npc),
            "emotion": "joy",
            "emotion_intensity": 2,
        })
    return processed


# =========================================================================
# PART D: EMOTIONAL CONTAGION BETWEEN CHARACTERS (PAD-based)
# =========================================================================
def apply_emotional_contagion(s: dict, script: list):
    """PAD-averaging emotional contagion with stress feedback."""
    dominant_emotions = []
    for block in script:
        if block.get("type") == "dialogue" and block.get("emotion"):
            emo_key = normalize_emotion_tag(block["emotion"])
            pad = get_emotion_pad(emo_key)
            intensity = block.get("emotion_intensity", 3) / 5.0
            dominant_emotions.append((pad, intensity, block.get("character", "?")))

    if not dominant_emotions:
        return

    avg_pleasure = sum(e[0][0] * e[1] for e in dominant_emotions) / len(dominant_emotions)
    avg_arousal = sum(e[0][1] * e[1] for e in dominant_emotions) / len(dominant_emotions)

    # Log contagion
    contagion_entry = {
        "turn": len(s.get("turns", [])),
        "pleasure": round(avg_pleasure, 3),
        "arousal": round(avg_arousal, 3),
        "atmosphere": "따뜻한 분위기" if avg_pleasure > 0.2 else ("긴장된 분위기" if avg_pleasure < -0.2 else "평온한 분위기"),
        "sources": [e[2] for e in dominant_emotions[:3]]
    }
    s.setdefault("memory", {}).setdefault("emotional_contagion_log", []).append(contagion_entry)
    if len(s["memory"]["emotional_contagion_log"]) > 10:
        s["memory"]["emotional_contagion_log"] = s["memory"]["emotional_contagion_log"][-10:]

    # Adjust CORE-4 stress based on emotional atmosphere
    core4 = s.get("core4", {})
    stress = core4.get("stress", {})
    if avg_pleasure < -0.3:
        stress["value"] = min(stress.get("max", 100), stress.get("value", 30) + 3)
    elif avg_pleasure > 0.3:
        stress["value"] = max(stress.get("min", 0), stress.get("value", 30) - 2)


# ─── Core turn engine ─────────────────────────────────────────
def run_dima_turn(s: dict, user_input: str) -> tuple:
    """Run one D.I.M.A turn and return (final_script, pulse_result, dima_response)."""
    system_instruction, main_prompt, pulse_result = build_dima_prompt(s, user_input)

    # Use tension-based temperature
    tension_info = calculate_tension_level(s)
    temperature = tension_info.get("temperature_hint", 0.75)

    raw = generate_llm(
        prompt=main_prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        response_schema=DIMA_SCHEMA,
    )
    # V5.0: Capture raw Gemini response for token metering
    dima_response = generate_llm.last_response

    script = (raw or {}).get("script") or []

    # Validate: must have at least one dialogue block with content
    if not any(
        isinstance(b, dict) and b.get("type") == "dialogue" and b.get("content")
        for b in script
    ):
        script = safe_local_script(s, prelude_narration="")

    processed = post_process_script(script, s)

    # Part D: Apply emotional contagion
    apply_emotional_contagion(s, processed)

    return processed, pulse_result, dima_response


# =========================================================================
# MAESTRO — Memory architect + relationship/CORE-4 adjuster (every 4 turns)
# =========================================================================
def _local_maestro_fallback(recent_4: list, s: Optional[dict] = None) -> dict:
    """Regex-based local fallback when Maestro LLM call fails.
    Extracts character names, emotion tags, and keywords from the last 4 turns.
    Also generates a fallback summary from short_term memory if available."""
    chars = set()
    emotions = set()
    keywords = []
    for turn in recent_4:
        for block in turn.get("script", []):
            if block.get("type") == "dialogue":
                char = block.get("character", "")
                if char:
                    chars.add(char)
                emo = block.get("emotion", "")
                if emo:
                    emotions.add(emo)
                content = block.get("content", "")
                # Extract first meaningful phrase as keyword
                words = re.findall(r'[\w가-힣]+', content)
                if words:
                    keywords.extend(words[:3])

    chars_str = ", ".join(chars) if chars else "등장인물"
    emotions_str = ", ".join(list(emotions)[:4]) if emotions else "neutral"
    kw_str = ", ".join(list(set(keywords))[:6]) if keywords else ""

    # Build fallback from short_term memory (last 5 entries)
    summary_parts = []
    if s:
        short_term = s.get("memory", {}).get("short_term", [])
        for entry in short_term[-5:]:
            summary_parts.append(str(entry))

    if summary_parts:
        summary = " → ".join(summary_parts[-3:])
    else:
        summary = f"{chars_str} 간 대화 진행. 감정: {emotions_str}."
        if kw_str:
            summary += f" 키워드: {kw_str}"

    # Build basic relationship deltas from detected emotions
    # so axes don't stay frozen at 0 when Maestro LLM fails
    fallback_deltas = {}
    positive_emotions = {"joy", "gentle_affection", "playful_tease", "wistful_nostalgia"}
    negative_emotions = {"anger", "sadness", "nervous_tension"}
    for char in chars:
        if emotions & positive_emotions:
            fallback_deltas[char] = {"axis": "agreeable", "delta": 1, "reason": "fallback: positive interaction detected"}
        elif emotions & negative_emotions:
            fallback_deltas[char] = {"axis": "adversarial", "delta": 1, "reason": "fallback: tension detected"}

    return {
        "long_term_summary": summary,
        "core_pin": None,
        "active_thought": None,
        "relationship_deltas": fallback_deltas,
        "core4_adjustments": {"energy": 0, "stress": 0, "intoxication": 0, "pain": 0},
    }


# ─── V5.0 Maestro V2: Pre-DIMA sync pipeline ────────────────
MAESTRO_PROMPT_V2 = """[ROLE] Maestro — 서사 기억 + 연출 기획 AI.
[GOAL] 직전 턴을 분석하고, 다음 턴의 서사 방향을 기획한다.

[ABSOLUTE RULES]
1. JSON만 출력. 마크다운/설명/코드펜스 절대 금지.
2. 등장하지 않은 캐릭터는 건드리지 마라.
3. 점수 변동: 일반 ±3~5, 강한 사건만 최대 ±10.

[SCHEMA — 이것만 출력하라]
{{
  "action_context": {{
    "<캐릭터명>": "<1~2문장: 이 캐릭터가 이번 턴에서 한 행동과 느낀 감정>"
  }},
  "emotion_update": {{
    "<캐릭터명>": {{"emotion": "<영어단어>", "intensity": <1~10>}}
  }},
  "rel_delta": {{
    "<캐릭터명>_to_<대상>": {{"호감": <±정수>, "신뢰": <±정수>, "긴장": <±정수>}}
  }},
  "scene_note": "<1문장: 이번 턴의 핵심 사건/분위기 변화>",
  "next_beat": {{
    "lead_character": "<다음 턴에서 행동을 주도할 캐릭터 이름>",
    "tactic": "<그 캐릭터가 사용할 전략. 예: '서툰 칭찬으로 거리 좁히기', '짓궂은 질문으로 본심 떠보기', '갑자기 진지해지며 과거 암시'>",
    "tension_direction": "<rise | fall | maintain — 다음 턴 긴장감 방향>"
  }}
}}

[THIS TURN DATA]
- 장소: {location}
- 등장인물: {on_screen_json}
- 유저 입력: "{user_input}"
- NPC 행동:
{script_summary}

[PREVIOUS CONTEXT]
{prev_context}

JSON만 출력하라."""


def parse_maestro_response(raw_text: str) -> Optional[dict]:
    """Maestro 응답에서 JSON을 견고하게 추출"""
    if not raw_text:
        return None
    # 1차: extract_first_json_block 사용
    result = extract_first_json_block(raw_text)
    if result and isinstance(result, dict):
        return result
    # 2차: 코드펜스 제거 후 직접 파싱
    cleaned = re.sub(r'```json\s*|\s*```', '', raw_text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    logger.warning(f"Maestro JSON parse failed (length={len(raw_text)})")
    return None


def apply_maestro_to_session(s: dict, data: dict):
    """Maestro 분석 결과를 세션 상태에 반영"""
    if not data or not isinstance(data, dict):
        return

    # 1. action_context (각 캐릭터가 직전에 뭘 했는지)
    ac = data.get("action_context")
    if ac and isinstance(ac, dict):
        s["action_context"] = ac

    # 2. emotion_update (캐릭터 감정 상태)
    eu = data.get("emotion_update")
    if eu and isinstance(eu, dict):
        for char_name, emo in eu.items():
            if not isinstance(emo, dict):
                continue
            char_state = s.get("characters", {}).get(char_name, {})
            ds = char_state.setdefault("dynamic_state", {})
            if emo.get("emotion"):
                ds["current_mood"] = emo["emotion"]
            if emo.get("intensity"):
                ds["mood_intensity"] = int(emo["intensity"])

    # 3. rel_delta (관계 변화) — 성격 기반 가중치 적용
    rd = data.get("rel_delta")
    if rd and isinstance(rd, dict):
        for key, delta in rd.items():
            if not isinstance(delta, dict):
                continue
            parts = key.split("_to_")
            char_name = parts[0] if parts else key
            rel = s.get("relationships", {}).get(char_name)
            if not rel:
                continue
            velocity = get_relationship_velocity(char_name)
            mult_map = {
                "호감": velocity.get("affection_mult", 1.0),
                "신뢰": velocity.get("trust_mult", 1.0),
                "긴장": velocity.get("tension_mult", 1.0),
            }
            for axis_kr, axis_en in [("호감", "affection"), ("신뢰", "trust"), ("긴장", "tension")]:
                if axis_kr in delta:
                    try:
                        raw_d = int(delta[axis_kr])
                        weighted_d = int(round(raw_d * mult_map.get(axis_kr, 1.0)))
                        old = rel.get(axis_en, 50)
                        rel[axis_en] = max(0, min(100, old + weighted_d))
                    except (ValueError, TypeError):
                        pass
            # axes 보조 업데이트
            hg = delta.get("호감", 0)
            if isinstance(hg, (int, float)):
                if hg > 0:
                    rel.setdefault("axes", {})["agreeable"] = rel.get("axes", {}).get("agreeable", 0) + 1
                elif hg < 0:
                    rel.setdefault("axes", {})["adversarial"] = rel.get("axes", {}).get("adversarial", 0) + 1

    # 4. scene_note → 단기 메모리에 추가
    note = data.get("scene_note", "")
    if note and isinstance(note, str):
        mem = s.setdefault("memory", {})
        st = mem.setdefault("short_term", [])
        st.append(note)
        mem["short_term"] = st[-8:]  # 최근 8개 유지

    # 5. next_beat → 세션에 저장 (DIMA director brief에서 참조)
    nb = data.get("next_beat")
    if nb and isinstance(nb, dict):
        s["next_beat"] = nb


def _run_maestro_preturn(s: dict) -> Optional[dict]:
    """직전 턴을 분석하여 Maestro 메모리 생성 (동기, DIMA 호출 직전 실행)"""
    turns = s.get("turns", [])
    if not turns:
        return None
    last_turn = turns[-1]
    on_screen = s.get("on_screen", [])
    location = s.get("world", {}).get("space", {}).get("current_location", "라운지")

    # 스크립트 요약 (토큰 절약)
    script_lines = []
    for b in last_turn.get("script", []):
        if not isinstance(b, dict):
            continue
        if b.get("type") == "dialogue":
            script_lines.append(
                f"{b.get('character', '?')}({b.get('emotion', '?')}): "
                f"{b.get('content', '')[:80]}"
            )
        elif b.get("type") == "narration":
            script_lines.append(f"(narr): {b.get('content', '')[:60]}")

    prev_ac = s.get("action_context", {})
    prev_text = json.dumps(prev_ac, ensure_ascii=False)[:400] if prev_ac else "없음"

    prompt = MAESTRO_PROMPT_V2.format(
        location=location,
        on_screen_json=json.dumps(on_screen, ensure_ascii=False),
        user_input=(last_turn.get("user_input") or "(침묵)")[:100],
        script_summary="\n".join(script_lines)[:600],
        prev_context=prev_text,
    )

    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        safety_settings=get_safety_settings(),
        max_output_tokens=2048,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=512),
    )

    result = safe_gemini_call(MODEL_MAESTRO, [prompt], config, timeout_sec=15)
    if not result:
        return None

    raw = _extract_text(result)
    if not raw:
        return None

    return parse_maestro_response(raw)


def run_maestro_sync(s: dict):
    """Run Maestro analysis every 4 turns to update long-term memory, relationships, and CORE-4."""
    turns = s.get("turns", [])
    if len(turns) < 4 or len(turns) % 4 != 0:
        return

    recent_4 = turns[-4:]
    recent_text = json.dumps(
        [{"turn_id": t.get("turn_id"), "user_input": t.get("user_input", ""), "script": t.get("script", [])} for t in recent_4],
        ensure_ascii=False
    )[:6000]

    on_screen = s.get("on_screen", [])
    player_name = s.get("player_name", "사용자")
    char_names = [n for n in on_screen if n != player_name]

    maestro_prompt = SAFETY_PREAMBLE + f"""
You are the Maestro, a narrative memory architect. Analyze the last 4 turns of an interactive novel and return a structured JSON summary.

Characters in scene: {', '.join(char_names)}
Player name: {player_name}

Last 4 turns:
{recent_text}

Also analyze and return:
- "core4_adjustments": {{"energy": int, "stress": int, "intoxication": int, "pain": int}}
  Positive values = increase, negative = decrease.
  Example: if a fight happened, stress +15, pain +10. If characters had a warm meal, energy +5, stress -5.

추가 분석 사항:
1. play_style_assessment: 지금까지의 대화 흐름에서 유저의 플레이 스타일을 판단하세요 (chat/rp/novel/game/counsel)
2. style_correction: D.I.M.A에게 전달할 연기 보정 지시 (예: "나레이션을 줄이고 대화 비율을 높여라", "더 문학적으로 써라")
3. world_inject: 다음 턴에 시스템 프롬프트에 추가할 세계관 정보 (필요시에만)
4. emotional_tone_summary: 현재 대화의 감정적 톤 요약

Return JSON with:
{{
  "long_term_summary": "이번 4턴의 핵심 줄거리 1~2문장 요약",
  "core_pin": "절대 잊으면 안 되는 핵심 사건이 있으면 기록. 없으면 null",
  "active_thought": "캐릭터가 새로 깨달은 생각/의심/결심 (Disco Elysium Thought Cabinet 형식). 없으면 null",
  "relationship_deltas": {{
    "캐릭터이름": {{
      "axis": "agreeable|adversarial|open|closed|bold|passive|reliable|unreliable|insightful|oblivious",
      "delta": 1,
      "reason": "왜 이 축이 변했는지 한 줄 설명"
    }}
  }},
  "core4_adjustments": {{"energy": 0, "stress": 0, "intoxication": 0, "pain": 0}},
  "play_style_assessment": "chat|rp|novel|game|counsel",
  "style_correction": "D.I.M.A 연기 보정 지시 (없으면 빈 문자열)",
  "world_inject": "세계관 추가 정보 (없으면 빈 문자열)",
  "emotional_tone_summary": "현재 감정 톤 요약"
}}
"""

    maestro_result = None
    config = genai_types.GenerateContentConfig(
        temperature=0.4,
        max_output_tokens=2048,
        response_mime_type="application/json",
        safety_settings=get_safety_settings(),
        thinking_config=genai_types.ThinkingConfig(thinking_budget=512),
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
    )

    response = call_gemini_with_fallback(
        contents=[maestro_prompt],
        config=config,
    )
    try:
        if response is not None:
            text = _extract_text(response)
            if text:
                maestro_result = extract_first_json_block(text)
    except (AttributeError, NameError, TypeError) as e:
        logger.warning(f"Maestro response parse failed: {e}")

    # If LLM failed, use local regex-based fallback
    if maestro_result is None:
        logger.info("Maestro LLM failed, using local fallback summary")
        maestro_result = _local_maestro_fallback(recent_4, s=s)

    memory = s.setdefault("memory", {"short_term": [], "long_term": [], "core_pins": [], "emotional_contagion_log": []})

    # Long-term memory
    lt = maestro_result.get("long_term_summary")
    if lt:
        memory.setdefault("long_term", []).append(lt)
        if len(memory["long_term"]) > 15:
            memory["long_term"] = memory["long_term"][-15:]

    # Core pin
    pin = maestro_result.get("core_pin")
    if pin:
        memory.setdefault("core_pins", []).append(pin)
        if len(memory["core_pins"]) > 20:
            memory["core_pins"] = memory["core_pins"][-20:]

    # Active thought (Disco Elysium Thought Cabinet)
    thought = maestro_result.get("active_thought")
    if thought:
        memory.setdefault("active_thoughts", []).append({
            "thought": thought,
            "turn_born": len(turns),
            "matured": False
        })
        # Mature old thoughts (> 8 turns old)
        for t in memory.get("active_thoughts", []):
            if len(turns) - t.get("turn_born", 0) > 8:
                t["matured"] = True

    # Relationship deltas (Scarlet Hollow style)
    deltas = maestro_result.get("relationship_deltas", {})
    for char_name, change in deltas.items():
        rel = s.get("relationships", {}).get(char_name)
        if not rel:
            # Try matching without exact case
            for rk in s.get("relationships", {}):
                if rk.lower() == char_name.lower():
                    rel = s["relationships"][rk]
                    break
        if not rel:
            logger.debug(f"Maestro relationship delta for unknown character: {char_name}")
            continue
        axis = change.get("axis", "")
        delta_val = change.get("delta", 0)
        axes_dict = rel.get("axes", {})
        if axis in axes_dict:
            axes_dict[axis] = max(0, axes_dict[axis] + delta_val)
            logger.info(f"Maestro axes update: {char_name}.{axis} += {delta_val} → {axes_dict[axis]}")
        else:
            logger.warning(f"Maestro returned unknown axis '{axis}' for {char_name}. Valid: {list(axes_dict.keys())}")
        rel["stage"] = calculate_relationship_stage(rel)

    # CORE-4 adjustments
    adjustments = maestro_result.get("core4_adjustments", {})
    for key, delta in adjustments.items():
        stat = s.get("core4", {}).get(key)
        if stat:
            stat["value"] = max(stat["min"], min(stat["max"], stat["value"] + delta))

    # V5.0: Maestro override — style correction and world inject
    s["maestro_override"] = {
        "style_correction": maestro_result.get("style_correction", ""),
        "world_inject": maestro_result.get("world_inject", ""),
    }
    # V5.0: Play style assessment from Maestro
    assessed_style = maestro_result.get("play_style_assessment")
    if assessed_style:
        if isinstance(assessed_style, dict):
            # analyzer와 동일한 confidence threshold 적용
            if assessed_style.get("confidence", 0) >= 0.8:
                s.setdefault("user_profile", {})["play_style"] = assessed_style.get("style", s.get("user_profile", {}).get("play_style", "chat"))
            s.setdefault("maestro_override", {})["play_style_assessment"] = assessed_style
        elif isinstance(assessed_style, str) and assessed_style in ("chat", "rp", "novel", "game", "counsel"):
            s.setdefault("user_profile", {})["play_style"] = assessed_style
            s["user_profile"]["style_confidence"] = max(
                0.7, s["user_profile"].get("style_confidence", 0)
            )

    logger.info(f"Maestro completed for turn {len(turns)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_pulse_payload(pulse_result: Optional[dict], s: dict) -> dict:
    """Build pulse payload dict for frontend from pulse analysis result."""
    return {"mode": (pulse_result or {}).get("mode", "REACTIVE")}


# Part K: Health endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "V5.0",
        "model_dima": MODEL_DIMA,
        "model_maestro": MODEL_MAESTRO,
        "characters_loaded": len(ALL_CHARACTER_NAMES),
        "supported_emotions": SUPPORTED_EMOTIONS,
        "illustration_model": MODEL_ILLUSTRATION,
    })


@app.route("/gallery", methods=["GET"])
def gallery():
    """Return list of illustration URLs for the current session."""
    sid = session.get("session_id")
    if not sid:
        return jsonify({"status": "ok", "images": []})
    safe_sid = _sanitize_sid(sid)
    session_illust_dir = ILLUSTRATIONS_DIR / safe_sid
    if not session_illust_dir.is_dir():
        return jsonify({"status": "ok", "images": []})
    images = sorted(
        f"/static/illustrations/{safe_sid}/{f.name}"
        for f in session_illust_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    return jsonify({"status": "ok", "images": images})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/bootstrap", methods=["POST"])
def bootstrap():
    data = request.get_json(force=True) or {}

    # Resume existing session
    save_code = data.get("save_code", "").strip()
    if save_code:
        s = load_session(save_code)
        if not s:
            return jsonify({"status": "error", "message": "save not found"}), 404
        session["session_id"] = save_code
        s["traffic_light"] = "GREEN"
        save_session(s)
        return jsonify({"status": "ok", "sid": save_code, "state": to_public_state(s)})

    # New session
    sid = data.get("sid") or f"S-{uuid.uuid4().hex[:8].upper()}"
    s = init_session(sid)
    player_name = (data.get("player_name") or "사용자").strip()
    s["player_name"] = player_name

    user_selected = data.get("on_screen_names")
    final_cast = normalize_cast(user_selected, player_name)

    if not final_cast:
        pool = [n for n in CHARACTERS_DB.keys() if n != player_name]
        if not pool:
            return jsonify({"status": "error", "message": "No available NPCs."}), 409
        final_cast.append(random.choice(pool))

    s["on_screen"] = final_cast
    merge_ui_settings(s, data.get("ui_settings"))

    # Part C: Initialize relationships
    init_relationships(s)

    seed_text = (data.get("seed_text") or "").strip() or "[PLAYER_PAUSE]"

    # Generate first turn
    _pulse = None
    _bootstrap_response = None
    try:
        final_script, _pulse, _bootstrap_response = run_dima_turn(s, seed_text)
    except Exception as e:
        logger.error(f"Bootstrap DIMA error: {e}")
        final_script = safe_local_script(s)

    # V5.0: Record token usage for bootstrap turn
    record_token_usage(s, _bootstrap_response, 1)

    first_turn = {
        "turn_id": 1,
        "ts": now_ts(),
        "user_input": data.get("seed_text", ""),
        "script": final_script,
        "emotion": extract_emotion_from_script(final_script),
        "ui_settings_snapshot": copy.deepcopy(s.get("ui_settings", {})),
    }
    s["turns"].append(first_turn)

    # --- Flow Digest Update for bootstrap ---
    seed_text = data.get("seed_text", "")
    digest_entry = _build_event_digest(1, seed_text, final_script)
    s.setdefault("flow_digest_10", []).append(digest_entry)

    # --- Short-Term Memory Update for bootstrap ---
    short_entry = _build_event_short_term(1, seed_text, final_script)
    s.setdefault("memory", {}).setdefault("short_term", []).append(short_entry)

    # --- 캐릭터별 최근 발언 인덱스 갱신 (부트스트랩) ---
    update_character_last(s, 1, final_script)

    # --- 캐릭터 참석 기록 갱신 (부트스트랩) ---
    update_character_presence(s, 1)

    s["traffic_light"] = "GREEN"
    session["session_id"] = sid
    update_all_relationship_stages(s)
    save_session(s)

    resp = {"status": "ok", "sid": sid, "state": to_public_state(s),
            "personal_colors": PERSONAL_COLORS,
            "pulse": _build_pulse_payload(_pulse, s)}
    return jsonify(resp)


@app.route("/execute-turn", methods=["POST"])
def execute_turn():
    sid = session.get("session_id")
    if not sid:
        return jsonify({"status": "error", "message": "세션이 만료되었습니다. 새로고침 해주세요."}), 401

    data = request.get_json(force=True, silent=True) or {}
    user_input = (
        data.get("user_input")
        or data.get("user_text")
        or data.get("text")
        or ""
    ).strip()

    if not user_input:
        user_input = "[CONTINUE_SCENE]"

    with get_session_lock(sid):
        s = load_session(sid)
        if not s:
            return jsonify({"status": "error", "message": "세션을 찾을 수 없습니다."}), 404

        # V5.0: Session migration (ensure new fields exist)
        s = _migrate_session(s)

        merge_ui_settings(s, data.get("ui_settings") or {})

        me = get_player_name(s)
        if not s.get("on_screen"):
            pool = [n for n in CHARACTERS_DB.keys() if n != me]
            if not pool:
                return jsonify({"status": "error", "message": "사용 가능한 NPC가 없습니다."}), 409
            s["on_screen"] = [random.choice(pool)]

        # Part C: Ensure relationships initialized
        init_relationships(s)

        # V5.0: Step 1 — Intent classification (no API call)
        profile = s.get("user_profile", {})
        intent_result = classify_user_intent(user_input, profile)
        update_user_profile(profile, intent_result, user_input)
        logger.info(f"Intent: {intent_result['play_style']} (conf={intent_result['style_confidence']:.2f}), heavy={intent_result['is_heavy_input']}")

        # V5.0: Step 2 — Heavy Input → AI Analyzer (conditional, with retry guard)
        analyzer_cache = s.get("analyzer_cache")
        turn_number = len(s.get("turns", []))
        should_run_analyzer = (
            intent_result["is_heavy_input"]
            and (
                not analyzer_cache
                or (
                    analyzer_cache.get("_error")
                    and turn_number - analyzer_cache.get("_error_turn", 0) >= analyzer_cache.get("_retry_after_turns", ANALYZER_RETRY_INTERVAL)
                )
            )
        )
        if should_run_analyzer:
            logger.info("Heavy input detected, running AI Analyzer...")
            analyzer_result = run_ai_analyzer(user_input, s.get("on_screen", []))
            if analyzer_result:
                if analyzer_result.get("_error"):
                    analyzer_result["_error_turn"] = turn_number
                    s["analyzer_cache"] = analyzer_result
                else:
                    s["analyzer_cache"] = analyzer_result
                    recommended = analyzer_result.get("recommended_play_style")
                    if recommended:
                        s["user_profile"]["play_style"] = recommended
                        s["user_profile"]["style_confidence"] = max(
                            0.8, s["user_profile"].get("style_confidence", 0)
                        )
                        logger.info(f"Analyzer override: play_style → {recommended}")

        # Part B: Apply CORE-4 natural decay
        apply_core4_decay(s)

        # === Maestro V2 동기 실행 — STEP 11: 적응형 호출 빈도 ===
        if should_call_maestro(s, user_input):
            try:
                maestro_data = _run_maestro_preturn(s)
                if maestro_data:
                    apply_maestro_to_session(s, maestro_data)
                    logger.info(f"Maestro called at turn {len(s['turns'])}")
            except Exception as e:
                logger.warning(f"Maestro preturn failed: {e}")
                # 실패해도 DIMA는 진행 (기존 context 사용)

        # V5.0: Step 3 — D.I.M.A turn (uses build_adaptive_instruction internally)
        final_script, pulse_result, dima_response = run_dima_turn(s, user_input)

        # V5.0: Step 4 — Token metering
        turn_number = len(s.get("turns", []))
        current_turn_usage = record_token_usage(s, dima_response, turn_number)

        # Increment scene turn counter
        sc = s.setdefault("scene_context", {})
        sc["turn_count_in_scene"] = sc.get("turn_count_in_scene", 0) + 1

        turn_id = (len(s.get("turns", [])) + 1)
        turn_payload = {
            "turn_id": turn_id,
            "ts": now_ts(),
            "user_input": user_input if user_input not in ("[CONTINUE_SCENE]", "[PLAYER_PAUSE]") else "",
            "script": final_script,
            "emotion": extract_emotion_from_script(final_script),
            "ui_settings_snapshot": copy.deepcopy(s.get("ui_settings", {})),
        }

        # ─── 삽화 생성 ────────────────────────
        on_screen = s.get("on_screen", [])
        illustration_url = None
        if s.get("ui_settings", {}).get("illustration", False):
            scene_desc_parts = []
            for block in final_script:
                if block.get("type") == "narration":
                    scene_desc_parts.append(block.get("content", ""))
                elif block.get("type") == "dialogue":
                    scene_desc_parts.append(
                        f"{block.get('character','')}: {block.get('content','')[:50]}"
                    )
            scene_desc = " ".join(scene_desc_parts)[:500]
            primary_emotion = extract_emotion_from_script(final_script)
            primary_char = next(
                (b.get("character") for b in final_script if b.get("type") == "dialogue"),
                on_screen[0] if on_screen else None
            )
            ref_slug = ENG_SLUG_MAP.get(primary_char) if primary_char else None
            illustration_url = generate_illustration(
                scene_description=scene_desc,
                character_names=on_screen,
                emotion=primary_emotion,
                session_id=s["session_id"],
                turn_number=len(s.get("turns", [])),
                reference_slug=ref_slug,
            )

        turn_payload["illustration_url"] = illustration_url
        s.setdefault("turns", []).append(turn_payload)

        # --- Flow Digest Update (event-based) ---
        digest_entry = _build_event_digest(turn_id, user_input, final_script)
        s.setdefault("flow_digest_10", []).append(digest_entry)
        if len(s["flow_digest_10"]) > 10:
            s["flow_digest_10"] = s["flow_digest_10"][-10:]

        # --- Short-Term Memory Update (event summary) ---
        short_entry = _build_event_short_term(turn_id, user_input, final_script)
        s.setdefault("memory", {}).setdefault("short_term", []).append(short_entry)
        if len(s["memory"]["short_term"]) > 20:
            s["memory"]["short_term"] = s["memory"]["short_term"][-20:]

        # --- STEP 5: 3-Tier Memory 갱신 ---
        update_memory_tiers(s, user_input, final_script)

        # --- 캐릭터별 최근 발언 인덱스 갱신 ---
        update_character_last(s, turn_id, final_script)

        # --- 캐릭터 참석 기록 갱신 ---
        update_character_presence(s, turn_id)

        # Recalculate relationship stages
        update_all_relationship_stages(s)

        trim_turns_after_maestro(s)

        s["traffic_light"] = "GREEN"
        save_session(s)

    # V5.0: Step 5 — Include token_usage in response
    resp = {"status": "ok", "sid": sid, "state": to_public_state(s),
            "personal_colors": PERSONAL_COLORS,
            "pulse": _build_pulse_payload(pulse_result, s),
            "illustration_url": illustration_url,
            "token_usage": {
                "this_turn": current_turn_usage,
                "session_total": {
                    "input": s["token_ledger"]["total_input_tokens"],
                    "output": s["token_ledger"]["total_output_tokens"],
                    "cached": s["token_ledger"]["total_cached_tokens"],
                }
            }}
    return jsonify(resp)


# Part B: Hot-swap endpoint
@app.route("/hot-swap", methods=["POST"])
def hot_swap():
    sid = session.get("session_id")
    if not sid:
        return jsonify({"error": "No active session"}), 400
    lock = get_session_lock(sid)
    with lock:
        s = load_session(sid)
        if not s:
            return jsonify({"error": "Session not found"}), 404
        payload = request.get_json(silent=True) or {}
        # core4 안전 업데이트
        incoming_core4 = payload.get("core4")
        if incoming_core4 and isinstance(incoming_core4, dict):
            for key in ("energy", "intoxication", "stress", "pain"):
                if key in incoming_core4:
                    val = incoming_core4[key]
                    if isinstance(val, (int, float)):
                        s["core4"][key]["value"] = max(
                            s["core4"][key]["min"],
                            min(s["core4"][key]["max"], int(val))
                        )
                    elif isinstance(val, dict) and "value" in val:
                        s["core4"][key]["value"] = max(
                            s["core4"][key]["min"],
                            min(s["core4"][key]["max"], int(val["value"]))
                        )
        # ui_settings 업데이트
        merge_ui_settings(s, payload.get("ui_settings"))
        save_session(s)
        return jsonify({"status": "ok", "state": to_public_state(s)})


@app.route("/get-character-profiles", methods=["GET"])
def get_character_profiles():
    return jsonify([
        {
            "name": n,
            "eng": CHARACTERS_DB.get(n, {}).get("metadata", {}).get("eng", n.lower()),
            "color": PERSONAL_COLORS.get(n, "#666666"),
        }
        for n in ALL_CHARACTER_NAMES
    ])


@app.route("/get-session-data", methods=["GET"])
def get_session_data():
    sid = session.get("session_id")
    if not sid:
        return jsonify({"status": "no_session"})
    s = load_session(sid)
    if s:
        return jsonify({"status": "ok", "state": to_public_state(s)})
    return jsonify({"status": "no_session"})


@app.route("/reset-session", methods=["GET"])
def reset_session_route():
    sid = session.get("session_id")
    if sid:
        with get_session_lock(sid):
            try:
                session_path(sid).unlink(missing_ok=True)
            except Exception:
                pass
        session.clear()
    return jsonify({"status": "ok"})


@app.route("/set_on_screen", methods=["POST"])
def set_on_screen():
    sid = session.get("session_id")
    if not sid:
        return jsonify({"status": "error", "message": "세션이 없습니다."}), 400

    data = request.get_json(force=True, silent=True) or {}
    names = data.get("on_screen_names", [])

    with get_session_lock(sid):
        s = load_session(sid)
        if not s:
            return jsonify({"status": "error", "message": "세션을 찾을 수 없습니다."}), 404

        player_name = get_player_name(s)
        s["on_screen"] = normalize_cast(names, player_name)
        init_relationships(s)
        save_session(s)

    return jsonify({"status": "ok", "on_screen": s["on_screen"]})


@app.route("/generate-illustration", methods=["POST"])
def generate_illustration():
    sid = session.get("session_id")
    if not sid:
        return jsonify({"status": "error", "message": "No session"}), 400

    s = load_session(sid)
    if not s or not s.get("turns"):
        return jsonify({"status": "error", "message": "No turns yet"}), 400

    last_turn = s["turns"][-1]
    narration_parts = [b["content"] for b in last_turn.get("script", []) if b.get("type") == "narration"]
    scene_desc = " ".join(narration_parts)[:300] if narration_parts else "기숙사 라운지의 따뜻한 오후"

    # Character appearances
    char_looks = []
    for name in s.get("on_screen", [])[:3]:
        if name == s.get("player_name"):
            continue
        char_db = CHARACTERS_DB.get(name, {})
        appearance = char_db.get("appearance", {})
        if isinstance(appearance, dict):
            brief = ", ".join(f"{k}: {v}" for k, v in list(appearance.items())[:5])
            char_looks.append(f"{name}({brief[:100]})")
        elif isinstance(appearance, str):
            char_looks.append(f"{name}({appearance[:100]})")

    # Emotion from last dialogue
    last_emotion = "gentle_affection"
    for b in reversed(last_turn.get("script", [])):
        if b.get("type") == "dialogue" and b.get("emotion"):
            last_emotion = normalize_emotion_tag(b["emotion"])
            break
    emotion_kr = EMOTION_TAXONOMY.get(last_emotion, {}).get("kr", "잔잔한")

    illustration_prompt = (
        f"한국 판타지 라이트노벨 삽화 스타일. 부드러운 수채화 터치, 따뜻한 조명. "
        f"장면: {scene_desc} "
        f"등장인물: {'; '.join(char_looks) if char_looks else '기숙사 전경'}. "
        f"전체 분위기: {emotion_kr}. "
        f"텍스트 없음, 워터마크 없음, 고품질 일러스트."
    )

    try:
        import base64
        response = call_gemini_with_timeout(
            model="gemini-2.5-flash-image",
            contents=[illustration_prompt],
            config=genai_types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                safety_settings=get_safety_settings(),
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            ),
            timeout_sec=60,
        )
        if response is None:
            return jsonify({"status": "error", "message": "삽화 생성 시간 초과"}), 504
        for part in response.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                img_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                return jsonify({
                    "status": "ok",
                    "image_base64": img_b64,
                    "mime_type": getattr(part.inline_data, 'mime_type', "image/png"),
                    "prompt_used": illustration_prompt
                })
        return jsonify({"status": "error", "message": "No image in response"}), 500
    except Exception as e:
        logger.warning(f"Illustration failed: {e}")
        return jsonify({"status": "error", "message": "삽화 생성에 실패했습니다."}), 500


NOVELIZE_PROMPT = """
다음은 인터랙티브 소설의 대화 스크립트입니다. 이것을 한국어 문학 소설로 변환하세요.

규칙:
1. 모든 대사를 자연스러운 소설 문체로 풀어쓰세요 (큰따옴표 사용)
2. narration 블록은 더 풍부한 묘사로 확장하세요
3. monologue는 캐릭터의 내면 독백으로 자연스럽게 삽입하세요
4. dialogue의 emotion을 행동 묘사로 표현하세요 (예: playful_tease → 입꼬리를 올리며)
5. 플레이어의 입력도 소설의 일부로 자연스럽게 통합하세요
6. 각 청크의 시작에 간단한 장면 전환 문구를 넣으세요

출력: 순수한 소설 텍스트만 반환. JSON이나 마크다운 코드블록 없이 순수 텍스트.
"""


@app.route("/novelize", methods=["POST"])
def novelize():
    sid = session.get("session_id")
    if not sid:
        return jsonify({"status": "error", "message": "세션이 없습니다."}), 400

    s = load_session(sid)
    if not s or not s.get("turns"):
        return jsonify({"status": "error", "message": "턴이 없습니다."}), 400

    data = request.get_json(force=True, silent=True) or {}
    turn_ids = data.get("turn_ids", [])
    all_turns = s.get("turns", [])

    # Filter turns by turn_ids if provided
    if turn_ids:
        turns = [t for t in all_turns if t.get("turn_id") in turn_ids]
    else:
        turns = all_turns

    if not turns:
        return jsonify({"status": "error", "message": "턴이 없습니다."}), 400

    chunk_size = 5
    chunks_result = []

    try:
        for chunk_idx, chunk_start in enumerate(range(0, len(turns), chunk_size)):
            chunk = turns[chunk_start:chunk_start + chunk_size]
            first_tid = chunk[0].get("turn_id", chunk_start + 1)
            last_tid = chunk[-1].get("turn_id", chunk_start + len(chunk))

            chunk_data = []
            for t in chunk:
                entry = {
                    "turn_id": t.get("turn_id"),
                    "user_input": t.get("user_input", ""),
                    "script": t.get("script", []),
                }
                chunk_data.append(entry)

            chunk_text = json.dumps(chunk_data, ensure_ascii=False)
            chunk_prompt = (
                f"{NOVELIZE_PROMPT}\n\n"
                f"### 턴 {first_tid}~{last_tid} 스크립트 ###\n"
                f"{chunk_text}"
            )

            novelize_config = genai_types.GenerateContentConfig(
                safety_settings=get_safety_settings(),
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            )
            chunk_response = call_gemini_with_fallback(
                contents=[chunk_prompt],
                config=novelize_config,
            )
            if chunk_response is None:
                logger.warning(f"Novelize chunk {chunk_idx+1} timed out")
                chunks_result.append({
                    "chunk_id": chunk_idx + 1,
                    "turns": f"{first_tid}-{last_tid}",
                    "text": "[이 구간은 생성에 실패하여 생략되었습니다]",
                })
                continue
            result_text = (chunk_response.text or "").strip()
            # Strip markdown code fences if present
            result_text = re.sub(r'^```[a-z]*\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)

            chunks_result.append({
                "chunk_id": chunk_idx + 1,
                "turns": f"{first_tid}-{last_tid}",
                "text": result_text,
            })

        return jsonify({
            "status": "ok",
            "chunks": chunks_result,
            "total_chunks": len(chunks_result),
        })
    except Exception as e:
        logger.warning(f"Novelize failed: {e}")
        # Return partial results if any chunks succeeded
        if chunks_result:
            return jsonify({
                "status": "partial",
                "chunks": chunks_result,
                "total_chunks": len(chunks_result),
                "message": "일부 청크 처리 중 오류가 발생했습니다.",
            })
        return jsonify({"status": "error", "message": "소설화에 실패했습니다."}), 500


@app.route("/novelization", methods=["POST"])
def novelization():
    sid = session.get("session_id")
    if not sid:
        return jsonify({"error": "No active session"}), 400
    lock = get_session_lock(sid)
    with lock:
        s = load_session(sid)
        if not s:
            return jsonify({"error": "Session not found"}), 404
    chapters = []
    for i, turn in enumerate(s.get("turns", [])):
        script = turn.get("script", [])
        lines = []
        for block in script:
            btype = block.get("type", "")
            content = block.get("content", "")
            char = block.get("character", "")
            if btype == "narration":
                lines.append(content)
            elif btype == "dialogue":
                lines.append(f'"{content}" \u2014 {char}')
            elif btype == "monologue":
                lines.append(f'({content})')
        chapters.append({
            "turn": i + 1,
            "text": "\n".join(lines),
            "illustration_url": turn.get("illustration_url"),
        })
    novel_data = {
        "title": f"\uc0ac\ub8e8\ub77c \uc544\ub730\ub9ac\uc5d0 \u2014 {s.get('session_id', '')}",
        "player_name": s.get("player_name", "\uc0ac\uc6a9\uc790"),
        "characters": s.get("on_screen", []),
        "core_memories": s.get("memory", {}).get("core_pins", []),
        "chapters": chapters,
        "created_at": now_ts(),
    }
    return jsonify({"novel": novel_data})


@app.route("/share-novel", methods=["POST"])
def share_novel():
    data = request.get_json(silent=True) or {}
    novel = data.get("novel")
    if not novel:
        return jsonify({"error": "No novel data"}), 400
    share_code = f"NV-{uuid.uuid4().hex[:8].upper()}"
    share_path = SHARED_NOVELS_DIR / f"{share_code}.json"
    try:
        share_path.write_text(
            json.dumps(novel, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to save shared novel: {e}")
        return jsonify({"error": "Save failed"}), 500
    return jsonify({"share_code": share_code})


@app.route("/read-novel/<share_code>", methods=["GET"])
def read_novel(share_code):
    if not re.fullmatch(r'NV-[A-F0-9]{8}', share_code):
        return jsonify({"error": "Invalid share code"}), 400
    filename = f"{share_code}.json"
    share_path = SHARED_NOVELS_DIR / filename
    if not share_path.is_file():
        return jsonify({"error": "Novel not found"}), 404
    try:
        novel = json.loads(share_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to read shared novel: {e}")
        return jsonify({"error": "Read failed"}), 500
    return jsonify({"novel": novel})


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500


# ─── Run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from waitress import serve
        logger.info("Sarura Atelier V5.0 — Waitress on port 5000")
        serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    except ImportError:
        logger.info("Sarura Atelier V5.0 — Flask dev server on port 5000")
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
