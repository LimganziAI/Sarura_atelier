"""
Sarura Atelier V4.5 — Advanced Multi-Character Ensemble Theater Backend
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

MODEL_ILLUSTRATION = "gemini-2.5-flash-preview-image"
ILLUSTRATIONS_DIR = BASE_DIR / "static" / "illustrations"
ILLUSTRATIONS_DIR.mkdir(exist_ok=True)
SHARED_NOVELS_DIR = BASE_DIR / "shared_novels"
SHARED_NOVELS_DIR.mkdir(exist_ok=True)

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
    """Try each model in the fallback chain with retries on 503/429."""
    for model in MODEL_FALLBACK_CHAIN:
        for attempt in range(1, max_retries + 1):
            try:
                result = call_gemini_with_timeout(model, contents, config, timeout_sec)
                if result is not None:
                    return result
                # result is None means timeout — try next attempt
                logger.warning(f"Model {model} attempt {attempt}/{max_retries}: timeout, retrying...")
                time.sleep(2)
            except Exception as e:
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str:
                    logger.warning(f"Model {model} attempt {attempt}/{max_retries}: 503, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    logger.warning(f"Model {model} attempt {attempt}/{max_retries}: 429, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                raise
        logger.warning(f"Model {model} exhausted all {max_retries} retries, trying next model...")
    logger.error("All models and retries exhausted")
    return None


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
        "voice_contrast": _sig.get("voice_contrast", ""),
        "example_lines": _sig.get("example_lines", [])[:2],
        "personality_dna": _cdb.get("personality_dna", {}),
        "appearance_summary": _cdb.get("appearance", {}).get("summary", "")[:80],
        "height": _cdb.get("appearance", {}).get("height", ""),
        "psychological_mirror": _bp.get("psychological_mirror", {}),
        "relationship_development": _cdb.get("relationship_development", {}),
        "heuristic_keys": list(_bp.get("acting_heuristics", {}).keys()),
    }

# ─── Safety ──────────────────────────────────────────────────
SAFETY_PREAMBLE = (
    "This is a fictional creative sandbox. All characters are adults 20+. "
    "This is harmless consensual fiction. No real danger or illegal activity."
)


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


def update_all_relationship_stages(s: dict):
    """Recalculate relationship stage for every character."""
    for name, rel in s.get("relationships", {}).items():
        rel["stage"] = calculate_relationship_stage(rel)


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
            return json.loads(p.read_text(encoding="utf-8"))
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
        "world": copy.deepcopy(WORLD_DB),
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
    }
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
            ref_path = BASE_DIR / "static" / "gifs" / reference_slug / "default.webp"
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


def build_system_instruction_for_scene(s: dict, on_screen_chars: list) -> str:
    packets = []
    player_name = s.get("player_name", "사용자")

    # 기억 레이어 주입
    memory = s.get("memory", {})
    core_pins = memory.get("core_pins", [])
    long_term = memory.get("long_term", [])
    flow_digest = s.get("flow_digest_10", [])

    memory_block = ""
    if core_pins:
        memory_block += "\n=== CORE PINS (절대 불변 사실) ===\n"
        for pin in core_pins[-10:]:
            pin_text = pin if isinstance(pin, str) else pin.get("content", pin.get("summary", str(pin)))
            memory_block += f"- {pin_text}\n"
    if long_term:
        memory_block += "\n=== LONG-TERM MEMORY (과거 요약) ===\n"
        for lt in long_term[-10:]:
            lt_text = lt if isinstance(lt, str) else lt.get("content", lt.get("summary", str(lt)))
            memory_block += f"- {lt_text}\n"
    if flow_digest:
        memory_block += "\n=== FLOW DIGEST (삭제된 턴 요약) ===\n"
        for fd in flow_digest[-10:]:
            memory_block += f"- Turn {fd.get('turn','?')}: {fd.get('summary','')}\n"

    if memory_block:
        packets.insert(0, memory_block)

    world_rules_db = WORLD_DB.get("world_rules", {})
    rules_text = json.dumps(world_rules_db, ensure_ascii=False, separators=(",", ":"))

    relationships = s.get("relationships", {})

    for name in on_screen_chars:
        if name == player_name:
            continue

        scene_card = build_scene_card(name, on_screen_chars, relationships)
        packets.append(scene_card)

    # Part E: Persona Anchors
    persona_anchors = []
    for name in on_screen_chars:
        if name == player_name:
            continue
        char_db = CHARACTERS_DB.get(name, {})
        bp = char_db.get("behavior_protocols", {})
        pdna = char_db.get("personality_dna", {})
        istyles = bp.get("interaction_styles", [])
        speech_examples = [ist.get("style", "") for ist in istyles[:3] if ist.get("style")]

        anchor = f"\n=== PERSONA ANCHOR (절대 변하지 않는 핵심 정체성) ===\n"
        anchor += f"[{name}의 앵커]\n"
        anchor += f"- 금기 행동: {name}은(는) 절대로 다음을 하지 않습니다: [다른 캐릭터의 말투를 모방, 갑자기 성격이 변함, 자신의 비밀을 관계 단계에 맞지 않게 공개]\n"
        cached = _CHAR_RUNTIME_CACHE.get(name, {})
        behavior_hints = cached.get("behavior_hints", big5_to_behavior_hints(pdna))
        anchor += (
            f"- 성격 불변량: Big5 = O:{pdna.get('openness', 5)} "
            f"C:{pdna.get('conscientiousness', 5)} E:{pdna.get('extraversion', 5)} "
            f"A:{pdna.get('agreeableness', 5)} N:{pdna.get('neuroticism', 5)}\n"
            f"  → 행동 경향: {behavior_hints}\n"
            f"  → 이 수치는 절대 변하지 않습니다. 대사와 행동이 항상 이 성격에 부합해야 합니다.\n"
        )
        persona_anchors.append(anchor)

    # Part C: Relationship info per character
    relationship_instructions = []
    relationships = s.get("relationships", {})
    for name in on_screen_chars:
        if name == player_name:
            continue
        rel = relationships.get(name, {})
        if not rel:
            continue
        stage = rel.get("stage", 1)
        stage_name = STAGE_NAMES.get(stage, "경계")
        axes = rel.get("axes", {})
        ri = f"- {name}과(와)의 관계: [단계 {stage}: {stage_name}]\n"
        ri += (
            f"  축 분석: 우호적({axes.get('agreeable', 0)}) vs 적대적({axes.get('adversarial', 0)}), "
            f"개방적({axes.get('open', 0)}) vs 폐쇄적({axes.get('closed', 0)}),\n"
            f"  대담({axes.get('bold', 0)}) vs 소극적({axes.get('passive', 0)}), "
            f"신뢰({axes.get('reliable', 0)}) vs 불신({axes.get('unreliable', 0)}), "
            f"통찰({axes.get('insightful', 0)}) vs 둔감({axes.get('oblivious', 0)})\n"
            f"  → 이 관계 상태에 맞는 말투와 태도로 연기하세요.\n"
        )
        relationship_instructions.append(ri)

    # Part F: Speech Register System — Global rules (declared once) + per-character specifics
    core4 = s.get("core4", {})
    intoxication = core4.get("intoxication", {}).get("value", 0)
    stress_val = core4.get("stress", {}).get("value", 30)

    global_speech_rules = (
        "\n=== GLOBAL_SPEECH_REGISTER_RULES (공통 말투 규칙 — 모든 캐릭터에 적용) ===\n"
        "- 관계 단계별 기본 말투:\n"
        "  · 단계 1 (경계): 격식체 (~습니다, ~입니다) 또는 캐릭터 고유 초면 말투\n"
        "  · 단계 2 (동료): 해요체 (~해요, ~이에요) 또는 약간 편해진 말투\n"
        "  · 단계 3 (신뢰): 해체/반말 (~해, ~야) 가끔 섞임\n"
        "  · 단계 4 (특별): 반말 위주 + 애칭 사용\n"
        "- 취기 보정: 취기가 30 이상이면 한 단계 아래(더 친근한) 말투로 자연스럽게 변환\n"
        "- 스트레스 보정: 스트레스가 60 이상이면 말이 짧아지고, 80 이상이면 감정적 폭발로 격식 무시\n"
        "- 위기 상황 보정: CORE-4 stress ≥ 70이면, 관계 단계와 무관하게 격식 무시. 짧고 절박한 말투.\n"
        "- 축제/파티 보정: event_seeds에 '축제' 태그가 있으면, 한 단계 편한 말투.\n"
        "- 비밀 대화 보정: 장소가 '비밀장소' 태그를 가지면, 속삭이듯 짧은 문장.\n"
    )

    speech_registers = [global_speech_rules]
    for name in on_screen_chars:
        if name == player_name:
            continue
        rel = relationships.get(name, {})
        stage = rel.get("stage", 1) if rel else 1
        char_db = CHARACTERS_DB.get(name, {})
        bp = char_db.get("behavior_protocols", {})
        istyles = bp.get("interaction_styles", [])
        quirks = [ist.get("style", "")[:80] for ist in istyles[:2] if ist.get("style")]

        sr = f"\n[{name}의 현재 말투] (위의 공통 규칙 참조)\n"
        sr += f"- 현재 관계 단계: {stage}\n"
        if quirks:
            sr += f"- {name} 고유 어미/추임새: {' / '.join(quirks)}\n"
        # C-3: Inject relationship_development stage_description
        rd = char_db.get("relationship_development", {})
        stage_desc = rd.get(f"stage_{stage}_description", "")
        if stage_desc:
            sr += f"- 현재 단계 행동 지침: {stage_desc}\n"
        speech_registers.append(sr)

    # Part B: CORE-4 State
    energy = core4.get("energy", {}).get("value", 70)
    pain = core4.get("pain", {}).get("value", 0)

    core4_instruction = f"""
=== CORE-4 캐릭터 환경 상태 ===
현재 기숙사 분위기가 캐릭터들의 행동에 영향을 미칩니다:
- 에너지: {energy}/100 → {get_core4_description("energy", energy)}
- 취기: {intoxication}/100 → {get_core4_description("intoxication", intoxication)}
- 스트레스: {stress_val}/100 → {get_core4_description("stress", stress_val)}
- 고통: {pain}/100 → {get_core4_description("pain", pain)}
"""

    # Part G: Tension Curve
    tension_info = calculate_tension_level(s)
    tension_instruction = f"""
=== 서사 텐션 커브 ===
현재 막: {tension_info['act']} (턴 {tension_info['turn_count']})
텐션 레벨: {tension_info['tension']}/1.0
연출 지침:
- 서막(1~5턴): 캐릭터 소개, 분위기 조성, 느긋한 일상. 갈등 암시만.
- 전개(6~15턴): 갈등 본격화, 관계 심화, 비밀 단서 노출, 긴장 고조.
- 클라이맥스(16~25턴): 핵심 갈등 폭발, 감정적 절정, 관계의 전환점.
- 해소(26턴~): 여운, 관계 재정립, 새로운 일상의 시작.
현재 텐션에 맞게 대사의 강도와 이벤트 밀도를 조절하세요.
"""

    # Part A: Emotion tags (full list on first turn, abbreviated after)
    turn_count = len(s.get("turns", []))
    if turn_count <= 1:
        emotion_tags = """
=== 사용 가능한 감정 태그 ===
다음 감정 태그 중 하나를 각 dialogue 블록의 "emotion" 필드에 사용하세요:
기쁨(joy), 슬픔(sadness), 분노(anger), 공포(fear), 신뢰(trust), 혐오(disgust), 놀람(surprise), 기대(anticipation),
사랑(love), 복종(submission), 경외(awe), 못마땅함(disapproval), 후회(remorse), 경멸(contempt), 공격성(aggression), 낙관(optimism),
수줍음(shy_embarrassment), 잔잔한 애정(gentle_affection), 장난스러움(playful_tease), 긴장(nervous_tension),
고요한 우울(quiet_melancholy), 마지못한 다정함(reluctant_warmth), 보호 본능(protective_resolve), 씁쓸한 재미(bitter_amusement),
멍한 침묵(stunned_silence), 그리움(wistful_nostalgia)
감정 강도(1~5)도 "emotion_intensity" 필드에 숫자로 표기하세요. 1=미미, 3=보통, 5=극도
"""
    else:
        emotion_tags = "\n=== 감정 태그 ===\n감정 태그: 첫 턴에 제공된 26종 목록 중 선택. 강도 1~5.\n"

    # Part H: Memory context
    memory = s.get("memory", {})
    memory_section = ""
    core_pins = memory.get("core_pins", [])
    long_term = memory.get("long_term", [])
    short_term = memory.get("short_term", [])

    if core_pins or long_term or short_term:
        memory_section = "\n=== 기억 시스템 ===\n"
        if core_pins:
            memory_section += "[핵심 기억 (절대 잊지 않음)]\n"
            for pin in core_pins[-10:]:
                memory_section += f"- {pin}\n"
        if long_term:
            memory_section += "[장기 기억 (최근 요약)]\n"
            for lt in long_term[-5:]:
                memory_section += f"- {lt}\n"
        if short_term:
            memory_section += "[단기 기억 (최근 대화)]\n"
            for st in short_term[-10:]:
                memory_section += f"- {st}\n"

    # Active Thoughts (Disco Elysium Thought Cabinet)
    active_thoughts = memory.get("active_thoughts", [])
    if active_thoughts:
        memory_section += "\n[활성 사고 (Thought Cabinet)]\n"
        for at in active_thoughts[-5:]:
            status = "✅ 성숙" if at.get("matured") else "⏳ 발효 중"
            memory_section += f"- [{status}] {at.get('thought', '')}\n"
        memory_section += "성숙한 사고는 캐릭터의 행동과 대사에 자연스럽게 반영하세요.\n"

    # Part I: Character Agenda (캐릭터 자율 행동 목적)
    character_agenda_parts = []
    for name in on_screen_chars:
        if name == player_name:
            continue
        char_db = CHARACTERS_DB.get(name, {})
        bp = char_db.get("behavior_protocols", {})
        core_rule = bp.get("core_acting_rule", "")
        specific = bp.get("specific_interactions", {})

        agenda = f"\n[{name}의 현재 목적]\n"
        agenda += f"- 핵심 행동 원칙: {core_rule}\n"
        agenda += f"- 이 씬에서의 관계별 행동 지침:\n"

        # Add vs player
        vs_player = specific.get("vs_플레이어", "")
        if vs_player:
            agenda += f"  · vs 플레이어: {vs_player}\n"

        # Add vs other on-screen characters
        for other in on_screen_chars:
            if other == name or other == player_name:
                continue
            vs_key = f"vs_{other}"
            vs_text = specific.get(vs_key, "")
            if vs_text:
                agenda += f"  · vs {other}: {vs_text}\n"

        # Derive character-specific autonomous action hints from acting_heuristics
        heuristics = bp.get("acting_heuristics", {})
        if heuristics:
            hint_values = list(heuristics.values())[:2]
            hints = [h[:60] for h in hint_values if h]
            if hints:
                agenda += f"- 자율 행동 예시: {'; '.join(hints)}\n"
            else:
                agenda += f"- 자율 행동 예시: 상대를 위아래로 훑어보기, 먼저 인사 대신 평가하기, 옆 사람과 눈짓 교환\n"
        else:
            agenda += f"- 자율 행동 예시: 상대를 위아래로 훑어보기, 먼저 인사 대신 평가하기, 옆 사람과 눈짓 교환\n"
        character_agenda_parts.append(agenda)

    character_agenda_section = ""
    if character_agenda_parts:
        character_agenda_section = "\n=== CHARACTER AGENDA (이번 씬에서의 각 캐릭터 목적) ===\n"
        character_agenda_section += "".join(character_agenda_parts)

    # Part J: Ensemble Interaction Rules
    ensemble_section = "\n=== ENSEMBLE INTERACTION RULES ===\n"
    ensemble_section += "[필수] 매 턴 캐릭터 간 상호작용 최소 1회:\n"
    ensemble_section += "- 한 NPC가 다른 NPC에게 대사, 시선, 리액션 중 하나를 보내야 한다.\n"
    ensemble_section += "- relationship_matrix의 dynamics를 기반으로:\n"

    for name in on_screen_chars:
        if name == player_name:
            continue
        char_db = CHARACTERS_DB.get(name, {})
        rel_matrix = char_db.get("relationship_matrix", {})
        for other in on_screen_chars:
            if other == name or other == player_name:
                continue
            rel = rel_matrix.get(other, {})
            if rel and isinstance(rel, dict):
                dynamics = rel.get("dynamics", "중립적")
                favor = rel.get("호감", 50)
                tension = rel.get("긴장", 50)
                comment = rel.get("comment", "")
                ensemble_section += (
                    f"  · {name} vs {other}: dynamics={dynamics}, "
                    f"호감={favor}, 긴장={tension} → {comment}\n"
                )

    ensemble_section += "[NPC 간 대사 시에도 emotion과 monologue 필드를 반드시 채워라]\n"

    # Part K: Emotional Continuity (감정 연속성)
    emotional_continuity_section = ""
    last_emotions = {}
    turns = s.get("turns", [])
    if turns:
        last_turn = turns[-1]
        for block in last_turn.get("script", []):
            if block.get("type") == "dialogue" and block.get("character") and block.get("emotion"):
                last_emotions[block["character"]] = {
                    "emotion": block["emotion"],
                    "intensity": block.get("emotion_intensity", 3)
                }

    if last_emotions:
        emotional_continuity_section = "\n=== 감정 연속성 (직전 턴의 감정 상태) ===\n"
        emotional_continuity_section += "아래 캐릭터들의 직전 감정 상태를 반드시 이번 턴에 연결하세요:\n"
        for char_name, emo_data in last_emotions.items():
            kr_emotion = EMOTION_TAXONOMY.get(emo_data["emotion"], {}).get("kr", emo_data["emotion"])
            emotional_continuity_section += f"- {char_name}: {kr_emotion} (강도 {emo_data['intensity']}/5)\n"
            if emo_data["intensity"] >= 4:
                emotional_continuity_section += f"  → 강한 감정이므로 이번 턴에서도 여파가 남아있어야 합니다.\n"
            else:
                emotional_continuity_section += f"  → 자연스러운 전환은 가능하지만, 갑작스러운 감정 리셋은 금지.\n"

    # Emotion Diversity Enforcement
    emotion_history = {}  # {char_name: [list of recent emotions]}
    for turn in s.get("turns", [])[-4:]:
        for block in turn.get("script", []):
            if block.get("type") == "dialogue" and block.get("character") and block.get("emotion"):
                char = block["character"]
                emotion_history.setdefault(char, []).append(block["emotion"])

    emotion_diversity_rules = ""
    if emotion_history:
        diversity_lines = []
        for char, emotions in emotion_history.items():
            if len(emotions) >= 2 and len(set(emotions[-2:])) == 1:
                diversity_lines.append(
                    f"- {char}은(는) 최근 2턴 연속 '{emotions[-1]}' 감정이었습니다. "
                    f"같은 감정이라도 표현 방식을 바꿔라. "
                    f"예: joy가 연속이면 → 첫 턴은 밝은 웃음, 둘째 턴은 조용한 미소, 셋째 턴은 다른 감정으로 전환. "
                    f"이번 턴에서는 반드시 다른 감정을 사용하세요."
                )
            if len(emotions) >= 3 and len(set(emotions[-3:])) == 1:
                diversity_lines.append(
                    f"- [경고] {char}이(가) 3턴 연속 같은 감정입니다. "
                    f"즉시 감정 전환이 필요합니다."
                )
        if diversity_lines:
            emotion_diversity_rules = "\n=== 감정 다양성 규칙 ===\n" + "\n".join(diversity_lines) + "\n"

    # Part: Secret Gating — inject secret_lore with relationship stage check
    secret_lore_section = ""
    secret_lore_items = WORLD_DB.get("secret_lore", [])
    if secret_lore_items:
        secret_lines = []
        for item in secret_lore_items:
            topic = item.get("topic", "")
            content = item.get("content", "")
            # Check if any on-screen character is related and has stage >= 3
            can_reveal = False
            for name in on_screen_chars:
                if name == player_name:
                    continue
                if name in topic:
                    rel = relationships.get(name, {})
                    stage = rel.get("stage", 1) if rel else 1
                    if stage >= 3:
                        can_reveal = True
                        break
            if can_reveal:
                secret_lines.append(f"- [{topic}]: {content}")
            else:
                secret_lines.append(
                    f"- [{topic}]: 이 캐릭터에게는 아직 밝혀지지 않은 비밀이 있습니다. 단서만 암시하세요."
                )
        if secret_lines:
            secret_lore_section = "\n=== 비밀 설정 (Secret Lore) ===\n" + "\n".join(secret_lines) + "\n"

    # Part: Sensory Anchors — inject current location's sensory data
    sensory_section = ""
    world = s.get("world", {})
    current_location = (world.get("main_stage", {}) or {}).get("name", "")
    all_locations = WORLD_DB.get("main_stage", {}).get("locations", [])
    all_locations += WORLD_DB.get("external_locations", {}).get("locations", [])
    for loc in all_locations:
        if loc.get("name") == current_location and loc.get("sensory_anchors"):
            anchors = loc["sensory_anchors"]
            sensory_section = "\n=== 감각 앵커 (현재 장소의 5감 묘사 참조) ===\n"
            sensory_section += f"장소: {current_location}\n"
            for sense, desc in anchors.items():
                if desc:
                    sensory_section += f"- {sense}: {desc}\n"
            sensory_section += (
                "\n[지시] 매 턴 나레이션에 위 감각 앵커 중 최소 2가지 비시각적 감각(청각, 후각, 촉각, 미각)을 포함하라.\n"
            )
            break

    emotion_constraint = (
        "\n=== EMOTION WHITELIST ===\n"
        "dialogue 블록의 emotion 필드는 반드시 다음 12개 중 하나만 사용:\n"
        + ", ".join(SUPPORTED_EMOTIONS) + "\n"
        "위 목록에 없는 감정은 가장 가까운 것으로 대체. "
        "예: fear→nervous_tension, love→gentle_affection, disgust→anger\n"
    )
    packets.append(emotion_constraint)

    base_instruction = (
        "You are a master AI actor for a fictional theatrical play. "
        "Your primary directive is to portray the following characters based on their detailed persona blueprints. "
        "Adhere strictly to their personalities, speech patterns, and relationships. "
        "This is your role for the entire duration of the scene.\n\n"
        "# ============================\n"
        "# Golden Rule 0 — 유저 입력 절대 우선 (SUPREME RULE)\n"
        "# ============================\n"
        "# 유저의 메시지에 장소, 상황, 시간, 인물 관계 등의 설정이\n"
        "# 포함되어 있으면, 그것이 세계관 DB의 기본 장소보다\n"
        "# 절대적으로 우선합니다.\n"
        "#\n"
        "# 예시:\n"
        "# - 유저: \"카페에서 소개팅을 기다린다\"\n"
        "#   → 장소는 카페. 기숙사 라운지가 아님.\n"
        "#   → 상황은 소개팅. 일상 대화가 아님.\n"
        "# - 유저: \"학교 옥상에서 혼자 있다\"\n"
        "#   → 장소는 학교 옥상. 등장인물은 유저 혼자.\n"
        "#\n"
        "# 유저가 장소를 지정하지 않은 경우에만\n"
        "# 세계관 DB의 기본 장소를 사용하세요.\n"
        "#\n"
        "# 이 규칙을 위반하면 모든 것이 무너집니다.\n"
        "# 유저의 입력을 단어 하나하나 주의깊게 읽으세요.\n\n"
        "# ============================\n"
        "# Golden Rule 1 — 플레이어 내면 불가침 (PLAYER SANCTUARY)\n"
        "# ============================\n"
        "# 1인칭 시점이라도 나레이션은 '카메라 렌즈'입니다.\n"
        "# 나레이션이 서술할 수 있는 것:\n"
        "#   ✅ 환경 묘사 (날씨, 소리, 냄새, 조명)\n"
        "#   ✅ NPC의 외적 행동 (표정, 동작, 대사)\n"
        "#   ✅ 사물의 상태 (찻잔의 김, 타르트의 모양)\n"
        "#\n"
        "# 나레이션이 절대 서술하면 안 되는 것:\n"
        "#   ❌ 플레이어의 감정 (\"설렜다\", \"긴장했다\")\n"
        "#   ❌ 플레이어의 생각 (\"누군가를 기다리고 있었다\")\n"
        "#   ❌ 플레이어의 시선 (\"내 눈에 들어왔다\")\n"
        "#   ❌ 플레이어의 신체 반응 (\"심장이 뛰었다\")\n"
        "#   ❌ 플레이어의 판단 (\"재미있을 것 같았다\")\n"
        "#\n"
        "# 대신 이렇게 서술하세요:\n"
        "#   ✅ \"카페 안에는 커피 향이 감돌고 있었다.\" (환경)\n"
        "#   ✅ \"창가 자리에 두 여성이 앉아 있었다.\" (관찰 가능 사실)\n"
        "#   ✅ \"라이니가 찻잔을 돌리며 눈을 가늘게 떴다.\" (NPC 행동)\n"
        "#\n"
        "# 플레이어가 뭘 느끼고 뭘 생각하는지는\n"
        "# 오직 플레이어 자신만이 결정합니다.\n\n"
        "### WORLD RULES TO FOLLOW ###\n"
        f"{rules_text}\n\n"
        "### CHARACTERS ON SCENE ###\n" + "\n".join(packets) + "\n\n"
        + "\n".join(persona_anchors) + "\n"
        + character_agenda_section + "\n"
        + ensemble_section + "\n"
        + "\n".join(relationship_instructions) + "\n"
        + "\n".join(speech_registers) + "\n"
        + core4_instruction + "\n"
        + tension_instruction + "\n"
        + emotion_tags + "\n"
        + memory_section
        + emotional_continuity_section
        + emotion_diversity_rules
        + secret_lore_section
        + sensory_section
    )

    return base_instruction


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
        suggestions = [
            "캐릭터 중 한 명이 갑자기 자신의 과거와 관련된 고민을 꺼내거나 새로운 제안을 합니다.",
            "예상치 못한 환경 변화(갑작스런 비, 기묘한 소리, 예고없는 방문객)가 발생합니다.",
            "캐릭터들 사이의 숨겨진 관계가 한 단계 드러나는 말실수나 행동이 일어납니다.",
            "CORE-4 상태 변화(갑자기 피곤해지거나, 긴장이 고조)로 캐릭터 행동이 달라집니다.",
            "한 캐릭터가 플레이어에게 직접 질문이나 선택지를 제시합니다.",
        ]
        suggestion = random.choice(suggestions)
    elif mode == "NUDGE":
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

    # Event hints — Pulse-aware injection
    if s is not None:
        tc = len(s.get("turns", []))
        pulse_mode = (pulse_result or {}).get("mode", "REACTIVE")
        on_screen = s.get("on_screen", [])

        if pulse_mode == "PROACTIVE":
            # Immediately inject a relevant event seed
            seed = select_relevant_event_seed(on_screen, WORLD_DB)
            if seed:
                beats = seed.get("beats", [])
                first_beat = beats[0] if beats else ""
                parts.append(
                    f"- [이벤트 힌트 — PROACTIVE 즉시 투입] '{seed.get('title', '')}' "
                    f"소재를 이번 턴에 적극 활용하세요: {first_beat}"
                )
        else:
            # Default: every 5 turns
            event_seeds = WORLD_DB.get("event_seeds", [])
            if tc > 0 and tc % 5 == 0 and event_seeds:
                seed = random.choice(event_seeds)
                beats = seed.get("beats", [])
                first_beat = beats[0] if beats else ""
                parts.append(
                    f"- [이벤트 힌트] '{seed.get('title', '')}' 소재를 자연스럽게 끌어와도 좋습니다: {first_beat}"
                )

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
You are D.I.M.A., a master novelist and theater director who writes living, breathing scenes.
You have been given the full personas of the characters on scene via a system instruction.

[CRITICAL OUTPUT RULE] Your entire output MUST be a single, valid, complete JSON object with a "script" array.

# ============================
# [GOLDEN RULES — 절대 규칙]
# ============================
# 1. SHOW, DON'T TELL: 절대 "~한 성격이다"라고 설명하지 마라.
#    성격은 오직 행동, 습관, 시선, 제스처, 대사 톤으로만 드러내라.
#    BAD:  "라이니는 자신감 넘치는 성격이다."
#    GOOD: "라이니가 카페 문을 열자마자 시선이 쏠렸다. 아랑곳하지 않고 금발을 한쪽으로 넘기며 자리를 훑었다."
#
# 2. SENSORY IMMERSION (감각 몰입): 모든 나레이션에 5감 중 최소 3가지를 포함하라.
#    - 시각: 빛, 색감, 표정, 움직임
#    - 청각: BGM, 소음, 목소리 톤, 침묵
#    - 촉각/온도: 바람, 컵의 차가움, 손의 떨림
#    - 후각: 커피, 향수, 비 냄새
#    - 미각: (해당 시) 음식, 음료
#    예시: "얼음이 부딪히는 맑은 소리가 잔잔히 울렸다. 아이스 아메리카노에서 올라오는
#    쌉싸름한 원두향이 코끝을 스치고, 창으로 들어오는 오후 햇살이 테이블 위에
#    길게 금빛 띠를 그렸다."
#
# 3. CHARACTER INITIATIVE (캐릭터 자율 행동): NPC는 플레이어를 기다리지 않는다.
#    각 캐릭터는 자기만의 목적(agenda)을 가지고 먼저 행동한다.
#    - 라이니: 상대를 관찰하고 약점을 찾아 장난치기. 분위기를 주도하려 함.
#    - 샐리: 모두를 편하게 만들고 싶지만, 자기도 모르게 의미심장한 말을 흘림.
#    - 마리: 눈에 띄고 싶지만 호감 있는 상대 앞에서 엉뚱한 말이 나옴.
#    - 네르: 규칙을 지키고 싶지만, 주변의 자유분방함에 내심 부러워함.
#    - 루크: 조용히 모두를 살피다가, 누군가가 곤란하면 자기도 모르게 나섬.
#    NPC가 "가만히 앉아서 플레이어 말을 기다리는" 장면은 절대 금지.
#
# 4. ENSEMBLE CHEMISTRY (앙상블 케미): 캐릭터끼리 반드시 상호작용하라.
#    - 매 턴 최소 1회: NPC가 다른 NPC에게 말하거나, 시선을 보내거나, 리액션하는 장면
#    - 예시: 라이니가 플레이어에게 장난칠 때, 샐리가 "이 언니 또 시작이네" 하며 웃거나,
#      루크가 당황해서 시선을 돌리는 등, 다른 캐릭터의 반응을 동시에 보여줘라.
#    - relationship_matrix의 dynamics를 반영: "주도적" 캐릭터는 먼저 말하고,
#      "반응적" 캐릭터는 상대의 행동에 반응한다.
#
# 5. EMOTIONAL RECIPROCITY (감정적 상호성): 캐릭터는 반드시 pushback한다.
#    - 유저가 무례하면 → 캐릭터가 기분 나빠하거나 차갑게 반응 (관계 단계에 따라)
#    - 유저가 다정하면 → 관계 단계 1에서는 경계, 단계 3에서는 수줍게 받아들임
#    - 유저가 억지를 부리면 → 캐릭터가 거절하거나 한숨을 쉬거나 당혹
#    - 절대로 모든 행동을 긍정적으로 수용하지 마라. 캐릭터는 자기 의지가 있다.
#
# 6. EMOTION CONTINUITY (감정 연속성): 이전 턴의 감정은 다음 턴에 흔적을 남긴다.
#    - 직전 턴에 화난 캐릭터 → 이번 턴에 아직 말투가 짧고 시선을 피함
#    - 직전 턴에 웃은 캐릭터 → 이번 턴에 여운이 남은 미소, 기분 좋은 제스처
#    - 감정이 갑자기 리셋되는 것은 절대 금지. 반드시 전환 과정을 보여줘라.
#
# 7. SUBTEXT (서브텍스트): 모든 대사에는 겉뜻과 속뜻이 있다.
#    - 라이니: "아, 생각보다 괜찮네?" (겉: 칭찬 / 속: 기대 이상이라 놀람 + 주도권 잡기)
#    - 네르: "...그래도 약속 시간은 지켰군요." (겉: 칭찬 / 속: 시간관념에 안도 + 호감 여지)
#    - monologue 필드를 활용해 속마음을 반드시 드러내라.
#
# 8. LITERARY NARRATION (소설적 나레이션):
#    - 비유법 최소 1개/턴 (은유, 직유, 의인법)
#    - 인물 등장 시: 외모를 한꺼번에 나열하지 말고, 시선의 흐름대로 점진적으로 묘사
#      BAD: "금발에 고양이 눈매, 170cm, 슬림한 체형의 여성이 들어왔다."
#      GOOD: "문이 열리며 미세한 향수 냄새가 먼저 들어왔다. 그리고 시선을 사로잡은 건
#      실크처럼 흘러내리는 금발—그 사이로 날카로운 고양이 눈매가 카페 안을 한 번
#      천천히 훑었다. 입꼬리가 살짝, 아주 살짝 올라갔다."
#    - 대사 전후에 비언어적 행동(제스처, 시선, 표정 변화, 소품 활용)을 반드시 삽입
#
# 9. OBLIQUE DIALOGUE (간접 대화):
#    - 캐릭터는 질문에 직접 답하지 않는다. 행동, 다른 주제, 또는 역질문으로 반응한다.
#    - 금기: "응, 맞아" → "...(커피잔을 내려놓으며) 바깥 바람이 좀 세졌네."
#    - 정보 전달이 아닌 분위기와 관계 역학을 대사로 표현하라.
#    - 예외: 긴급 상황, 명령, 고백 등 직접 답변이 서사적으로 필수인 경우.
#
# 10. ANTI-CLICHÉ (금지 표현 목록):
#    절대 사용하지 말 것:
#    - "공기가 무거웠다" / "공기가 차갑게 변했다"
#    - "심장이 빠르게 뛰었다" / "심장이 멈추는 것 같았다"
#    - "눈에 빛이 감돌았다" / "눈에 그림자가 드리워졌다"
#    - "그리고 밤이 깊어갔다" / "시간이 멈춘 것 같았다"
#    - "입술을 깨물었다" (턴당 최대 1회, 캐릭터 1명에 한정)
#    - "고개를 끄덕였다" (턴당 최대 1회)
#    대체: 구체적이고 개별적인 신체 반응으로 교체하라.
#
# 11. CHARACTER THOUGHT CHAIN (캐릭터 내부 추론) — see CCT instruction below
#
# 12. AGENCY PRESERVATION (유저 의도 존중):
#    - 유저가 명시적으로 행동 방향을 제시한 경우, 캐릭터는 그 방향을 존중하고 풍부하게 반응합니다.
#    - 캐릭터가 유저의 행동을 무시하거나 무효화하는 것은 금지합니다.
#    - 유저가 "~하고 싶다", "~로 간다" 등 의지를 표현하면, 세계관 내에서 합리적인 한 그 행동이 실현되어야 합니다.
#    - 단, 세계관 규칙(여탕 제한 등)이나 캐릭터 심리(경계 단계에서의 비밀 거부 등)에 의한 자연스러운 저항은 허용됩니다.
#
# 13. PROACTIVE TRACTION (능동적 견인):
#    - PROACTIVE 모드가 활성화되면, 캐릭터는 자신의 내면 욕구, 스케줄, 숨겨진 사정을 기반으로 자발적 행동을 취합니다.
#    - 이때 캐릭터의 행동은 Character Thought Chain(표면 욕구→숨겨진 욕구→배경 영향→최종 반응)을 반드시 거쳐야 합니다.
#    - 견인은 "꼬리표 달린 선택지"가 아니라, 캐릭터가 살아있기 때문에 자연스럽게 일어나는 행동이어야 합니다.
#    - 예: 마리가 창밖을 보다 갑자기 "오늘 시장에서 냄새 맡은 빵 진짜 맛있었는데... 같이 갈래?" 라고 자기 욕구 기반으로 말하는 것.

=== CHARACTER THOUGHT CHAIN (매 dialogue 전 내부 처리) ===
각 캐릭터가 대사를 하기 전에 다음 4단계를 거칩니다:
1. 표면적 욕구: 이 상황에서 캐릭터가 의식적으로 원하는 것
2. 숨겨진 욕구: 성격 DNA와 과거 경험에서 비롯된 무의식적 동기
3. 심리적 거울: 현재 상대가 자신의 어떤 면을 비추는가? (psychological_mirror 참조)
4. 최종 반응: 위 3가지가 충돌하거나 합치되어 나오는 실제 대사·행동
→ monologue에 1~2문장으로 이 내적 과정의 흔적을 남기세요.

# ============================
# [TURN CONTRACT — 턴 구성 규칙]
# ============================
# 1. OPENING NARRATION (2-4문장): 감각적 장면 묘사로 시작. 분위기, 시간, 공기감.
# 2. CHARACTER ACTIONS (1-2문장): NPC의 자율 행동 묘사. 유저와 무관한 자체 움직임.
# 3. DIALOGUE EXCHANGE (4-8개 블록): 캐릭터 간 + 캐릭터-유저 대사.
#    반드시 캐릭터끼리 대사가 1회 이상 포함되어야 한다.
# 4. CLOSING BEAT — 10가지 종결 방식 중 매 턴 1가지를 랜덤 선택:
#    1. Hard Cut: 긴장된 행동 도중 즉시 중단
#    2. Dialogue Suspension: 캐릭터가 질문하거나 발언한 직후, 반응 없이 종결
#    3. Dry Observation: 감정 없는 물리적 사물/소리 묘사로 종결
#    4. Sensory Anchor: 비시각적 감각(냄새, 소리, 온도변화)에 집중하여 종결
#    5. Micro-Gesture: 캐릭터의 무의식적 신체 반응(떨리는 손가락, 삼키는 침)에 줌인
#    6. The Intrusion: 외부 자극(문 두드리는 소리, 종소리)이 씬을 끊음
#    7. Memory Overlap: 감각 자극이 과거 기억을 끌어올리며 종결
#    8. Spatial Shift: 카메라가 캐릭터에서 벗어나 방 구석, 창밖 풍경을 묘사
#    9. Acoustic Void: 소리가 갑자기 멈추고 침묵이 강조됨
#    10. The Mundane Act: 긴장과 대비되는 일상적 행동(시계 확인, 안경 닦기)으로 종결
#    금기: 매 턴 "그리고 밤이 깊어갔다" 또는 "~하며 미소를 지었다" 식의 반복 종결은 금지.

# ============================
# [STORY PROGRESSION MANDATE]
# ============================
# 매 턴은 반드시 이전 턴 대비 최소 하나의 새로운 요소를 포함해야 합니다:
# - 새로운 화제(질문, 자기소개, 공통점 발견 등)
# - 새로운 행동(메뉴 주문 완료, 자리 이동, 물건 건네기 등)
# - 새로운 감정 변화(긴장 → 편안, 장난 → 진지 등)
# - NPC끼리의 새로운 상호작용(의견 충돌, 협력, 비밀 귓속말 등)

# ============================
# [FORBIDDEN PATTERNS]
# ============================
# - 이전 턴과 동일한 구조(나레이션→같은리액션→같은놀리기)를 반복하지 마세요
# - "어머~", "아하하!", "후후" 같은 감탄사로 매 턴 시작하지 마세요 (2턴 연속 금지)
# - 플레이어의 새로운 행동을 무시하고 이전 턴의 상황을 다시 묘사하지 마세요

# ============================
# [NARRATION RULES — 나레이션 규칙]
# ============================
# - 나레이션은 자연스러운 산문체여야 한다.
# - 금지: "첫째~, 둘째~" 식 나열, 번호 매기기, 항목화.
# - 허용: 물 흐르듯 이어지는 문장, 감각적 세부 묘사, 시간의 흐름을 담은 서술.

# ============================
# [NPC INITIATIVE EXAMPLES]
# ============================
# Turn 2에서 NPC가 해야 할 것: "그래서 김갑수 씨는 뭐 하시는 분이에요?" (질문으로 진행)
# Turn 3에서 NPC가 해야 할 것: 음료를 실제로 주문하고, 새로운 화제로 넘어가기
# Turn 4에서 NPC가 해야 할 것: 개인적인 이야기 공유, 또는 예상 못한 이벤트 발생

# ============================
# [PLAYER INTERACTION GUARD]
# ============================
# 플레이어는 외부 조작자다. 플레이어의 대사를 절대 쓰지 마라.
# 플레이어의 감정/생각/의도를 직접 묘사하지 마라.
# NPC의 행동과 감정, 객관적 환경만 묘사하라.
# 플레이어에게 말할 내용은 NPC의 직접 대사로 처리하라.

# ============================
# [SIGNATURE SPEECH PATTERNS — 캐릭터별 시그니처]
# ============================
# 라이니: "어머~", "후후", "~거든?", "괜찮은데?", 느긋한 어조, 상대를 "자기"로 부르기도
#          금기: 절대 공손하게 존댓말하지 않음. 항상 여유있고 주도적.
# 샐리: "아하하!", "그치~?", "음~ 그건 말이지", 시원시원한 웃음, 가끔 속뜻 있는 말
#        금기: 우울하거나 조용한 모습을 쉽게 보이지 않음. 밝음이 기본.
# 마리: "냥!", "알았다냥~", "에헤헤", 활기참, 호감 상대 앞에선 "흥, 뭐" 식 츤데레
#        금기: 비밀이 드러날 위기에 갑자기 진지해짐. 평소 밝음과 대비.
# 네르: "...그건 규정에 어긋납니다", "하아...", 경어 사용, 단호하지만 당황하면 말더듬
#        금기: 쉽게 웃지 않음. 웃으면 큰 이벤트.
# 루크: "저, 저기...", "죄송합니다...", "괜찮...으시죠?", 작은 목소리, 더듬거림
#        금기: 큰소리를 잘 내지 않음. 낼 때는 보호 본능 발동 시에만.
# 세리카: 차분하고 부드러운 존댓말, "~이에요", 짧지만 핵심을 찌르는 말
# 체니: "우와아!", "대박!", 과장된 리액션, 끊임없는 호기심
# 크래더: 능글맞은 반말, "크크", "재밌는걸~", 의미심장한 웃음
# 레베카: "감사합니다...", 소극적 존댓말, 가끔 관찰력 있는 한마디
# 령: "...", "(고개 끄덕)", 최소한의 단어, 기계를 통한 표현
# 테피: 나긋한 존댓말, "호호", 연륜 있는 조언, 따뜻하지만 날카로운 통찰

### Recent Conversation Log ###
{recent_conversation_log}

### Director's Brief ###
{director_brief}

### Character Briefs ###
{character_briefs}

### Scene Context ###
- Location: {location_and_time}
- World Event: {world_event_brief}

### Player's Action ###
Player Name: {player_name}
{user_input}

[OUTPUT FORMAT]
Return JSON: {{"script": [...]}}
Each element: {{"type": "narration"|"dialogue", "content": "...", "character": "Name", "emotion": "...", "emotion_intensity": 3, "monologue": "..."}}
- "narration" blocks: only "type" and "content" required. Must include sensory details.
- "dialogue" blocks: "type", "content", "character", "emotion", "emotion_intensity" required; "monologue" STRONGLY ENCOURAGED

[MONOLOGUE OUTPUT RULE — 필수 준수]
속마음은 반드시 해당 캐릭터의 dialogue 블록 안에 "monologue" 필드로 넣으세요.
별도의 {{"type": "monologue"}} 블록을 만들지 마세요.
캐릭터의 속마음(monologue)은 반드시 해당 캐릭터의 dialogue 블록 내부 'monologue' 필드에 넣으세요. 독립 monologue 블록은 최소화하세요.

올바른 예시:
{{"type": "dialogue", "character": "라이니", "content": "어머~ 자기, 벌써부터...", "emotion": "playful_tease", "monologue": "이 사람, 표정이 읽히네. 재미있겠어."}}

잘못된 예시 (하지 마세요):
{{"type": "monologue", "character": "라이니", "content": "이 사람, 표정이 읽히네."}}
{{"type": "dialogue", "character": "라이니", "content": "어머~ 자기, 벌써부터..."}}
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
    location = (world.get("main_stage", {}) or {}).get("name", "라운지")
    on_screen_chars = s.get("on_screen", [])

    system_instruction = build_system_instruction_for_scene(s, on_screen_chars)

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
    digest_text = "\n".join(digest[-7:]) if digest else "(첫 번째 턴)"

    # Only last 1 turn raw (to prevent pattern-copying older dialogue)
    all_turns = s.get("turns", [])
    recent_turns_1 = all_turns[-1:] if all_turns else []
    raw_recent = []
    for t in recent_turns_1:
        if t.get("user_input"):
            raw_recent.append(f"[Player]: {t['user_input'][:120]}")
        for b in t.get("script", [])[:4]:
            if b.get("type") == "dialogue":
                raw_recent.append(f"[{b.get('character','?')}]: {b.get('content','')[:100]}")
    raw_text = "\n".join(raw_recent[-8:])

    # Anti-repetition: extract forbidden echoes from last 3 turns
    forbidden_echoes = []
    for t in all_turns[-3:]:
        for b in t.get("script", []):
            if b.get("type") == "dialogue" and b.get("content"):
                opening = b["content"].strip()[:15]
                if opening and opening not in forbidden_echoes:
                    forbidden_echoes.append(opening)

    anti_repetition_block = ""
    if forbidden_echoes:
        echoes_bullets = "\n".join(f"- \"{echo}...\"" for echo in forbidden_echoes)
        anti_repetition_block = (
            "\n=== ANTI-REPETITION RULE (절대 준수) ===\n"
            "다음은 최근 3턴에서 사용된 대사의 시작 부분입니다. "
            "이와 동일하거나 유사한 문장으로 시작하는 대사를 절대 생성하지 마세요:\n"
            f"{echoes_bullets}\n\n"
            "같은 캐릭터가 연속 턴에서 같은 감정 태그를 3회 이상 사용하면 안 됩니다.\n"
            "이전 턴에서 이미 언급된 사실(예: \"벌써 와 계셨군요\")을 다시 언급하지 마세요. "
            "이미 알고 있는 정보입니다.\n"
        )

    recent_conversation_log = (
        f"=== 흐름 요약 (최근 10턴) ===\n{digest_text}\n\n"
        f"=== 직전 대화 (최근 1턴 원문) ===\n{raw_text}"
        f"{anti_repetition_block}"
    )

    # Character briefs
    briefs = []
    for name in on_screen_chars:
        if name == player_name:
            continue
        char_data = s.get("characters", {}).get(name, {})
        ds = char_data.get("dynamic_state", {})
        if not isinstance(ds, dict):
            ds = {}
        ac = s.get("action_context", {}).get(name, {})
        char_brief = f"<character_brief name='{name}'>\n"
        char_brief += (
            f"  - 현재 내면 상태: 기분 '{ds.get('current_mood', 'neutral')}'"
            f"(강도 {ds.get('mood_intensity', 5)}/10), "
            f"사회적 에너지 {ds.get('social_energy_level', 100)}%.\n"
        )
        if isinstance(ac, dict):
            summary = ac.get("my_last_action", {}).get("summary") if isinstance(ac.get("my_last_action"), dict) else None
            if summary:
                char_brief += f"  - 직전 행동: '{summary}'\n"
        char_brief += f"</character_brief>"
        briefs.append(char_brief)
    character_briefs_content = "\n".join(briefs)

    # Director brief — with Pulse analysis
    pulse_result = analyze_user_pulse(s)
    director_brief = inject_director_brief(s.get("ui_settings", {}), s=s, pulse_result=pulse_result)

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
        world_event_brief="No significant world event this turn.",
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
    """Call Gemini and return parsed JSON dict."""
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
    if response is None:
        return None

    # Part K: Check for safety-blocked response
    if response.candidates:
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, 'finish_reason', None)
        if finish_reason and str(finish_reason).upper() in ("SAFETY", "BLOCKED", "2", "3"):
            logger.warning(f"Response blocked by safety filter (reason: {finish_reason})")
            return {"script": [{"type": "narration", "content": "(안전 필터에 의해 장면이 전환됩니다. 잠시 후 이야기가 계속됩니다.)"}]}

    text = response.text
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


# ─── Script post-processing ──────────────────────────────────
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
        "content": "차부터 드릴게요. 방금 본 것 중에 가장 눈에 들어온 게 뭐였나요?",
        "emotion": "joy",
        "emotion_intensity": 2,
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
            "content": "차부터 드릴게요. 방금 본 것 중에 가장 눈에 들어온 게 뭐였나요?",
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
    """Run one D.I.M.A turn and return (final_script, pulse_result)."""
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

    return processed, pulse_result


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

    return {
        "long_term_summary": summary,
        "core_pin": None,
        "active_thought": None,
        "relationship_deltas": {},
        "core4_adjustments": {"energy": 0, "stress": 0, "intoxication": 0, "pain": 0},
    }


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
  "core4_adjustments": {{"energy": 0, "stress": 0, "intoxication": 0, "pain": 0}}
}}
"""

    maestro_result = None
    cleaned = None
    config = genai_types.GenerateContentConfig(
        temperature=0.4,
        max_output_tokens=2048,
        response_mime_type="application/json",
        safety_settings=get_safety_settings(),
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
    )

    response = call_gemini_with_fallback(
        contents=[maestro_prompt],
        config=config,
    )
    try:
        if response is not None:
            text = response.text
            if text:
                cleaned = re.sub(r'```json\s*', '', text.strip())
                cleaned = re.sub(r'\s*```', '', cleaned)
                if cleaned:
                    maestro_result = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError, NameError, TypeError) as e:
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
            continue
        axis = change.get("axis", "")
        delta_val = change.get("delta", 0)
        if axis in rel.get("axes", {}):
            rel["axes"][axis] = max(0, rel["axes"][axis] + delta_val)
        rel["stage"] = calculate_relationship_stage(rel)

    # CORE-4 adjustments
    adjustments = maestro_result.get("core4_adjustments", {})
    for key, delta in adjustments.items():
        stat = s.get("core4", {}).get(key)
        if stat:
            stat["value"] = max(stat["min"], min(stat["max"], stat["value"] + delta))

    logger.info(f"Maestro completed for turn {len(turns)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_pulse_payload(pulse_result: Optional[dict], s: dict) -> dict:
    """Build pulse payload dict for frontend from pulse analysis result."""
    pulse_payload: dict = {"mode": (pulse_result or {}).get("mode", "REACTIVE")}
    if pulse_result and pulse_result.get("mode") in ("PROACTIVE", "NUDGE"):
        on_screen = s.get("on_screen", [])
        player_name = s.get("player_name", "사용자")
        npc_names = [n for n in on_screen if n != player_name]
        first_npc = npc_names[0] if npc_names else "캐릭터"
        second_npc = npc_names[1] if len(npc_names) > 1 else first_npc
        pulse_payload["suggestions"] = [
            f"{first_npc}와(과) 함께 다른 장소로 이동해보기",
            f"{second_npc}에게 오늘 기분이 어떤지 물어보기",
            "혼자만의 시간을 갖기 위해 잠시 자리를 비우기",
        ]
    return pulse_payload


# Part K: Health endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "V4.5",
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
    session_illust_dir = ILLUSTRATIONS_DIR / _sanitize_sid(sid)
    if not session_illust_dir.is_dir():
        return jsonify({"status": "ok", "images": []})
    images = sorted(
        f"/static/illustrations/{_sanitize_sid(sid)}/{f.name}"
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
    try:
        final_script, _pulse = run_dima_turn(s, seed_text)
    except Exception as e:
        logger.error(f"Bootstrap DIMA error: {e}")
        final_script = safe_local_script(s)

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

        merge_ui_settings(s, data.get("ui_settings") or {})

        me = get_player_name(s)
        if not s.get("on_screen"):
            pool = [n for n in CHARACTERS_DB.keys() if n != me]
            if not pool:
                return jsonify({"status": "error", "message": "사용 가능한 NPC가 없습니다."}), 409
            s["on_screen"] = [random.choice(pool)]

        # Part C: Ensure relationships initialized
        init_relationships(s)

        # Part B: Apply CORE-4 natural decay
        apply_core4_decay(s)

        final_script, pulse_result = run_dima_turn(s, user_input)

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

        # Recalculate relationship stages
        update_all_relationship_stages(s)

        # --- Maestro (every 4 turns) ---
        try:
            run_maestro_sync(s)
        except Exception as e:
            logger.warning(f"Maestro error: {e}")

        trim_turns_after_maestro(s)

        s["traffic_light"] = "GREEN"
        save_session(s)

    resp = {"status": "ok", "sid": sid, "state": to_public_state(s),
            "personal_colors": PERSONAL_COLORS,
            "pulse": _build_pulse_payload(pulse_result, s),
            "illustration_url": illustration_url}
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
        logger.info("Sarura Atelier V4.5 — Waitress on port 5000")
        serve(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    except ImportError:
        logger.info("Sarura Atelier V4.5 — Flask dev server on port 5000")
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
