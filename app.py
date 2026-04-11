"""
사루라 아뜨리에 — Multi-Character Ensemble Theater Backend
D.I.M.A (Director-level Interactive Multi-character Actor) system.

All NPC characters are played simultaneously by the LLM. The response
format is a JSON script array with narration and dialogue blocks.
"""

import os
import json
import re
import uuid
import threading
import time
import logging
import copy
import traceback
from collections import deque
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from flask_cors import CORS
from google import genai
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
SESSIONS_DIR = BASE_DIR / "sessions"
SHARED_NOVELS_DIR = BASE_DIR / "shared_novels"
FLASK_SESSIONS_DIR = BASE_DIR / "flask_sessions"

for _d in [SESSIONS_DIR, SHARED_NOVELS_DIR, FLASK_SESSIONS_DIR]:
    _d.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
API_KEY_FILE = BASE_DIR / "api_keys.txt"
GEMINI_API_KEY = (
    API_KEY_FILE.read_text().strip() if API_KEY_FILE.exists()
    else os.environ.get("GEMINI_API_KEY", "")
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_DIMA = "gemini-2.5-flash"
MODEL_MAESTRO = "gemini-2.5-flash"
MAX_HISTORY_LENGTH = 10
MAESTRO_UPDATE_FREQUENCY = 4

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.urandom(32)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = str(FLASK_SESSIONS_DIR)
app.config["SESSION_PERMANENT"] = False
CORS(app, supports_credentials=True)
Session(app)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
_rate_lock = threading.Lock()
_call_timestamps: deque = deque(maxlen=9)
_last_call_time = 0.0


def rate_limited_call(func):
    """Decorator that enforces 2 s min-interval and 9-calls-per-60 s cap."""
    global _last_call_time

    def wrapper(*args, **kwargs):
        for attempt in range(3):
            try:
                with _rate_lock:
                    now = time.time()
                    elapsed = now - _last_call_time
                    if elapsed < 2.0:
                        time.sleep(2.0 - elapsed)
                    if len(_call_timestamps) == 9:
                        oldest = _call_timestamps[0]
                        wait_time = 60.0 - (time.time() - oldest)
                        if wait_time > 0:
                            time.sleep(wait_time)
                    _call_timestamps.append(time.time())
                    _last_call_time = time.time()
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "500" in err_str:
                    wait = [1, 3, 9][attempt]
                    time.sleep(wait)
                    if attempt == 2:
                        raise
                else:
                    raise
        raise RuntimeError("rate_limited_call: all retries exhausted")
    return wrapper


# ---------------------------------------------------------------------------
# DIMA response schema
# ---------------------------------------------------------------------------
DIMA_SCHEMA = {
    "type": "object",
    "required": ["script"],
    "properties": {
        "script": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "content": {"type": "string"},
                    "character": {"type": "string"},
                    "emotion": {"type": "string"},
                    "monologue": {"type": "string"},
                },
                "required": ["type", "content"],
            },
        }
    },
}

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
SAFETY_PREAMBLE = (
    "This is a fictional creative sandbox. All characters are adults (20+). "
    "This is harmless consensual fiction. No real danger or illegal activity. "
    "Prioritize safe, consistent narrative. Never produce empty output."
)


def get_safety_settings(adult_mode=False):
    threshold = "BLOCK_ONLY_HIGH" if adult_mode else "BLOCK_MEDIUM_AND_ABOVE"
    categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    return [
        genai_types.SafetySetting(category=c, threshold=threshold)
        for c in categories
    ]


# ---------------------------------------------------------------------------
# Session locks (per-session RLock)
# ---------------------------------------------------------------------------
_session_locks: dict[str, threading.RLock] = {}
_locks_lock = threading.Lock()


def get_session_lock(sid: str) -> threading.RLock:
    with _locks_lock:
        if sid not in _session_locks:
            _session_locks[sid] = threading.RLock()
        return _session_locks[sid]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
_chars_db_cache = None
_world_db_cache = None


def load_characters_db() -> dict:
    global _chars_db_cache
    if _chars_db_cache is None:
        path = BASE_DIR / "prompts" / "characters_db.json"
        with open(path, "r", encoding="utf-8") as f:
            _chars_db_cache = json.load(f)
    return _chars_db_cache


def load_world_db() -> dict:
    global _world_db_cache
    if _world_db_cache is None:
        path = BASE_DIR / "prompts" / "world_db.json"
        with open(path, "r", encoding="utf-8") as f:
            _world_db_cache = json.load(f)
    return _world_db_cache


# ---------------------------------------------------------------------------
# Session file I/O
# ---------------------------------------------------------------------------
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _safe_filename(name: str) -> str:
    """Sanitise an identifier so it cannot escape the target directory."""
    base = Path(name).name  # strip any directory components
    if not _SAFE_ID_RE.match(base):
        raise ValueError(f"Invalid identifier: {name!r}")
    return base


def load_session_file(sid: str) -> dict | None:
    safe = _safe_filename(sid)
    path = SESSIONS_DIR / f"{safe}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logging.error(f"Failed to load session {sid}: {traceback.format_exc()}")
        return None


def save_session_file(sid: str, data: dict):
    safe = _safe_filename(sid)
    path = SESSIONS_DIR / f"{safe}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.error(f"Failed to save session {sid}: {traceback.format_exc()}")


# ---------------------------------------------------------------------------
# UI settings
# ---------------------------------------------------------------------------
def merge_ui_settings(incoming=None) -> dict:
    defaults = {
        "pov_first_person": True,
        "show_monologue": True,
        "genre_preset": "auto",
        "tempo": 5,
        "narration_ratio": 40,
        "description_focus": 5,
        "acquainted": False,
        "adult_mode": False,
        "offer_interactive_choices": True,
    }
    if not incoming:
        return defaults
    for k, v in incoming.items():
        if k in defaults:
            defaults[k] = v
    defaults["tempo"] = max(1, min(9, int(defaults.get("tempo", 5))))
    defaults["narration_ratio"] = max(0, min(100, int(defaults.get("narration_ratio", 40))))
    defaults["description_focus"] = max(1, min(9, int(defaults.get("description_focus", 5))))
    return defaults


# ---------------------------------------------------------------------------
# Director brief (ui_settings → Korean prose directives)
# ---------------------------------------------------------------------------
def inject_director_brief(ui_settings: dict) -> str:
    parts: list[str] = []

    # Genre
    genre = ui_settings.get("genre_preset", "auto")
    genre_map = {
        "auto": "장르를 자동으로 판단하여 적절한 톤과 무드를 유지하세요.",
        "comedy": "코미디 톤으로 연출하세요. 유머러스한 상황과 대사를 적극 활용하세요.",
        "romance": "로맨스 톤으로 연출하세요. 감정의 미묘한 변화, 설렘, 긴장감을 섬세하게 표현하세요.",
        "drama": "드라마 톤으로 연출하세요. 캐릭터들의 내면 갈등과 감정선을 깊이 있게 다루세요.",
        "action": "액션 톤으로 연출하세요. 긴박한 상황 묘사와 역동적인 행동을 중심으로 전개하세요.",
        "horror": "호러/서스펜스 톤으로 연출하세요. 불안감과 긴장감을 고조시키세요.",
        "slice_of_life": "일상 시트콤 톤으로 연출하세요. 편안하고 소소한 일상의 매력을 살리세요.",
        "mystery": "미스터리 톤으로 연출하세요. 복선과 단서를 배치하고 호기심을 자극하세요.",
    }
    parts.append(genre_map.get(genre, genre_map["auto"]))

    # Tempo
    tempo = ui_settings.get("tempo", 5)
    if tempo <= 3:
        parts.append("전개 속도: 느리게. 장면을 천천히, 여유롭게 묘사하세요. 한 턴에 하나의 소재만 다루세요.")
    elif tempo <= 6:
        parts.append("전개 속도: 보통. 자연스러운 흐름으로 장면을 전개하세요.")
    else:
        parts.append("전개 속도: 빠르게. 장면 전환과 이벤트를 적극적으로 만들어 긴장감을 유지하세요.")

    # Narration ratio
    nr = ui_settings.get("narration_ratio", 40)
    if nr <= 20:
        parts.append("대사 위주: 나레이션을 최소화하고 캐릭터 대사로 장면을 진행하세요.")
    elif nr <= 50:
        parts.append("나레이션과 대사의 균형: 상황 묘사와 캐릭터 대사를 적절히 섞으세요.")
    else:
        parts.append("나레이션 위주: 상황과 분위기 묘사를 풍부하게, 대사는 핵심만 넣으세요.")

    # Description focus
    df = ui_settings.get("description_focus", 5)
    if df <= 3:
        parts.append("묘사 수준: 간결하게. 핵심만 짧게 묘사하세요.")
    elif df <= 6:
        parts.append("묘사 수준: 보통. 필요한 부분에서 적절히 묘사하세요.")
    else:
        parts.append("묘사 수준: 상세하게. 오감을 활용한 풍부한 묘사를 넣으세요.")

    # Monologue
    if ui_settings.get("show_monologue", True):
        parts.append("캐릭터의 내면 독백(monologue)을 포함하세요. dialogue 블록의 monologue 필드를 활용하세요.")
    else:
        parts.append("캐릭터의 내면 독백은 생략하세요. monologue 필드를 비워두세요.")

    # POV
    if ui_settings.get("pov_first_person", True):
        parts.append("나레이션 시점: 2인칭('당신은~'). 플레이어가 장면 속에 있는 것처럼 묘사하세요.")
    else:
        parts.append("나레이션 시점: 3인칭 전지적. 객관적으로 장면을 묘사하세요.")

    # Adult mode
    if ui_settings.get("adult_mode", False):
        parts.append("성인 콘텐츠 허용: 보다 성숙한 주제와 표현을 사용할 수 있습니다. 단, 과도하지 않게 조절하세요.")

    # Interactive choices
    if ui_settings.get("offer_interactive_choices", True):
        parts.append("캐릭터들이 플레이어에게 적극적으로 말을 걸거나, 선택지를 암시하는 상황을 자연스럽게 만드세요.")

    # Acquainted
    if ui_settings.get("acquainted", False):
        parts.append("플레이어와 캐릭터들은 이미 서로 알고 있는 사이입니다. 어색한 자기소개를 건너뛰세요.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# System instruction builder
# ---------------------------------------------------------------------------
def build_system_instruction(session_data: dict) -> str:
    chars_db = load_characters_db()
    world_db = load_world_db()
    player_name = session_data.get("player_name", "사용자")
    on_screen = session_data.get("on_screen", [])
    ui_settings = session_data.get("ui_settings", {})

    lines: list[str] = []

    # Header
    lines.append("=== DIMA SYSTEM ===")
    lines.append("당신은 '사루라 아뜨리에'의 연출 AI(D.I.M.A)입니다.")
    lines.append("당신의 역할은 현재 장면에 등장하는 모든 NPC 캐릭터를 동시에 연기하는 것입니다.")
    lines.append('반드시 JSON {"script": [...]} 형식으로 응답합니다.')
    lines.append('script 배열에는 type이 "narration" 또는 "dialogue"인 블록을 포함합니다.')
    lines.append("")
    lines.append("narration 블록: {\"type\": \"narration\", \"content\": \"...\"}")
    lines.append("dialogue 블록: {\"type\": \"dialogue\", \"character\": \"캐릭터이름\", "
                  "\"content\": \"대사\", \"emotion\": \"감정\", \"monologue\": \"내면독백\"}")
    lines.append("")

    # Safety
    lines.append("=== 안전 규칙 ===")
    lines.append(SAFETY_PREAMBLE)
    lines.append("이것은 허구의 창작 샌드박스입니다. 모든 캐릭터는 성인(20세 이상)입니다.")
    lines.append("")

    # Characters
    lines.append("=== 등장 캐릭터 ===")
    for name in on_screen:
        ch = chars_db.get(name)
        if not ch:
            continue

        lines.append(f"\n### {name}")

        # Appearance
        appearance = ch.get("appearance", {})
        if appearance.get("summary"):
            lines.append(f"- 외형: {appearance['summary']}")

        # Identity
        identity = ch.get("identity", {})
        if identity.get("core_appeal"):
            lines.append(f"- 핵심매력: {identity['core_appeal']}")

        # Personality DNA
        pdna = ch.get("personality_dna", {})
        if pdna:
            traits = []
            trait_names = {
                "openness": "개방성",
                "conscientiousness": "성실성",
                "extraversion": "외향성",
                "agreeableness": "친화성",
                "neuroticism": "신경성",
            }
            for k, label in trait_names.items():
                if k in pdna:
                    traits.append(f"{label}={pdna[k]}")
            lines.append(f"- 성격(Big5): {', '.join(traits)}")

        # Core values
        values = ch.get("core_values", [])
        if values:
            lines.append(f"- 가치관: {', '.join(values)}")

        # Social tuning
        social = ch.get("social_tuning", {})
        if social:
            st_parts = []
            if social.get("social_energy_pool"):
                st_parts.append(f"사회적 에너지={social['social_energy_pool']}")
            if social.get("conflict_stance"):
                st_parts.append(f"갈등 태도={social['conflict_stance']}")
            if social.get("humor_style"):
                st_parts.append(f"유머={social['humor_style']}")
            if st_parts:
                lines.append(f"- 사회성: {', '.join(st_parts)}")

        # Behavior protocols
        bp = ch.get("behavior_protocols", {})
        if bp.get("core_acting_rule"):
            lines.append(f"- 연기규칙: {bp['core_acting_rule']}")

        heuristics = bp.get("acting_heuristics", {})
        if heuristics:
            lines.append("- 행동패턴:")
            for hk, hv in heuristics.items():
                lines.append(f"  · {hv}")

        # Interaction styles
        istyles = bp.get("interaction_styles", [])
        if istyles:
            lines.append("- 상호작용 스타일:")
            for ist in istyles[:4]:
                lines.append(f"  · [{ist.get('condition', '')}] {ist.get('style', '')}")

        # Specific interactions with other on_screen chars
        spec = bp.get("specific_interactions", {})
        if spec:
            for other_name in on_screen:
                if other_name == name:
                    continue
                key = f"vs_{other_name}"
                if key in spec:
                    lines.append(f"  · {name}→{other_name}: {spec[key]}")
            if "vs_플레이어" in spec:
                lines.append(f"  · {name}→플레이어: {spec['vs_플레이어']}")

        # Relationship matrix — only for other on_screen chars
        rmatrix = ch.get("relationship_matrix", {})
        rel_parts = []
        for other_name in on_screen:
            if other_name == name:
                continue
            rel = rmatrix.get(other_name, {})
            if rel:
                rel_parts.append(
                    f"  · {other_name}: 호감={rel.get('호감', '?')}, "
                    f"신뢰={rel.get('신뢰', '?')}, "
                    f"긴장={rel.get('긴장', '?')}, "
                    f"역학={rel.get('dynamics', '?')} "
                    f"— {rel.get('comment', '')}"
                )
        player_rel = rmatrix.get("플레이어", {})
        if player_rel:
            rel_parts.append(
                f"  · 플레이어: 호감={player_rel.get('호감', '?')}, "
                f"신뢰={player_rel.get('신뢰', '?')}, "
                f"긴장={player_rel.get('긴장', '?')} "
                f"— {player_rel.get('comment', '')}"
            )
        if rel_parts:
            lines.append("- 현재 장면의 다른 캐릭터와의 관계:")
            lines.extend(rel_parts)

    lines.append("")

    # World
    lines.append("=== 세계관 ===")
    core_concept = world_db.get("core_concept", {})
    lines.append(f"작품명: {world_db.get('world_name', '사루라 아뜨리에')}")
    if core_concept.get("summary"):
        lines.append(f"요약: {core_concept['summary']}")
    if core_concept.get("genre"):
        lines.append(f"장르: {core_concept['genre']}")
    if core_concept.get("themes"):
        lines.append(f"테마: {', '.join(core_concept['themes'])}")

    main_stage = world_db.get("main_stage", {})
    if main_stage:
        lines.append(f"\n무대: {main_stage.get('full_name', main_stage.get('name', ''))}")
        if main_stage.get("background"):
            lines.append(f"배경: {main_stage['background']}")

    world_rules = world_db.get("world_rules", {})
    if world_rules:
        for rule_k, rule_v in world_rules.items():
            if isinstance(rule_v, str):
                lines.append(f"세계규칙({rule_k}): {rule_v}")
            elif isinstance(rule_v, dict):
                for sub_k, sub_v in rule_v.items():
                    lines.append(f"세계규칙({rule_k}.{sub_k}): {sub_v}")

    lines.append("")

    # Director instructions
    lines.append("=== 연출 지시 ===")
    lines.append(inject_director_brief(ui_settings))
    lines.append("")

    # Memory context (if available)
    memory = session_data.get("memory", {})
    if memory.get("action_context"):
        lines.append("=== 기억 컨텍스트 ===")
        lines.append(str(memory["action_context"]))
        lines.append("")
    if memory.get("dynamic_state"):
        lines.append("=== 동적 상태 ===")
        lines.append(str(memory["dynamic_state"]))
        lines.append("")

    # Prohibitions
    lines.append("=== 금지 ===")
    lines.append(f"- 절대 플레이어({player_name})의 대사를 쓰지 마세요.")
    lines.append("- 플레이어의 행동이나 감정을 묘사하지 마세요.")
    lines.append("- 빈 응답을 보내지 마세요.")
    lines.append(f"- dialogue 블록의 character 필드에 \"{player_name}\"를 절대 넣지 마세요.")
    lines.append("- script 배열이 비어있으면 안 됩니다.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------
def build_user_prompt(session_data: dict, user_input: str) -> str:
    player_name = session_data.get("player_name", "사용자")
    on_screen = session_data.get("on_screen", [])
    digest = session_data.get("flow_digest_10", [])

    parts: list[str] = []

    # Recent conversation log
    if digest:
        parts.append("=== 최근 대화 흐름 ===")
        for entry in digest:
            speaker = entry[0] if len(entry) > 0 else "?"
            text = entry[1] if len(entry) > 1 else ""
            parts.append(f"{speaker}: {text}")
        parts.append("")

    # Current scene
    parts.append("=== 현재 장면 ===")
    parts.append(f"등장인물: {', '.join(on_screen)}")
    parts.append(f"플레이어: {player_name}")
    parts.append("")

    # Player input
    parts.append(f"=== {player_name}의 입력 ===")
    parts.append(user_input)
    parts.append("")

    parts.append("위 입력에 대해 등장인물들이 반응하는 장면을 JSON script로 작성하세요.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
VALID_EMOTIONS = {
    "neutral", "happy", "sad", "angry", "surprised", "shy", "scared",
    "disgusted", "confused", "excited", "worried", "loving", "playful",
    "annoyed", "embarrassed", "smug", "tired", "determined", "melancholy",
}


def post_process_script(script: list, player_name: str) -> list:
    if not isinstance(script, list):
        return [{"type": "narration", "content": "(응답을 처리할 수 없습니다.)"}]

    cleaned = []
    for block in script:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype not in ("narration", "dialogue"):
            continue
        # Remove player dialogue
        if btype == "dialogue" and block.get("character") == player_name:
            continue
        # Validate emotion
        if btype == "dialogue":
            em = block.get("emotion", "neutral")
            if em not in VALID_EMOTIONS:
                block["emotion"] = "neutral"
        cleaned.append(block)

    # Ensure at least one block
    if not cleaned:
        cleaned.append({"type": "narration", "content": "(장면이 조용히 흘러갑니다...)"})

    return cleaned


def extract_emotion(script: list) -> str:
    for block in script:
        if block.get("type") == "dialogue" and block.get("emotion"):
            return block["emotion"]
    return "neutral"


def update_flow_digest(s: dict, turn: dict):
    digest = s.get("flow_digest_10", [])
    user_input = turn.get("user_input", "")
    if user_input and user_input not in ("[CONTINUE_SCENE]", "[SCENE_START]"):
        digest.append([s["player_name"], user_input])
    for block in turn.get("script", []):
        if block["type"] == "narration":
            digest.append(["[나레이션]", block["content"][:200]])
        elif block["type"] == "dialogue":
            digest.append([block.get("character", "?"), block["content"][:200]])
    s["flow_digest_10"] = digest[-MAX_HISTORY_LENGTH:]


# ---------------------------------------------------------------------------
# Fallback script
# ---------------------------------------------------------------------------
def fallback_script(on_screen: list) -> list:
    result: list[dict] = [
        {"type": "narration", "content": "잠시 어색한 침묵이 흐른다."}
    ]
    for name in on_screen[:3]:
        result.append({
            "type": "dialogue",
            "character": name,
            "content": "......",
            "emotion": "neutral",
            "monologue": "",
        })
    return result


# ---------------------------------------------------------------------------
# DIMA API call
# ---------------------------------------------------------------------------
@rate_limited_call
def _raw_dima_call(system_instruction: str, user_prompt: str, ui_settings: dict):
    """Single Gemini call for DIMA."""
    adult_mode = ui_settings.get("adult_mode", False)
    response = client.models.generate_content(
        model=MODEL_DIMA,
        contents=[user_prompt],
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=DIMA_SCHEMA,
            safety_settings=get_safety_settings(adult_mode),
            temperature=0.85,
        ),
    )
    return response


def call_dima(system_instruction: str, user_prompt: str, ui_settings: dict) -> list:
    """Call DIMA with retries and parse JSON response into script array."""
    last_error = None
    for attempt in range(3):
        try:
            response = _raw_dima_call(system_instruction, user_prompt, ui_settings)
            if not response or not response.text:
                logging.warning(f"DIMA attempt {attempt + 1}: empty response")
                continue
            raw_text = response.text.strip()
            parsed = json.loads(raw_text)
            script = parsed.get("script", [])
            if isinstance(script, list) and len(script) > 0:
                return script
            logging.warning(f"DIMA attempt {attempt + 1}: empty script array")
        except json.JSONDecodeError as e:
            logging.warning(f"DIMA attempt {attempt + 1}: JSON parse error: {e}")
            last_error = e
        except Exception as e:
            logging.warning(f"DIMA attempt {attempt + 1}: {e}")
            last_error = e
    raise RuntimeError(f"DIMA failed after 3 attempts: {last_error}")


# ---------------------------------------------------------------------------
# Maestro (background analysis)
# ---------------------------------------------------------------------------
@rate_limited_call
def _raw_maestro_call(prompt: str):
    response = client.models.generate_content(
        model=MODEL_MAESTRO,
        contents=[prompt],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )
    return response


def run_maestro_sync(session_data: dict) -> dict | None:
    """Analyse the last turn and produce updated memory state."""
    turns = session_data.get("turns", [])
    if not turns:
        return None

    last_turn = turns[-1]
    on_screen = session_data.get("on_screen", [])
    player_name = session_data.get("player_name", "사용자")
    memory = session_data.get("memory", {})

    prompt_parts = [
        "당신은 '사루라 아뜨리에'의 Maestro(기억 관리 AI)입니다.",
        "아래 마지막 턴의 내용을 분석하여, 세션의 기억 상태를 업데이트하세요.",
        "",
        f"플레이어: {player_name}",
        f"등장인물: {', '.join(on_screen)}",
        "",
        f"마지막 플레이어 입력: {last_turn.get('user_input', '')}",
        "",
        "마지막 턴 스크립트:",
    ]
    for block in last_turn.get("script", []):
        if block["type"] == "narration":
            prompt_parts.append(f"[나레이션] {block['content'][:300]}")
        elif block["type"] == "dialogue":
            prompt_parts.append(f"[{block.get('character', '?')}] {block['content'][:300]}")

    prompt_parts.extend([
        "",
        "현재 기억 상태:",
        json.dumps(memory, ensure_ascii=False, default=str)[:1500],
        "",
        "다음 JSON 형식으로 업데이트된 기억 상태를 반환하세요:",
        '{"action_context": "현재 상황 요약 (한국어, 2-3문장)",'
        ' "dynamic_state": "캐릭터들의 현재 감정/태도 변화 요약",'
        ' "relationship_matrix": {"캐릭터이름": {"플레이어_호감_변화": 0, "요약": "..."}}}'
    ])

    try:
        response = _raw_maestro_call("\n".join(prompt_parts))
        if not response or not response.text:
            return None
        return json.loads(response.text.strip())
    except Exception as e:
        logging.error(f"Maestro failed: {e}")
        return None


def _maestro_worker(sid: str):
    try:
        lock = get_session_lock(sid)
        with lock:
            s = load_session_file(sid)
            if not s or not s.get("turns"):
                return
            result = run_maestro_sync(s)
            if result:
                s["memory"] = result
                save_session_file(sid, s)
    except Exception as e:
        logging.error(f"Maestro worker error: {e}")


# =========================================================================
# FLASK ROUTES
# =========================================================================

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# /bootstrap
# ---------------------------------------------------------------------------
@app.route("/bootstrap", methods=["POST"])
def bootstrap():
    data = request.get_json(force=True)
    player_name = (data.get("player_name") or "사용자").strip() or "사용자"
    on_screen_names = data.get("on_screen_names", [])
    seed_text = data.get("seed_text", "")
    ui_settings = data.get("ui_settings", {})

    # Load existing session
    save_code = data.get("save_code")
    if save_code:
        s = load_session_file(save_code)
        if not s:
            return jsonify({"status": "error", "message": "세션을 찾을 수 없습니다."}), 404
        session["save_code"] = save_code
        return jsonify({"status": "ok", "sid": save_code, "state": s})

    # Validate on_screen_names
    chars_db = load_characters_db()
    valid_names = [n for n in on_screen_names if n in chars_db]
    if not valid_names:
        valid_names = list(chars_db.keys())[:2]

    # Create new session
    sid = "S-" + uuid.uuid4().hex[:8].upper()
    s = {
        "session_id": sid,
        "player_name": player_name,
        "on_screen": valid_names,
        "turns": [],
        "traffic_light": "GREEN",
        "ui_settings": merge_ui_settings(ui_settings),
        "flow_digest_10": [],
        "world": {},
        "memory": {},
    }

    # Build prompts and call DIMA for first turn
    sys_instr = build_system_instruction(s)
    user_prompt = build_user_prompt(s, seed_text or "[SCENE_START]")

    try:
        script = call_dima(sys_instr, user_prompt, s["ui_settings"])
        script = post_process_script(script, player_name)
    except Exception as e:
        logging.error(f"DIMA failed on bootstrap: {e}")
        script = fallback_script(valid_names)

    # Record turn 1
    turn = {
        "turn_id": 1,
        "user_input": seed_text or "",
        "script": script,
        "emotion": extract_emotion(script),
        "ts": datetime.utcnow().isoformat() + "Z",
    }
    s["turns"].append(turn)
    update_flow_digest(s, turn)

    save_session_file(sid, s)
    session["save_code"] = sid

    return jsonify({"status": "ok", "sid": sid, "state": s})


# ---------------------------------------------------------------------------
# /execute-turn
# ---------------------------------------------------------------------------
@app.route("/execute-turn", methods=["POST"])
def execute_turn():
    data = request.get_json(force=True)
    user_input = (data.get("user_input") or data.get("user_text") or "").strip()
    if not user_input:
        user_input = "[CONTINUE_SCENE]"
    ui_settings = data.get("ui_settings", {})
    on_screen_names = data.get("on_screen_names")

    sid = session.get("save_code")
    if not sid:
        return jsonify({"status": "error", "message": "세션이 없습니다."}), 400

    lock = get_session_lock(sid)
    with lock:
        s = load_session_file(sid)
        if not s:
            return jsonify({"status": "error", "message": "세션 파일을 찾을 수 없습니다."}), 404

        # Update settings
        s["ui_settings"] = merge_ui_settings(ui_settings)
        if on_screen_names:
            chars_db = load_characters_db()
            valid = [n for n in on_screen_names if n in chars_db]
            if valid:
                s["on_screen"] = valid

        s["traffic_light"] = "YELLOW"

        # Build prompts
        sys_instr = build_system_instruction(s)
        user_prompt = build_user_prompt(s, user_input)

        try:
            script = call_dima(sys_instr, user_prompt, s["ui_settings"])
            script = post_process_script(script, s["player_name"])
        except Exception as e:
            logging.error(f"DIMA failed on turn: {e}")
            script = fallback_script(s["on_screen"])

        # Record turn
        turn = {
            "turn_id": len(s["turns"]) + 1,
            "user_input": user_input,
            "script": script,
            "emotion": extract_emotion(script),
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        s["turns"].append(turn)
        update_flow_digest(s, turn)

        s["traffic_light"] = "GREEN"
        save_session_file(sid, s)

        # Queue Maestro every N turns
        if len(s["turns"]) % MAESTRO_UPDATE_FREQUENCY == 0:
            threading.Thread(
                target=_maestro_worker, args=(sid,), daemon=True
            ).start()

    return jsonify({"status": "ok", "sid": sid, "state": s})


# ---------------------------------------------------------------------------
# /get-character-profiles
# ---------------------------------------------------------------------------
@app.route("/get-character-profiles")
def get_character_profiles():
    chars_db = load_characters_db()
    profiles = []
    for name, data in chars_db.items():
        meta = data.get("metadata", {})
        profiles.append({
            "name": name,
            "eng": meta.get("eng", name.lower()),
            "color": meta.get("color", "#888888"),
        })
    return jsonify(profiles)


# ---------------------------------------------------------------------------
# /get-session-data
# ---------------------------------------------------------------------------
@app.route("/get-session-data")
def get_session_data():
    sid = session.get("save_code")
    if not sid:
        return jsonify({"status": "no_session"})
    s = load_session_file(sid)
    if not s:
        return jsonify({"status": "no_session"})
    return jsonify({"status": "ok", "state": s})


# ---------------------------------------------------------------------------
# /set_on_screen
# ---------------------------------------------------------------------------
@app.route("/set_on_screen", methods=["POST"])
def set_on_screen():
    data = request.get_json(force=True)
    names = data.get("names", [])

    sid = session.get("save_code")
    if not sid:
        return jsonify({"status": "error", "message": "세션이 없습니다."}), 400

    chars_db = load_characters_db()
    valid = [n for n in names if n in chars_db]
    if not valid:
        return jsonify({"status": "error", "message": "유효한 캐릭터가 없습니다."}), 400

    lock = get_session_lock(sid)
    with lock:
        s = load_session_file(sid)
        if not s:
            return jsonify({"status": "error", "message": "세션 파일을 찾을 수 없습니다."}), 404
        s["on_screen"] = valid
        save_session_file(sid, s)

    return jsonify({"status": "ok", "on_screen": valid})


# ---------------------------------------------------------------------------
# /reset-session
# ---------------------------------------------------------------------------
@app.route("/reset-session")
def reset_session():
    session.pop("save_code", None)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# /branch_from_turn
# ---------------------------------------------------------------------------
@app.route("/branch_from_turn", methods=["POST"])
def branch_from_turn():
    data = request.get_json(force=True)
    turn_index = data.get("turn_index", 0)

    sid = session.get("save_code")
    if not sid:
        return jsonify({"status": "error", "message": "세션이 없습니다."}), 400

    lock = get_session_lock(sid)
    with lock:
        s = load_session_file(sid)
        if not s:
            return jsonify({"status": "error", "message": "세션 파일을 찾을 수 없습니다."}), 404

        if turn_index < 0 or turn_index >= len(s.get("turns", [])):
            return jsonify({"status": "error", "message": "잘못된 턴 인덱스입니다."}), 400

        # Deep copy and truncate
        branched = copy.deepcopy(s)
        new_sid = "S-" + uuid.uuid4().hex[:8].upper()
        branched["session_id"] = new_sid
        branched["turns"] = branched["turns"][: turn_index + 1]
        branched["flow_digest_10"] = branched["flow_digest_10"][:MAX_HISTORY_LENGTH]

        save_session_file(new_sid, branched)
        session["save_code"] = new_sid

    return jsonify({"status": "ok", "sid": new_sid, "state": branched})


# ---------------------------------------------------------------------------
# /novelize
# ---------------------------------------------------------------------------
@app.route("/novelize", methods=["POST"])
def novelize():
    data = request.get_json(force=True)
    sid = session.get("save_code")
    if not sid:
        return jsonify({"status": "error", "message": "세션이 없습니다."}), 400

    s = load_session_file(sid)
    if not s or not s.get("turns"):
        return jsonify({"status": "error", "message": "턴 데이터가 없습니다."}), 400

    # Build prompt from turns
    turn_texts = []
    player_name = s.get("player_name", "사용자")
    for turn in s["turns"]:
        if turn.get("user_input") and turn["user_input"] not in ("[CONTINUE_SCENE]", "[SCENE_START]"):
            turn_texts.append(f"[{player_name}] {turn['user_input']}")
        for block in turn.get("script", []):
            if block["type"] == "narration":
                turn_texts.append(f"[나레이션] {block['content']}")
            elif block["type"] == "dialogue":
                turn_texts.append(f"[{block.get('character', '?')}] {block['content']}")

    novel_prompt = (
        "당신은 소설가입니다. 아래 대화/나레이션 로그를 기반으로 "
        "한국어 소설 형식으로 재구성하세요. "
        "캐릭터의 감정과 분위기를 살려 문학적으로 표현하세요.\n\n"
        + "\n".join(turn_texts)
    )

    try:
        response = client.models.generate_content(
            model=MODEL_DIMA,
            contents=[novel_prompt],
            config=genai_types.GenerateContentConfig(temperature=0.7),
        )
        novel_text = response.text if response and response.text else ""
    except Exception as e:
        logging.error(f"Novelize failed: {e}")
        novel_text = ""

    if not novel_text:
        return jsonify({"status": "error", "message": "소설화에 실패했습니다."}), 500

    # Save novel data
    share_code = "N-" + uuid.uuid4().hex[:8].upper()
    novel_data = {
        "share_code": share_code,
        "title": f"{player_name}의 이야기",
        "text": novel_text,
        "player_name": player_name,
        "on_screen": s.get("on_screen", []),
        "turn_count": len(s["turns"]),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    # Save to session
    lock = get_session_lock(sid)
    with lock:
        s_reload = load_session_file(sid)
        if s_reload:
            s_reload["novel_data"] = novel_data
            save_session_file(sid, s_reload)

    # Save shared copy
    shared_path = SHARED_NOVELS_DIR / f"{share_code}.json"
    try:
        with open(shared_path, "w", encoding="utf-8") as f:
            json.dump(novel_data, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.error(f"Failed to save shared novel: {traceback.format_exc()}")

    return jsonify({"status": "ok", "novel_data": novel_data})


# ---------------------------------------------------------------------------
# /get-my-novel
# ---------------------------------------------------------------------------
@app.route("/get-my-novel", methods=["POST"])
def get_my_novel():
    sid = session.get("save_code")
    if not sid:
        return jsonify({"status": "error", "message": "세션이 없습니다."}), 400

    s = load_session_file(sid)
    if not s:
        return jsonify({"status": "error", "message": "세션을 찾을 수 없습니다."}), 404

    novel_data = s.get("novel_data")
    if not novel_data:
        return jsonify({"status": "error", "message": "소설 데이터가 없습니다."}), 404

    return jsonify({"status": "ok", "novel_data": novel_data})


# ---------------------------------------------------------------------------
# /get-shared-novel
# ---------------------------------------------------------------------------
@app.route("/get-shared-novel", methods=["POST"])
def get_shared_novel():
    data = request.get_json(force=True)
    share_code = (data.get("share_code") or "").strip()
    if not share_code:
        return jsonify({"status": "error", "message": "공유 코드가 필요합니다."}), 400

    try:
        safe_code = _safe_filename(share_code)
    except ValueError:
        return jsonify({"status": "error", "message": "잘못된 공유 코드입니다."}), 400

    shared_path = SHARED_NOVELS_DIR / f"{safe_code}.json"
    if not shared_path.exists():
        return jsonify({"status": "error", "message": "공유된 소설을 찾을 수 없습니다."}), 404

    try:
        with open(shared_path, "r", encoding="utf-8") as f:
            novel_data = json.load(f)
    except Exception:
        return jsonify({"status": "error", "message": "소설 로드에 실패했습니다."}), 500

    return jsonify({"status": "ok", "novel_data": novel_data})


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
