"""
Sarura Atelier — Local Flask Backend (R03.11+ compatible)
Uses google-genai SDK, matching the reference Colab backend architecture.
"""

import os, json, re, uuid, copy, time, threading, logging
from pathlib import Path
from datetime import datetime
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

MODEL_DIMA = "gemini-2.5-flash"

# ─── Flask App ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(32)
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
                    "monologue": {"type": "string"},
                },
                "required": ["type", "content"],
            },
        }
    },
    "required": ["script"],
}

# ─── Allowed emotions ────────────────────────────────────────
EMOTION_SET_20 = [
    "neutral", "agreement", "annoyance", "apology", "confident assertion",
    "contempt", "curiosity", "direct flirting", "ecstatic joy", "explosive anger",
    "gentle tease", "greeting", "melancholy", "nervous laughter", "playful denial",
    "quiet affection", "reluctant admission", "shocked silence", "shy embarrassment",
    "suspicion",
]


def get_allowed_emotions(ui_settings: dict) -> list:
    return EMOTION_SET_20


# ─── Session helpers ─────────────────────────────────────────
_session_locks: Dict[str, threading.RLock] = {}
_session_locks_lock = threading.Lock()


def get_session_lock(sid: str) -> threading.RLock:
    with _session_locks_lock:
        if sid not in _session_locks:
            _session_locks[sid] = threading.RLock()
        return _session_locks[sid]


def session_path(sid: str) -> Path:
    return SESSIONS_DIR / f"{sid}.json"


def now_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


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
    p.write_text(
        json.dumps(_to_jsonable(s), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def to_public_state(s: dict) -> dict:
    """Convert session to a JSON-safe public state dict."""
    return _to_jsonable(copy.deepcopy(s))


# ─── Session init / migrate ──────────────────────────────────
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
        "memory": {},
    }
    return s


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
    return "neutral"


# ─── Build system instruction for D.I.M.A ────────────────────
def build_system_instruction_for_scene(s: dict, on_screen_chars: list) -> str:
    packets = []
    player_name = s.get("player_name", "사용자")

    world_rules_db = WORLD_DB.get("world_rules", {})
    rules_text = json.dumps(world_rules_db, ensure_ascii=False, indent=2)

    for name in on_screen_chars:
        if name == player_name:
            continue

        char_db = CHARACTERS_DB.get(name, {})
        full_persona = {
            "character_name": name,
            "appearance": char_db.get("appearance"),
            "identity": char_db.get("identity"),
            "personality_dna": char_db.get("personality_dna"),
            "social_tuning": char_db.get("social_tuning"),
            "core_values": char_db.get("core_values"),
            "behavior_protocols": char_db.get("behavior_protocols"),
        }

        # Relationship context
        relations_brief = []
        full_rel_matrix = char_db.get("relationship_matrix", {})
        for other in on_screen_chars:
            if name == other:
                continue
            rel = full_rel_matrix.get(other)
            if rel and isinstance(rel, dict):
                relations_brief.append(
                    f"- vs {other}: {rel.get('comment', '')} "
                    f"(호감:{rel.get('호감', 50)}, 신뢰:{rel.get('신뢰', 50)})"
                )
        full_persona["relationships_in_this_scene"] = (
            "\n".join(relations_brief) if relations_brief
            else "No specific relationships in this scene."
        )
        packets.append(json.dumps(full_persona, ensure_ascii=False, indent=2))

    return (
        "You are a master AI actor for a fictional theatrical play. "
        "Your primary directive is to portray the following characters based on their detailed persona blueprints. "
        "Adhere strictly to their personalities, speech patterns, and relationships. "
        "This is your role for the entire duration of the scene.\n\n"
        "### WORLD RULES TO FOLLOW ###\n"
        f"{rules_text}\n\n"
        "### CHARACTERS ON SCENE ###\n" + "\n".join(packets)
    )


def inject_director_brief(ui_settings: dict) -> str:
    parts = []
    if ui_settings.get("pov_first_person"):
        parts.append("- [시점]: 1인칭 시점으로 NPC들이 플레이어에게 직접 말하는 것처럼 연기하라.")
    else:
        parts.append("- [시점]: 3인칭 관찰자 시점으로 서술하라.")

    if ui_settings.get("show_monologue"):
        parts.append("- [내면 독백]: 각 대사에 monologue 필드를 채워 캐릭터의 내면을 드러내라.")

    genre = ui_settings.get("genre_preset", "auto")
    if genre and genre != "auto":
        parts.append(f"- [장르]: '{genre}' 장르의 분위기와 톤에 맞추어 연기하라.")

    tempo = ui_settings.get("tempo", 5)
    parts.append(f"- [템포]: {tempo}/10 (낮을수록 느리고 묘사적, 높을수록 빠르고 액션 중심)")

    narr_ratio = ui_settings.get("narration_ratio", 40)
    parts.append(f"- [나레이션 비율]: {narr_ratio}% (나레이션과 대사의 비율)")

    desc_focus = ui_settings.get("description_focus", 5)
    parts.append(f"- [묘사 집중도]: {desc_focus}/10")

    return "\n".join(parts)


# ─── Build D.I.M.A prompt ────────────────────────────────────
DIMA_PROMPT_TEMPLATE = SAFETY_PREAMBLE + """
You are D.I.M.A. (Dynamic Immersive Montage Agent), a master storyteller and character director.
You have been given the full personas of the characters on scene via a system instruction.
Your task is to use that established knowledge to act out the current turn.

[CRITICAL OUTPUT RULE] Your entire output MUST be a single, valid, complete JSON object with a "script" array.

# [TURN CONTRACT]
# 1. NARRATION: Start with 2-4 sentences of rich narration describing the scene/atmosphere/non-verbal actions.
# 2. DIALOGUE: Then, provide 3-6 lines of proactive, in-character dialogue from the NPCs.
# 3. MOMENTUM: Advance the story, don't wait passively.

# [PLAYER INTERACTION GUARD]
# Player is an external operator. Do NOT write lines for the player.
# Never output dialogue whose speaker equals the player's name.
# If a line would go to the player, convert it to short narration that invites the user's reply.

# [Narrator's POV Limitation]
# Never describe the player's emotions/thoughts/intentions directly.
# Only describe NPC actions/emotions and objective surroundings.

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

### Allowed Emotions ###
{emotion_set_json}

[OUTPUT FORMAT]
Return JSON: {{"script": [...]}}
Each element: {{"type": "narration"|"dialogue"|"monologue", "content": "...", "character": "Name", "emotion": "...", "monologue": "..."}}
- "narration" blocks: only "type" and "content" required
- "dialogue" blocks: "type", "content", "character", "emotion" required; "monologue" optional
- "monologue" blocks: "type", "content", "character" required
"""


def _short(txt: str, n: int = 100) -> str:
    s = str(txt or "")
    return s if len(s) <= n else s[:n] + "..."


def build_dima_prompt(s: dict, user_input: str) -> tuple:
    """Returns (system_instruction, main_prompt)."""
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
        user_input_for_prompt = f"[This is a line of dialogue from the player]: '{user_input}'"

    # Recent conversation log
    flow_digest = s.get("flow_digest_10", [])
    if isinstance(flow_digest, deque):
        flow_digest = list(flow_digest)
    log_entries = []
    for entry in flow_digest:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            speaker, line = entry
            log_entries.append(f"- {speaker}: {_short(line)}")
    recent_conversation_log = "\n".join(log_entries) if log_entries else "이번 씬의 첫 대화입니다."

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

    # Director brief
    director_brief = inject_director_brief(s.get("ui_settings", {}))
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

    emotion_set_json = json.dumps(get_allowed_emotions(s.get("ui_settings", {})), ensure_ascii=False)

    main_prompt = DIMA_PROMPT_TEMPLATE.format(
        recent_conversation_log=recent_conversation_log,
        world_event_brief="No significant world event this turn.",
        character_briefs=character_briefs_content,
        director_brief=director_brief,
        location_and_time=f"{location}",
        player_name=player_name,
        user_input=user_input_for_prompt,
        emotion_set_json=emotion_set_json,
    )

    return system_instruction, main_prompt


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
        **config_kwargs,
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_DIMA,
                contents=[genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=prompt)]
                )],
                config=config,
            )
            text = response.text
            if text:
                # Try direct parse
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Try to extract JSON from mixed output
                    cleaned = re.sub(r'```json\s*|\s*```', '', text.strip())
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
            logger.warning(f"Empty LLM response on attempt {attempt+1}")
        except Exception as e:
            emsg = str(e).lower()
            logger.error(f"LLM call error (attempt {attempt+1}): {e}")
            if "429" in emsg or "quota" in emsg:
                time.sleep(15 * (attempt + 1))
            elif "500" in emsg:
                time.sleep(3 * (attempt + 1))
            else:
                time.sleep(2)

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
        "emotion": "greeting",
    })
    return script


def post_process_script(script: list, s: dict) -> list:
    """Validate and clean up script blocks."""
    if not isinstance(script, list):
        return [{"type": "narration", "content": "(AI 응답 형식 오류)"}]

    player_name = get_player_name(s)
    on_screen_set = set(s.get("on_screen", []))
    allowed_emotions = get_allowed_emotions(s.get("ui_settings", {}))
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
            if block.get("emotion") not in allowed_emotions:
                block["emotion"] = "greeting" if "greeting" in allowed_emotions else "neutral"
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
            "emotion": "greeting" if "greeting" in allowed_emotions else "neutral",
        })
    return processed


# ─── Core turn engine ─────────────────────────────────────────
def run_dima_turn(s: dict, user_input: str) -> list:
    """Run one D.I.M.A turn and return the final script array."""
    system_instruction, main_prompt = build_dima_prompt(s, user_input)

    raw = generate_llm(
        prompt=main_prompt,
        system_instruction=system_instruction,
        temperature=0.75,
        response_schema=DIMA_SCHEMA,
    )

    script = (raw or {}).get("script") or []

    # Validate: must have at least one dialogue block with content
    if not any(
        isinstance(b, dict) and b.get("type") == "dialogue" and b.get("content")
        for b in script
    ):
        script = safe_local_script(s, prelude_narration="")

    return post_process_script(script, s)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


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
        import random
        pool = [n for n in CHARACTERS_DB.keys() if n != player_name]
        if not pool:
            return jsonify({"status": "error", "message": "No available NPCs."}), 409
        final_cast.append(random.choice(pool))

    s["on_screen"] = final_cast
    merge_ui_settings(s, data.get("ui_settings"))

    seed_text = (data.get("seed_text") or "").strip() or "[PLAYER_PAUSE]"

    # Generate first turn
    try:
        final_script = run_dima_turn(s, seed_text)
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

    # Update flow digest
    flow = s.get("flow_digest_10", [])
    if isinstance(flow, list):
        flow = deque(flow, maxlen=10)
    for b in final_script:
        if b.get("type") == "dialogue":
            flow.append((b["character"], b["content"]))
        elif b.get("type") == "narration":
            flow.append(("[System]", b["content"]))
    s["flow_digest_10"] = list(flow)

    s["traffic_light"] = "GREEN"
    session["session_id"] = sid
    save_session(s)

    return jsonify({"status": "ok", "sid": sid, "state": to_public_state(s)})


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
            import random
            pool = [n for n in CHARACTERS_DB.keys() if n != me]
            if not pool:
                return jsonify({"status": "error", "message": "사용 가능한 NPC가 없습니다."}), 409
            s["on_screen"] = [random.choice(pool)]

        final_script = run_dima_turn(s, user_input)

        turn_id = (len(s.get("turns", [])) + 1)
        turn_payload = {
            "turn_id": turn_id,
            "ts": now_ts(),
            "user_input": user_input if user_input not in ("[CONTINUE_SCENE]", "[PLAYER_PAUSE]") else "",
            "script": final_script,
            "emotion": extract_emotion_from_script(final_script),
            "ui_settings_snapshot": copy.deepcopy(s.get("ui_settings", {})),
        }
        s.setdefault("turns", []).append(turn_payload)

        # Update flow digest
        flow = s.get("flow_digest_10", [])
        if isinstance(flow, list):
            flow = deque(flow, maxlen=10)
        if turn_payload["user_input"]:
            flow.append((f"[{me}]", turn_payload["user_input"]))
        for b in final_script:
            if b.get("type") == "dialogue":
                flow.append((b.get("character", "?"), b.get("content", "")))
            elif b.get("type") == "narration":
                flow.append(("[System]", b.get("content", "")))
        s["flow_digest_10"] = list(flow)

        s["traffic_light"] = "GREEN"
        save_session(s)

    return jsonify({"status": "ok", "sid": sid, "state": to_public_state(s)})


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


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500


# ─── Run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"★ Sarura Atelier → http://localhost:{port}")
    from waitress import serve
    serve(app, host="0.0.0.0", port=port)
