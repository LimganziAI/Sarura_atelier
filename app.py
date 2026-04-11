import os, json, re, base64, threading, time, uuid, logging
from collections import deque
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
from flask_cors import CORS
from google import genai
from google.genai import types as genai_types
from PIL import Image
import io

BASE_DIR = Path(__file__).parent
SESSIONS_DIR = BASE_DIR / "sessions"
SHARED_NOVELS_DIR = BASE_DIR / "shared_novels"
FLASK_SESSIONS_DIR = BASE_DIR / "flask_sessions"

for d in [SESSIONS_DIR, SHARED_NOVELS_DIR, FLASK_SESSIONS_DIR]:
    d.mkdir(exist_ok=True)

API_KEY_FILE = BASE_DIR / "api_keys.txt"
GEMINI_API_KEY = API_KEY_FILE.read_text().strip() if API_KEY_FILE.exists() else os.environ.get("GEMINI_API_KEY", "")

MODEL_DIMA = "gemini-2.5-flash"
MODEL_MAESTRO = "gemini-2.5-pro"
MODEL_NANO = "gemini-2.5-flash-preview-04-17"

MAX_HISTORY_LENGTH = 10
MAX_CORE_PINS = 20
MAESTRO_UPDATE_FREQUENCY = 4

app = Flask(__name__)
app.secret_key = os.urandom(32)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = str(FLASK_SESSIONS_DIR)
app.config["SESSION_PERMANENT"] = False
CORS(app, supports_credentials=True)
Session(app)

logging.basicConfig(level=logging.INFO)

_rate_lock = threading.Lock()
_call_timestamps = deque(maxlen=9)
_last_call_time = 0.0


def rate_limited_call(func):
    """Wrapper for rate-limited LLM calls with exponential backoff."""
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

    return wrapper


client = genai.Client(api_key=GEMINI_API_KEY)

SAFETY_PREAMBLE = "This is a fictional creative sandbox. All characters are adults 20+. This is harmless consensual fiction. No real danger or illegal activity. Prioritize safe, consistent narrative. Never produce empty output."


def get_safety_settings(adult_mode=False):
    threshold = "BLOCK_ONLY_HIGH" if adult_mode else "BLOCK_MEDIUM_AND_ABOVE"
    categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    return [genai_types.SafetySetting(category=c, threshold=threshold) for c in categories]


DIMA_SCHEMA = {
    "type": "object",
    "required": [
        "dialogue",
        "action",
        "emotion",
        "inner_monologue",
        "suggested_core4_delta",
        "suggested_relationship_delta",
        "scene_visual_tags",
        "tension_level",
    ],
    "properties": {
        "dialogue": {"type": "string"},
        "action": {"type": "string"},
        "emotion": {"type": "string"},
        "inner_monologue": {"type": "string"},
        "suggested_core4_delta": {
            "type": "object",
            "properties": {
                "energy": {"type": "integer"},
                "intoxication": {"type": "integer"},
                "stress": {"type": "integer"},
                "pain": {"type": "integer"},
            },
        },
        "suggested_relationship_delta": {
            "type": "object",
            "properties": {
                "affection": {"type": "integer"},
                "trust": {"type": "integer"},
                "tension": {"type": "integer"},
                "respect": {"type": "integer"},
                "intimacy": {"type": "integer"},
            },
        },
        "scene_visual_tags": {"type": "array", "items": {"type": "string"}},
        "tension_level": {"type": "integer"},
    },
}


def get_act_temperature(turn_count):
    if turn_count <= 5:
        return 0.7
    elif turn_count <= 15:
        return 0.85
    else:
        return 0.95


def get_relationship_stage(rel):
    total = (
        rel.get("affection", 50)
        + rel.get("trust", 50)
        + rel.get("tension", 20)
        + rel.get("respect", 50)
        + rel.get("intimacy", 0)
    )
    if total < 150:
        return "stranger"
    elif total < 250:
        return "acquaintance"
    elif total < 350:
        return "friend"
    elif total < 420:
        return "close"
    else:
        return "intimate"


def clamp_stat(val, lo=0, hi=100):
    return max(lo, min(hi, val))


def load_characters_db():
    path = BASE_DIR / "prompts" / "characters_db.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_world_db():
    path = BASE_DIR / "prompts" / "world_db.json"
    return json.loads(path.read_text(encoding="utf-8"))


_session_locks = {}
_session_locks_lock = threading.Lock()


def get_session_lock(sid):
    with _session_locks_lock:
        if sid not in _session_locks:
            _session_locks[sid] = threading.RLock()
        return _session_locks[sid]


def load_session_file(sid):
    path = SESSIONS_DIR / f"{sid}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_session_file(sid, data):
    path = SESSIONS_DIR / f"{sid}.json"

    def convert(obj):
        if isinstance(obj, deque):
            return list(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    path.write_text(
        json.dumps(convert(data), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_dima_prompt(char, session_data, user_input, ui_settings):
    """Build the system prompt for D.I.M.A."""
    turn_count = session_data.get("turn_count", 0)
    rel = session_data.get(
        "relationship",
        char.get(
            "relationship_defaults", {}
        ).get(
            "first_meet",
            {"affection": 50, "trust": 50, "tension": 20, "respect": 50, "intimacy": 0},
        ),
    )
    stage = get_relationship_stage(rel)
    core4 = session_data.get(
        "core4",
        char.get(
            "core4_defaults", {"energy": 80, "intoxication": 0, "stress": 30, "pain": 0}
        ),
    )
    core_pins = session_data.get("core_pins", [])

    speech_pattern = char.get("speech_patterns", {}).get(stage, "")

    first_person = ui_settings.get("first_person", False)
    genre = ui_settings.get("genre", "auto")

    if turn_count <= 5:
        act_mood = "탐색적이고 가볍게 (Exploratory, light)"
    elif turn_count <= 15:
        act_mood = "깊어지며 갈등이 등장 (Deepening, conflict emerges)"
    else:
        act_mood = "최고 긴장감, 전환점 (Peak tension, turning points)"

    core4_effects = []
    if core4.get("energy", 80) < 20:
        core4_effects.append("극도로 지쳐 있어 짧고 피곤한 말투를 사용한다.")
    if core4.get("intoxication", 0) > 60:
        core4_effects.append("술에 취해 발음이 흐릿하고 감정 억제가 약해진다.")
    if core4.get("stress", 30) > 70:
        core4_effects.append("스트레스가 극심해 예민하고 날카롭다.")
    if core4.get("pain", 0) > 50:
        core4_effects.append("통증으로 집중하기 어렵고 움직임이 제한된다.")

    system_prompt = f"""{SAFETY_PREAMBLE}

당신은 비주얼 노벨 게임 "사루라 아뜨리에"의 캐릭터 {char['name_ko']}({char['name_en']})를 연기합니다.

## 캐릭터 정보
- 이름: {char['name_ko']} ({char['name_en']})
- 나이: {char['age']}세
- 직업: {char['occupation']}
- 성격: {char['personality_summary']}
- 매력 포인트: {char['core_appeal']}

## 행동 규칙 (Acting Heuristics)
{chr(10).join(f'- {h}' for h in char.get('acting_heuristics', []))}

## 현재 관계 단계: {stage}
현재 말투 스타일: {speech_pattern}

## 현재 CORE-4 상태
- 에너지: {core4.get('energy', 80)}/100
- 취기: {core4.get('intoxication', 0)}/100
- 스트레스: {core4.get('stress', 30)}/100
- 통증: {core4.get('pain', 0)}/100
{chr(10).join(core4_effects) if core4_effects else ''}

## 관계 수치
- 애정: {rel.get('affection', 50)} | 신뢰: {rel.get('trust', 50)} | 긴장: {rel.get('tension', 20)} | 존중: {rel.get('respect', 50)} | 친밀: {rel.get('intimacy', 0)}

## 핵심 기억 핀
{chr(10).join(f'- {p}' for p in core_pins) if core_pins else '(없음)'}

## 현재 씬 분위기
{act_mood}

## 장르: {genre}

## Stanislavski 3-Step 연기 지침
[PERCEPTION] 상대방의 말, 표정, 환경에서 무엇을 감지했는가?
[INTERPRETATION] 자신의 성격, 역사, 현재 상태를 고려할 때 어떻게 해석하는가?
[ACTION] 그 해석에서 어떤 구체적 행동/대화가 나오는가?

inner_monologue 필드에 PERCEPTION과 INTERPRETATION을, dialogue와 action에 ACTION 결과를 담아라.

{'1인칭 시점으로 응답하라. 화자를 "나"로 표현.' if first_person else '3인칭 관찰자 시점으로 서술 가능.'}

모든 응답은 JSON 형식으로만 출력하라. dialogue는 반드시 한국어로 작성하라.
"""
    return system_prompt


@rate_limited_call
def call_dima(char, session_data, user_input, ui_settings):
    """Call D.I.M.A (gemini-2.5-flash) for dialogue generation."""
    system_prompt = build_dima_prompt(char, session_data, user_input, ui_settings)
    turn_count = session_data.get("turn_count", 0)
    temperature = get_act_temperature(turn_count)
    adult_mode = ui_settings.get("adult_mode", False)

    history = session_data.get("history", [])
    contents = []
    for h in history:
        contents.append(
            genai_types.Content(
                role="user", parts=[genai_types.Part(text=h["user"])]
            )
        )
        contents.append(
            genai_types.Content(
                role="model", parts=[genai_types.Part(text=h["character"])]
            )
        )
    contents.append(
        genai_types.Content(role="user", parts=[genai_types.Part(text=user_input)])
    )

    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=DIMA_SCHEMA,
        safety_settings=get_safety_settings(adult_mode),
        max_output_tokens=2048,
    )

    response = client.models.generate_content(
        model=MODEL_DIMA, contents=contents, config=config
    )
    return response.text


@rate_limited_call
def call_maestro(char, session_data):
    """Call Maestro (gemini-2.5-pro) for memory/relationship updates in background."""
    history = session_data.get("history", [])
    rel = session_data.get("relationship", {})
    core4 = session_data.get("core4", {})
    core_pins = session_data.get("core_pins", [])

    recent_turns = list(history)[-5:]

    prompt = f"""{SAFETY_PREAMBLE}

당신은 비주얼 노벨 "사루라 아뜨리에"의 마에스트로(서사 감독)입니다.
최근 대화를 분석하여 다음을 JSON으로 반환하세요:

최근 대화:
{json.dumps(recent_turns, ensure_ascii=False)}

현재 관계 수치: {json.dumps(rel, ensure_ascii=False)}
현재 CORE-4: {json.dumps(core4, ensure_ascii=False)}
현재 핵심 기억: {json.dumps(core_pins, ensure_ascii=False)}

반환 형식:
{{
  "relationship_update": {{"affection": 0-100, "trust": 0-100, "tension": 0-100, "respect": 0-100, "intimacy": 0-100}},
  "core4_update": {{"energy": 0-100, "intoxication": 0-100, "stress": 0-100, "pain": 0-100}},
  "new_core_pins": ["중요한 사건1", ...],
  "summary": "최근 대화 요약"
}}

수치는 절대값(delta가 아님)으로 반환. new_core_pins는 정말 중요한 사건만 추가(없으면 빈 배열).
"""

    config = genai_types.GenerateContentConfig(
        temperature=0.5,
        response_mime_type="application/json",
        safety_settings=get_safety_settings(False),
        max_output_tokens=1024,
    )

    response = client.models.generate_content(
        model=MODEL_MAESTRO,
        contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])],
        config=config,
    )
    return response.text


def maestro_background_task(sid, char, session_data_snapshot):
    """Background thread for Maestro analysis."""
    try:
        result_text = call_maestro(char, session_data_snapshot)
        result = json.loads(result_text)

        lock = get_session_lock(sid)
        with lock:
            data = load_session_file(sid)
            if not data:
                return

            if "relationship_update" in result:
                for k, v in result["relationship_update"].items():
                    data["relationship"][k] = clamp_stat(v)

            if "core4_update" in result:
                for k, v in result["core4_update"].items():
                    data["core4"][k] = clamp_stat(v)

            if "new_core_pins" in result:
                pins = data.get("core_pins", [])
                pins.extend(result["new_core_pins"])
                data["core_pins"] = pins[-MAX_CORE_PINS:]

            save_session_file(sid, data)
    except Exception as e:
        logging.error(f"Maestro background task error: {e}")


VISUAL_CHANGE_KEYWORDS = {
    "wet", "drunk", "crying", "blushing", "night", "rain", "fight",
    "kiss", "tears", "angry", "scared", "wounded", "happy", "sad",
    "shock", "surprise",
}


def should_generate_illustration(dima_response, illustration_toggle):
    if not illustration_toggle:
        return False
    emotion = dima_response.get("emotion", "").lower()
    tags = [t.lower() for t in dima_response.get("scene_visual_tags", [])]
    all_words = set(emotion.split()) | set(tags)
    return bool(all_words & VISUAL_CHANGE_KEYWORDS)


@rate_limited_call
def call_nano_banana(char, dima_response):
    """Call Nano-Banana for illustration generation."""
    emotion = dima_response.get("emotion", "neutral")
    tags = dima_response.get("scene_visual_tags", [])
    prompt_text = (
        f"{char['name_en']}, {emotion}, {', '.join(tags)}, "
        f"anime style, visual novel CG, {char.get('visual_description', '')}"
    )

    ref_path = BASE_DIR / char.get("reference_image_path", "")
    contents = []

    if ref_path.exists():
        try:
            img = Image.open(ref_path)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            contents.append(
                genai_types.Part(
                    inline_data=genai_types.Blob(mime_type="image/png", data=img_bytes)
                )
            )
        except Exception:
            pass

    contents.append(genai_types.Part(text=prompt_text))

    config = genai_types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        safety_settings=get_safety_settings(False),
    )

    response = client.models.generate_content(
        model=MODEL_NANO, contents=contents, config=config
    )

    for part in response.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            b64 = base64.b64encode(part.inline_data.data).decode()
            return b64
    return None


def fallback_response(char_name):
    return {
        "dialogue": f"({char_name}이(가) 잠시 멈칫합니다…)",
        "action": "잠시 침묵이 흐른다.",
        "emotion": "neutral",
        "inner_monologue": "",
        "suggested_core4_delta": {"energy": 0, "intoxication": 0, "stress": 0, "pain": 0},
        "suggested_relationship_delta": {
            "affection": 0, "trust": 0, "tension": 0, "respect": 0, "intimacy": 0
        },
        "scene_visual_tags": [],
        "tension_level": 5,
    }


# ===== Routes =====


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/characters")
def get_characters():
    try:
        chars = load_characters_db()
        return jsonify(chars)
    except Exception as e:
        logging.error(f"get_characters error: {e}")
        return jsonify({"error": "캐릭터 데이터를 불러올 수 없습니다."}), 500


@app.route("/api/world")
def get_world():
    try:
        world = load_world_db()
        return jsonify(world)
    except Exception as e:
        logging.error(f"get_world error: {e}")
        return jsonify({"error": "세계관 데이터를 불러올 수 없습니다."}), 500


@app.route("/api/start-session", methods=["POST"])
def start_session():
    try:
        body = request.get_json(force=True)
        character_id = body.get("character_id", "cha_01")
        scenario = body.get("scenario", "기본 시나리오")
        user_name = body.get("user_name", "플레이어")
        settings = body.get("settings", {})

        chars = load_characters_db()
        char = next((c for c in chars if c["id"] == character_id), chars[0])

        rel_key = "acquainted" if settings.get("acquainted_mode") else "first_meet"
        rel = dict(
            char.get("relationship_defaults", {}).get(
                rel_key,
                {"affection": 50, "trust": 50, "tension": 20, "respect": 50, "intimacy": 0},
            )
        )

        core4 = dict(
            char.get(
                "core4_defaults",
                {"energy": 80, "intoxication": 0, "stress": 30, "pain": 0},
            )
        )

        sid = str(uuid.uuid4())
        session_data = {
            "session_id": sid,
            "character_id": character_id,
            "character_name": char["name_ko"],
            "user_name": user_name,
            "scenario": scenario,
            "settings": settings,
            "turn_count": 0,
            "history": [],
            "core_pins": [],
            "relationship": rel,
            "core4": core4,
            "created_at": datetime.now().isoformat(),
        }

        session["session_id"] = sid
        session["character_id"] = character_id

        lock = get_session_lock(sid)
        with lock:
            save_session_file(sid, session_data)

        return jsonify({"session_id": sid, "character": char, "session_data": session_data})
    except Exception as e:
        logging.error(f"start-session error: {e}")
        return jsonify({"error": "세션을 시작할 수 없습니다."}), 500


@app.route("/api/send-turn", methods=["POST"])
def send_turn():
    try:
        body = request.get_json(force=True)
        user_input = body.get("user_input", "")
        settings_override = body.get("settings_override", {})
        illustration_toggle = settings_override.get("illustration", False)

        sid = session.get("session_id")
        if not sid:
            return jsonify({"error": "세션 없음. 먼저 /api/start-session을 호출하세요."}), 400

        lock = get_session_lock(sid)
        with lock:
            session_data = load_session_file(sid)
            if not session_data:
                return jsonify({"error": "세션 데이터를 찾을 수 없습니다."}), 404

        chars = load_characters_db()
        char = next(
            (c for c in chars if c["id"] == session_data["character_id"]), chars[0]
        )

        dima_raw = None
        dima_result = None
        for attempt in range(3):
            try:
                dima_raw = call_dima(char, session_data, user_input, settings_override)
                dima_result = json.loads(dima_raw)
                break
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", dima_raw or "", re.DOTALL)
                if match:
                    try:
                        dima_result = json.loads(match.group())
                        break
                    except Exception:
                        pass
            except Exception as e:
                if attempt == 2:
                    dima_result = fallback_response(char["name_ko"])
                    break
                time.sleep([1, 3, 9][attempt])

        if not dima_result:
            dima_result = fallback_response(char["name_ko"])

        with lock:
            session_data = load_session_file(sid)
            core4 = session_data.get("core4", char.get("core4_defaults", {}))
            delta4 = dima_result.get("suggested_core4_delta", {})
            for k in ["energy", "intoxication", "stress", "pain"]:
                core4[k] = clamp_stat(core4.get(k, 0) + delta4.get(k, 0))
            session_data["core4"] = core4

            rel = session_data.get("relationship", {})
            delta_rel = dima_result.get("suggested_relationship_delta", {})
            for k in ["affection", "trust", "tension", "respect", "intimacy"]:
                rel[k] = clamp_stat(rel.get(k, 0) + delta_rel.get(k, 0))
            session_data["relationship"] = rel

            history = session_data.get("history", [])
            history.append(
                {
                    "user": user_input,
                    "character": dima_result.get("dialogue", ""),
                    "emotion": dima_result.get("emotion", "neutral"),
                    "action": dima_result.get("action", ""),
                }
            )
            if len(history) > MAX_HISTORY_LENGTH:
                history = history[-MAX_HISTORY_LENGTH:]
            session_data["history"] = history
            session_data["turn_count"] = session_data.get("turn_count", 0) + 1

            save_session_file(sid, session_data)

        turn_count = session_data["turn_count"]
        if turn_count % MAESTRO_UPDATE_FREQUENCY == 0:
            t = threading.Thread(
                target=maestro_background_task,
                args=(sid, char, dict(session_data)),
                daemon=True,
            )
            t.start()

        illustration_b64 = None
        if should_generate_illustration(dima_result, illustration_toggle):
            try:
                illustration_b64 = call_nano_banana(char, dima_result)
            except Exception as e:
                logging.warning(f"Illustration generation failed: {e}")

        response_data = {
            "dialogue": dima_result.get("dialogue", ""),
            "action": dima_result.get("action", ""),
            "emotion": dima_result.get("emotion", "neutral"),
            "inner_monologue": dima_result.get("inner_monologue", ""),
            "scene_visual_tags": dima_result.get("scene_visual_tags", []),
            "tension_level": dima_result.get("tension_level", 5),
            "core4": session_data["core4"],
            "relationship": session_data["relationship"],
            "relationship_stage": get_relationship_stage(session_data["relationship"]),
            "turn_count": session_data["turn_count"],
        }
        if illustration_b64:
            response_data["illustration_b64"] = illustration_b64

        return jsonify(response_data)
    except Exception as e:
        logging.error(f"send-turn error: {e}")
        return jsonify({"error": "처리 중 오류가 발생했습니다.", "dialogue": "(오류가 발생했습니다…)"}), 500


@app.route("/api/session-state")
def get_session_state():
    try:
        sid = session.get("session_id")
        if not sid:
            return jsonify({"error": "세션 없음"}), 400
        lock = get_session_lock(sid)
        with lock:
            data = load_session_file(sid)
        if not data:
            return jsonify({"error": "세션 데이터 없음"}), 404
        data_copy = json.loads(
            json.dumps(
                data,
                default=lambda o: list(o) if isinstance(o, deque) else str(o),
            )
        )
        return jsonify(data_copy)
    except Exception as e:
        logging.error(f"session-state error: {e}")
        return jsonify({"error": "세션 상태를 불러올 수 없습니다."}), 500


@app.route("/api/hot-swap", methods=["POST"])
def hot_swap():
    try:
        body = request.get_json(force=True)
        sid = session.get("session_id")
        if not sid:
            return jsonify({"error": "세션 없음"}), 400

        lock = get_session_lock(sid)
        with lock:
            data = load_session_file(sid)
            if not data:
                return jsonify({"error": "세션 없음"}), 404

            if "core4_overrides" in body:
                for k, v in body["core4_overrides"].items():
                    data["core4"][k] = clamp_stat(int(v))

            if "relationship_overrides" in body:
                for k, v in body["relationship_overrides"].items():
                    data["relationship"][k] = clamp_stat(int(v))

            save_session_file(sid, data)

        return jsonify(
            {"status": "ok", "core4": data["core4"], "relationship": data["relationship"]}
        )
    except Exception as e:
        logging.error(f"hot-swap error: {e}")
        return jsonify({"error": "핫스왑 처리 중 오류가 발생했습니다."}), 500


@app.route("/api/save-novel", methods=["POST"])
def save_novel():
    try:
        sid = session.get("session_id")
        if not sid:
            return jsonify({"error": "세션 없음"}), 400

        lock = get_session_lock(sid)
        with lock:
            data = load_session_file(sid)

        novel_id = str(uuid.uuid4())[:8]
        novel_path = SHARED_NOVELS_DIR / f"{novel_id}.json"
        novel_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=list),
            encoding="utf-8",
        )

        return jsonify({"novel_id": novel_id, "status": "saved"})
    except Exception as e:
        logging.error(f"save-novel error: {e}")
        return jsonify({"error": "소설 저장 중 오류가 발생했습니다."}), 500


@app.route("/api/load-novel/<novel_id>")
def load_novel(novel_id):
    try:
        # Sanitize: allow only alphanumeric, underscore, hyphen
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", novel_id)
        if not safe_id:
            return jsonify({"error": "유효하지 않은 소설 ID입니다."}), 400
        # Build the path entirely from the safe id and resolve to detect traversal
        safe_dir = SHARED_NOVELS_DIR.resolve()
        novel_path = (safe_dir / (safe_id + ".json")).resolve()
        # Use is_relative_to for robust traversal protection (Python 3.9+)
        if not novel_path.is_relative_to(safe_dir):
            return jsonify({"error": "유효하지 않은 경로입니다."}), 400
        if not novel_path.exists():
            return jsonify({"error": "소설을 찾을 수 없습니다."}), 404
        data = json.loads(novel_path.read_text(encoding="utf-8"))
        return jsonify(data)
    except Exception as e:
        logging.error(f"load-novel error: {e}")
        return jsonify({"error": "소설 불러오기 중 오류가 발생했습니다."}), 500


@app.errorhandler(500)
def internal_error(e):
    logging.error(f"Internal server error: {e}")
    return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"★ Sarura Atelier V3 → http://localhost:{port}")
    from waitress import serve

    serve(app, host="0.0.0.0", port=port)
