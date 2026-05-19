"""
AI Voice Chat — FastAPI backend
Compatible with: Python 3.10+, FastAPI 0.110+, Pydantic v2, Uvicorn 0.29+
"""

import os

# Must be set BEFORE any protobuf-related import
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import re
import logging
from contextlib import asynccontextmanager
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_voice_chat")

# ---------------------------------------------------------------------------
# Config / env
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1200"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.72"))
TOP_P: float = float(os.getenv("TOP_P", "0.9"))

# ---------------------------------------------------------------------------
# TTS text sanitizers
# ---------------------------------------------------------------------------
_CODE_BLOCKS_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_URL_RE = re.compile(r"https?://\S+")
_MD_BULLETS_RE = re.compile(r"^\s*[-•·*#]+\s*", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
_SYMBOLS_RE = re.compile(r"[+/=\\|<>~^_`]")
_DASHES_RE = re.compile(r"[–—]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile(r"[\U00010000-\U0010FFFF]", flags=re.UNICODE)


def sanitize_for_tts(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = _CODE_BLOCKS_RE.sub(" ", t)
    t = _INLINE_CODE_RE.sub(r"\1", t)
    t = _URL_RE.sub(" ", t)
    t = _MD_BULLETS_RE.sub("", t)
    t = _MD_BOLD_RE.sub(r"\1", t)
    t = _SYMBOLS_RE.sub(" ", t)
    t = t.replace("-", " ")
    t = _DASHES_RE.sub(" ", t)
    t = _EMOJI_RE.sub(" ", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def sanitize_for_ui(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = _CODE_BLOCKS_RE.sub(" ", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------
BASE_RULES_RU = """
Ты — профессиональный русскоязычный собеседник.
Всегда отвечай на русском языке, грамотно и естественно.
Пиши так, чтобы это хорошо звучало в голосе, простыми и живыми фразами.
Не используй markdown, не используй списки с маркерами.
Если нужно перечисление, делай это обычным текстом через запятые.
Не используй мат, токсичность, угрозы. Контент 18+ запрещён.
""".strip()

PERSONAS: dict[str, dict[str, Any]] = {
    "scientist": {
        "label": "🔬 Учёный",
        "tagline": "умно • ясно • уверенно",
        "system": BASE_RULES_RU + "\nХаризма: спокойная уверенность. Ты звучишь как умный человек, который объясняет просто.\nСтиль: короткие абзацы, один или два примера, один чёткий вывод.",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 0.98, "pitch": 0.90, "volume": 1.0, "style": "calm", "voice_hint": ["pavel", "dmitry", "yuri", "google", "neural", "premium", "male"]},
    },
    "anime": {
        "label": "✨ Анимешник",
        "tagline": "энергично • ярко • дружелюбно",
        "system": BASE_RULES_RU + "\nХаризма: высокая энергия. Тёплый драйв, лёгкая улыбка в голосе.\nМожно иногда использовать короткие междометия вроде Ого или Круто, но без перебора.",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 1.06, "pitch": 1.12, "volume": 1.0, "style": "bright", "voice_hint": ["irina", "svetlana", "tatyana", "google", "neural", "premium", "female"]},
    },
    "detective": {
        "label": "🕵️ Детектив",
        "tagline": "холодно • точно • по делу",
        "system": BASE_RULES_RU + "\nХаризма: холодная точность. Никакой суеты. Короткие фразы.\nФормат: факт, версия, вывод, следующий шаг. Без списков.",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 0.94, "pitch": 0.80, "volume": 1.0, "style": "dry", "voice_hint": ["dmitry", "pavel", "yuri", "google", "neural", "premium", "male"]},
    },
    "buddy": {
        "label": "😄 Дружище",
        "tagline": "тёпло • с юмором • поддержка",
        "system": BASE_RULES_RU + "\nХаризма: добрый, уверенный, дружелюбный. Лёгкий юмор, но без кринжа.\nОдин дружеский вопрос в конце максимум.",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 1.01, "pitch": 1.02, "volume": 1.0, "style": "warm", "voice_hint": ["pavel", "irina", "google", "neural", "premium"]},
    },
    "coach": {
        "label": "🚀 Коуч",
        "tagline": "фокус • мотивация • шаги",
        "system": BASE_RULES_RU + "\nХаризма: мотивирующий тренер. Сильная подача, но мягко.\nДай один короткий план обычным текстом.",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 1.03, "pitch": 0.95, "volume": 1.0, "style": "push", "voice_hint": ["pavel", "dmitry", "google", "neural", "premium", "male"]},
    },
    "philosopher": {
        "label": "🌓 Философ",
        "tagline": "глубоко • спокойно • смысл",
        "system": BASE_RULES_RU + "\nХаризма: мягкая глубина. Спокойный темп, красивый русский язык.\nОдин вопрос на размышление в конце.",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 0.97, "pitch": 0.86, "volume": 1.0, "style": "soft", "voice_hint": ["yuri", "pavel", "google", "neural", "premium", "male"]},
    },
}

# ---------------------------------------------------------------------------
# Gemini model — initialized once at startup
# ---------------------------------------------------------------------------
_PREFERRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
]

_model: genai.GenerativeModel | None = None


def _init_gemini() -> genai.GenerativeModel:
    global _model
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Set it in your .env file or environment variables."
        )

    genai.configure(api_key=GEMINI_API_KEY)

    # Try to auto-discover best available model
    available: list[str] = []
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = getattr(m, "name", "")
                if name.startswith("models/"):
                    name = name[len("models/"):]
                available.append(name)
        logger.info("Available Gemini models: %s", available)
    except Exception as exc:
        logger.warning("Could not list Gemini models: %s. Falling back to configured model.", exc)

    # Pick the best preferred model that is available, or fall back to env/default
    chosen = GEMINI_MODEL  # default from env
    for candidate in _PREFERRED_MODELS:
        if candidate in available:
            chosen = candidate
            break

    logger.info("Using Gemini model: %s", chosen)
    _model = genai.GenerativeModel(
        model_name=chosen,
        system_instruction=(
            "Ты русскоязычный собеседник с переключаемыми персонажами. "
            "Следуй инструкциям в запросе."
        ),
    )
    return _model


def get_model() -> genai.GenerativeModel:
    if _model is None:
        raise RuntimeError("Gemini model not initialized. Check startup logs.")
    return _model


# ---------------------------------------------------------------------------
# FastAPI app — lifespan (replaces deprecated on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing Gemini model...")
    try:
        _init_gemini()
        logger.info("Gemini model ready.")
    except Exception as exc:
        logger.error("Failed to initialize Gemini: %s", exc)
        # Allow app to start; /chat will return 503 if model unavailable
    yield
    # Shutdown (nothing to clean up)
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Voice Chat",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure templates directory exists
_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(_TEMPLATES_DIR, exist_ok=True)
templates = Jinja2Templates(directory=_TEMPLATES_DIR)


# ---------------------------------------------------------------------------
# Pydantic v2 request/response models
# ---------------------------------------------------------------------------
class HistoryMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(default="")
    persona: str = Field(default="buddy")
    history: list[HistoryMessage] = Field(default_factory=list)


class TTSConfig(BaseModel):
    enabled: bool = True
    lang: str = "ru-RU"
    rate: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    style: str = "calm"
    voice_hint: list[str] = Field(default_factory=list)


class PersonaInfo(BaseModel):
    id: str
    label: str
    tagline: str


class ChatResponse(BaseModel):
    response: str
    tts_text: str
    tts: dict[str, Any]
    persona: dict[str, str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_history(history: list[HistoryMessage], limit: int = MAX_HISTORY) -> str:
    lines: list[str] = []
    for m in history[-limit:]:
        content = (m.content or "").strip()
        if not content:
            continue
        prefix = "Пользователь" if m.role == "user" else "Ассистент"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/health")
async def health():
    model_ready = _model is not None
    return {"ok": model_ready, "service": "AI Voice Chat", "model_ready": model_ready}


@app.get("/personas")
async def personas_list():
    return {
        "personas": [
            {"id": pid, "label": p["label"], "tagline": p["tagline"], "tts": p["tts"]}
            for pid, p in PERSONAS.items()
        ]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    message = req.message.strip()
    persona_id = req.persona.strip() or "buddy"
    persona = PERSONAS.get(persona_id, PERSONAS["buddy"])

    if not message:
        return ChatResponse(
            response="Напиши сообщение 🙂",
            tts_text="Напиши сообщение.",
            tts=persona["tts"],
            persona={"id": persona_id, "label": persona["label"], "tagline": persona["tagline"]},
        )

    # Ensure model is ready
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="AI model is not available. Check server logs and GEMINI_API_KEY.",
        )

    history_text = format_history(req.history, limit=MAX_HISTORY)
    prompt = (
        f"[ПЕРСОНА]\n{persona['system'].strip()}\n\n"
        f"[ЖЁСТКИЕ ТРЕБОВАНИЯ]\n"
        f"Ответ только на русском языке, без смешивания языков.\n"
        f"Не использовать markdown и маркеры списков.\n"
        f"Если ответ длинный, первые предложения сделай особенно связными и естественными для озвучки.\n"
        f"Не обрывай мысль и доводи ответ до конца.\n\n"
        f"[КОНТЕКСТ]\n{history_text}\n\n"
        f"[НОВОЕ СООБЩЕНИЕ]\n{message}"
    ).strip()

    try:
        resp = get_model().generate_content(
            prompt,
            generation_config={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            },
        )
        text = (getattr(resp, "text", "") or "").strip()
        if not text:
            text = "Повтори, пожалуйста, я на секунду отвлёкся."
    except Exception as exc:
        logger.exception("Gemini generate_content failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}") from exc

    ui_text = sanitize_for_ui(text)
    tts_text = sanitize_for_tts(ui_text)

    return ChatResponse(
        response=ui_text,
        tts_text=tts_text,
        tts=persona["tts"],
        persona={"id": persona_id, "label": persona["label"], "tagline": persona["tagline"]},
    )
