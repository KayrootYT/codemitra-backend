import os
from threading import Lock
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="CodeMitra API", version="2.1.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://kayrootyt.github.io,http://localhost:3000,http://127.0.0.1:5500",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

PREFERRED_MODELS = [
    os.getenv("MODEL", "deepseek-ai/deepseek-v4.1-flash"),
    "z-ai/glm-5.3-flash",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "google/gemma-4-31b-it",
    "openai/gpt-oss-20b",
]
last_successful_model = None
model_lock = Lock()


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    lang: Literal["hi", "en"] = "hi"
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4500)
    voice_id: str | None = Field(default=None, max_length=128)
    style: Literal["friendly", "calm", "energetic", "professional"] = "friendly"
    gender: Literal["male", "female"] = "female"
    lang: Literal["hi", "en"] = "hi"


def get_client() -> OpenAI:
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="NVIDIA_API_KEY is not configured")
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=key,
        timeout=45,
        max_retries=0,
    )


def build_messages(request: ChatRequest) -> list[dict]:
    language = (
        "Hindi (use Roman or Devanagari script to match the user)"
        if request.lang == "hi"
        else "English"
    )
    messages = [{
        "role": "system",
        "content": (
            "You are CodeMitra, a helpful expert coding assistant. "
            f"Reply in {language}. Be accurate, practical, and clear. "
            "Explain assumptions, provide runnable code in fenced markdown blocks, "
            "and never claim code was executed unless it actually was."
        ),
    }]
    for item in request.history[-10:]:
        messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": request.message})
    return messages


def model_order() -> list[str]:
    with model_lock:
        preferred = [last_successful_model] if last_successful_model else []
    return preferred + [model for model in PREFERRED_MODELS if model not in preferred]


def eleven_key() -> str:
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="Natural voice is not configured. Add ELEVENLABS_API_KEY on the backend.")
    return key


@app.get("/")
def health():
    return {"status": "ok", "service": "CodeMitra API", "version": "2.1.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/voice/config")
def voice_config():
    return {
        "provider": "elevenlabs" if os.getenv("ELEVENLABS_API_KEY") else "browser",
        "male_available": bool(os.getenv("VOICE_MALE_ID")),
        "female_available": bool(os.getenv("VOICE_FEMALE_ID")),
        "clone_available": bool(os.getenv("ELEVENLABS_API_KEY")),
        "styles": ["friendly", "calm", "energetic", "professional"],
    }


@app.post("/voice/speak")
async def voice_speak(request: SpeechRequest):
    key = eleven_key()
    voice_id = request.voice_id or (
        os.getenv("VOICE_MALE_ID") if request.gender == "male" else os.getenv("VOICE_FEMALE_ID")
    )
    if not voice_id:
        raise HTTPException(status_code=503, detail="Set VOICE_MALE_ID or VOICE_FEMALE_ID in backend environment.")
    style_prompts = {
        "friendly": "Speak warmly and naturally, like a helpful friend.",
        "calm": "Speak calmly, gently, and at a relaxed pace.",
        "energetic": "Speak with positive energy and lively but natural emphasis.",
        "professional": "Speak clearly and confidently in a polished professional tone.",
    }
    payload = {
        "text": request.text,
        "model_id": "eleven_v4_turbo",
        "language_code": "hi" if request.lang == "hi" else "en",
        "apply_text_normalization": "auto",
        "voice_settings": {"stability": 0.48, "similarity_boost": 0.78, "style": 0.35, "use_speaker_boost": True},
    }
    # Delivery direction is prepended as a non-spoken model instruction.
    payload["text"] = f"[{style_prompts[request.style]}] {request.text}"
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            result = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": key, "Accept": "audio/mpeg", "Content-Type": "application/json"},
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Voice provider connection failed: {str(exc)[:160]}")
    if result.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Voice generation failed ({result.status_code}): {result.text[:300]}")
    return Response(content=result.content, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@app.post("/voice/clone")
async def clone_voice(
    name: str = Form(..., min_length=2, max_length=80),
    consent: bool = Form(...),
    audio: UploadFile = File(...),
):
    if not consent:
        raise HTTPException(status_code=400, detail="Explicit voice-owner consent is required.")
    key = eleven_key()
    allowed_types = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/webm", "audio/ogg"}
    if audio.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Upload an MP3, WAV, M4A, WebM, or OGG audio recording.")
    data = await audio.read(20 * 1024 * 1024 + 1)
    if len(data) < 10_000:
        raise HTTPException(status_code=400, detail="Audio is too short. Use a clear recording of at least 10 seconds.")
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file must be 20 MB or smaller.")
    files = {"files": (audio.filename or "voice-sample.mp3", data, audio.content_type)}
    form = {"name": name, "description": "CodeMitra user-created voice profile"}
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            result = await client.post(
                "https://api.elevenlabs.io/v1/voices/add",
                headers={"xi-api-key": key},
                data=form,
                files=files,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Voice cloning service connection failed: {str(exc)[:160]}")
    if result.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Voice cloning failed ({result.status_code}): {result.text[:300]}")
    body = result.json()
    if not body.get("voice_id"):
        raise HTTPException(status_code=502, detail="Voice provider did not return a voice ID.")
    return {"voice_id": body["voice_id"], "name": name, "message": "Voice profile created. Keep consent and original recording."}


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    client = get_client()
    messages = build_messages(request)

    def generate():
        global last_successful_model
        failures = []
        for model in model_order():
            received_content = False
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1800,
                    stream=True,
                )
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    text = chunk.choices[0].delta.content
                    if text:
                        received_content = True
                        with model_lock:
                            last_successful_model = model
                        yield text
                if received_content:
                    return
                failures.append(f"{model}: empty response")
            except Exception as exc:
                failures.append(f"{model}: {str(exc)[:140]}")
                if received_content:
                    yield "\n\n[Response interrupted by the AI provider. Please retry to get a complete answer.]"
                    return
        yield "CodeMitra could not get a response from the available AI models. Please try again later."

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
