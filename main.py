import os
import json
import asyncio
import wave
import base64
import io
import zipfile
from pathlib import PurePosixPath
from threading import Lock
from time import monotonic
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="CodeMitra API", version="2.3.0")

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
rate_lock = Lock()
rate_events: dict[str, list[float]] = {}
piper_lock = Lock()
piper_cached_name: str | None = None
piper_cached_voice = None
PIPER_VOICES = {
    "male": ("rohan", "hi_IN-rohan-medium"),
    "female": ("priyamvada", "hi_IN-priyamvada-medium"),
}


def enforce_rate_limit(request: Request, bucket: str, limit: int, window_seconds: int) -> None:
    """Small per-client in-memory guard for expensive voice provider calls."""
    client_ip = request.client.host if request.client else "unknown"
    key = f"{bucket}:{client_ip}"
    now = monotonic()
    with rate_lock:
        recent = [stamp for stamp in rate_events.get(key, []) if now - stamp < window_seconds]
        if len(recent) >= limit:
            raise HTTPException(status_code=429, detail="Too many requests. Please wait and try again.")
        recent.append(now)
        rate_events[key] = recent


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


def build_messages(request: ChatRequest, attachment_text=None, image_parts=None) -> list[dict]:
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
            "and never claim code was executed unless it actually was. For complete project requests, emit every file in its own fenced code block with filename=relative/path.ext in the opening info string, for example python filename=src/main.py. Include all essential files and no placeholders."
        ),
    }]
    for item in request.history[-10:]:
        messages.append({"role": item.role, "content": item.content})
    user_text = request.message
    if attachment_text:
        user_text += "\n\nATTACHED FILE CONTENTS:\n" + "\n\n".join(attachment_text)
    if image_parts:
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}] + image_parts})
    else:
        messages.append({"role": "user", "content": user_text})
    return messages




TEXT_EXTENSIONS = {".txt", ".md", ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".kt", ".java", ".gradle", ".properties", ".sql", ".sh", ".csv", ".log", ".dart", ".swift", ".c", ".cpp", ".h", ".go", ".rs", ".php", ".rb", ".vue", ".svelte"}

async def read_attachments(uploads: list[UploadFile]):
    texts, images, total = [], [], 0
    for upload in uploads[:10]:
        data = await upload.read(15 * 1024 * 1024 + 1)
        total += len(data)
        if len(data) > 15 * 1024 * 1024 or total > 30 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Attachments exceed 30 MB total.")
        name = (upload.filename or "attachment").replace(chr(92), "/").split("/")[-1][:160]
        ctype = upload.content_type or "application/octet-stream"
        ext = PurePosixPath(name).suffix.lower()
        if ctype in {"image/jpeg", "image/png", "image/webp"}:
            encoded = base64.b64encode(data).decode("ascii")
            images.append({"type": "image_url", "image_url": {"url": "data:" + ctype + ";base64," + encoded}})
            texts.append("[Image attached: " + name + "]")
        elif ext == ".zip" or ctype in {"application/zip", "application/x-zip-compressed"}:
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    expanded = 0
                    for info in archive.infolist()[:80]:
                        if info.is_dir() or info.file_size > 1000000:
                            continue
                        path = info.filename.replace(chr(92), "/")
                        if ".." in PurePosixPath(path).parts:
                            continue
                        expanded += info.file_size
                        if expanded > 8000000:
                            break
                        if PurePosixPath(path).suffix.lower() in TEXT_EXTENSIONS:
                            texts.append("--- ZIP FILE: " + path + " ---\n" + archive.read(info).decode("utf-8", errors="replace")[:12000])
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Invalid ZIP archive.")
        elif ext == ".pdf" or ctype == "application/pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                texts.append("--- PDF: " + name + " ---\n" + "\n".join((p.extract_text() or "") for p in reader.pages[:30])[:30000])
            except Exception:
                texts.append("[PDF text extraction unavailable for " + name + "]")
        elif ext == ".docx":
            try:
                from xml.etree import ElementTree
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    root = ElementTree.fromstring(archive.read("word/document.xml"))
                content = " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
                texts.append("--- DOCX: " + name + " ---\n" + content[:30000])
            except Exception:
                texts.append("[DOCX text extraction unavailable for " + name + "]")
        elif ext in TEXT_EXTENSIONS or ctype.startswith("text/"):
            texts.append("--- FILE: " + name + " ---\n" + data.decode("utf-8", errors="replace")[:20000])
        else:
            texts.append("[Binary attachment: " + name + "]")
    return texts, images


def model_order() -> list[str]:
    with model_lock:
        preferred = [last_successful_model] if last_successful_model else []
    return preferred + [model for model in PREFERRED_MODELS if model not in preferred]


def eleven_key() -> str:
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="ElevenLabs is not configured.")
    return key


def synthesize_piper(text: str, gender: str) -> bytes:
    """CPU-friendly offline Hindi TTS. Downloads each voice model on first use."""
    global piper_cached_name, piper_cached_voice
    speaker, model_name = PIPER_VOICES[gender]
    model_dir = os.getenv("PIPER_MODEL_DIR", "/tmp/codemitra-piper")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name + ".onnx")
    config_path = model_path + ".json"
    base_url = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        f"hi/hi_IN/{speaker}/medium/{model_name}"
    )

    with piper_lock:
        import httpx
        from piper import PiperVoice

        for path, url, max_bytes in (
            (model_path, base_url + ".onnx", 80 * 1024 * 1024),
            (config_path, base_url + ".onnx.json", 64 * 1024),
        ):
            if not os.path.isfile(path) or os.path.getsize(path) < 100:
                temp_path = path + ".part"
                try:
                    with httpx.stream("GET", url, follow_redirects=True, timeout=180) as response:
                        response.raise_for_status()
                        total = 0
                        with open(temp_path, "wb") as output:
                            for chunk in response.iter_bytes():
                                total += len(chunk)
                                if total > max_bytes:
                                    raise RuntimeError("Piper voice download exceeded its size limit.")
                                output.write(chunk)
                    if total < 100:
                        raise RuntimeError("Piper voice download was empty.")
                    os.replace(temp_path, path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        if piper_cached_name != model_name or piper_cached_voice is None:
            piper_cached_voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=False)
            piper_cached_name = model_name

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            piper_cached_voice.synthesize_wav(text, wav_file)
        return buffer.getvalue()


@app.get("/")
def health():
    return {"status": "ok", "service": "CodeMitra API", "version": "2.3.0"}


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
async def voice_speak(request: SpeechRequest, http_request: Request):
    enforce_rate_limit(http_request, "voice_speak", 30, 60)
    key = eleven_key()
    voice_id = request.voice_id or (
        os.getenv("VOICE_MALE_ID") if request.gender == "male" else os.getenv("VOICE_FEMALE_ID")
    )
    if not voice_id:
        raise HTTPException(status_code=503, detail="Set VOICE_MALE_ID or VOICE_FEMALE_ID in backend environment.")
    payload = {
        "text": request.text,
        "model_id": "eleven_v4_turbo",
        "language_code": "hi" if request.lang == "hi" else "en",
        "apply_text_normalization": "auto",
        "voice_settings": {
            "stability": 0.58 if request.style == "professional" else 0.42 if request.style == "energetic" else 0.52,
            "similarity_boost": 0.78,
            "style": 0.15 if request.style == "professional" else 0.55 if request.style == "energetic" else 0.35,
            "use_speaker_boost": True,
        },
    }
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
    http_request: Request,
    name: str = Form(..., min_length=2, max_length=80),
    consent: bool = Form(...),
    audio: UploadFile = File(...),
):
    enforce_rate_limit(http_request, "voice_clone", 3, 3600)
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
async def chat_stream(
    message: str = Form(..., min_length=1, max_length=12000),
    lang: Literal["hi", "en"] = Form("hi"),
    history: str = Form("[]"),
    files: list[UploadFile] = File(default=[]),
):
    try:
        request = ChatRequest(message=message, lang=lang, history=json.loads(history))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid chat history: " + str(exc)[:120])
    attachment_text, image_parts = await read_attachments(files)
    client = get_client()
    messages = build_messages(request, attachment_text, image_parts)

    def generate():
        global last_successful_model
        for model in model_order():
            received_content = False
            try:
                stream = client.chat.completions.create(model=model, messages=messages, max_tokens=1800, stream=True)
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    piece = chunk.choices[0].delta.content
                    if piece:
                        received_content = True
                        with model_lock:
                            last_successful_model = model
                        yield piece
                if received_content:
                    return
            except Exception:
                if received_content:
                    yield "\n\n[Response interrupted by the AI provider. Please retry.]"
                    return
        yield "CodeMitra could not get a response from the available AI models. Please try again later."

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})



class ProjectFile(BaseModel):
    path: str = Field(min_length=1, max_length=240)
    content: str = Field(max_length=2000000)


class ProjectZipRequest(BaseModel):
    project_name: str = Field(default="CodeMitra-Project", min_length=1, max_length=80)
    files: list[ProjectFile] = Field(min_length=1, max_length=100)


@app.post("/project/zip")
def create_project_zip(request: ProjectZipRequest):
    buffer = io.BytesIO()
    seen = set()
    total_chars = 0
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in request.files:
            total_chars += len(item.content)
            if total_chars > 15000000:
                raise HTTPException(status_code=413, detail="Project ZIP content exceeds 15 MB.")
            normalized = item.path.replace(chr(92), "/").strip().lstrip("/")
            parts = PurePosixPath(normalized).parts
            if not normalized or any(part in {"..", "."} for part in parts) or ":" in normalized:
                raise HTTPException(status_code=400, detail="Unsafe project file path.")
            path = "/".join(parts)
            if path in seen:
                raise HTTPException(status_code=400, detail="Duplicate project file path.")
            seen.add(path)
            archive.writestr(path, item.content)
    name = "".join(c for c in request.project_name if c.isalnum() or c in "-_ ")[:60].strip() or "CodeMitra-Project"
    return Response(content=buffer.getvalue(), media_type="application/zip", headers={"Content-Disposition": "attachment; filename=\"" + name + ".zip\"", "Cache-Control": "no-store"})
