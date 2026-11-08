import os
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="CodeMitra API", version="2.0.0")

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


@app.get("/")
def health():
    return {"status": "ok", "service": "CodeMitra API", "version": "2.0.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    client = get_client()
    messages = build_messages(request)

    def generate():
        global last_successful_model
        failures = []
        for model in model_order():
            collected = []
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
                        collected.append(text)
                answer = "".join(collected)
                if not answer.strip():
                    failures.append(f"{model}: empty response")
                    continue
                with model_lock:
                    last_successful_model = model
                # Buffer each attempt so a failed provider cannot leak a partial
                # answer before the next fallback model starts.
                for start in range(0, len(answer), 96):
                    yield answer[start:start + 96]
                return
            except Exception as exc:
                failures.append(f"{model}: {str(exc)[:140]}")
        yield "CodeMitra could not get a response from the available AI models. Please try again later."

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
