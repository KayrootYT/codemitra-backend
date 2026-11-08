import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
    timeout=40,
)

MODEL = os.environ.get("MODEL", "deepseek-ai/deepseek-v4.1-flash")

PREFERRED = [
    MODEL,
    "deepseek-ai/deepseek-v4.1-flash",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.3-flash",
    "minimaxai/minimax-m3.1-flash",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-nano-3-30b-a3b",
    "google/gemma-4-31b-it",
    "openai/gpt-oss-20b",
    "mistralai/mistral-large-4-1t-a52b",
    "poolside/laguna-xs-2.1",
]

working = {"model": None}


class Req(BaseModel):
    message: str
    lang: str = "hi"


def live_models():
    try:
        return [m.id for m in client.models.list().data]
    except Exception:
        return []


def candidates():
    live = live_models()
    out = []
    if working["model"]:
        out.append(working["model"])
    for m in PREFERRED:
        if m not in out and (not live or m in live):
            out.append(m)
    return out or PREFERRED


@app.get("/")
def home():
    return {"status": "ok", "model": working["model"] or MODEL}


@app.get("/models")
def models():
    ids = live_models()
    return ids[:300] if ids else {"error": "could not fetch models"}


@app.post("/chat")
def chat(r: Req):
    lang = ("Hindi (Roman or Devanagari, match the user)"
            if r.lang == "hi" else "English")
    msgs = [
        {"role": "system",
         "content": f"You are CodeMitra, an expert coding assistant. "
                    f"Reply in {lang}. Keep code in markdown code blocks."},
        {"role": "user", "content": r.message},
    ]
    errors = []
    for m in candidates():
        try:
            resp = client.chat.completions.create(
                model=m, messages=msgs, max_tokens=1000)
            text = resp.choices[0].message.content
            if not text:
                raise ValueError("empty reply")
            working["model"] = m
            return {"reply": text}
        except Exception as e:
            if working["model"] == m:
                working["model"] = None
            errors.append(f"{m}: {str(e)[:70]}")
    return {"reply": "Sab models fail hue:\n" + "\n".join(errors)}
