import os, time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
    timeout=25,
    max_retries=0,
)

PREFERRED = [
    "deepseek-ai/deepseek-v4.1-flash",
    "z-ai/glm-5.3-flash",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "google/gemma-4-31b-it",
    "openai/gpt-oss-20b",
]

state = {"model": None}


class Req(BaseModel):
    message: str
    lang: str = "hi"
    history: list = []


def build(r: Req):
    lang = ("Hindi (Roman or Devanagari, match the user)"
            if r.lang == "hi" else "English")
    msgs = [{"role": "system",
             "content": "You are CodeMitra, an expert coding assistant. "
                        f"Reply in {lang}. Be concise and practical. "
                        "Put code in markdown code blocks."}]
    for h in r.history[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": r.message})
    return msgs


def order():
    out = [state["model"]] if state["model"] else []
    return out + [m for m in PREFERRED if m not in out]


@app.get("/")
def home():
    return {"status": "ok", "model": state["model"]}


@app.post("/chat/stream")
def chat_stream(r: Req):
    msgs = build(r)

    def gen():
        errors = []
        for m in order():
            try:
                stream = client.chat.completions.create(
                    model=m, messages=msgs, max_tokens=1200, stream=True)
                got = False
                for ch in stream:
                    if not ch.choices:
                        continue
                    t = ch.choices[0].delta.content
                    if t:
                        got = True
                        yield t
                if got:
                    state["model"] = m
                    return
                errors.append(f"{m}: empty")
            except Exception as e:
                if state["model"] == m:
                    state["model"] = None
                errors.append(f"{m}: {str(e)[:60]}")
        yield "Sab models fail hue:\n" + "\n".join(errors)

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
