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
)
MODEL = os.environ.get("MODEL", "qwen/qwen2.5-coder-32b-instruct")

class Req(BaseModel):
    message: str
    lang: str = "hi"

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/chat")
def chat(r: Req):
    lang = "Hindi (Roman or Devanagari, match the user)" if r.lang == "hi" else "English"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"You are CodeMitra, an expert coding assistant. Reply in {lang}. Keep code in markdown code blocks."},
            {"role": "user", "content": r.message},
        ],
        max_tokens=1000,
    )
    return {"reply": resp.choices[0].message.content}
