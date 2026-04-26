from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generate import generate_text, load_model

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

model, stoi, itos, device = load_model()
stats = {"requests": 0, "total_latency_ms": 0.0}


class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 250
    temperature: float = 0.8
    top_k: int = 40


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    try:
        text, metrics = generate_text(
            model=model,
            prompt=request.prompt,
            stoi=stoi,
            itos=itos,
            device=device,
            max_new_tokens=max(1, min(request.max_new_tokens, 500)),
            temperature=max(0.01, min(request.temperature, 2.0)),
            top_k=max(1, min(request.top_k, len(stoi))),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stats["requests"] += 1
    stats["total_latency_ms"] += metrics["latency_ms"]
    metrics["request_count"] = stats["requests"]
    metrics["avg_latency_ms"] = round(stats["total_latency_ms"] / stats["requests"], 2)
    return {"text": text, "metrics": metrics}


@app.get("/metrics")
def metrics():
    avg_latency = 0.0
    if stats["requests"]:
        avg_latency = stats["total_latency_ms"] / stats["requests"]
    return {
        "requests": stats["requests"],
        "avg_latency_ms": round(avg_latency, 2),
        "device": device,
    }
