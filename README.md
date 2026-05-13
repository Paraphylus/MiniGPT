# MiniGPT 3

This is a small character-level Transformer LLM project made from scratch in pyTorch and trained on the Tiny Shakespeare dataset.

It includes:
- model code (`minigpt.py`)
- training script (`train.py`)
- generation/inference script (`generate.py`)
- a simple FastAPI chat app (`app.py` + `static/index.html`)

The app runs on CPU by default and works well for local experimentation(all recorded metrics are from an i7 12th gen CPU only).

## Performance Snapshot

Measured locally on a Lenovo Yoga 14s CPU setup after the generation-path
optimizations in this repo. CPU timings can vary between runs, so use
`python benchmark.py` for the latest number on your machine.

- Machine: **Lenovo Yoga 14s**
- Environment: **Windows 11**, **Python 3.13.7**, **PyTorch 2.11.0+cpu**, **10 torch CPU threads**
- Device: **CPU** (`torch.cuda.is_available() == False` in this environment)
- Model size: **7,541,760 parameters**
- Vocabulary size: **65 characters**
- Training corpus size: **1,115,394 characters** (Tiny Shakespeare)
- Benchmark script: **5 prompts**, **80 generated chars each**, **1 warmup generation**
- Benchmark result: **0.71 s** average latency, **0.68 s** median latency, **113.99 chars/sec** average, **116.88 chars/sec** median
- Fastest/slowest prompt latency in that run: **0.59 s** / **0.83 s**
- Sample CPU generation run: **220 generated chars** in **3.01 s** (**72.99 chars/sec**)
- Docker image build (no cache, CPU torch): **~287 s** on the shown setup

GPU inference should be faster, but these numbers do not include a GPU
benchmark because this environment is using CPU-only PyTorch.

## Optimization Notes

Generation was optimized to reduce CPU overhead during autoregressive decoding:

- Switched generation from `torch.no_grad()` to `torch.inference_mode()` for lower inference overhead.
- Preallocated the token buffer instead of repeatedly growing it with `torch.cat()`.
- Cached the causal attention mask inside the model instead of rebuilding it every forward pass.
- Added a `return_last_logits` path so generation only projects the newest token through the output head.
- Improved `benchmark.py` with one warmup generation plus average/median latency and throughput reporting.

Measured improvement on the Lenovo Yoga 14s CPU setup:

- Original README benchmark: **0.86 s** average latency, **93.60 chars/sec**
- Updated benchmark run: **0.71 s** average latency, **113.99 chars/sec**
- Latency improvement: **0.15 s faster** per 80-character benchmark generation, about **17% lower latency**
- Throughput improvement: **20.39 more chars/sec**, about **22% higher throughput**
- Original 220-character sample run: **6.94 s**, **31.71 chars/sec**
- Updated 220-character sample run: **3.01 s**, **72.99 chars/sec**

## What It Does

- Loads a pretrained checkpoint: `MiniGPT_Tiny_Shakespeare.pth`
- Accepts a prompt (for example: `ROMEO:`)
- Generates Shakespeare-style continuation text
- Shows basic runtime metrics (latency, chars/sec, request count)

## Project Structure

```text
.
|- app.py
|- generate.py
|- minigpt.py
|- train.py
|- data.py
|- benchmark.py
|- static/
|  \- index.html
|- MiniGPT_Tiny_Shakespeare.pth
|- tiny_shakespeare.txt
|- Dockerfile
|- docker-compose.yml
|- requirements.txt
```

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

3. Start the API server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

4. Open:

```text
http://localhost:8000
```

## Run with Docker

Build and run with Compose:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000
```

If you want a clean rebuild:

```bash
docker compose build --no-cache
docker compose up
```

## Notes

- The container is configured for CPU-only PyTorch.
- First Docker build may take a few minutes because PyTorch CPU wheels are large.
- Generation length is adjustable in the UI (`Chars`, up to 500).

## Quick API Example

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"ROMEO:\",\"max_new_tokens\":250}"
```

## License

Use this project for learning and experimentation.