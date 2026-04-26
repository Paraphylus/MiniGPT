# MiniGPT 3

This is a small character-level Transformer project trained on the Tiny Shakespeare dataset.

It includes:
- model code (`minigpt.py`)
- training script (`train.py`)
- generation/inference script (`generate.py`)
- a simple FastAPI chat app (`app.py` + `static/index.html`)

The app runs on CPU by default and works well for local experimentation.

## Performance Snapshot

Numbers below are from the runs/benchmarks shared for this project:

- Model size: **7.5M parameters**
- Training corpus size: **~1.1M characters** (Tiny Shakespeare)
- Benchmark result: **859 ms** average response time, **93.6 chars/sec**
- Sample CPU generation run: **220 output chars** in **6937.23 ms** (**31.71 chars/sec**)
- Docker image build (no cache, CPU torch): **~287 s** on the shown setup

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
