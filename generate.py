import time
from pathlib import Path

import torch

from data import build_vocab, decode, encode, load_text
from minigpt import MiniGPT

BASE_DIR = Path(__file__).resolve().parent


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path="MiniGPT_Tiny_Shakespeare.pth", device=None):
    device = device or get_device()
    text_path = BASE_DIR / "tiny_shakespeare.txt"
    chars, stoi, itos = build_vocab(load_text(text_path) if text_path.exists() else None)
    model = MiniGPT(len(chars)).to(device)
    checkpoint_path = BASE_DIR / checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, stoi, itos, device


@torch.no_grad()
def generate_text(
    model,
    prompt,
    stoi,
    itos,
    device,
    max_new_tokens=250,
    temperature=0.8,
    top_k=40,
):
    started = time.perf_counter()
    idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)
    max_context = model.pos_emb.num_embeddings
    top_k = min(top_k, len(stoi))

    for _ in range(max_new_tokens):
        context = idx[:, -max_context:]
        logits = model(context)
        logits = logits[:, -1, :] / max(temperature, 0.01)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, top_k)
        next_token = topk_idx[
            torch.arange(topk_idx.size(0), device=device),
            torch.multinomial(topk_probs, 1).squeeze(1),
        ].unsqueeze(1)
        idx = torch.cat([idx, next_token], dim=1)

    text = decode(idx[0].tolist(), itos)
    latency_ms = (time.perf_counter() - started) * 1000
    return text, {
        "latency_ms": round(latency_ms, 2),
        "prompt_chars": len(prompt),
        "generated_chars": max_new_tokens,
        "total_chars": len(text),
        "chars_per_second": round(max_new_tokens / (latency_ms / 1000), 2),
        "device": device,
    }


if __name__ == "__main__":
    model, stoi, itos, device = load_model()
    text, metrics = generate_text(model, "ROMEO:", stoi, itos, device)
    print(text)
    print(metrics)
