import platform
from pathlib import Path
from statistics import median

import torch
from generate import generate_text, load_model


PROMPTS = [
    "ROMEO:",
    "JULIET:",
    "Once upon a time",
    "To be or not to be",
    "KING:",
]
WARMUP_PROMPT = "ROMEO:"
MAX_NEW_TOKENS = 80


def average(values):
    return sum(values) / len(values) if values else 0


def seconds(milliseconds):
    return milliseconds / 1000


def main():
    model, stoi, itos, device = load_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dataset_path = Path("tiny_shakespeare.txt")
    dataset_chars = len(dataset_path.read_text(encoding="utf-8")) if dataset_path.exists() else 0

    latencies = []
    chars_per_second = []

    generate_text(
        model=model,
        prompt=WARMUP_PROMPT,
        stoi=stoi,
        itos=itos,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print("MiniGPT Benchmark")
    print("=================")
    print(f"Device: {device}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torch CPU threads: {torch.get_num_threads()}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Vocabulary size: {len(stoi):,}")
    print(f"Dataset characters: {dataset_chars:,}")
    print(f"Prompts tested: {len(PROMPTS)}")
    print(f"Generated chars per prompt: {MAX_NEW_TOKENS}")
    print("Warmup generations: 1")
    print()

    for prompt in PROMPTS:
        text, metrics = generate_text(
            model=model,
            prompt=prompt,
            stoi=stoi,
            itos=itos,
            device=device,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        latencies.append(metrics["latency_ms"])
        chars_per_second.append(metrics["chars_per_second"])

        print(f"Prompt: {prompt!r}")
        print(f"Latency: {seconds(metrics['latency_ms']):.2f} s")
        print(f"Chars/sec: {metrics['chars_per_second']}")
        print(f"Generated chars: {metrics['generated_chars']}")
        print(f"Total chars: {metrics['total_chars']}")
        print(f"Sample: {text[:120]!r}")
        print()

    print("Summary")
    print("=======")
    print(f"Average latency: {seconds(average(latencies)):.2f} s")
    print(f"Median latency: {seconds(median(latencies)):.2f} s")
    print(f"Fastest latency: {seconds(min(latencies)):.2f} s")
    print(f"Slowest latency: {seconds(max(latencies)):.2f} s")
    print(f"Average chars/sec: {average(chars_per_second):.2f}")
    print(f"Median chars/sec: {median(chars_per_second):.2f}")


if __name__ == "__main__":
    main()
