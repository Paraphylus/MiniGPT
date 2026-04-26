from pathlib import Path

from generate import generate_text, load_model


PROMPTS = [
    "ROMEO:",
    "JULIET:",
    "Once upon a time",
    "To be or not to be",
    "KING:",
]


def average(values):
    return sum(values) / len(values) if values else 0


def main():
    model, stoi, itos, device = load_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dataset_path = Path("tiny_shakespeare.txt")
    dataset_chars = len(dataset_path.read_text(encoding="utf-8")) if dataset_path.exists() else 0

    latencies = []
    chars_per_second = []

    print("MiniGPT Benchmark")
    print("=================")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Vocabulary size: {len(stoi):,}")
    print(f"Dataset characters: {dataset_chars:,}")
    print(f"Prompts tested: {len(PROMPTS)}")
    print()

    for prompt in PROMPTS:
        text, metrics = generate_text(
            model=model,
            prompt=prompt,
            stoi=stoi,
            itos=itos,
            device=device,
            max_new_tokens=80,
        )
        latencies.append(metrics["latency_ms"])
        chars_per_second.append(metrics["chars_per_second"])

        print(f"Prompt: {prompt!r}")
        print(f"Latency: {metrics['latency_ms']} ms")
        print(f"Chars/sec: {metrics['chars_per_second']}")
        print(f"Output chars: {metrics['total_chars']}")
        print(f"Sample: {text[:120]!r}")
        print()

    print("Summary")
    print("=======")
    print(f"Average latency: {average(latencies):.2f} ms")
    print(f"Fastest latency: {min(latencies):.2f} ms")
    print(f"Slowest latency: {max(latencies):.2f} ms")
    print(f"Average chars/sec: {average(chars_per_second):.2f}")


if __name__ == "__main__":
    main()
